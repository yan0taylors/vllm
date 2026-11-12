# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Edge-side prefix negotiation client (SSE gateway form).

The protocol shape is OpenAI-compatible so a gateway forwards it
transparently:

* request: POST to the control endpoint, body is the manifest shadow
  (``messages`` + ``stream`` only), headers carry ``X-Edge-Cloud-*`` and an
  optional Authorization Bearer. Image requests declare
  ``edge-cloud-prefix-v2`` and attach the ``X-Edge-Cloud-MM-ABI``
  fingerprint. The original messages, sampling parameters and image data
  never leave the edge.
* response: an SSE stream — the first chunk's delta.content embeds the
  ``edge_cloud_probe`` JSON control message; then empty delta frames;
  heartbeat comments; the final chunk is the standard OpenAI usage +
  ``[DONE]``. The gateway (which does not drop delta.content) can bill on
  it.

Usage consumption: a background task drains the SSE tail for the usage
frame and logs it edge-side (the data-plane ``LwdC2eNotify`` path remains
the accounting channel). The probe client reads its tunables from the
``LwdCoordinationConfig`` snapshot; the bearer token comes from the
``VLLM_LWD_CLOUD_API_KEY`` environment variable and is never logged.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from typing import Any

from vllm import envs
from vllm.config.lwd import LwdCoordinationConfig
from vllm.lwd_logger import lwd_logger
from vllm.v1.lwd_control.lwd_coordination.lwd_mm_identity import (
    mm_abi_header_value,
)
from vllm.v1.lwd_control.lwd_coordination.lwd_prefix import (
    HEADER_EDGE_ID,
    HEADER_MM_ABI,
    HEADER_PROTOCOL,
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_MM,
    LwdPrefixHasher,
    LwdPrefixManifest,
    LwdProbeResult,
)

# Upper bound of one SSE control message JSON in characters (keeps a
# malicious or broken response from ballooning memory).
_MAX_CONTROL_MESSAGE_CHARS = 16 * 1024

# Probe session connection limit: usage drains each hold one connection
# until the request finishes (about the inference duration). Sharing the
# pool with probes at aiohttp's default limit of 100 would queue new probes
# behind drains under load. 0 = unlimited; the real cap is the number of
# in-flight edge requests.
_LWD_PROBE_MAX_CONNS = 0


async def _iter_sse_data(
    content: AsyncIterable[bytes],
) -> AsyncIterator[str]:
    """Read SSE events, preserving multiline data and ignoring heartbeats."""
    data: list[str] = []
    size = 0
    async for raw_line in content:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                payload = "\n".join(data)
                data.clear()
                size = 0
                yield payload
        elif line == "data" or line.startswith("data:"):
            value = line.partition(":")[2].removeprefix(" ")
            size += len(value) + 1
            if size > _MAX_CONTROL_MESSAGE_CHARS:
                raise ValueError(
                    "edge-cloud SSE event exceeds the control message limit"
                )
            data.append(value)
    if data:
        yield "\n".join(data)


class LwdProbeRuntime:
    """Dedicated background thread for prefix probes: one persistent event
    loop plus one shared client session.

    Probes (HMAC chaining + HTTP round trips) run off the engine's thread,
    concurrent requests interleave on the single loop, and the connection
    pool stays resident instead of a new TCP connection per request.
    ``submit`` schedules a coroutine on the loop from any thread.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._session: Any = None
        self._thread = threading.Thread(
            target=self._run, name="lwd-prefix-probe", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def stop(self) -> None:
        """Stop the loop and close the session (recycles an old instance on
        the submit-failure rebuild path, preventing thread/connection
        leaks). Non-blocking: the cleanup runs asynchronously on the loop."""

        def _cleanup() -> None:
            if self._session is not None:
                asyncio.ensure_future(self._session.close())
            self._loop.call_later(1.0, self._loop.stop)

        # The loop may already be dead; cleanup is best-effort.
        with contextlib.suppress(Exception):
            self._loop.call_soon_threadsafe(_cleanup)

    async def ensure_session(self) -> Any:
        if self._session is None:
            import aiohttp

            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=_LWD_PROBE_MAX_CONNS)
            )
        return self._session

    def submit(self, coro: Any) -> asyncio.Future:
        """Schedule a coroutine on the resident loop; the returned future is
        awaited or cancelled from the calling thread."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)


class LwdEdgePrefixClient:
    """Edge admission client: HMAC manifest shadow body + SSE probe parsing.

    Tunables (consumer id, connect timeout, probe timeout) come from the
    ``LwdCoordinationConfig``; ``control_url`` is the endpoint of one cloud
    (the single-value form's URL when not given explicitly)."""

    def __init__(
        self,
        coordination: LwdCoordinationConfig,
        block_size: int,
        tenant_key: bytes,
        edge_id: int = 0,
        control_url: str | None = None,
        processor_fingerprint: bytes | None = None,
    ) -> None:
        if control_url is None:
            control_url = coordination.control_url
        if control_url is None:
            raise ValueError(
                "control_url is required: pass the per-cloud URL or set the "
                "single-value control_url"
            )
        self._control_url = control_url
        self._hasher = LwdPrefixHasher(
            tenant_key, block_size, processor_fingerprint
        )
        # MM-ABI header value for image requests (rendered from the local
        # processor fingerprint); without a fingerprint v2 cannot speak.
        self._mm_abi_header: str | None = (
            mm_abi_header_value(processor_fingerprint)
            if processor_fingerprint is not None
            else None
        )
        self._edge_id = edge_id
        self._consumer_id = coordination.consumer_id
        self._connect_timeout = coordination.connect_timeout
        # End-to-end deadline of the whole probe (connect + first frame).
        # The cloud-side deadline is this minus a headroom, so normally the
        # cloud answers first; this timeout only fires when the cloud or
        # gateway truly never answers.
        self._probe_timeout = coordination.probe_timeout
        # Background drain tasks of the probe streams (asyncio holds only
        # weak references, so this set must hold the strong ones; a set
        # rather than a req_id-indexed dict — see _start_drain).
        self._drain_tasks: set[Any] = set()
        # Gateway bearer token (sensitive, never logged): read from the
        # environment; None = not sent.
        api_key = envs.VLLM_LWD_CLOUD_API_KEY
        if api_key is not None and (
            not api_key
            or not api_key.isascii()
            or not api_key.isprintable()
            or any(c.isspace() for c in api_key)
        ):
            raise ValueError(
                "VLLM_LWD_CLOUD_API_KEY must contain a non-empty ASCII "
                "API token without whitespace"
            )
        self._api_key: str | None = api_key

    @property
    def block_size(self) -> int:
        return self._hasher.block_size

    @property
    def control_url(self) -> str:
        return self._control_url

    def build_manifest(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        media_items: Sequence[Any] = (),
    ) -> LwdPrefixManifest:
        """Build the manifest with the tenant-key chained HMAC (the original
        prompt never leaves the edge)."""
        return self._hasher.build_manifest(
            request_id, prompt_token_ids, media_items=media_items
        )

    def build_control_request(
        self,
        manifest: LwdPrefixManifest,
        model: str | None = None,
        *,
        multimodal: bool = False,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Whitelisted shadow body: manifest/stream only, sampling parameters
        never cross the network.

        The required control headers (including
        ``X-Edge-Cloud-Prompt-Tokens``) all come from
        ``LwdPrefixManifest.to_headers()``; this only adds the edge identity
        and gateway-related headers. ``model`` is passed through optionally
        (the gateway picks its channel by it). ``multimodal`` is decided by
        the caller from the media items used at chaining time:
        media-covered blocks use the 0x04/0x05 domains, so the protocol must
        declare v2 and carry the MM-ABI fingerprint, otherwise the cloud
        cannot identify the chaining convention."""
        headers = manifest.to_headers()
        if multimodal:
            if self._mm_abi_header is None:
                raise ValueError(
                    "multimodal probe needs a processor fingerprint for "
                    f"{HEADER_MM_ABI}"
                )
            headers[HEADER_PROTOCOL] = PROTOCOL_VERSION_MM
            headers[HEADER_MM_ABI] = self._mm_abi_header
        headers[HEADER_EDGE_ID] = str(self._edge_id)
        headers["Accept"] = "text/event-stream"
        if self._consumer_id:
            headers["X-Mse-Consumer"] = self._consumer_id
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        body: dict[str, Any] = {
            "messages": manifest.to_messages(),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if model is not None:
            body["model"] = model
        return headers, body

    async def negotiate(
        self,
        session: Any,
        request_id: str,
        prompt_token_ids: list[int],
        manifest: LwdPrefixManifest | None = None,
        media_items: Sequence[Any] = (),
        model: str | None = None,
    ) -> LwdProbeResult:
        """POST the probe and parse the Probe from the SSE delta.content;
        failures raise (fail-closed).

        The hit is consumed by the caller (on a hit it advances
        request.num_computed_tokens so the edge only embeds/sends the tail).
        A manifest built by the caller can be passed to reuse the chain
        (avoiding a second O(N) chaining for one probe); by default it is
        built here (media_items must match the caller's chaining convention,
        so the default path cannot miss the multimodal identity).

        The whole span (connect + first frame) is clamped by
        ``probe_timeout``. The cloud-side handler deadline is
        ``probe_timeout - 5``, so the normal slow path is the **cloud
        answering first** (200 or 503); this timeout only fires when the
        cloud or gateway truly never answers. On abandonment the billing
        stream is handed to the drain path (see ``_post_and_read_probe``)."""
        if manifest is None:
            manifest = self.build_manifest(
                request_id, prompt_token_ids, media_items
            )
        # Non-empty media identity <=> the chain mixed in the 0x04/0x05
        # domains <=> must speak v2.
        multimodal = bool(media_items)
        headers, body = self.build_control_request(
            manifest, model=model, multimodal=multimodal
        )
        try:
            response, sse_data, probe = await asyncio.wait_for(
                self._post_and_read_probe(
                    session, headers, body,
                    request_id=request_id,
                    prompt_token_ids=prompt_token_ids,
                    multimodal=multimodal,
                ),
                self._probe_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"prefix probe timed out after {self._probe_timeout}s "
                f"url={self._control_url}"
            ) from exc
        # The probe is in hand, but the stream must NOT be closed here: the
        # cloud-side handler waits for the inference to finish before it
        # sends the usage tail chunk (up to 600s), and the real token usage
        # lands in that frame.
        self._start_drain(request_id, response, sse_data)

        lwd_logger.info(
            "[edge-client] negotiate req=%s hit_tokens=%d cloud_id=%s",
            request_id, probe.hit_tokens, probe.cloud_id,
        )
        return probe

    async def _post_and_read_probe(
        self,
        session: Any,
        headers: dict[str, str],
        body: dict[str, Any],
        *,
        request_id: str,
        prompt_token_ids: list[int],
        multimodal: bool,
    ) -> tuple[Any, AsyncIterator[str], LwdProbeResult]:
        """Send the shadow body and read the Probe; on abandonment decide
        "hand to drain or close".

        Once 200 is in hand, the cloud-side handler is already attached to
        this request's usage. Whatever the abandonment reason afterwards —
        timeout, cancellation or a failed echo check — the stream must NOT
        be closed: an early edge-side disconnect makes the gateway record
        client_gone/context canceled and this request's real usage is lost
        forever."""
        import aiohttp

        timeout = aiohttp.ClientTimeout(
            total=None, connect=self._connect_timeout
        )
        response = await session.post(
            self._control_url, json=body, headers=headers, timeout=timeout
        )
        # Non-None <=> 200 is in hand and the stream iterator is held (the
        # abandonment path decides drain vs close by this).
        sse_data: AsyncIterator[str] | None = None
        try:
            if response.status != 200:
                detail = await self._read_error_body(response)
                raise RuntimeError(
                    f"prefix negotiation HTTP {response.status} "
                    f"url={self._control_url} body={detail}"
                )
            # The drain task shares this iterator: the frames after the
            # probe frame are read on by the task.
            sse_data = _iter_sse_data(response.content)
            probe = await self._read_probe_from_sse(
                sse_data, multimodal=multimodal
            )
            if probe.request_id != request_id:
                raise RuntimeError("cloud returned a different request ID")
            # Block granularity is not required to match: with hybrid models
            # (Mamba/SWA) the cloud hash granularity can be smaller than the
            # edge KV block granularity; the caller re-learns the chain from
            # probe.block_size.
            if probe.hit_tokens > len(prompt_token_ids):
                raise RuntimeError("cloud prefix hit exceeds the prompt length")
        except BaseException:
            if sse_data is not None:
                self._start_drain(request_id, response, sse_data)
            else:
                response.close()
            raise
        assert sse_data is not None  # the success path always holds the stream
        return response, sse_data, probe

    def _start_drain(
        self,
        request_id: str,
        response: Any,
        sse_data: AsyncIterator[str],
    ) -> Any:
        """Hand the frames after the probe frame to a background drain task
        (same loop, off the negotiate return path).

        The registration must be a **set**, not a req_id-indexed dict:
        asyncio holds only weak references, and a coroutine parked on a
        network read has no pending callback, so this registration is its
        only strong reference. Re-submitting the same req_id (re-add/retry)
        would overwrite the previous stream's only strong reference in a
        dict, letting the GC drop that stream before its usage is read —
        the gateway records client_gone and that usage is lost. The task is
        returned so tests can assert the registration."""
        task = asyncio.create_task(
            self._drain_stream(request_id, response, sse_data)
        )
        self._drain_tasks.add(task)
        task.add_done_callback(self._drain_tasks.discard)
        return task

    async def _drain_stream(
        self,
        request_id: str,
        response: Any,
        sse_data: AsyncIterator[str],
    ) -> None:
        """Drain the probe stream to the usage tail chunk: the receive path
        that lets the gateway bill.

        This side only keeps a log record (accounting follows the cloud's
        C2E path back to the edge), but the stream must be read to the end:
        only then does the cloud-side handler reach the tail chunk + DONE.
        The cloud-side usage wait is capped at 600s and still ends the
        stream with DONE (without a usage frame), so the task count is
        bounded (= in-flight requests) and needs no extra timeout. A missing
        usage frame is logged as a warning — the gateway has no upstream
        usage to bill for this request."""
        usage: dict | None = None
        try:
            async for payload in sse_data:
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage") is not None:
                    usage = chunk["usage"]
            if usage is None:
                lwd_logger.warning(
                    "[edge-client] usage missing in probe stream req=%s",
                    request_id,
                )
                return
            details = usage.get("prompt_tokens_details") or {}
            lwd_logger.info(
                "[edge-client] usage req=%s prompt=%s completion=%s "
                "total=%s cached=%s",
                request_id,
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                usage.get("total_tokens"),
                details.get("cached_tokens"),
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001  a failed drain only loses this
            # gateway-side billing record; inference is unaffected
            lwd_logger.warning(
                "[edge-client] probe stream drain failed req=%s",
                request_id,
                exc_info=True,
            )
        finally:
            try:
                response.close()
            except Exception:  # noqa: BLE001  a failed close needs no raise
                lwd_logger.debug(
                    "[edge-client] probe stream close failed req=%s",
                    request_id,
                    exc_info=True,
                )

    @staticmethod
    async def _read_error_body(response: Any, limit: int = 512) -> str:
        """Read the error body on a non-200 (truncated + whitespace folded:
        gateways often return a full HTML page)."""
        try:
            raw = await response.read(limit)
            return " ".join(raw.decode("utf-8", "replace").split())
        except Exception as exc:  # noqa: BLE001  diagnostics never re-raise
            return f"<unreadable: {type(exc).__name__}>"

    async def _read_probe_from_sse(
        self, sse_data: AsyncIterator[str], *, multimodal: bool
    ) -> LwdProbeResult:
        """Parse the first edge_cloud_probe control message from the SSE
        stream (delta.content accumulation).

        The echo is always validated (fail-closed): the cloud-acked protocol
        must be the one this side spoke — a v2 image request silently
        degraded to v1 means the media-covered blocks were accounted with
        the text convention and hits would misalign; v2 must also echo the
        matching MM-ABI fingerprint, and a text probe must not carry it. A
        usage or error frame before the probe means the stream order is
        wrong: raise instead of treating it as a truncated handshake. The
        gateway may split delta.content across frames, so the object is
        judged only once assembled."""
        expected_protocol = (
            PROTOCOL_VERSION_MM if multimodal else PROTOCOL_VERSION
        )
        content_buf = ""
        async for payload in sse_data:
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if not isinstance(chunk, dict):
                raise ValueError("cloud returned a non-object SSE chunk")
            if chunk.get("usage") is not None or chunk.get("error") is not None:
                raise RuntimeError(
                    "cloud returned usage or an error before the probe"
                )
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content_buf += delta.get("content") or ""
            if not content_buf:
                continue
            try:
                message = json.loads(content_buf)
            except json.JSONDecodeError:
                continue  # a split that is not complete yet, keep reading
            if not isinstance(message, dict):
                raise ValueError("cloud probe must contain a JSON object")
            if "protocol" not in message:
                continue  # split right after the first key: read one more frame
            if message["protocol"] != expected_protocol:
                raise RuntimeError(
                    f"cloud did not acknowledge {expected_protocol} "
                    f"(returned {message['protocol']!r})"
                )
            mm_abi = message.get("mm_abi")
            if multimodal:
                if mm_abi != self._mm_abi_header:
                    raise RuntimeError(
                        "cloud MM-ABI fingerprint does not match the local "
                        "processor fingerprint"
                    )
            elif mm_abi is not None:
                raise RuntimeError("cloud returned MM-ABI for a text-only probe")
            return LwdProbeResult.from_control_message(message)
        raise RuntimeError("edge-cloud probe control message not found in SSE stream")
