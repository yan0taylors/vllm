# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cloud-side HTTP coordination plane: probe and usage over an SSE gateway.

Adapted from the source branch's ``lwd_cloud_control.py`` (FastAPI parent
process + bridge + child-process processor over multiprocessing queues) to
the target engine structure: each cloud DP engine owns its whole
coordination endpoint. The engine process hosts the uvicorn thread and the
``LwdCloudControlBridge``; probe commands cross from the HTTP event loop to
the engine step loop over one thread-safe command queue (woken by a
WAKEUP on the engine input queue, the same primitive the shutdown signal
path uses), and answers/usages cross back over one event queue to the
bridge's dispatch thread. Answers therefore always run on the engine
thread: the block pool is only ever touched there.

The per-engine hosting replaces the source branch's parent-hosted fanout
(one server, one command queue per cloud DP engine): rank r binds
``listen_port + r`` (rank 0 binds exactly ``listen_port``), mirroring the
per-DP ctrl_port convention of the topology, and each engine answers
probes from its own cache.

SSE event order (gateway-compatible OpenAI streaming, unchanged from the
source implementation):

1. the probe result as a JSON control message inside ``delta.content``
   (gateway DTO conversion never drops the message body);
2. an empty ``delta`` frame that flushes the gateway's last-chunk buffer
   (NewAPI and friends);
3. an SSE comment heartbeat every 10 seconds while usage is pending, so
   upstream idle timeouts do not fire;
4. a standard OpenAI usage chunk plus ``[DONE]`` once inference settles —
   and when no real usage can be obtained, a comment plus ``[DONE]`` only:
   a fabricated zero-usage frame would let the gateway bill zero tokens.

Protocol gating: text rides ``edge-cloud-prefix-v1``; media rides
``edge-cloud-prefix-v2`` and must carry the ``X-Edge-Cloud-MM-ABI``
fingerprint (the link ABI of the media-covered blocks). The two stacks
never degrade into each other. ``X-Edge-Cloud-*`` response headers stay for
direct debugging and gateway observability; they do not take part in the
negotiation.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import json
import queue
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.lwd_logger import lwd_logger
from vllm.v1.lwd_control.lwd_coordination.lwd_mm_identity import (
    compute_processor_fingerprint,
    mm_abi_header_value,
    validate_mm_abi_header,
)
from vllm.v1.lwd_control.lwd_coordination.lwd_prefix import (
    HEADER_EDGE_ID,
    HEADER_MM_ABI,
    HEADER_PROTOCOL,
    HEADER_REQUEST_ID,
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_MM,
    LwdPrefixManifest,
    LwdProbeResult,
    LwdUsage,
)
from vllm.v1.lwd_control.lwd_coordination.lwd_request_namespace import (
    lwd_parse_request_edge,
    lwd_wrap_request_id,
)

# SSE heartbeat interval (seconds): must stay below the gateway's
# stream_idle_timeout with a comfortable margin.
_LWD_USAGE_HEARTBEAT_INTERVAL = 10.0
# Total wait for usage (seconds): covers a full inference; the stream ends
# without a usage frame after it (never a fabricated zero).
_LWD_USAGE_WAIT_TIMEOUT = 600.0
# Cloud probe cap = the edge's lwd_coordination.probe_timeout minus this
# headroom: the cloud must answer first (200 or 503), otherwise the edge may
# already have failed open while the cloud still waits for the engine loop,
# leaving the reservation and the billing stream hanging for nothing. On a
# 503 the edge fails open exactly as it does today.
_LWD_CLOUD_PROBE_HEADROOM_S = 5.0

_DISPATCH_THREAD_NAME = "lwd-ctrl-dispatch"


@dataclass(frozen=True)
class LwdProbeCommand:
    """One probe request handed from the HTTP loop to the engine thread."""

    manifest: LwdPrefixManifest
    # [Lwd][perf] enqueue moment on the parent-side monotonic clock. The
    # HTTP loop and the engine thread share one process, so the timestamps
    # are comparable; across nodes they are not.
    t_cmd: float


@dataclass(frozen=True)
class LwdProbeAnswer:
    """Probe outcome handed back from the engine thread to the HTTP loop."""

    request_id: str
    ok: bool
    result: LwdProbeResult | None = None
    error: str = ""


@dataclass(frozen=True)
class LwdUsageEvent:
    """Settled usage of one request, pushed by the engine at finish time."""

    request_id: str
    usage: LwdUsage


class LwdCloudControlBridge:
    """Connect the HTTP coroutines with the engine thread over two queues.

    Waiting uses one asyncio.Future per request, so an in-flight probe
    occupies no thread: the dispatch thread only moves events back onto the
    event loop with ``loop.call_soon_threadsafe`` and the handler coroutine
    awaits its future directly. ``_loop`` is captured inside ``probe()``
    (i.e. in a handler coroutine): that is the loop uvicorn actually runs,
    and taking it on the startup thread could deliver to a loop that never
    runs.

    Why not ``run_in_executor(None, waiter.get)`` for the usage wait: a
    usage wait lives until the request's inference settles (measured ~8s
    live); under the executor form one in-flight request pins one executor
    thread, the default pool cap ``min(32, cpu+4)`` saturates, and new
    probes cannot even queue to wait for a result (live evidence 8cfb9f499:
    busy_usage=32, exec_delay_ms=-1, probe total_ms=60000). Worse, a thread
    already running after ``wait_for`` timed out cannot be cancelled while
    its waiter has been removed from the dict, so every timeout leaks one
    thread permanently. Under the future form a timeout only cancels one
    coroutine: in-flight count and thread count decouple.
    """

    def __init__(
        self,
        command_queue: Any,
        event_queue: Any,
        wakeup: Callable[[], None] | None = None,
    ) -> None:
        self._command_queue = command_queue
        self._event_queue = event_queue
        self._wakeup = wakeup
        self._probe_futures: dict[str, asyncio.Future] = {}
        self._usage_futures: dict[str, asyncio.Future] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()
        self._closed = False
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_events,
            name=_DISPATCH_THREAD_NAME,
            daemon=True,
        )

    def start(self) -> None:
        self._dispatch_thread.start()

    def close(self) -> None:
        """Stop the dispatch thread and fail the futures still waiting.

        Registered as an atexit hook by ``lwd_start_cloud_control_server``:
        uvicorn runs on a daemon thread with no shutdown channel, so without
        this hook the dispatch thread could only be hard-killed at exit.
        """
        self._closed = True
        with self._lock:
            pending = [
                *self._probe_futures.values(), *self._usage_futures.values()
            ]
            self._probe_futures.clear()
            self._usage_futures.clear()
            loop = self._loop
        alive = [future for future in pending if not future.done()]
        if loop is None or not alive:
            return
        error = RuntimeError("cloud control plane stopped")

        def _fail_pending() -> None:
            for future in alive:
                if future.done():
                    continue
                future.set_exception(error)
                # A usage future is often left unclaimed (its handler ended
                # with the edge disconnecting): take the exception once to
                # clear the "was never retrieved" log at loop shutdown.
                future.exception()

        try:
            loop.call_soon_threadsafe(_fail_pending)
        except RuntimeError:
            # The loop is already stopped: nobody is waiting on these
            # futures anymore, dropping them is enough.
            lwd_logger.info(
                "[coord] close: loop already stopped, drop %d waiter(s)",
                len(alive),
            )

    async def probe(
        self, manifest: LwdPrefixManifest, timeout: float = 20.0
    ) -> LwdProbeResult:
        """Send a probe command and await its answer; a timeout raises
        TimeoutError (the handler turns it into a 503).

        Both the probe and the usage future are registered here: the usage
        may be pushed by the engine at any time after the probe, and a late
        registration would lose it. The waiter registers before sending the
        command, so no event can land before the registration.

        The timeout is derived by the caller from the edge's
        ``lwd_coordination.probe_timeout`` minus headroom, so the cloud
        answers first (200 or 503) instead of the edge giving up: under
        load the engine step can stretch, a probe command may sit until the
        cap, and in that case its result only serves observability.

        A repeated req_id is a normal path (the same id re-added by a client
        retry or an abort-then-resubmit), so a duplicate supersedes instead
        of being rejected: the old usage future resolves with None (that
        generation's usage belongs to the new stream; the old one ends
        without a usage frame) and the old probe future fails (the old
        handler answers 503; no billing was issued because no 200 left yet).

        Cloud-internal DP fan-out of the source branch is gone with the
        parent-hosted server: every cloud DP engine answers its own
        endpoint, so the bridge holds exactly one command queue.
        """
        request_id = manifest.request_id
        loop = asyncio.get_running_loop()
        self._loop = loop
        with self._lock:
            stale_probe = self._probe_futures.pop(request_id, None)
            stale_usage = self._usage_futures.pop(request_id, None)
            probe_future: asyncio.Future = loop.create_future()
            usage_future: asyncio.Future = loop.create_future()
            self._probe_futures[request_id] = probe_future
            self._usage_futures[request_id] = usage_future
        if stale_probe is not None or stale_usage is not None:
            lwd_logger.warning(
                "[coord] duplicate control request req=%s: superseding the "
                "previous registration",
                request_id,
            )
            if stale_usage is not None and not stale_usage.done():
                stale_usage.set_result(None)
            if stale_probe is not None and not stale_probe.done():
                stale_probe.set_exception(RuntimeError(
                    f"control request {request_id} superseded by a newer probe"))
        self._command_queue.put(LwdProbeCommand(manifest, time.monotonic()))
        if self._wakeup is not None:
            # A WAKEUP on the engine input queue lets an idle engine reach
            # its next coordination poll immediately; engines already
            # stepping consume the command at the next poll anyway.
            # A shutdown race is harmless; the command remains visible to
            # any engine still polling.
            with contextlib.suppress(Exception):
                self._wakeup()
        try:
            return await asyncio.wait_for(probe_future, timeout)
        except asyncio.TimeoutError:
            # Python 3.10's asyncio timeout is not the built-in
            # TimeoutError; normalize it so the handler's quiet 503 path
            # is one exception type on every supported interpreter.
            failure: BaseException = TimeoutError(
                f"prefix probe timed out after {timeout}s"
            )
        except BaseException as exc:
            failure = exc
        # The probe failed (timeout/cancel/cloud error): the usage future
        # is removed with it — the handler will never wait for a usage,
        # and leaving it would be an unclaimed dict entry (the success
        # path removes it in _deliver_event).
        with self._lock:
            if self._probe_futures.get(request_id) is probe_future:
                self._probe_futures.pop(request_id, None)
            if self._usage_futures.get(request_id) is usage_future:
                self._usage_futures.pop(request_id, None)
        raise failure

    def release(self, request_id: str) -> None:
        """Remove this control request's registration (idempotent).

        Must be called when the SSE generator ends: when the client
        disconnects right after the startup message, ``wait_usage`` was
        never called and the dict entries would otherwise wait for the
        process-wide ``close()``. A late settlement then takes the orphan
        path and is dropped.
        """
        with self._lock:
            self._probe_futures.pop(request_id, None)
            self._usage_futures.pop(request_id, None)

    async def wait_usage(
        self, request_id: str, timeout: float = _LWD_USAGE_WAIT_TIMEOUT
    ) -> LwdUsage | None:
        """Await the usage event of the request; None on timeout or when no
        control request is registered (the handler ends the stream without
        a usage frame). The future was registered by ``probe()``, so an
        early settlement is not lost."""
        with self._lock:
            future = self._usage_futures.get(request_id)
        if future is None:
            lwd_logger.warning(
                "[coord] usage wait without control request req=%s", request_id
            )
            return None
        try:
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            lwd_logger.warning("[coord] usage wait timeout req=%s", request_id)
            return None
        finally:
            with self._lock:
                # Remove only our own entry: a newer probe may have
                # superseded the registration in the meantime.
                if self._usage_futures.get(request_id) is future:
                    self._usage_futures.pop(request_id, None)

    def publish_usage(self, request_id: str, usage: LwdUsage) -> None:
        """Push a settled usage event (engine thread; consumed by the
        dispatch thread and delivered to the waiting SSE handler)."""
        try:
            self._event_queue.put(LwdUsageEvent(request_id, usage))
        except Exception:  # noqa: BLE001
            lwd_logger.exception(
                "[coord] publish_usage failed req=%s", request_id)

    def _dispatch_events(self) -> None:
        """One resident thread: move events onto the event loop; it never
        touches a future itself (loop affinity belongs to the loop)."""
        while not self._closed:
            try:
                event = self._event_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            request_id = event.request_id
            with self._lock:
                table = (
                    self._probe_futures
                    if isinstance(event, LwdProbeAnswer)
                    else self._usage_futures
                )
                registered = request_id in table
                loop = self._loop
            # One line per event: debug, matching the hot-path log policy.
            lwd_logger.debug(
                "[coord] dispatch got event kind=%s req=%s ok=%s",
                type(event).__name__,
                request_id,
                getattr(event, "ok", "-"),
            )
            if not registered or loop is None:
                # The waiter already timed out or disconnected (late
                # settlement), or the event precedes the first probe (a
                # data-plane request can finish without one): drop it.
                lwd_logger.warning(
                    "[coord] orphan %s event req=%s (no waiter/loop)",
                    type(event).__name__,
                    request_id,
                )
                continue
            try:
                loop.call_soon_threadsafe(self._deliver_event, event)
            except RuntimeError:
                # The loop is closed (uvicorn shutdown racing atexit):
                # no coroutine can still receive a settlement, and pumping
                # on would log one traceback per event: stop the thread,
                # symmetric to the guard in close().
                lwd_logger.info(
                    "[coord] dispatch stopping: loop closed (%d control "
                    "request(s) abandoned)",
                    len(self._probe_futures) + len(self._usage_futures),
                )
                return

    def _deliver_event(self, event: Any) -> None:
        """Settle a future on the event loop thread (the only place that
        calls set_result/set_exception)."""
        request_id = event.request_id
        if isinstance(event, LwdProbeAnswer):
            with self._lock:
                future = self._probe_futures.pop(request_id, None)
            if future is None or future.done():
                return  # Timed out or cancelled: the edge already failed open.
            if event.ok:
                future.set_result(event.result)
            else:
                future.set_exception(
                    RuntimeError(event.error or "probe failed"))
            return
        if isinstance(event, LwdUsageEvent):
            # The settlement carries no generation marker, so it can only be
            # delivered to the currently registered future. "One rid has one
            # in-flight attempt" is a whole-stack invariant (the edge nonce
            # and abort tombstones, the single-reservation overwrite and the
            # gate window maintain it), not something the cloud deduplicates
            # gently: reason about concurrency from those three places.
            with self._lock:
                future = self._usage_futures.get(request_id)
            if future is None or future.done():
                lwd_logger.warning(
                    "[coord] usage event arrived too late req=%s", request_id
                )
                return
            future.set_result(event.usage)


class LwdCloudControlProcessor:
    """Engine-thread side: consume probe commands non-blockingly.

    The command source is a thread-safe queue filled by the HTTP loop; the
    engine step loop drains it after every input-queue poll (and a WAKEUP
    posted with each command makes an idle engine reach that poll). The
    query itself runs here, on the engine thread: the block pool is not
    thread-safe and the read-only hit lookup must not run anywhere else.
    """

    def __init__(self, command_queue: Any, event_queue: Any) -> None:
        self._command_queue = command_queue
        self._event_queue = event_queue

    def poll(self, query: Any) -> int:
        """Consume pending commands; returns how many ran this round.

        A count above one means probes waited for the engine step head and
        were consumed in a batch — direct evidence that probe service is
        bounded by the cloud step rate.
        """
        consumed = 0
        while True:
            try:
                command = self._command_queue.get_nowait()
            except queue.Empty:
                return consumed
            consumed += 1
            lwd_logger.debug(
                "[coord] poll consumed command type=%s", type(command).__name__
            )
            if not isinstance(command, LwdProbeCommand):
                lwd_logger.warning(
                    "[coord] drop unknown command %s", type(command).__name__
                )
                continue
            manifest = command.manifest
            # [Lwd][perf] cmd_wait_ms = time from the HTTP enqueue to the
            # engine taking the command (queueing + WAKEUP + step head).
            # Close to the edge-visible http_ms means the engine step head
            # was the bottleneck; near zero means the latency sat on the
            # answer/settlement side.
            t_take = time.monotonic()
            cmd_wait_ms = (t_take - command.t_cmd) * 1000.0
            try:
                result = query.probe(manifest)
                lwd_logger.info(
                    "[coord][perf] probe_consumed req=%s cmd_wait_ms=%.1f "
                    "probe_ms=%.1f",
                    manifest.request_id,
                    cmd_wait_ms,
                    (time.monotonic() - t_take) * 1000.0,
                )
                event: Any = LwdProbeAnswer(
                    request_id=manifest.request_id,
                    ok=True,
                    result=result,
                )
            except Exception as exc:  # noqa: BLE001
                lwd_logger.exception("[coord] probe failed")
                event = LwdProbeAnswer(
                    request_id=manifest.request_id,
                    ok=False,
                    error=str(exc),
                )
            self._event_queue.put(event)


def _negotiate_lwd_protocol_headers(
    headers: Mapping[str, str],
    expected_mm_abi: str | None,
    *,
    enforce_mm_abi_match: bool,
) -> str | None:
    """Return the validated MM-ABI header value of a v2 request; None for
    v1.

    v1 keeps its original semantics; a v1 request carrying an MM-ABI header
    and a v2 request missing or malformed MM-ABI are both rejected — the two
    protocol stacks must not degrade into each other (a media link counted
    under the text rules would silently misplace its hits). With
    ``enforce_mm_abi_match`` on, a v2 header value must also equal the
    locally rendered fingerprint: model identifiers of heterogeneous
    edge/cloud images usually differ, so this is off by default and the
    check is existence/format only.
    """
    normalized = {k.lower(): v for k, v in headers.items()}
    protocol = normalized.get(HEADER_PROTOCOL.lower())
    mm_abi = normalized.get(HEADER_MM_ABI.lower())
    if protocol == PROTOCOL_VERSION_MM:
        value = validate_mm_abi_header(mm_abi)
        if enforce_mm_abi_match:
            if expected_mm_abi is None:
                raise ValueError(
                    f"{PROTOCOL_VERSION_MM} requires a locally computed "
                    f"{HEADER_MM_ABI} fingerprint on the cloud"
                )
            if value != expected_mm_abi:
                raise ValueError(
                    f"{HEADER_MM_ABI} fingerprint does not match this cloud"
                )
        return value
    if mm_abi is not None:
        raise ValueError(f"{HEADER_MM_ABI} requires protocol {PROTOCOL_VERSION_MM}")
    return None


def _sse_data(obj: Any) -> str:
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n"


def _build_chunk(
    request_id: str, model: str, choices: list, usage: dict | None = None
) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def _resolve_edge_id(
    headers: Mapping[str, str], edge_ids: frozenset[int]
) -> int | None:
    """Resolve the source edge of a probe request from its header.

    The edge id decides the cloud-internal namespace key, so an unknown id
    is rejected instead of silently falling back to 0 (which would
    misattribute the probe against the data-plane namespace). Without a
    header the single edge of a one-edge cloud is unambiguous; more edges
    cannot be guessed and the request fails.
    """
    normalized = {k.lower(): v for k, v in headers.items()}
    raw = normalized.get(HEADER_EDGE_ID.lower())
    if raw is None:
        if len(edge_ids) == 1:
            return next(iter(edge_ids))
        return None
    if not (raw.isascii() and raw.isdecimal()):
        return None
    edge_id = int(raw)
    if edge_ids and edge_id not in edge_ids:
        return None
    return edge_id


def create_lwd_cloud_control_app(
    bridge: LwdCloudControlBridge,
    processor_fingerprint: bytes | None = None,
    *,
    enforce_mm_abi_match: bool = False,
    probe_timeout: float = 25.0,
    edge_ids: frozenset[int] = frozenset(),
) -> Any:
    """FastAPI control application; successful answers stream as
    OpenAI-compatible SSE.

    Errors keep using JSONResponse (gateways handle non-2xx JSON bodies
    natively). ``processor_fingerprint`` is the local cloud fingerprint and
    only arbitrates v2 when ``enforce_mm_abi_match`` is on. ``probe_timeout``
    is this process's hard cap for one probe (derived by the caller from
    the edge's probe_timeout minus headroom). ``edge_ids`` are the edge
    instances this cloud serves; probe requests naming another edge are
    rejected before the engine is touched.
    """
    app = FastAPI()
    expected_mm_abi = (
        mm_abi_header_value(processor_fingerprint)
        if processor_fingerprint is not None
        else None
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Gateway liveness probe (takes no part in the negotiation)."""
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        try:
            headers = dict(request.headers)
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": f"bad request: {exc}"}, status_code=400)
        try:
            mm_abi_header = _negotiate_lwd_protocol_headers(
                headers, expected_mm_abi,
                enforce_mm_abi_match=enforce_mm_abi_match,
            )
            parse_headers: Mapping[str, str] = headers
            if mm_abi_header is not None:
                # The v2 manifest wire format is isomorphic to v1: after the
                # protocol marker and ABI gate pass, parse the body with the
                # v1 parser (it only accepts the v1 marker).
                parse_headers = {
                    k: v
                    for k, v in headers.items()
                    if k.lower() != HEADER_PROTOCOL.lower()
                }
                parse_headers[HEADER_PROTOCOL] = PROTOCOL_VERSION
            manifest = LwdPrefixManifest.from_openai_request(
                parse_headers, body
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        edge_id = _resolve_edge_id(headers, edge_ids)
        if edge_id is None:
            return JSONResponse(
                {"error": f"invalid {HEADER_EDGE_ID} for this cloud"},
                status_code=400,
            )
        # Namespace wrap at the HTTP boundary: the engine-internal id of the
        # probe equals the data-plane request key (the ingest seam wraps
        # edge messages the same way), so the usage settlement finds the
        # same request the probe announced.
        manifest = dataclass_replace(
            manifest, request_id=lwd_wrap_request_id(edge_id, manifest.request_id)
        )

        try:
            lwd_logger.debug(
                "[coord] handler entering probe req=%s edge=%d",
                manifest.request_id, edge_id,
            )
            result = await bridge.probe(manifest, timeout=probe_timeout)
            lwd_logger.debug(
                "[coord] handler probe resolved req=%s hit=%d",
                manifest.request_id, result.hit_tokens,
            )
        except TimeoutError as exc:
            # The cap was derived from the edge's probe_timeout minus
            # headroom, so normally the cloud answers first (200/503).
            # Reaching here means the engine loop did not consume the probe
            # command within that window (the step stretched under load);
            # the edge most probably failed open already and this answer is
            # observability only — one warning line, no traceback.
            lwd_logger.warning(
                "[coord] probe timeout req=%s edge=%d: %s "
                "(engine loop did not answer in time; edge already "
                "failed-open)",
                manifest.request_id, edge_id, exc,
            )
            return JSONResponse({"error": str(exc)}, status_code=503)
        except Exception as exc:  # noqa: BLE001
            lwd_logger.exception("[coord] handler probe raised")
            return JSONResponse({"error": str(exc)}, status_code=503)

        # The probe result rides delta.content (gateway DTO conversion keeps
        # the message body); the response headers stay for direct debugging
        # and gateway observability. Usage rides the last chunk of the same
        # SSE stream, where the gateway bills from. What the edge sees is
        # the original id with the namespace prefix stripped.
        edge_of_id, display_request_id = lwd_parse_request_edge(
            result.request_id
        )
        if edge_of_id is None:
            display_request_id = result.request_id
        model = body.get("model", "edge-cloud-prefix")

        # Echo the negotiated marker: the edge verifies the cloud answered
        # under its own protocol (no silent v1 downgrade).
        protocol = (
            PROTOCOL_VERSION_MM if mm_abi_header is not None else PROTOCOL_VERSION
        )
        control_message = result.to_control_message(protocol, mm_abi_header)
        control_message["request_id"] = display_request_id

        async def _events_inner():
            # 1. Probe control message (inside delta.content).
            yield _sse_data(_build_chunk(
                display_request_id,
                model,
                [{"index": 0, "delta": {
                    "role": "assistant",
                    "content": json.dumps(control_message, separators=(",", ":")),
                }, "finish_reason": None}],
            ))
            # 2. Empty delta frame: flush the gateway's last-chunk buffer.
            yield _sse_data(_build_chunk(
                display_request_id,
                model,
                [{"index": 0, "delta": {}, "finish_reason": None}],
            ))
            # 3. Await usage (pushed by the engine when inference settles).
            usage: LwdUsage | None = None
            usage_task = asyncio.create_task(
                bridge.wait_usage(manifest.request_id))
            try:
                while not usage_task.done():
                    done, _ = await asyncio.wait(
                        {usage_task}, timeout=_LWD_USAGE_HEARTBEAT_INTERVAL)
                    if not done:
                        yield ": edge-cloud-usage-pending\n\n"
                usage = usage_task.result()
            except asyncio.CancelledError:
                if not usage_task.done():
                    usage_task.cancel()
                await asyncio.gather(usage_task, return_exceptions=True)
                raise
            # 4. Usage end chunk + [DONE].
            if usage is None:
                # No real usage means no usage frame: a fabricated zero
                # would let the gateway bill zero tokens (a free ride on a
                # zero quota). The stream still ends with a comment plus
                # [DONE] — cutting it half-way would trip the gateway's
                # client_gone/error accounting, which is worse than "this
                # request has no upstream usage".
                lwd_logger.error(
                    "[coord] usage unavailable req=%s: ending stream "
                    "without usage frame (never bill fabricated zeros)",
                    manifest.request_id,
                )
                yield ": edge-cloud-usage-unavailable\n\n"
                yield "data: [DONE]\n\n"
                return
            yield _sse_data(_build_chunk(
                display_request_id,
                model,
                [],
                usage=usage.to_openai_dict(),
            ))
            yield "data: [DONE]\n\n"

        async def events():
            try:
                async for frame in _events_inner():
                    yield frame
            finally:
                # Normal completion, early edge disconnect or generator
                # cancellation all must drop the registration: otherwise the
                # usage future hangs in the dict until process close().
                bridge.release(manifest.request_id)

        headers_out = {
            **result.to_headers(),
            # The engine keys the request by the wrapped id; the response
            # header carries the stripped form: the edge sent the original
            # id, and cross-checking edge/cloud books by request id needs
            # one shape on both sides.
            HEADER_REQUEST_ID: display_request_id,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        if mm_abi_header is not None:
            headers_out[HEADER_PROTOCOL] = PROTOCOL_VERSION_MM
            headers_out[HEADER_MM_ABI] = mm_abi_header
        return StreamingResponse(
            events(), media_type="text/event-stream", headers=headers_out
        )

    return app


def lwd_start_cloud_control_server(
    vllm_config: Any,
    *,
    command_queue: Any,
    event_queue: Any,
    wakeup: Callable[[], None] | None = None,
) -> LwdCloudControlBridge | None:
    """Start this cloud DP engine's coordination HTTP server.

    Returns the started bridge (the caller wires the command queue into the
    engine step loop and pushes usage events through the bridge), or None
    when coordination is disabled, absent or the role is edge — in which
    case nothing is started and no behavior changes.

    Called on the engine thread during assembly: uvicorn and the bridge
    both run as daemon threads of the engine process. Rank r of the cloud
    instance binds ``listen_port + r``; rank 0 binds exactly
    ``listen_port``.
    """
    lwd_config = getattr(vllm_config, "lwd_config", None)
    if lwd_config is None:
        return None
    coord_cfg = getattr(lwd_config, "coordination", None)
    if coord_cfg is None or not coord_cfg.enabled:
        return None
    if lwd_config.role != "cloud":
        return None

    import uvicorn

    dp_idx = int(vllm_config.parallel_config.data_parallel_index)
    edge_ids = frozenset(
        connection.connection_id.edge_id
        for connection in lwd_config.connections
        if connection.connection_id.dp_idx == dp_idx
    )
    bridge = LwdCloudControlBridge(command_queue, event_queue, wakeup=wakeup)
    bridge.start()
    # uvicorn runs on a daemon thread with no shutdown channel: hook atexit
    # so the dispatch thread stops cleanly and waiting futures fail (the
    # daemon hard-kill is only the fallback).
    atexit.register(bridge.close)
    # The probe cap derives from the edge's probe_timeout minus headroom:
    # the cloud answers first (200 or 503). Fields still landing with the
    # parallel edge-side MR read via getattr with the source defaults.
    cloud_probe_timeout = max(
        float(getattr(coord_cfg, "probe_timeout", 25.0))
        - _LWD_CLOUD_PROBE_HEADROOM_S,
        1.0,
    )
    enforce_mm_abi_match = bool(
        getattr(coord_cfg, "enforce_mm_abi_match", False)
    )
    app = create_lwd_cloud_control_app(
        bridge,
        compute_processor_fingerprint(vllm_config.model_config),
        enforce_mm_abi_match=enforce_mm_abi_match,
        probe_timeout=cloud_probe_timeout,
        edge_ids=edge_ids,
    )
    listen_port = int(coord_cfg.listen_port) + dp_idx
    threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={
            "host": coord_cfg.listen_host,
            "port": listen_port,
            "log_level": "warning",
        },
        name="lwd-cloud-ctrl",
        daemon=True,
    ).start()
    lwd_logger.info(
        "[coord] cloud control HTTP server listening on %s:%s (dp_idx=%d, "
        "%d edge(s) served, probe_timeout=%.1fs enforce_mm_abi_match=%s)",
        coord_cfg.listen_host, listen_port, dp_idx, len(edge_ids),
        cloud_probe_timeout, enforce_mm_abi_match,
    )
    return bridge
