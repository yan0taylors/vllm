# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Two-phase admission of the coordination probe.

With coordination enabled, a new request first travels to its probe
endpoint (the single-value gateway URL or one multi-value URL picked by
the scheduler's rotation) off the engine thread: ``LwdProbeRuntime``
owns one daemon thread, one event loop and one shared session, so the
HMAC chaining and the HTTP round trip never block the scheduling loop.
``submit`` parks the request in an inflight ledger and returns; the
completion callback posts the outcome on a thread-safe queue and wakes
the engine with a ``LwdProbeWake`` marker (the same WAKEUP-hook class
the c2e notify queue uses); ``drain`` -- engine thread -- registers the
receipt's routing pin in the scheduler and finalizes the admission
(announce + enqueue, or the prefix-control wait).

Fail-open: a probe that fails, times out or is abandoned by a silent
runtime routes by the scheduler's fallback chain instead (peek position
/ first cloud / rotation), never blocking or dropping the request. A
request aborted while inflight is removed from the ledger, so a late
receipt finds no entry to claim; the probe stream itself is left to the
client's drain task (the gateway keeps its billing path).

Probe wiring depth (by design, this commit): the probe decides the
routing (cloud + dp pin) only. The receipt's prefix hit and block-size
learning stay with the data-plane prefix flow (LwdPrefixReadyNotify and
the edge prefix control), and placement_only without a tenant key keeps
probes off (the client always chains a manifest, which needs the key).
"""

from __future__ import annotations

import contextlib
import hashlib
import queue
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.config.lwd import LwdCoordinationConfig
from vllm.lwd_logger import lwd_logger
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.lwd_control.lwd_coordination.lwd_edge_client import (
    LwdEdgePrefixClient,
    LwdProbeRuntime,
)
from vllm.v1.lwd_control.lwd_coordination.lwd_mm_identity import (
    compute_processor_fingerprint,
)
from vllm.v1.lwd_control.lwd_coordination.lwd_prefix import LwdProbeResult

if TYPE_CHECKING:
    from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_engine import (
        LwdEdgeEngineCore,
    )
    from vllm.v1.request import Request

# Watchdog headroom over coordination.probe_timeout: the probe span itself
# is clamped by the client's probe_timeout, the watchdog only covers a
# silently dead runtime (a future that never completes), so it must fire
# strictly after the client deadline.
_LWD_PROBE_WATCHDOG_HEADROOM_S = 5.0


@dataclass(frozen=True)
class LwdProbeWake:
    """WAKEUP payload: at least one probe outcome is queued for drain."""


@dataclass(frozen=True)
class _LwdProbeOutcome:
    """One completed (or failed) probe, posted by the completion callback
    or the watchdog from the probe thread."""

    request_id: str
    nonce: int
    target_cloud: int | None
    result: LwdProbeResult | None


@dataclass(frozen=True)
class _LwdProbeMediaItem:
    """Image content identity for the v2 probe chain (same material the
    wire protocol chains into media-covered blocks)."""

    modality: str
    digest: bytes
    offset: int
    length: int


class LwdEdgeProbeAdmission:
    """Engine-thread owner of the probe lifecycle; the routing decisions
    stay with the scheduler (target rotation, receipt pin, health)."""

    def __init__(
        self,
        engine: LwdEdgeEngineCore,
        coordination: LwdCoordinationConfig,
        tenant_key: bytes,
    ) -> None:
        self._engine = engine
        self._coordination = coordination
        self._tenant_key = tenant_key
        self._runtime: LwdProbeRuntime | None = None
        self._clients: dict[int | None, LwdEdgePrefixClient] = {}
        """Per submission-target clients (None = the single-value URL)."""
        self._inflight: dict[str, tuple[Request, int, threading.Timer]] = {}
        """request_id -> (request, submission nonce, watchdog timer)."""
        self._nonce = 0
        self._done: queue.Queue = queue.Queue()
        self._processor_fingerprint = compute_processor_fingerprint(
            engine.vllm_config.model_config
        )

    @classmethod
    def create(
        cls, engine: LwdEdgeEngineCore
    ) -> LwdEdgeProbeAdmission | None:
        """Build the admission layer, or None when probes stay off.

        Probes need an enabled coordination block with an endpoint, and a
        readable tenant key (the client always chains a manifest;
        placement_only without a key cannot chain and keeps the fallback
        routing, announced once)."""
        lwd_config = engine.vllm_config.lwd_config
        coordination = getattr(lwd_config, "coordination", None)
        if coordination is None or not coordination.enabled:
            return None
        if not (coordination.control_url or coordination.control_url_map):
            return None
        if coordination.tenant_key_file is None:
            lwd_logger.warning(
                "[edge-probe] coordination placement_only without "
                "tenant_key_file: the edge client always chains a "
                "manifest, so probes stay off and requests route by the "
                "fallback chain"
            )
            return None
        try:
            with open(coordination.tenant_key_file, "rb") as key_file:
                tenant_key = key_file.read()
        except OSError as exc:
            raise RuntimeError(
                "[LWD] edge coordination tenant_key_file "
                f"{coordination.tenant_key_file!r} is unreadable: {exc}"
            ) from exc
        return cls(engine, coordination, tenant_key)

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._inflight

    # ------------------------------------------------------------------ #
    # Submission (engine thread, request entry)                          #
    # ------------------------------------------------------------------ #
    def submit(self, request: Request) -> bool:
        """Park the request and submit its probe. True = inflight (the
        admission finalizes on drain); False = no endpoint or submission
        failed (the caller proceeds synchronously, fail-open)."""
        engine = self._engine
        target_cloud, url = engine.scheduler.lwd_edge_next_probe_target()
        if url is None:
            return False
        client = self._client_for(target_cloud, url)
        request_id = request.request_id
        prompt_token_ids = list(request.prompt_token_ids)
        media_items = self._media_items(request)
        model_name = self._served_model_name()
        if self._runtime is None:
            self._runtime = LwdProbeRuntime()
        runtime = self._runtime
        self._nonce += 1
        nonce = self._nonce
        try:
            future = runtime.submit(
                self._job(
                    client,
                    runtime,
                    request_id,
                    prompt_token_ids,
                    media_items,
                    model_name,
                )
            )
        except Exception:  # noqa: BLE001  a dead runtime fails open
            lwd_logger.warning(
                "[edge-probe] submit failed (fail-open, routed by the "
                "fallback chain)",
                exc_info=True,
            )
            runtime.stop()
            self._runtime = None
            return False
        # The watchdog covers a silently dead runtime only; the probe span
        # itself is clamped by the client's probe_timeout.
        timer = threading.Timer(
            self._coordination.probe_timeout + _LWD_PROBE_WATCHDOG_HEADROOM_S,
            self._watchdog,
            args=(future, request_id, nonce, target_cloud),
        )
        timer.daemon = True
        self._inflight[request_id] = (request, nonce, timer)

        def _done(fut: Any) -> None:
            timer.cancel()
            try:
                result = fut.result()
            except Exception:  # noqa: BLE001  fail-open: no receipt
                lwd_logger.warning(
                    "[edge-probe] probe failed (fail-open, routed by the "
                    "fallback chain)",
                    exc_info=True,
                )
                result = None
            self._post(request_id, nonce, target_cloud, result)

        future.add_done_callback(_done)
        timer.start()
        lwd_logger.info(
            "[edge-probe] submitted req=%s target=%s url=%s",
            request_id,
            target_cloud,
            url,
        )
        return True

    async def _job(
        self,
        client: LwdEdgePrefixClient,
        runtime: LwdProbeRuntime,
        request_id: str,
        prompt_token_ids: list[int],
        media_items: tuple[_LwdProbeMediaItem, ...],
        model_name: str | None,
    ) -> LwdProbeResult:
        session = await runtime.ensure_session()
        return await client.negotiate(
            session,
            request_id,
            prompt_token_ids,
            media_items=media_items,
            model=model_name,
        )

    def _watchdog(
        self,
        future: Any,
        request_id: str,
        nonce: int,
        target_cloud: int | None,
    ) -> None:
        """Fail-open on a silent runtime (the future never completes);
        racing the completion callback is harmless -- the drain drops the
        entry that no longer matches the inflight ledger."""
        if future.done():
            return
        future.cancel()
        lwd_logger.warning(
            "[edge-probe] watchdog timeout (fail-open, routed by the "
            "fallback chain): req=%s",
            request_id,
        )
        self._post(request_id, nonce, target_cloud, None)

    def _post(
        self,
        request_id: str,
        nonce: int,
        target_cloud: int | None,
        result: LwdProbeResult | None,
    ) -> None:
        """Probe thread: queue the outcome and wake the engine."""
        self._done.put(_LwdProbeOutcome(request_id, nonce, target_cloud, result))
        # A dying engine needs no wake; the outcome stays queued.
        with contextlib.suppress(Exception):
            self._engine.input_queue.put_nowait(
                (EngineCoreRequestType.WAKEUP, LwdProbeWake())
            )

    # ------------------------------------------------------------------ #
    # Drain and abort (engine thread)                                    #
    # ------------------------------------------------------------------ #
    def drain(self) -> None:
        """Consume completed probes: health accounting, receipt pinning,
        then the normal admission finalize. A late outcome for a request
        that left the ledger (aborted or superseded) is dropped."""
        while True:
            try:
                outcome = self._done.get_nowait()
            except queue.Empty:
                return
            entry = self._inflight.get(outcome.request_id)
            if entry is None or entry[1] != outcome.nonce:
                continue
            request, _nonce, timer = entry
            timer.cancel()
            self._inflight.pop(outcome.request_id, None)
            scheduler = self._engine.scheduler
            if outcome.target_cloud is not None:
                scheduler.lwd_edge_note_probe_health(
                    outcome.target_cloud, outcome.result is not None
                )
            if outcome.result is not None:
                scheduler.lwd_edge_note_probe_receipt(
                    outcome.request_id,
                    outcome.target_cloud,
                    outcome.result.cloud_id,
                    outcome.result.dp_idx,
                )
            self._engine._lwd_admit_or_wait(request)

    def prepare_abort(self, request_ids: list[str]) -> None:
        """Abort inflight probes: drop the ledger entries so a late
        outcome claims nothing. The requests were never announced to any
        cloud (no AbortNotify); they are enqueued here so the caller's
        native abort right after gives the frontend its terminal state,
        mirroring the prefix-control waiter abort."""
        for request_id in request_ids:
            entry = self._inflight.pop(request_id, None)
            if entry is None:
                continue
            request, _nonce, timer = entry
            timer.cancel()
            self._engine._lwd_enqueue_request(request)

    def shutdown(self) -> None:
        """Stop the probe runtime (the daemon watchdog timers of inflight
        requests die with the process; their outcomes are never drained)."""
        for _request, _nonce, timer in self._inflight.values():
            timer.cancel()
        if self._runtime is not None:
            self._runtime.stop()

    # ------------------------------------------------------------------ #
    # Client and identity helpers                                        #
    # ------------------------------------------------------------------ #
    def _client_for(
        self, target_cloud: int | None, url: str
    ) -> LwdEdgePrefixClient:
        """Per-target client cache: the hasher and key material are built
        once per submission target."""
        client = self._clients.get(target_cloud)
        if client is not None:
            return client
        client = LwdEdgePrefixClient(
            self._coordination,
            self._probe_block_size(),
            self._tenant_key,
            edge_id=self._engine.vllm_config.lwd_config.instance_id,
            control_url=url,
            processor_fingerprint=self._processor_fingerprint,
        )
        self._clients[target_cloud] = client
        return client

    def _probe_block_size(self) -> int:
        """Chaining granularity of the probe manifest: the learned cloud
        hash block size when the data-plane prefix flow announced one,
        else the local KV block size."""
        block_size = getattr(
            self._engine.scheduler, "_lwd_prefix_hash_block_size", 0
        )
        if block_size > 0:
            return block_size
        return self._engine.vllm_config.cache_config.block_size

    @staticmethod
    def _media_items(request: Request) -> tuple[_LwdProbeMediaItem, ...]:
        """Image-only content identity (the protocol's modality domain);
        prompt_embeds passthrough carries no grid identity, text requests
        chain the plain v1 form."""
        items: list[_LwdProbeMediaItem] = []
        for feature in getattr(request, "mm_features", ()):
            if feature.modality != "image":
                continue
            position = getattr(feature, "mm_position", None)
            if position is None:
                continue
            items.append(
                _LwdProbeMediaItem(
                    modality=feature.modality,
                    digest=hashlib.sha256(
                        feature.identifier.encode("utf-8")
                    ).digest(),
                    offset=position.offset,
                    length=position.length,
                )
            )
        return tuple(items)

    def _served_model_name(self) -> str | None:
        """Model name for gateway channel routing (the gateway picks its
        upstream by it; harmless to the direct cloud form)."""
        model_config = getattr(self._engine.vllm_config, "model_config", None)
        if model_config is None:
            return None
        return getattr(model_config, "served_model_name", None) or getattr(
            model_config, "model", None
        )
