# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cloud engine cores and their LWD control-plane seams.

Range notifies post extended early-receive hints, request notifies become
native ADDs with prompt-length and M-RoPE metadata, and abort notifies use the
native ABORT path. Step metadata is flushed to the edge in post_step. The
transport accepts independent per-edge sessions; MR4 owns multi-edge request
namespaces and role-side output accounting. The coordination plane adds an
HTTP probe/usage endpoint: its commands are consumed by the engine step
loop (LwdProbeQueryAdapter answers read-only), and finished requests settle
their usage back onto the endpoint's SSE stream.

The LWD seams live in LwdCloudEngineCoreMixin and are orthogonal to the
engine base: LwdCloudEngineCore (EngineCoreProc) serves one DP rank of a
cloud instance and non-MoE multi-DP deployments; LwdCloudDPEngineCore
(DPEngineCoreProc) serves MoE multi-DP deployments, inheriting the native
DP semantics (wave state machine, dummy batches, START_DP_WAVE, stats
publishing) unchanged.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from collections.abc import Iterable
from typing import Any

import msgspec
import torch

from vllm.config.lwd import LwdConnectionId
from vllm.lwd_logger import lwd_logger
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc, EngineShutdownState
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_scheduler import (
    LwdCloudScheduler,
)
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_prefix_cache import (
    init_lwd_prefix_cache,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_engine import (
    LwdBaseEngineCore,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    lwd_spec_factor,
)
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import LwdPeer
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LWD_SAMPLING_DEFAULTS,
    LwdAbortNotify,
    LwdC2eNotify,
    LwdCloudParamsNotify,
    LwdCloudToEdgeMessage,
    LwdEdgeToCloudMessage,
    LwdMessage,
    LwdPrefixReadyNotify,
    LwdRangeNotify,
    LwdRefillNotify,
    LwdRequestNotify,
    lwd_decode_edge_to_cloud,
    lwd_decode_structured_outputs,
    lwd_message_connection_id,
)
from vllm.v1.lwd_control.lwd_control_diag import LwdControlDiag
from vllm.v1.lwd_control.lwd_coordination.lwd_cloud_control import (
    LwdCloudControlProcessor,
    lwd_start_cloud_control_server,
)
from vllm.v1.lwd_control.lwd_coordination.lwd_prefix import (
    LwdPrefixManifest,
    LwdProbeResult,
    LwdUsage,
)
from vllm.v1.lwd_control.lwd_coordination.lwd_request_namespace import (
    lwd_unwrap_request_id,
    lwd_wrap_request_id,
)
from vllm.v1.request import Request

# Sleep between retries of a full outbound queue: step metadata must not be
# lost, the edge posts its receive of the exact size from it.
_LWD_SEND_RETRY_SLEEP_S = 0.05
_LWD_INGEST_JOIN_TIMEOUT_S = 5.0
_LWD_RETIRED_CONNECTION_CAP = 4096


_LwdIngestItem = tuple[LwdPeer, LwdEdgeToCloudMessage]


class LwdProbeQueryAdapter:
    """Read-only hit query behind the HTTP probe commands.

    The probe answers routing and hit-boundary questions only: reservations
    and pinning stay with the data-plane request path (the single
    reservation authority of ``LwdPrefixReservations.reserve`` in
    add_request), so this adapter never pins blocks, never creates a
    reservation and needs no TTL sweep — no orphan reservations can exist.
    It links the probe manifest to the native ``find_longest_cache_hit``
    and reports the answering engine's identity.
    """

    def __init__(
        self,
        manager: Any,
        hash_block_size: int,
        alignment_tokens: int,
        instance_id: str,
        cloud_id: int,
        dp_idx: int,
    ) -> None:
        self._manager = manager
        self._hash_block_size = int(hash_block_size or 0)
        self._alignment_tokens = int(alignment_tokens or 0)
        self._instance_id = instance_id
        self._cloud_id = cloud_id
        self._dp_idx = dp_idx

    def probe(self, manifest: LwdPrefixManifest) -> LwdProbeResult:
        hash_bs = self._hash_block_size if self._hash_block_size > 0 else 1
        hit = 0
        manager = self._manager
        hashes = list(getattr(manifest, "full_block_hashes", None) or [])
        if manager is not None and self._hash_block_size > 0 and hashes:
            # The manifest chain is at the cloud hash granularity (a
            # multiple of hash_block_size); the native pool records blocks
            # at the vllm block size, so expand by the ratio before the
            # lookup.
            native_bs = int(getattr(manager, "block_size", 0) or 0)
            ratio = (
                max(1, self._hash_block_size // native_bs)
                if native_bs > 0 else 1
            )
            chain = (
                [h for h in hashes for _ in range(ratio)]
                if ratio > 1 else hashes
            )
            _groups, hit, _boundary = manager.coordinator.find_longest_cache_hit(
                chain,
                max_cache_hit_length=max(0, manifest.prompt_tokens - 1),
            )
        # Hits floor to whole cloud hash blocks: the edge builds and trims
        # its chain at that granularity, and LwdProbeResult requires
        # hit_tokens == hit_blocks * block_size.
        hit = (int(hit) // hash_bs) * hash_bs
        hit_blocks = hit // hash_bs
        return LwdProbeResult(
            request_id=manifest.request_id,
            instance_id=self._instance_id,
            block_size=hash_bs,
            hit_blocks=hit_blocks,
            hit_tokens=hit_blocks * hash_bs,
            alignment_tokens=self._alignment_tokens,
            cloud_id=self._cloud_id,
            dp_idx=self._dp_idx,
        )


def _lwd_namescope_edge_message(
    msg: LwdMessage, connection_id: LwdConnectionId
) -> LwdMessage:
    """Wrap every request id of an inbound edge message in the cloud namespace.

    The single seam where wire ids become engine-internal ids: the ingest
    thread calls it right after resolving the source connection, so the
    scheduler, arrival ledgers, reservation keys and usage settlement all see
    one collision-free key space even when two edges reuse one client id.
    """
    edge_id = connection_id.edge_id
    if isinstance(msg, LwdRangeNotify):
        if not msg.items:
            return msg
        return msgspec.structs.replace(
            msg,
            items=[
                msgspec.structs.replace(
                    item, request_id=lwd_wrap_request_id(edge_id, item.request_id)
                )
                for item in msg.items
            ],
        )
    if isinstance(msg, (LwdRequestNotify, LwdAbortNotify)):
        return msgspec.structs.replace(
            msg, request_id=lwd_wrap_request_id(edge_id, msg.request_id)
        )
    return msg


class LwdCloudEngineCoreMixin(LwdBaseEngineCore):
    """The LWD seams of a cloud engine, shared by both engine bases."""

    lwd_scheduler_cls = LwdCloudScheduler
    scheduler: LwdCloudScheduler

    def __init__(self, *args: Any, lwd_dp_size: int, **kwargs: Any) -> None:
        vllm_config = kwargs["vllm_config"]
        if (
            vllm_config.cache_config.enable_prefix_caching
            and getattr(vllm_config, "kv_transfer_config", None) is not None
        ):
            raise ValueError(
                "[LWD] prefix caching with KV connector is unsupported: "
                "external hits bypass the negotiated READY boundary"
            )
        # Pre-recv hint MQ must exist before super().__init__ spawns the
        # executor workers (same process): create_hint_mq() publishes the
        # handle through the environment, and only processes spawned after
        # that see it. The sideband MQ exists because rpc_broadcast_mq is
        # drained by the worker's single-threaded busy loop, which sits
        # inside a several-hundred-ms prefill forward - hints queued behind
        # it would lose the whole point of pre-posting the receive.
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import create_hint_mq

        dp_idx = vllm_config.parallel_config.data_parallel_index
        connections = tuple(
            connection.connection_id
            for connection in vllm_config.lwd_config.connections
            if connection.dp_idx == dp_idx
        ) or (LwdConnectionId(0, vllm_config.lwd_config.instance_id, dp_idx),)
        self._lwd_hint_mqs = {
            connection: create_hint_mq(connection) for connection in connections
        }
        self._lwd_hint_mq = next(iter(self._lwd_hint_mqs.values()))
        super().__init__(*args, lwd_dp_size=lwd_dp_size, **kwargs)
        # The cloud endpoint worker created its done-report MQ during
        # init_device; now that the executor is up, fetch its handle over
        # the executor's collective RPC (the executor itself stays free of
        # LWD plumbing). Exactly one worker - the endpoint rank - must
        # report a handle; anything else is a wiring error.
        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

        self._lwd_done_mqs = {}
        for connection in self._lwd_hint_mqs:
            handles = [
                handle
                for handle in self.model_executor.collective_rpc(
                    "lwd_early_recv_done_handle", args=(connection,)
                )
                if handle is not None
            ]
            if len(handles) != 1:
                raise RuntimeError(
                    "[LWD] early receive expects one endpoint worker reporting "
                    f"completions for {connection}, got {len(handles)}"
                )
            self._lwd_done_mqs[connection] = MessageQueue.create_from_handle(
                handles[0], 0
            )
        self._lwd_done_mq = next(iter(self._lwd_done_mqs.values()))
        lwd_logger.info(
            "[pre-recv] %d hint MQ(s) created, done MQs attached via RPC",
            len(self._lwd_hint_mqs),
        )

    def _lwd_init_prefix_cache(self) -> None:
        state = init_lwd_prefix_cache(
            self.vllm_config,
            self.scheduler.kv_cache_config,
            self.scheduler.kv_cache_manager,
            use_eagle=bool(getattr(self.scheduler, "use_eagle", False)),
        )
        self._lwd_prefix_state = state
        self._lwd_prefix_enabled = state.enabled
        self._lwd_hash_block_size = state.hash_block_size
        if state.enabled:
            self.scheduler.lwd_prefix_reservations = state.reservations
            self.scheduler.lwd_refill_sender = self._lwd_send_refill

    def _lwd_assemble(self) -> None:
        if self.vllm_config.model_config.skip_tokenizer_init:
            # StructuredOutputManager builds its tokenizer and compile
            # executor only with a tokenizer; without one the first
            # constrained request would crash at grammar_init. Fail fast at
            # startup instead (the cloud slice ships the tokenizer per F2
            # metadata rules).
            raise ValueError(
                "[LWD] the cloud engine requires its tokenizer "
                "(skip_tokenizer_init=True is not supported): grammar "
                "compilation and the sampling bitmask happen on the cloud"
            )
        self._lwd_init_prefix_cache()
        # Multi-edge flag of this DP rank: request ids stay unwrapped on the
        # wire of a single-edge cloud (byte-identical to the pre-namespace
        # format), while a multi-edge cloud must never see an unnamespaced
        # id leaving the engine (it could not be attributed to a source).
        dp_idx = self.vllm_config.parallel_config.data_parallel_index
        self._lwd_multi_edge = len(
            {
                connection.connection_id.edge_id
                for connection in self.lwd_config.connections
                if connection.connection_id.dp_idx == dp_idx
            }
        ) > 1
        self._lwd_ingest_thread: threading.Thread | None = None
        self._lwd_edge: LwdPeer | None = None
        self._lwd_peers: dict[LwdConnectionId, LwdPeer] = {}
        self._lwd_prompt_lens: dict[str, int] = {}
        self._lwd_retired_prompt_lens: dict[str, int] = {}
        self._lwd_request_nonces: dict[str, int] = {}
        self._lwd_retired_nonces: dict[str, int] = {}
        self._lwd_request_connections: dict[str, LwdConnectionId] = {}
        self._lwd_retired_connections: dict[str, LwdConnectionId] = {}
        self._lwd_retired_req_ids: set[str] = set()
        self._lwd_rejected_nonces: dict[str, int] = {}
        self._lwd_pending_seqnos: dict[
            LwdConnectionId, dict[int, set[str]]
        ] = {}
        self._lwd_request_seqnos: dict[str, set[tuple[LwdConnectionId, int]]] = {}
        self._lwd_latest_seqnos: dict[LwdConnectionId, int] = {}
        self._lwd_failed_connections: set[LwdConnectionId] = set()
        sched_config = self.vllm_config.scheduler_config
        # Scheduling limits are checked by the edge. The coordinator address
        # lets the edge
        # trigger the first DP wave; its published host may not be routable
        # from the edge, so it is rewritten to this DP's topology address.
        self._lwd_cloud_params = LwdCloudParamsNotify(
            max_num_batched_tokens=sched_config.max_num_batched_tokens,
            max_num_seqs=sched_config.max_num_seqs,
            num_speculative_tokens=lwd_spec_factor(self.vllm_config) - 1,
            dp_coordinator_addr=self._lwd_dp_coordinator_addr(),
            prefix_caching=self._lwd_prefix_enabled,
            hash_block_size=self._lwd_hash_block_size,
            prefix_alignment_tokens=self._lwd_prefix_state.alignment_tokens,
            prefix_alignment_required=self._lwd_prefix_state.alignment_required,
            prefix_use_eagle=self._lwd_prefix_state.use_eagle,
        )
        self._lwd_register_recv_hint()
        # Edge messages are handled off the IO thread: building a request
        # allocates its placeholder embeds and the pre-recv hint may block,
        # which must not delay step metadata going out.
        self._lwd_ingest_queue: queue.SimpleQueue[_LwdIngestItem | None] = (
            queue.SimpleQueue()
        )
        self._lwd_ingest_thread = threading.Thread(
            target=self._lwd_ingest_loop, name="lwd-cloud-ingest", daemon=True
        )
        self._lwd_ingest_thread.start()
        plane = self._lwd_start_control_plane()
        # Coordination attributes exist before any thread starts, so the
        # busy-loop guards hold even when coordination is disabled or the
        # assembly fails halfway.
        self._lwd_coordination_commands: queue.Queue | None = None
        self._lwd_coordination_bridge: Any = None
        self._lwd_coordination_processor: LwdCloudControlProcessor | None = None
        self._lwd_coordination_query: LwdProbeQueryAdapter | None = None
        self._lwd_setup_coordination()
        lwd_logger.info("[cloud] engine assembled: control plane on %s", plane.endpoint)

    def _lwd_dp_coordinator_addr(self) -> str:
        """Routable coordinator stats endpoint, or empty without one."""
        address = getattr(self, "frontend_stats_publish_address", None)
        if not self.has_coordinator or not address:
            return ""
        if not str(address).startswith("tcp://"):
            return ""
        _, _, port = str(address).rpartition(":")
        dp_rank = self.vllm_config.parallel_config.data_parallel_index
        local = self.lwd_config.instance.dp[dp_rank]
        return f"tcp://{local.addr}:{port}"

    def _lwd_setup_coordination(self) -> None:
        """Start this engine's HTTP coordination endpoint and wire its
        command queue into the step loop.

        The server function owns the enabled/role checks: coordination off,
        absent, or an edge role leaves every attribute empty and the engine
        loop untouched (zero behavior change). With it armed, probe commands
        are consumed by ``_lwd_poll_coordination`` on the engine thread —
        the only thread allowed to touch the block pool — and usage events
        are pushed back through the bridge at finish settlement.
        """
        coord_cfg = getattr(self.lwd_config, "coordination", None)
        if coord_cfg is None or not coord_cfg.enabled:
            return
        command_queue: queue.Queue = queue.Queue()
        event_queue: queue.Queue = queue.Queue()
        bridge = lwd_start_cloud_control_server(
            self.vllm_config,
            command_queue=command_queue,
            event_queue=event_queue,
            wakeup=self._lwd_coordination_wakeup,
        )
        if bridge is None:
            return
        self._lwd_coordination_commands = command_queue
        self._lwd_coordination_bridge = bridge
        self._lwd_coordination_query = LwdProbeQueryAdapter(
            manager=getattr(self.scheduler, "kv_cache_manager", None),
            hash_block_size=self._lwd_hash_block_size,
            alignment_tokens=(
                self._lwd_prefix_state.alignment_tokens
                if self._lwd_prefix_state is not None
                else 0
            ),
            instance_id=str(
                coord_cfg.instance_id or f"cloud-{self.lwd_config.instance_id}"
            ),
            cloud_id=self.lwd_config.instance_id,
            dp_idx=self.vllm_config.parallel_config.data_parallel_index,
        )
        self._lwd_coordination_processor = LwdCloudControlProcessor(
            command_queue, event_queue
        )
        lwd_logger.info(
            "[coord] cloud engine coordination plane armed "
            "(hash_block_size=%d, query=read-only)",
            self._lwd_hash_block_size,
        )

    def _lwd_coordination_wakeup(self) -> None:
        """Wake an idle engine for one pending coordination command.

        The same primitive the shutdown signal path uses: the native WAKEUP
        payload is a no-op whose only job is returning
        ``_process_input_queue`` from its blocking wait, and the command
        itself is consumed by the poll that follows. Multiple posts are
        harmless.
        """
        # A shutdown race leaves the command queued; that is harmless.
        with contextlib.suppress(Exception):
            self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

    def _lwd_poll_coordination(self) -> None:
        """Consume coordination commands (engine thread, non-blocking).

        No reservation TTL sweep is needed: the read-only query creates no
        reservations, so none can be orphaned.
        """
        processor = self._lwd_coordination_processor
        if processor is None:
            return
        started = time.monotonic()
        consumed = processor.poll(self._lwd_coordination_query)
        if consumed:
            lwd_logger.info(
                "[coord] probe poll consumed=%d poll_ms=%.2f",
                consumed,
                (time.monotonic() - started) * 1000.0,
            )

    def _process_input_queue(self) -> None:
        """Idle-wait override: with the coordination plane armed, one round
        of probe commands is drained after every input-queue poll.

        The query itself runs here on the engine thread: the block pool is
        not thread-safe and the HTTP loop never touches it. Idle
        reachability: every command posts a WAKEUP, so the native blocking
        get returns immediately, and has_work counts the pending commands so
        the blocking loop exits and reaches this poll.
        """
        if self._lwd_coordination_processor is None:
            return super()._process_input_queue()
        super()._process_input_queue()
        self._lwd_poll_coordination()

    def has_work(self) -> bool:
        """Pending coordination commands count as work: the native
        ``_process_input_queue`` blocks its wait loop while has_work is
        false, and a WAKEUP alone does not end that loop — the queued
        commands must hold the loop open until the poll consumes them (an
        idle engine answers probes without waiting for a step)."""
        commands = self._lwd_coordination_commands
        return super().has_work() or (
            commands is not None and not commands.empty()
        )

    def _lwd_publish_finished_usage(
        self, engine_core_outputs: Any
    ) -> dict[str, LwdUsage]:
        """Settle finished requests onto the coordination SSE stream.

        Token counts come from the scheduler's ``lwd_finished_records``
        (captured by the ``_free_request`` choke point — abort and error
        terminations land there too, so the ledger, not the output rows, is
        the terminal-state authority); ``cached_tokens`` comes from the
        reservation hit snapshot (the reservation itself was released at
        first allocation, and reading it back would report zero — exactly
        the bug the snapshot ledger fixes). Records are consumed here;
        without a coordination bridge nothing is read or pushed.
        """
        bridge = self._lwd_coordination_bridge
        if bridge is None:
            return {}
        records = getattr(self.scheduler, "lwd_finished_records", None) or {}
        if not records:
            return {}
        finished: set[str] = set()
        for outputs in engine_core_outputs.values():
            for output in outputs.outputs:
                if output.finish_reason is not None:
                    finished.add(output.request_id)
            for request_id in outputs.finished_requests or ():
                finished.add(request_id)
        # The ledger is the authority: outputs only cover rows carrying a
        # terminal output; abort/error terminations emit no row at all.
        finished |= set(records)
        reservations = getattr(self.scheduler, "lwd_prefix_reservations", None)
        usages: dict[str, LwdUsage] = {}
        for request_id in finished:
            snapshot = records.pop(request_id, None)
            if snapshot is None:
                continue
            prompt_tokens, completion_tokens = snapshot
            cached = (
                reservations.cached_hit_of(request_id)
                if reservations is not None
                else 0
            )
            usage = LwdUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=int(cached or 0),
            )
            usages[request_id] = usage
            bridge.publish_usage(request_id, usage)
        return usages

    def _lwd_decode(self, payload: bytes) -> LwdMessage:
        return lwd_decode_edge_to_cloud(payload)

    def _lwd_handle_peer_registered(self, peer: LwdPeer) -> None:
        """Engine thread: remember the edge and send it the scheduling
        limits before anything else."""
        self._lwd_edge = peer
        self._lwd_peers[self._lwd_connection_of(peer)] = peer
        params = msgspec.structs.replace(self._lwd_cloud_params)
        self._lwd_send_until_queued(peer, params)
        lwd_logger.info(
            "[cloud] edge %s registered; scheduling limits sent: %s",
            peer,
            params,
        )

    def _lwd_send_until_queued(
        self, peer: LwdPeer, message: LwdCloudToEdgeMessage
    ) -> None:
        """Retry a full per-peer FIFO until queued, or until shutdown starts."""
        self._lwd_stamp(peer, message)
        while not self._lwd_send(peer, message):
            if self.shutdown_state != EngineShutdownState.RUNNING:
                lwd_logger.error(
                    "[cloud] %s to edge %s dropped: shutting down with the "
                    "outbound queue full",
                    type(message).__name__,
                    peer,
                )
                return
            time.sleep(_LWD_SEND_RETRY_SLEEP_S)

    def _lwd_peer_for_connection(self, connection_id: LwdConnectionId) -> LwdPeer:
        peer = getattr(self, "_lwd_peers", {}).get(connection_id)
        if peer is None:
            peers = getattr(self, "_lwd_peers", {})
            if len(peers) == 1:
                peer = next(iter(peers.values()))
        if peer is None and len(getattr(self, "_lwd_peers", {})) <= 1:
            peer = getattr(self, "_lwd_edge", None)
        if peer is None:
            raise RuntimeError(f"[LWD] no registered peer for {connection_id}")
        return peer

    def _lwd_peer_for_request(self, request_id: str) -> LwdPeer:
        connection_id = getattr(self, "_lwd_request_connections", {}).get(
            request_id
        ) or getattr(self, "_lwd_retired_connections", {}).get(request_id)
        if connection_id is None:
            if len(getattr(self, "_lwd_peers", {})) <= 1:
                peer = getattr(self, "_lwd_edge", None)
                if peer is not None:
                    return peer
            raise RuntimeError(f"[LWD] request route missing for {request_id}")
        return self._lwd_peer_for_connection(connection_id)

    def _lwd_wire_request_id(self, request_id: str) -> str:
        """Restore the edge's original id for an outbound wire message."""
        return lwd_unwrap_request_id(
            request_id, multi_edge=getattr(self, "_lwd_multi_edge", False)
        )

    def _lwd_send_prefix_ready(
        self, request_id: str, hit_tokens: int, nonce: int, error: str = ""
    ) -> None:
        peer = self._lwd_peer_for_request(request_id)
        self._lwd_send_until_queued(
            peer,
            LwdPrefixReadyNotify(
                self._lwd_wire_request_id(request_id), hit_tokens, nonce, error
            ),
        )
        lwd_logger.info(
            "[prefix] READY req=%s hit=%d nonce=%d error=%r",
            request_id,
            hit_tokens,
            nonce,
            error,
        )

    def _lwd_send_refill(
        self, request_id: str, start_row: int, end_row: int, prefix_nonce: int
    ) -> None:
        peer = self._lwd_peer_for_request(request_id)
        self._lwd_send_until_queued(
            peer,
            LwdRefillNotify(
                self._lwd_wire_request_id(request_id),
                start_row,
                end_row,
                prefix_nonce,
            ),
        )
        lwd_logger.info(
            "[prefix] REFILL req=%s rows=[%d, %d) nonce=%d",
            request_id,
            start_row,
            end_row,
            prefix_nonce,
        )

    def add_request(self, request: Request, request_wave: int = 0) -> None:
        nonce = getattr(request, "lwd_prefix_nonce", 0)
        if not nonce:
            if self._lwd_prefix_enabled:
                lwd_logger.error(
                    "[prefix] reject req=%s: missing prefix nonce",
                    request.request_id,
                )
                return
            super().add_request(request, request_wave)
            return

        request_id = request.request_id
        if request_id in getattr(self.scheduler, "requests", {}):
            self._lwd_send_prefix_ready(
                request_id, 0, nonce, error="duplicate active request id"
            )
            return
        reservations = getattr(self.scheduler, "lwd_prefix_reservations", None)
        if reservations is None:
            self._lwd_send_prefix_ready(
                request_id, 0, nonce, error="cloud prefix caching is disabled"
            )
            return
        if not getattr(request, "lwd_wire_hashes_adopted", False):
            self._lwd_send_prefix_ready(
                request_id,
                0,
                nonce,
                error="prefix hash chain does not match cloud hash block size",
            )
            return
        try:
            hit_tokens = reservations.reserve(request)
        except Exception as exc:
            lwd_logger.exception("[prefix] reserve failed req=%s", request_id)
            self._lwd_send_prefix_ready(request_id, 0, nonce, error=str(exc))
            return
        try:
            super().add_request(request, request_wave)
        except Exception:
            reservations.release(request_id)
            self._lwd_send_prefix_ready(
                request_id, 0, nonce, error="cloud admission failed"
            )
            raise
        self._lwd_send_prefix_ready(request_id, hit_tokens, nonce)

    def _lwd_register_recv_hint(self) -> None:
        """[LWD][pre-recv] Early receive: the hint delivery channel =
        engine -> cloud endpoint worker comm thread. It goes through
        the sideband MQ this engine creates in __init__ (not
        rpc_broadcast_mq: that one is consumed single-threaded by the
        busy_loop and gets blocked by a several-hundred-ms prefill
        forward, so the hint would no longer be "early")."""
        hint_mqs = getattr(self, "_lwd_hint_mqs", None)
        if hint_mqs:
            from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
                register_hint_sender,
            )

            for connection, hint_mq in hint_mqs.items():
                if hint_mq is None:
                    continue
                # Blocking enqueue: once the gate is on, hints are a
                # correctness dependency — drop one and nobody pre-posts
                # that recv, the watermark stops advancing, and batches
                # behind the gate are never dispatched.
                register_hint_sender(
                    lambda hint, mq=hint_mq: mq.enqueue(hint), connection
                )
            lwd_logger.info(
                "[pre-recv] hint senders registered for %d connection(s)",
                len(hint_mqs),
            )
        else:
            # PD_mix is the only scheduling mode: a missing hint MQ =
            # hints never go out and the gate never opens; this must be
            # explicitly visible
            lwd_logger.warning(
                "[pre-recv] hint MQ missing: pre-recv gate will never open"
            )

    def _lwd_dispatch_message(self, peer: LwdPeer, msg: LwdMessage) -> bool:
        """IO thread: hand all edge business notifies to the ingest thread."""
        if not isinstance(msg, (LwdRangeNotify, LwdRequestNotify, LwdAbortNotify)):
            return False
        self._lwd_ingest_queue.put((peer, msg))
        return True

    def _lwd_ingest_loop(self) -> None:
        while (item := self._lwd_ingest_queue.get()) is not None:
            try:
                self._lwd_handle_edge_message(*item)
            except Exception as e:
                lwd_logger.exception(
                    "[cloud] ingest of %s failed", type(item[1]).__name__
                )
                self._lwd_fail(f"ingest of {type(item[1]).__name__}: {e!r}")
                return

    def _lwd_handle_edge_message(
        self, peer: LwdPeer, msg: LwdEdgeToCloudMessage
    ) -> None:
        lwd_logger.info(
            "[cloud] ingest msg=%s seqno=%s",
            type(msg).__name__,
            getattr(msg, "seqno", "-"),
        )
        connection_id = self._lwd_connection_of(peer)
        if connection_id in getattr(self, "_lwd_failed_connections", set()):
            if isinstance(msg, LwdRequestNotify):
                from vllm.v1.engine import FinishReason

                self._lwd_send_until_queued(
                    peer,
                    LwdC2eNotify(
                        hidden_num_elements=0,
                        top_id_ths=[[]],
                        num_accepted_tokens=[0],
                        req_ids=[msg.request_id],
                        finish_reasons=[int(FinishReason.ERROR)],
                        down_seqno=-1,
                    ),
                )
            return
        msg = _lwd_namescope_edge_message(msg, connection_id)
        if isinstance(msg, LwdRangeNotify):
            self._lwd_handle_range_notify(msg, connection_id=connection_id)
        elif isinstance(msg, LwdRequestNotify):
            self._lwd_request_connections[msg.request_id] = connection_id
            getattr(self, "_lwd_retired_connections", {}).pop(msg.request_id, None)
            self._lwd_handle_request_notify(msg)
        elif isinstance(msg, LwdAbortNotify):
            self._lwd_handle_abort_notify(msg)

    def _lwd_handle_range_notify(
        self,
        notify: LwdRangeNotify,
        *,
        connection_id: LwdConnectionId | None = None,
    ) -> None:
        """Post each new extended hint exactly once, per source connection."""
        connection_id = connection_id or lwd_message_connection_id(notify)
        rejected = getattr(self, "_lwd_rejected_nonces", {})
        if any(
            rejected.get(item.request_id) == item.prefix_nonce for item in notify.items
        ):
            lwd_logger.info(
                "[cloud] ignore RangeNotify for rejected generation seqno=%d",
                notify.seqno,
            )
            return
        current_nonces = getattr(self, "_lwd_request_nonces", {})
        if any(
            item.request_id in current_nonces
            and current_nonces[item.request_id] != item.prefix_nonce
            for item in notify.items
        ):
            lwd_logger.warning(
                "[cloud] ignore stale RangeNotify seqno=%d", notify.seqno
            )
            return
        latest_seqnos = getattr(self, "_lwd_latest_seqnos", None)
        if latest_seqnos is None:
            latest_seqnos = self._lwd_latest_seqnos = {}
        legacy_latest = (
            getattr(self, "_lwd_latest_seqno", -1)
            if connection_id == LwdConnectionId(0, 0, 0)
            else -1
        )
        latest = latest_seqnos.get(connection_id, legacy_latest)
        if notify.seqno <= latest:
            lwd_logger.info(
                "[cloud] ignore duplicate RangeNotify seqno=%d connection=%s",
                notify.seqno,
                connection_id,
            )
            return
        items = notify.items
        request_ids = [item.request_id for item in items]
        prompt_lens = [
            self._lwd_prompt_lens.get(
                request_id, self._lwd_retired_prompt_lens.get(request_id, 0)
            )
            for request_id in request_ids
        ]
        if any(prompt_len <= 0 for prompt_len in prompt_lens):
            raise RuntimeError(
                f"[LWD] RangeNotify seqno={notify.seqno} arrived before "
                "RequestNotify prompt lengths were recorded"
            )
        self._lwd_track_range(
            notify.seqno, request_ids, connection_id=connection_id
        )
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
            make_recv_hint,
            post_irecv_hint,
        )

        num_tokens = sum(item.num_tokens for item in items)
        lwd_logger.info(
            "[Lwd][pre-recv] HINT seqno=%d num_tokens=%d ts=%.3f",
            notify.seqno,
            num_tokens,
            time.monotonic(),
        )
        hint_fields: dict[str, Any] = {
            "req_ids": request_ids,
            "rows": [item.num_tokens for item in items],
            "prompt_offsets": [item.offset for item in items],
            "prompt_lens": prompt_lens,
            "prefix_nonces": [item.prefix_nonce for item in items],
        }
        if any(item.is_refill for item in items):
            hint_fields["refills"] = [item.is_refill for item in items]
        post_irecv_hint(
            make_recv_hint(
                notify.seqno,
                num_tokens,
                0,
                connection_id=connection_id,
                **hint_fields,
            )
        )

    def _lwd_track_range(
        self,
        seqno: int,
        request_ids: list[str],
        *,
        connection_id: LwdConnectionId | None = None,
    ) -> None:
        connection_id = connection_id or LwdConnectionId(0, 0, 0)
        latest_seqnos = getattr(self, "_lwd_latest_seqnos", None)
        if latest_seqnos is None:
            latest_seqnos = self._lwd_latest_seqnos = {}
        latest_seqnos[connection_id] = seqno
        if connection_id == LwdConnectionId(0, 0, 0) and hasattr(
            self, "_lwd_latest_seqno"
        ):
            self._lwd_latest_seqno = seqno
        pending_seqnos = getattr(self, "_lwd_pending_seqnos", None)
        if pending_seqnos is None:
            pending_seqnos = self._lwd_pending_seqnos = {}
        pending_seqnos.setdefault(connection_id, {})[seqno] = set(request_ids)
        for request_id in request_ids:
            self._lwd_request_seqnos.setdefault(request_id, set()).add(
                (connection_id, seqno)
            )

    def _lwd_complete_seqno(
        self,
        seqno: int,
        *,
        connection_id: LwdConnectionId | None = None,
    ) -> None:
        connection_id = connection_id or LwdConnectionId(0, 0, 0)
        pending_by_seqno = getattr(self, "_lwd_pending_seqnos", {}).get(
            connection_id
        )
        if pending_by_seqno is None:
            return
        request_ids = pending_by_seqno.pop(seqno, None)
        if not pending_by_seqno:
            self._lwd_pending_seqnos.pop(connection_id, None)
        if request_ids is None:
            return
        for request_id in request_ids:
            pending = self._lwd_request_seqnos.get(request_id)
            if pending is not None:
                pending.discard((connection_id, seqno))
            if pending:
                continue
            self._lwd_request_seqnos.pop(request_id, None)
            if request_id in self._lwd_retired_req_ids:
                self._lwd_clear_retired(request_id)

    def _lwd_handle_request_notify(self, notify: LwdRequestNotify) -> None:
        request_id = notify.request_id
        request_connections = getattr(self, "_lwd_request_connections", None)
        if request_connections is not None:
            request_connections[request_id] = lwd_message_connection_id(notify)
        current = getattr(self, "_lwd_request_nonces", {}).get(request_id)
        if (self._lwd_request_seqnos.get(request_id) and current is None) or (
            current is not None and current != notify.prefix_nonce
        ):
            self._lwd_rejected_nonces[request_id] = notify.prefix_nonce
            self._lwd_send_prefix_ready(
                request_id,
                0,
                notify.prefix_nonce,
                error="request id reused before previous generation drained",
            )
            lwd_logger.warning(
                "[cloud] reject early reuse req=%s nonce=%d",
                request_id,
                notify.prefix_nonce,
            )
            return
        self._lwd_prompt_lens[request_id] = notify.num_prompt_tokens
        self._lwd_request_nonces[request_id] = notify.prefix_nonce
        request = self._lwd_build_request(notify)
        if request.use_structured_output:
            # Wire-rebuilt requests bypass the native auto-compile entry
            # (core.py::preprocess_add_request); without this explicit init
            # the request would sit in WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
            # forever with no error.
            self.structured_output_manager.grammar_init(request)
        self.input_queue.put_nowait((EngineCoreRequestType.ADD, (request, 0)))
        lwd_logger.info(
            "[cloud] request added req=%s tokens=%d nonce=%d",
            request_id,
            notify.num_prompt_tokens,
            notify.prefix_nonce,
        )
        if not self.engines_running:
            self.engines_running = True
            lwd_logger.info(
                "[cloud] first request wakes a paused wave (local fallback)"
            )

    def _lwd_handle_abort_notify(self, notify: LwdAbortNotify) -> None:
        request_id = notify.request_id
        current = getattr(self, "_lwd_request_nonces", {}).get(request_id)
        retired = getattr(self, "_lwd_retired_nonces", {}).get(request_id)
        if request_id in self._lwd_retired_req_ids and retired == notify.prefix_nonce:
            lwd_logger.info(
                "[cloud] ignore duplicate abort req=%s nonce=%d",
                request_id,
                notify.prefix_nonce,
            )
            return
        if (
            getattr(self, "_lwd_rejected_nonces", {}).get(request_id)
            == notify.prefix_nonce
        ):
            lwd_logger.info(
                "[cloud] ignore abort for rejected generation req=%s nonce=%d",
                request_id,
                notify.prefix_nonce,
            )
            return
        if current is not None and current != notify.prefix_nonce:
            lwd_logger.warning(
                "[cloud] ignore stale abort req=%s nonce=%d current=%d",
                request_id,
                notify.prefix_nonce,
                current,
            )
            return
        if current is None and retired is not None and retired != notify.prefix_nonce:
            lwd_logger.warning(
                "[cloud] ignore stale abort req=%s nonce=%d retired=%d",
                request_id,
                notify.prefix_nonce,
                retired,
            )
            return
        if current is None and retired is None and notify.prefix_nonce != 0:
            lwd_logger.warning(
                "[cloud] ignore abort for unknown generation req=%s nonce=%d",
                request_id,
                notify.prefix_nonce,
            )
            return
        self._lwd_retire_request(request_id, preserve_prompt=True)
        lwd_logger.info(
            "[cloud] abort req=%s nonce=%d", request_id, notify.prefix_nonce
        )
        self.aborts_queue.put_nowait([request_id])
        self.input_queue.put_nowait((EngineCoreRequestType.ABORT, [request_id]))

    def _lwd_retire_request(
        self, request_id: str, preserve_prompt: bool = False
    ) -> None:
        clear_arrival = getattr(self.scheduler, "_lwd_clear_arrival", None)
        if clear_arrival is not None:
            clear_arrival(request_id)
        else:
            self.scheduler.lwd_recv_rows.pop(request_id, None)
        retired_nonces = getattr(self, "_lwd_retired_nonces", None)
        if retired_nonces is not None:
            retired_nonces[request_id] = getattr(self, "_lwd_request_nonces", {}).get(
                request_id, 0
            )
        if not self._lwd_request_seqnos.get(request_id):
            self._lwd_clear_retired(request_id)
            return
        self._lwd_retired_req_ids.add(request_id)
        if preserve_prompt:
            prompt_len = self._lwd_prompt_lens.get(request_id)
            if prompt_len is not None:
                self._lwd_retired_prompt_lens[request_id] = prompt_len

    def _lwd_clear_retired(self, request_id: str) -> None:
        if any(
            request_id in notify.req_ids
            for notify in getattr(self.scheduler, "pending_c2e", ())
        ):
            # Final and zero-accepted DOWN notifications may trail native
            # retirement. Keep the source route until post_step has flushed it.
            self._lwd_retired_req_ids.add(request_id)
            return
        connection_id = getattr(self, "_lwd_request_connections", {}).pop(
            request_id, None
        )
        if connection_id is not None:
            retired = getattr(self, "_lwd_retired_connections", None)
            if retired is None:
                retired = self._lwd_retired_connections = {}
            retired[request_id] = connection_id
            while len(retired) > _LWD_RETIRED_CONNECTION_CAP:
                retired.pop(next(iter(retired)))
        self._lwd_retired_req_ids.discard(request_id)
        self._lwd_retired_prompt_lens.pop(request_id, None)
        getattr(self, "_lwd_retired_nonces", {}).pop(request_id, None)
        getattr(self, "_lwd_request_nonces", {}).pop(request_id, None)
        self._lwd_request_seqnos.pop(request_id, None)
        self._lwd_prompt_lens.pop(request_id, None)
        getattr(self, "_lwd_rejected_nonces", {}).pop(request_id, None)

    def shutdown(self) -> None:
        # Before the executor goes: a pre-recv hint blocks on its queue.
        if self._lwd_ingest_thread is not None:
            self._lwd_ingest_queue.put(None)
            self._lwd_ingest_thread.join(_LWD_INGEST_JOIN_TIMEOUT_S)
        super().shutdown()

    def _lwd_build_request(self, req_notify: LwdRequestNotify) -> Request:
        """Build a placeholder request whose initial hashes come from the edge."""
        sampling_params = self._lwd_build_sampling_params(req_notify)
        prompt_embeds = torch.empty(
            req_notify.num_prompt_tokens,
            self.vllm_config.model_config.get_hidden_size(),
            dtype=self.vllm_config.model_config.dtype,
            device="meta",
        )
        local_hasher = getattr(self, "request_block_hasher", None)
        hash_block_size = getattr(self, "_lwd_hash_block_size", 0)
        expected = (
            req_notify.num_prompt_tokens // hash_block_size
            if local_hasher is not None and hash_block_size > 0
            else 0
        )

        def block_hasher(request: Request) -> list[bytes]:
            if request.num_output_tokens > 0:
                return []
            if (
                local_hasher is not None
                and not request.block_hashes
                and len(req_notify.block_hashes) == expected
            ):
                return list(req_notify.block_hashes)
            assert local_hasher is not None
            return local_hasher(request)

        request = Request(
            request_id=req_notify.request_id,
            prompt_token_ids=None,
            prompt_embeds=prompt_embeds,
            sampling_params=sampling_params,
            pooling_params=None,
            block_hasher=block_hasher if local_hasher is not None else None,
        )
        request.lwd_mrope_positions = req_notify.mrope_positions
        request.lwd_prefix_nonce = req_notify.prefix_nonce
        request.lwd_connection_id = lwd_message_connection_id(req_notify)
        request.lwd_wire_hashes_adopted = (
            local_hasher is not None and len(req_notify.block_hashes) == expected
        )
        return request

    @staticmethod
    def _lwd_build_sampling_params(req_notify: LwdRequestNotify) -> SamplingParams:
        """Rebuild sampling state carried by the request notify."""
        values = {f: getattr(req_notify, f) for f in LWD_SAMPLING_DEFAULTS}
        values["stop_token_ids"] = list(values["stop_token_ids"])
        sampling_params = SamplingParams(
            **values,
            skip_reading_prefix_cache=req_notify.skip_reading_prefix_cache,
        )
        sampling_params.update_from_generation_config({}, req_notify.eos_token_id)
        if req_notify.structured_outputs:
            # Decoded after update_from_generation_config so no later native
            # hook re-validates and overwrites the edge-chosen `_backend`.
            # Fail fast: a half-parsed request must never enter the queue.
            try:
                sampling_params.structured_outputs = lwd_decode_structured_outputs(
                    req_notify.structured_outputs, StructuredOutputsParams
                )
            except ValueError as exc:
                raise ValueError(
                    f"[LWD][cloud] request {req_notify.request_id}: {exc}"
                ) from exc
        LwdControlDiag.cloud_request_admitted(req_notify, sampling_params)
        return sampling_params

    def step_with_batch_queue(self):
        """Drain the early-recv watermark at the head of the step (done
        MQ -> gate readiness predicate); settle finished-request usage onto
        the coordination stream; the rest is native."""
        self._lwd_drain_irecv_completions()
        engine_core_outputs, model_executed = super().step_with_batch_queue()
        if engine_core_outputs:
            self._lwd_publish_finished_usage(engine_core_outputs)
        self._lwd_retire_finished_outputs(engine_core_outputs)
        return engine_core_outputs, model_executed

    def step(self):
        """Same for the synchronous step path: drain the watermark at the
        head of the step, settle usage, the rest is native."""
        self._lwd_drain_irecv_completions()
        engine_core_outputs, model_executed = super().step()
        if engine_core_outputs:
            self._lwd_publish_finished_usage(engine_core_outputs)
        self._lwd_retire_finished_outputs(engine_core_outputs)
        return engine_core_outputs, model_executed

    def _lwd_retire_finished_outputs(self, engine_core_outputs: Any) -> None:
        if not engine_core_outputs:
            return
        for outputs in engine_core_outputs.values():
            for request_id in outputs.finished_requests or ():
                self._lwd_retire_request(request_id)
            for output in outputs.outputs:
                if output.finish_reason is not None:
                    self._lwd_retire_request(output.request_id)

    def _lwd_drain_irecv_completions(self) -> None:
        """Drain each connection without sharing seqnos or arrival ledgers."""
        done_mqs = getattr(self, "_lwd_done_mqs", None)
        if not done_mqs:
            legacy_done_mq = getattr(self, "_lwd_done_mq", None)
            if legacy_done_mq is None:
                return
            done_mqs = {LwdConnectionId(0, 0, 0): legacy_done_mq}
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
            record_irecv_completions,
        )

        default_connection = LwdConnectionId(0, 0, 0)
        watermark_items: list[Any] = []
        for connection, done_mq in done_mqs.items():
            if done_mq is None:
                continue
            while True:
                try:
                    item = done_mq.dequeue(timeout=0)
                except TimeoutError:
                    break
                except Exception:
                    lwd_logger.exception(
                        "[pre-recv] done MQ drain failed for %s", connection
                    )
                    break
                if isinstance(item, tuple) and item and item[0] == "refill_done":
                    if len(item) == 2:
                        self.scheduler.lwd_refill_done(str(item[1]))
                    elif len(item) == 3:
                        self.scheduler.lwd_refill_done(str(item[1]), int(item[2]))
                    continue
                if (
                    isinstance(item, tuple)
                    and len(item) == 4
                    and item[1] == "refill_done"
                ):
                    if item[0] != connection:
                        raise ValueError("[LWD] refill_done crossed connections")
                    self.scheduler.lwd_refill_done(str(item[2]), int(item[3]))
                    continue
                if (
                    isinstance(item, tuple)
                    and len(item) == 4
                    and item[1] == "error"
                ):
                    if item[0] != connection:
                        raise ValueError("[LWD] receive error crossed connections")
                    self._lwd_fail_connection(connection, item[2], item[3])
                    continue
                if (
                    isinstance(item, tuple)
                    and len(item) == 3
                    and item[0] == "error"
                ):
                    self._lwd_fail_connection(connection, item[1], item[2])
                    continue
                if connection in getattr(self, "_lwd_failed_connections", set()):
                    continue
                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[0], tuple)
                ):
                    key, arrived = item
                    if len(key) == 3:
                        if key[0] != connection:
                            raise ValueError("[LWD] arrival crossed connections")
                        watermark_items.append(key)
                    elif len(key) == 2:
                        watermark_items.append(
                            key
                            if connection == default_connection
                            else (connection, *key)
                        )
                    else:
                        raise ValueError(f"[LWD] invalid arrival key {key}")
                    self._lwd_note_arrived(
                        key[-1], arrived, connection_id=connection
                    )
                else:
                    watermark_items.append(
                        item
                        if connection == default_connection
                        else (connection, *item)
                    )
        if not watermark_items:
            return
        advanced = record_irecv_completions(watermark_items)
        for item in advanced:
            if len(item) == 2:
                self._lwd_complete_seqno(item[1])
            else:
                self._lwd_complete_seqno(item[2], connection_id=item[0])
        lwd_logger.info(
            "[Lwd][pre-recv] DRAIN n=%d advanced=%s ts=%.3f",
            len(watermark_items),
            advanced,
            time.monotonic(),
        )

    def _lwd_fail_connection(
        self, connection: LwdConnectionId, seqno: int, error: Any
    ) -> None:
        """Abort one failed connection while preserving all other channels."""
        failed = getattr(self, "_lwd_failed_connections", None)
        if failed is None:
            failed = self._lwd_failed_connections = set()
        if connection in failed:
            return
        failed.add(connection)
        lwd_logger.error(
            "[pre-recv] connection %s failed at seqno=%d: %s",
            connection,
            seqno,
            error,
        )
        request_ids = [
            request_id
            for request_id, owner in getattr(
                self, "_lwd_request_connections", {}
            ).items()
            if owner == connection
        ]
        for request_id in request_ids:
            self._lwd_retire_request(request_id, preserve_prompt=True)
            self.aborts_queue.put_nowait([request_id])
            self.input_queue.put_nowait((EngineCoreRequestType.ABORT, [request_id]))

    def _lwd_note_arrived(
        self,
        seqno: int,
        arrived: Iterable[tuple[Any, ...]] | None,
        *,
        connection_id: LwdConnectionId | None = None,
    ) -> None:
        """Merge arrivals only into their owning connection and request."""
        connection_id = connection_id or LwdConnectionId(0, 0, 0)
        request_ids = getattr(self, "_lwd_pending_seqnos", {}).get(
            connection_id, {}
        ).get(seqno)
        if not arrived or request_ids is None:
            return
        touched: list[tuple[str, int, int]] = []
        for arrival in arrived:
            if len(arrival) == 2:
                request_id, rows = arrival
                resume_base = None
            elif len(arrival) == 3:
                request_id, rows, resume_base = arrival
                resume_base = int(resume_base)
            else:
                raise ValueError(f"[LWD] invalid arrival entry {arrival}")
            try:
                request_id, rows = str(request_id), int(rows)
            except (TypeError, ValueError):
                continue
            if (
                rows <= 0
                or request_id not in request_ids
                or request_id in self._lwd_retired_req_ids
            ):
                continue
            previous = self.scheduler.lwd_recv_rows.get(request_id, 0)
            coverage_of = getattr(self.scheduler, "_lwd_arrival_coverage", None)
            request = getattr(self.scheduler, "requests", {}).get(request_id)
            if coverage_of is not None and request is not None:
                previous = coverage_of(request)
            note_arrival = getattr(self.scheduler, "lwd_note_arrival", None)
            if note_arrival is None:
                if resume_base is not None:
                    raise RuntimeError("[LWD] scheduler lacks wire-arrival support")
                if rows > previous:
                    self.scheduler.lwd_recv_rows[request_id] = rows
            else:
                note_arrival(request_id, rows, resume_base=resume_base)
            current = self.scheduler.lwd_recv_rows.get(request_id, previous)
            if current != previous:
                touched.append((request_id, previous, current))
        if touched:
            lwd_logger.info("[cloud] arrival high watermarks updated: %s", touched)

    def _lwd_notify_connection(self, notify: LwdC2eNotify) -> LwdConnectionId:
        connection_id = lwd_message_connection_id(notify)
        if connection_id != LwdConnectionId(0, 0, 0):
            return connection_id
        owners = {
            owner
            for request_id in notify.req_ids
            if (
                owner := getattr(self, "_lwd_request_connections", {}).get(
                    request_id
                )
                or getattr(self, "_lwd_retired_connections", {}).get(request_id)
            )
            is not None
        }
        if len(owners) == 1:
            return next(iter(owners))
        if len(getattr(self, "_lwd_peers", {})) > 1:
            raise RuntimeError("[LWD] C2E batch has no unique source connection")
        return connection_id

    def post_step(self, model_executed: bool) -> None:
        """Flush pending c2e to the edge at the end of the step (the
        scheduler only queues them; both native step paths reach this hook)."""
        if self.scheduler.pending_c2e:
            multi_edge = getattr(self, "_lwd_multi_edge", False)
            finished_ids: list[str] = []
            for notify in self.scheduler.pending_c2e:
                if not notify.req_ids:
                    continue
                connection_id = self._lwd_notify_connection(notify)
                peer = self._lwd_peer_for_connection(connection_id)
                # Wire ids leave the engine namespaced ledgers only here:
                # resolve the source connection while the internal ids are
                # still intact, then restore the edge's original ids.
                self._lwd_send_until_queued(
                    peer,
                    msgspec.structs.replace(
                        notify,
                        req_ids=[
                            lwd_unwrap_request_id(request_id, multi_edge=multi_edge)
                            for request_id in notify.req_ids
                        ],
                    ),
                )
                for index, request_id in enumerate(notify.req_ids):
                    if (
                        notify.finish_reasons
                        and notify.finish_reasons[index] != -1
                    ):
                        finished_ids.append(request_id)
            self.scheduler.pending_c2e.clear()
            for request_id in finished_ids:
                self._lwd_clear_retired(request_id)
        super().post_step(model_executed)


class LwdCloudEngineCore(LwdCloudEngineCoreMixin, EngineCoreProc):
    """Cloud PO engine of one DP rank (dp=1 and non-MoE multi-DP): the
    engine processes of a multi-DP non-MoE cloud are independent, so the
    native EngineCoreProc base is the whole story."""


class LwdCloudDPEngineCore(LwdCloudEngineCoreMixin, DPEngineCoreProc):
    """Cloud PO engine of a MoE multi-DP cloud: the native DP engine base
    carries the wave state machine, dummy batches, START_DP_WAVE handling
    and stats publishing, so the MoE engines stay in lockstep exactly as a
    centralized DP deployment; the LWD seams come from the mixin."""
