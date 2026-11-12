# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cloud engine cores serving multiple edge connections per DP rank.

Range notifies post extended early-receive hints, request notifies become
native ADDs with prompt-length and M-RoPE metadata, and abort notifies use the
native ABORT path. Step metadata is flushed to the edge in post_step.

The LWD seams live in LwdCloudEngineCoreMixin and are orthogonal to the
engine base: LwdCloudEngineCore (EngineCoreProc) serves one DP rank of a
cloud instance and non-MoE multi-DP deployments; LwdCloudDPEngineCore
(DPEngineCoreProc) serves MoE multi-DP deployments, inheriting the native
DP semantics (wave state machine, dummy batches, START_DP_WAVE, stats
publishing) unchanged.
"""  # 【MR2】模块说明更新：一个云 DP rank 现在同时服务多条边连接，hint/done/路由都按连接分开

from __future__ import annotations

import json  # 【MR2】引入 json：解析边侧传来的结构化输出参数
import queue
import threading
import time
from collections.abc import Iterable
from typing import Any

import msgspec  # 【MR2】引入 msgspec：用 structs.replace 生成带边前缀的新消息，不改原对象
import torch

from vllm.config.lwd import LwdConnectionId  # 【MR2】云侧所有按连接的状态都以 LwdConnectionId 为 key
from vllm.lwd_logger import lwd_logger
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc, EngineShutdownState
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_scheduler import (
    LwdCloudScheduler,
)
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_prefix_cache import (
    LwdProbeQueryAdapter,  # 【MR2】引入前缀探测查询适配器：云端负责回答边侧的探测请求
    init_lwd_prefix_cache,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_engine import (
    LwdBaseEngineCore,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    lwd_spec_factor,
)
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import LwdPeer
from vllm.v1.lwd_control.lwd_comm.lwd_id_adapter import unwrap_req_id, wrap_req_id  # 【MR2】引入 wrap/unwrap_req_id：云端请求 id 加边前缀，回包时还原
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LWD_SAMPLING_DEFAULTS,
    LwdAbortNotify,
    LwdC2eNotify,  # 【MR2】LwdC2eNotify：给已失败连接回 ERROR 完成通知时使用
    LwdCloudParamsNotify,
    LwdCloudToEdgeMessage,
    LwdEdgeToCloudMessage,
    LwdMessage,
    LwdPrefixReadyNotify,
    LwdRangeNotify,
    LwdRefillNotify,
    LwdRequestNotify,
    lwd_decode_edge_to_cloud,
)
from vllm.v1.lwd_control.lwd_control_diag import LwdControlDiag
from vllm.v1.request import Request

# Sleep between retries of a full outbound queue: step metadata must not be
# lost, the edge posts its receive of the exact size from it.
_LWD_SEND_RETRY_SLEEP_S = 0.05
_LWD_RETIRED_ROUTE_CAP = 4096  # 【MR2】退休路由表上限，防止长期运行无限增长
# Debug-level cloud concurrency snapshot interval.  # 【MR2】云侧并发快照日志间隔
_LWD_CLOUD_CONC_LOG_INTERVAL_S = 10.0


def _lwd_decode_structured_outputs(payload: str) -> StructuredOutputsParams | None:  # 【MR2】把边侧传来的 JSON 还原成 StructuredOutputsParams
    """Rebuild the frontend-validated structured-output parameters."""
    if not payload:
        return None
    data = json.loads(payload)  # 【MR2】解析 JSON 载荷
    backend = data.pop("_backend", None)  # 【MR2】backend 是私有字段，构造前先单独取出
    data.pop("_backend_was_auto", None)
    params = StructuredOutputsParams(**data)  # 【MR2】按普通字段重建参数对象
    params._backend = backend  # 【MR2】再写回 backend
    return params


def _lwd_log_sent_to_edge(peer: LwdPeer, message: LwdCloudToEdgeMessage) -> None:  # 【MR2】云端发送消息的统一日志：和边侧 “[edge] c2e received” 配对
    """Full-payload log of one cloud->edge control-plane message once queued
    for the peer; pairs with the edge-side `[edge] c2e received` log. The
    per-step c2e notify gets per-row detail (top tree / token payloads are
    reduced to lengths to bound log volume)."""
    if isinstance(message, LwdC2eNotify):  # 【MR2】C2E 逐行打细节，但 top tree / token 载荷只打长度，控制日志体积
        lwd_logger.info(  # 【MR2】C2E 的关键字段：seqno、请求数、accepted、finish、nonce
            "[cloud] c2e sent to edge %s seqno=%d reqs=%d hidden_numel=%d "
            "req_ids=%s accepted=%s finish=%s nonces=%s top_tree_lens=%s "
            "token_lens=%s",
            peer,
            message.down_seqno,
            len(message.req_ids),
            message.hidden_num_elements,
            message.req_ids,
            message.num_accepted_tokens,
            message.finish_reasons,
            message.prefix_nonces,
            [len(top) for top in message.top_id_ths],  # 【MR2】top tree 只打长度
            [len(tokens) for tokens in message.token_ids],  # 【MR2】token 列表只打长度
        )
    else:
        lwd_logger.info(  # 【MR2】其它消息直接打印类型和内容
            "[cloud] %s sent to edge %s: %s",
            type(message).__name__,
            peer,
            message,
        )


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

        dp_idx = vllm_config.parallel_config.data_parallel_index  # 【MR2】本进程负责的云 DP 下标
        connections = tuple(  # 【MR2】从拓扑里取属于这个 DP 的所有连接（一条边一条）
            connection.connection_id
            for connection in vllm_config.lwd_config.connections
            if connection.dp_idx == dp_idx
        ) or (LwdConnectionId(0, vllm_config.lwd_config.instance_id, dp_idx),)  # 【MR2】拓扑没有显式连接时退化为单连接默认值，保持 1E1C 兼容
        self._lwd_hint_mqs = {  # 【MR2】每条连接一个 hint 队列，避免多边消息混在一起
            connection: create_hint_mq(connection) for connection in connections
        }
        super().__init__(*args, lwd_dp_size=lwd_dp_size, **kwargs)
        # The cloud endpoint worker created its done-report MQ during
        # init_device; now that the executor is up, fetch its handle over
        # the executor's collective RPC (the executor itself stays free of
        # LWD plumbing). Exactly one worker - the endpoint rank - must
        # report a handle; anything else is a wiring error.
        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

        self._lwd_done_mqs = {}  # 【MR2】每条连接一个 done 队列
        for connection in self._lwd_hint_mqs:
            handles = [  # 【MR2】通过 RPC 向 worker 取该连接的 done MQ handle
                handle
                for handle in self.model_executor.collective_rpc(
                    "lwd_early_recv_done_handle", args=(connection,)
                )
                if handle is not None
            ]
            if len(handles) != 1:  # 【MR2】每条连接必须恰好一个端点 worker 上报，否则是装配错误
                raise RuntimeError(
                    "[LWD] early receive expects one endpoint worker reporting "
                    f"completions for {connection}, got {len(handles)}"
                )
            self._lwd_done_mqs[connection] = MessageQueue.create_from_handle(  # 【MR2】用 handle 在引擎侧重建消息队列
                handles[0], 0
            )
        lwd_logger.info(  # 【MR2】打印创建的队列数量，便于确认每连接都建好
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
        self.lwd_config.validate_prefix_caching(  # 【MR2】启动时校验前缀缓存配置与云侧能力一致
            self.vllm_config.cache_config.enable_prefix_caching
        )
        self._lwd_init_prefix_cache()
        self._lwd_edges: set[LwdPeer] = set()  # 【MR2】记录已注册的边侧 peer
        self._lwd_prompt_lens: dict[str, int] = {}
        self._lwd_retired_prompt_lens: dict[str, int] = {}
        self._lwd_request_nonces: dict[str, int] = {}
        self._lwd_retired_nonces: dict[str, int] = {}
        self._lwd_retired_req_ids: set[str] = set()
        self._lwd_rejected_nonces: dict[str, int] = {}
        self._lwd_request_routes: dict[str, LwdPeer] = {}  # 【MR2】请求 -> 来源 peer 的路由表（云端以此决定回包发给谁）
        self._lwd_retired_routes: dict[str, LwdPeer] = {}  # 【MR2】退休路由：请求结束后仍可能收到迟到的 DOWN 元数据
        self._lwd_pending_seqnos: dict[  # 【MR2】每条连接各自的 pending seqno -> 请求集合
            LwdConnectionId, dict[int, set[str]]
        ] = {}
        self._lwd_request_seqnos: dict[str, set[int]] = {}
        self._lwd_latest_seqnos: dict[LwdConnectionId, int] = {}  # 【MR2】每条连接已收到的最大 seqno
        self._lwd_failed_connections: set[LwdConnectionId] = set()  # 【MR2】已失败的连接集合，失败后只拒收该连接的新消息
        self.scheduler.lwd_hash_block_size = self._lwd_hash_block_size  # 【MR2】把 hash 块大小交给调度器，探测和调度用同一粒度
        self._lwd_prefix_hits: dict[str, int] = {}  # 【MR2】每个请求当前的前缀命中数（usage 统计用）
        prefix_hash_block_size = self._lwd_hash_block_size  # 【MR2】保存 hash 块大小快照
        coordination = self.lwd_config.coordination
        self._lwd_probe_query = None  # 【MR2】前缀探测查询器：没有协调配置时保持 None
        # The probe query serves probes whenever coordination is enabled,  # 【MR2】只要协调开启就创建探测查询，不依赖是否开前缀缓存
        # NOT only with prefix caching: probe replies also carry placement  # 【MR2】探测回复同时携带位置/路由信息（engine_index、cloud_id）和健康状态
        # and routing data (engine_index / cloud_id) plus health
        # bookkeeping, which must keep flowing with prefix caching off.
        # With caching disabled the NoPrefixCache KV coordinator returns an
        # empty hit, so probes answer with zero hits instead of hanging in
        # a consumer-less HTTP command queue until the 20s timeout.
        if coordination is not None and coordination.enabled:  # 【MR2】协调启用时创建探测查询适配器
            self._lwd_probe_query = LwdProbeQueryAdapter(  # 【MR2】基于本 DP 的 KV manager
                manager=self.scheduler.kv_cache_manager,
                hash_block_size=self._lwd_hash_block_size,  # 【MR2】按压缩状态边界对齐命中
                alignment_tokens=self._lwd_prefix_state.alignment_tokens,
                instance_id=f"cloud-{self.lwd_config.instance_id}",  # 【MR2】实例标识用于协调器区分云实例
                cloud_id=self.lwd_config.instance_id,  # 【MR2】cloud_id 就是云实例编号
                # With caching off the query only carries placement/health;
                # the manifest chain is built at the edge's own granularity,
                # so the cloud hash-block-size contract does not apply.
                prefix_caching=self._lwd_prefix_enabled,  # 【MR2】缓存关闭时探测只做位置/健康，命中按 0 返回
            )
        self.scheduler.lwd_prefix_hits = self._lwd_prefix_hits  # 【MR2】调度器共享同一份命中表
        self._lwd_coord_pending: queue.Queue[dict[str, Any]] = queue.Queue()  # 【MR2】协调命令的本地队列（IO 线程 -> 引擎线程）
        self._lwd_control_processor = None  # 【MR2】控制命令处理器
        self.cloud_control_event_queue = getattr(  # 【MR2】serving 层注入的协调事件队列
            self, "cloud_control_event_queue", None
        )
        command_queue = getattr(self, "cloud_control_command_queue", None)  # 【MR2】serving 层注入的协调命令队列
        if self._lwd_probe_query is not None:  # 【MR2】启用探测但没有 HTTP 队列属于装配错误
            if command_queue is None or self.cloud_control_event_queue is None:
                raise RuntimeError(
                    "[LWD] prefix caching requires HTTP coordination queues"
                )
            from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_control import (
                LwdCloudControlProcessor,
            )

            self._lwd_control_processor = LwdCloudControlProcessor(  # 【MR2】处理器把探测结果写进本地队列
                self._lwd_coord_pending, self.cloud_control_event_queue
            )
            threading.Thread(  # 【MR2】独立线程把 HTTP 命令搬到本地并唤醒引擎
                target=self._lwd_coord_wake_loop,
                name="lwd-ctrl-wake",
                daemon=True,
            ).start()
        self._lwd_cloud_conc_last_log = 0.0  # 【MR2】并发日志时间戳
        self._lwd_engine_index = int(  # 【MR2】引擎索引就是 DP 下标
            self.vllm_config.parallel_config.data_parallel_index
        )
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
            prefix_hash_block_size=prefix_hash_block_size,  # 【MR2】把 hash 块大小随云参数下发给边侧
        )
        self._lwd_register_recv_hint()
        # Business messages are handed to the engine input queue. Request,  # 【MR2】业务消息统一投递到引擎输入队列，请求/预留/路由只有一个所有者：引擎线程
        # reservation, routing and coordinator state therefore have one owner:
        # the engine thread, same as the native scheduler and KV manager.
        plane = self._lwd_start_control_plane()
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

    def _lwd_decode(self, payload: bytes) -> LwdMessage:
        return lwd_decode_edge_to_cloud(payload)

    def _lwd_handle_peer_registered(self, peer: LwdPeer) -> None:
        """Engine thread: remember the edge and send it the scheduling
        limits before anything else."""
        self._lwd_edges.add(peer)  # 【MR2】先把 peer 记入集合，再下发调度参数
        self._lwd_send_until_queued(peer, self._lwd_cloud_params)
        lwd_logger.info(
            "[cloud] edge %s registered; scheduling limits sent: %s",
            peer,
            self._lwd_cloud_params,
        )

    def _lwd_send_until_queued(
        self, peer: LwdPeer, message: LwdCloudToEdgeMessage
    ) -> None:
        """Retry a full outbound queue until the message is queued; gives up
        with an error once shutdown is requested, so a stalled edge cannot
        block it."""
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
        _lwd_log_sent_to_edge(peer, message)  # 【MR2】消息成功入队后统一打发送日志

    def _lwd_send_prefix_ready(
        self, request_id: str, hit_tokens: int, nonce: int, error: str = ""
    ) -> None:
        peer = self._lwd_request_routes.get(request_id)  # 【MR2】按请求路由表找回包 peer
        if peer is None:
            raise RuntimeError(f"[LWD] prefix reply route missing for {request_id}")  # 【MR2】找不到路由说明状态机错乱，直接报错
        self._lwd_send_until_queued(
            peer,  # 【MR2】回包前把请求 id 还原成边侧认识的形式
            LwdPrefixReadyNotify(  # 【MR2】PrefixReady 只带回边侧请求 id、命中数和 nonce
                unwrap_req_id(request_id), hit_tokens, nonce, error
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
        peer = self._lwd_request_routes.get(request_id)  # 【MR2】refill 同样按请求路由找 peer
        if peer is None:
            raise RuntimeError(f"[LWD] refill route missing for {request_id}")  # 【MR2】路由丢失直接失败
        self._lwd_send_until_queued(
            peer,  # 【MR2】refill 通知也还原请求 id
            LwdRefillNotify(
                unwrap_req_id(request_id), start_row, end_row, prefix_nonce
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
        self._lwd_prefix_hits[request_id] = hit_tokens  # 【MR2】记录本次命中数，供结束时的 usage 使用
        self._lwd_send_prefix_ready(request_id, hit_tokens, nonce)

    def _lwd_register_recv_hint(self) -> None:
        """[LWD][pre-recv] Early receive: the hint delivery channel =
        engine -> cloud endpoint worker comm thread. It goes through
        the sideband MQ this engine creates in __init__ (not
        rpc_broadcast_mq: that one is consumed single-threaded by the
        busy_loop and gets blocked by a several-hundred-ms prefill
        forward, so the hint would no longer be "early")."""
        if self._lwd_hint_mqs:  # 【MR2】只有创建了 hint 队列才注册发送器
            from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
                register_hint_sender,
            )

            for connection, hint_mq in self._lwd_hint_mqs.items():  # 【MR2】逐连接注册 hint 发送器
                if hint_mq is None:
                    continue
                # Blocking enqueue: once the gate is on, hints are a
                # correctness dependency; dropping one stalls its connection.
                register_hint_sender(  # 【MR2】闭包按连接绑定各自的队列，不会串到别的连接
                    lambda hint, mq=hint_mq: mq.enqueue(hint), connection
                )
            lwd_logger.info(  # 【MR2】打印注册的连接数
                "[pre-recv] hint senders registered for %d connection(s)",
                len(self._lwd_hint_mqs),
            )
        else:
            # PD_mix is the only scheduling mode: a missing hint MQ =
            # hints never go out and the gate never opens; this must be
            # explicitly visible
            lwd_logger.warning(
                "[pre-recv] hint MQ missing: pre-recv gate will never open"
            )

    def _lwd_dispatch_message(self, peer: LwdPeer, msg: LwdMessage) -> bool:
        """IO thread: preserve order and hand business messages to the engine."""  # 【MR2】IO 线程：保持消息顺序，把业务消息交给引擎线程
        if not isinstance(  # 【MR2】只转发 Range/Request/Abort 三类业务消息
            msg,
            (
                LwdRangeNotify,
                LwdRequestNotify,
                LwdAbortNotify,
            ),
        ):
            return False
        self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, (peer, msg)))  # 【MR2】把（peer, msg）打包进输入队列，由引擎线程处理
        return True

    def _handle_client_request(  # 【MR2】覆盖客户端请求处理：识别内部（peer, msg）消息
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        if (  # 【MR2】WAKEUP + (peer, msg) 是控制面投递格式
            request_type == EngineCoreRequestType.WAKEUP
            and isinstance(request, tuple)
            and len(request) == 2
            and isinstance(request[0], LwdPeer)
        ):
            peer, msg = request  # 【MR2】解包 peer 和消息
            try:
                self._lwd_handle_edge_message(peer, msg)  # 【MR2】在引擎线程里处理，保证状态单线程访问
            except Exception as exc:  # 【MR2】处理异常时记录堆栈
                lwd_logger.exception(
                    "[cloud] ingest of %s failed", type(msg).__name__  # 【MR2】记录 ingest 失败的异常堆栈
                )
                self._lwd_fail(f"ingest of {type(msg).__name__}: {exc!r}")  # 【MR2】吞掉异常会让引擎状态不一致，直接置为失败
            return  # 【MR2】内部消息处理完就返回
        super()._handle_client_request(request_type, request)

    def _lwd_handle_edge_message(
        self, peer: LwdPeer, msg: LwdEdgeToCloudMessage
    ) -> None:
        lwd_logger.info(
            "[cloud] ingest msg=%s seqno=%s",
            type(msg).__name__,
            getattr(msg, "seqno", "-"),
        )
        connection_id = self._lwd_connection_of(peer)  # 【MR2】由 peer 推出连接编号，不再从消息字段读
        if connection_id in self._lwd_failed_connections:  # 【MR2】该连接已失败：只拒绝这条连接的新请求
            if isinstance(msg, LwdRequestNotify):
                from vllm.v1.engine import FinishReason

                self._lwd_send_until_queued(
                    peer,
                    LwdC2eNotify(  # 【MR2】给失败连接回一条 ERROR C2E，让边侧尽快结束请求
                        hidden_num_elements=0,
                        top_id_ths=[[]],
                        num_accepted_tokens=[0],
                        req_ids=[msg.request_id],
                        finish_reasons=[int(FinishReason.ERROR)],
                    ),
                )
            return
        if isinstance(msg, LwdRangeNotify):
            # Every request id in the batch enters the cloud namespace once.  # 【MR2】RangeNotify：批内每个请求 id 都加边前缀，进入云端命名空间
            msg = msgspec.structs.replace(  # 【MR2】msgspec 结构不可变，用 replace 生成新消息
                msg,
                items=[
                    msgspec.structs.replace(
                        item, request_id=wrap_req_id(peer.instance_id, item.request_id)  # 【MR2】wrap_req_id 生成 e{edge}-{req_id}
                    )
                    for item in msg.items
                ],
            )
            self._lwd_handle_range_notify(connection_id, msg)  # 【MR2】按连接处理 RangeNotify
            return
        if not isinstance(msg, (LwdRequestNotify, LwdAbortNotify)):
            return
        canonical_id = wrap_req_id(peer.instance_id, msg.request_id)  # 【MR2】Request/Abort 也统一加边前缀
        msg = msgspec.structs.replace(msg, request_id=canonical_id)  # 【MR2】替换成带前缀的新消息
        if isinstance(msg, LwdRequestNotify):
            self._lwd_retired_routes.pop(canonical_id, None)  # 【MR2】新请求复用旧 id 时，清掉上一代的路由墓碑
            self._lwd_request_routes[canonical_id] = peer  # 【MR2】记录该请求的来源 peer，后续所有回包都发这里
            self._lwd_handle_request_notify(msg)
        else:  # 【MR2】否则按 abort 通知处理（同样已加边前缀）
            self._lwd_handle_abort_notify(msg)

    def _lwd_connection_of(self, peer: LwdPeer) -> LwdConnectionId:  # 【MR2】由 peer 计算连接编号：(边实例, 本云实例, 边侧 DP rank)
        return LwdConnectionId(
            peer.instance_id, self.lwd_config.instance_id, peer.dp_rank
        )

    def _lwd_handle_range_notify(  # 【MR2】按连接处理 RangeNotify
        self, connection_id: LwdConnectionId, notify: LwdRangeNotify
    ) -> None:
        """Post each new extended hint exactly once, per source connection."""
        rejected = getattr(self, "_lwd_rejected_nonces", {})
        if any(
            rejected.get(item.request_id) == item.prefix_nonce for item in notify.items
        ):
            return
        current_nonces = getattr(self, "_lwd_request_nonces", {})
        if any(
            item.request_id in current_nonces
            and current_nonces[item.request_id] != item.prefix_nonce
            for item in notify.items
        ):
            lwd_logger.warning("[cloud] ignore stale RangeNotify seqno=%d", notify.seqno)  # 【MR2】nonce 已过期的 Range 直接丢弃
            return
        latest = self._lwd_latest_seqnos.get(connection_id, -1)  # 【MR2】取该连接已收到的最大 seqno
        if notify.seqno <= latest:  # 【MR2】seqno 不递增视为重复通知，忽略
            lwd_logger.info(
                "[cloud] ignore duplicate RangeNotify seqno=%d", notify.seqno
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
        self._lwd_track_range(connection_id, notify.seqno, request_ids)  # 【MR2】登记该连接该 seqno 对应的请求集合
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
        post_irecv_hint(  # 【MR2】hint 必须带连接编号，worker 才能找到对应通道
            make_recv_hint(
                notify.seqno,
                num_tokens,
                connection_id=connection_id,
                **hint_fields,
            )
        )

    def _lwd_track_range(  # 【MR2】记录连接内的 seqno 进度
        self,
        connection_id: LwdConnectionId,
        seqno: int,
        request_ids: list[str],
    ) -> None:
        self._lwd_latest_seqnos[connection_id] = seqno  # 【MR2】更新该连接的最新 seqno
        self._lwd_pending_seqnos.setdefault(connection_id, {})[seqno] = set(  # 【MR2】pending 表按连接分开，seqno 相同也不会串
            request_ids
        )
        for request_id in request_ids:
            self._lwd_request_seqnos.setdefault(request_id, set()).add(seqno)

    def _lwd_complete_seqno(  # 【MR2】某条连接的 seqno 完成
        self, connection_id: LwdConnectionId, seqno: int
    ) -> None:
        pending_by_seqno = self._lwd_pending_seqnos.get(connection_id)
        if pending_by_seqno is None:
            return
        request_ids = pending_by_seqno.pop(seqno, None)  # 【MR2】从 pending 中移除，空表顺手删除
        if not pending_by_seqno:
            self._lwd_pending_seqnos.pop(connection_id, None)
        if request_ids is None:
            return
        for request_id in request_ids:
            pending = self._lwd_request_seqnos.get(request_id)
            if pending is not None:
                pending.discard(seqno)
            if pending:
                continue
            self._lwd_request_seqnos.pop(request_id, None)
            if request_id in self._lwd_retired_req_ids:
                self._lwd_clear_retired(request_id)

    def _lwd_handle_request_notify(self, notify: LwdRequestNotify) -> None:
        request_id = notify.request_id
        current = getattr(self, "_lwd_request_nonces", {}).get(request_id)
        if (self._lwd_request_seqnos.get(request_id) and current is None) or (
            current is not None and current != notify.prefix_nonce
        ):
            self._lwd_rejected_nonces[request_id] = notify.prefix_nonce
            if notify.prefix_nonce:  # 【MR2】nonce 非零且请求 id 被复用：回错误而不是让云端崩溃
                self._lwd_send_prefix_ready(  # 【MR2】给边侧回 0 命中的 PrefixReady，带错误原因
                    request_id,
                    0,
                    notify.prefix_nonce,
                    error="request id reused before previous generation drained",
                )
                return  # 【MR2】没有 nonce 时仍按 fail-fast 处理
            raise RuntimeError(
                f"[LWD] request_id {request_id} reused before previous generation drained"
            )
        self._lwd_prompt_lens[request_id] = notify.num_prompt_tokens
        if notify.prefix_nonce:  # 【MR2】记录本轮请求的 nonce，后续 Range/Abort 校验用
            self._lwd_request_nonces[request_id] = notify.prefix_nonce
        request = self._lwd_build_request(notify)
        if request.use_structured_output:
            # This path constructs Request directly and therefore bypasses  # 【MR2】这里直接构造 Request，绕过了原生的语法初始化钩子，需要手动补
            # EngineCore.preprocess_add_request, the native grammar-init hook.
            self.structured_output_manager.grammar_init(request)  # 【MR2】云侧编译结构化输出语法
        self.input_queue.put_nowait((EngineCoreRequestType.ADD, (request, 0)))
        lwd_logger.info(
            "[cloud] request added req=%s tokens=%d nonce=%d",
            request_id,
            notify.num_prompt_tokens,
            notify.prefix_nonce,
        )
        # A coordinated MoE DP engine must let the native ADD path observe the
        # paused state and emit start_wave. Waking only this rank here suppresses  # 【MR2】有协调器的 MoE DP 必须走原生 ADD 路径，让 wave 状态机发出 start_wave
        # that notification and can leave its peers asleep for the next collective.
        native_dp_wake = isinstance(self, DPEngineCoreProc) and self.has_coordinator
        if not self.engines_running and not native_dp_wake:  # 【MR2】协调 DP 不在这里抢先把 engines_running 置 True
            self.engines_running = True
            lwd_logger.info(
                "[cloud] first request wakes a paused wave (local fallback)"
            )

    def _lwd_handle_abort_notify(self, notify: LwdAbortNotify) -> None:
        request_id = notify.request_id
        current = getattr(self, "_lwd_request_nonces", {}).get(request_id)
        retired = getattr(self, "_lwd_retired_nonces", {}).get(request_id)
        if request_id in self._lwd_retired_req_ids and retired == notify.prefix_nonce:
            return
        if (
            getattr(self, "_lwd_rejected_nonces", {}).get(request_id)
            == notify.prefix_nonce
        ):
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
            return
        if current is None and retired is None and notify.prefix_nonce != 0:
            lwd_logger.warning(
                "[cloud] ignore abort for unknown generation req=%s nonce=%d",
                request_id,
                notify.prefix_nonce,
            )
            return
        self.scheduler._lwd_refill_gate.discard(request_id)  # 【MR2】abort 时清掉该请求的 refill 门控
        self.scheduler._lwd_refill_pending.pop(request_id, None)  # 【MR2】清掉 refill 等待项
        reservations = getattr(self.scheduler, "lwd_prefix_reservations", None)  # 【MR2】取前缀预留管理器
        if reservations is not None:
            reservations.release(request_id)  # 【MR2】释放该请求占用的前缀预留
        self._lwd_retire_request(request_id, preserve_prompt=True)
        lwd_logger.info(
            "[cloud] abort req=%s nonce=%d", request_id, notify.prefix_nonce
        )
        self.aborts_queue.put_nowait([request_id])
        self.input_queue.put_nowait((EngineCoreRequestType.ABORT, [request_id]))

    def _lwd_retire_request(
        self, request_id: str, preserve_prompt: bool = False
    ) -> None:
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
        if any(request_id in notify.req_ids for notify in self.scheduler.pending_c2e):  # 【MR2】还有待发 C2E 涉及该请求时，先保留墓碑
            # A final or zero-accepted notification still needs the original
            # route. Keep the tombstone until post_step has flushed it.
            self._lwd_retired_req_ids.add(request_id)
            return  # 【MR2】等 post_step 冲刷完再清路由
        route = self._lwd_request_routes.pop(request_id, None)  # 【MR2】路由移到退休表
        if route is not None:
            # DOWN metadata can trail native request retirement by one or more
            # asynchronous steps. Keep a bounded tombstone for those batches.
            self._lwd_retired_routes[request_id] = route  # 【MR2】DOWN 元数据可能比原生退休晚若干步
            while len(self._lwd_retired_routes) > _LWD_RETIRED_ROUTE_CAP:  # 【MR2】退休路由超过上限就淘汰最旧的
                self._lwd_retired_routes.pop(next(iter(self._lwd_retired_routes)))
        self._lwd_retired_req_ids.discard(request_id)
        self._lwd_retired_prompt_lens.pop(request_id, None)
        getattr(self, "_lwd_retired_nonces", {}).pop(request_id, None)
        getattr(self, "_lwd_request_nonces", {}).pop(request_id, None)
        self._lwd_request_seqnos.pop(request_id, None)
        self._lwd_prompt_lens.pop(request_id, None)
        getattr(self, "_lwd_rejected_nonces", {}).pop(request_id, None)

    def shutdown(self) -> None:
        super().shutdown()

    def _lwd_build_request(self, req_notify: LwdRequestNotify) -> Request:
        """Build a placeholder request and adopt the edge's native wire hashes
        for single-instance prefix negotiation."""  # 【MR2】注释：云端只持有 prompt embedding 的 meta 占位，请求对象按边侧通知构建
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
            # The cloud holds only a meta placeholder for the prompt  # 【MR2】云端只有 prompt embedding 的 meta 占位，真实行稍后从 UP 通道到达
            # embeddings (the real rows arrive later over the UP channel), so
            # a local token/embedding hash raises "Cannot copy out of meta  # 【MR2】本地对 meta tensor 做 hash 会报错，所以这里只能采用边侧送来的链
            # tensor".  Adopt the edge's wire chain only when it matches the
            # cloud hash granularity; otherwise leave the request without
            # hashes and let add_request fail closed through
            # lwd_wire_hashes_adopted.
            if (  # 【MR2】只有长度匹配云侧粒度才采用
                local_hasher is not None
                and not request.block_hashes
                and len(req_notify.block_hashes) == expected
            ):
                return list(req_notify.block_hashes)
            return []  # 【MR2】否则返回空 hash 链，让 add_request 走 fail-closed

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
        request.lwd_prefix_probe_succeeded = req_notify.prefix_probe_succeeded  # 【MR2】把边侧探测是否成功传给请求对象
        request.lwd_wire_hashes_adopted = (
            local_hasher is not None and len(req_notify.block_hashes) == expected
        )
        request.block_hashes = list(req_notify.block_hashes)  # 【MR2】采用边侧的 block hash 链
        return request

    @staticmethod
    def _lwd_build_sampling_params(req_notify: LwdRequestNotify) -> SamplingParams:
        """Rebuild sampling state carried by the request notify."""
        values = {f: getattr(req_notify, f) for f in LWD_SAMPLING_DEFAULTS}
        values["stop_token_ids"] = list(values["stop_token_ids"])
        sampling_params = SamplingParams(
            **values,
            skip_reading_prefix_cache=req_notify.skip_reading_prefix_cache,
            structured_outputs=_lwd_decode_structured_outputs(  # 【MR2】结构化输出从 JSON 还原后交给 SamplingParams
                req_notify.structured_outputs
            ),
        )
        sampling_params.update_from_generation_config({}, req_notify.eos_token_id)
        LwdControlDiag.cloud_request_admitted(req_notify, sampling_params)
        return sampling_params

    def _lwd_coord_wake_loop(self) -> None:  # 【MR2】协调唤醒线程：HTTP 命令搬到本地队列后唤醒引擎
        """Move MP commands to a local queue and wake the engine thread."""
        command_queue = self.cloud_control_command_queue
        while True:
            try:
                command = command_queue.get()  # 【MR2】阻塞等待命令；对端关闭时退出
            except (EOFError, OSError):
                return
            self._lwd_coord_pending.put(command)  # 【MR2】先放进本地队列，保证引擎线程按序处理
            self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))  # 【MR2】唤醒引擎线程

    def _lwd_poll_coordination(self) -> None:  # 【MR2】在引擎线程轮询协调命令
        probe_query = getattr(self, "_lwd_probe_query", None)
        if probe_query is None:
            return
        processor = getattr(
            self, "cloud_control_processor", self._lwd_control_processor
        )
        if processor is not None:  # 【MR2】处理所有待处理命令（前缀探测等）
            processor.poll(probe_query)

    def has_work(self) -> bool:  # 【MR2】覆盖 has_work：空闲时也给协调命令一个轮询点
        # Native WAKEUP reevaluates this predicate even when the scheduler is
        # idle, giving pending HTTP probe commands a non-blocking poll point.
        self._lwd_poll_coordination()  # 【MR2】先轮询再判断是否还有工作
        return super().has_work()

    def step_with_batch_queue(self):
        """Drain the early-recv watermark at the head of the step (done
        MQ -> gate readiness predicate); the rest is native."""
        self._lwd_poll_coordination()  # 【MR2】步进前轮询协调命令
        self._lwd_drain_irecv_completions()
        engine_core_outputs, model_executed = super().step_with_batch_queue()
        self._lwd_publish_finished_usage(engine_core_outputs)  # 【MR2】步进后发布完成请求的 usage
        self._lwd_retire_finished_outputs(engine_core_outputs)
        return engine_core_outputs, model_executed

    def step(self):
        """Same for the synchronous step path: drain the watermark at
        the head of the step, the rest is native."""
        self._lwd_poll_coordination()  # 【MR2】同步 step 路径同样先轮询
        self._lwd_drain_irecv_completions()
        engine_core_outputs, model_executed = super().step()
        self._lwd_publish_finished_usage(engine_core_outputs)  # 【MR2】同步路径同样发布 usage
        self._lwd_retire_finished_outputs(engine_core_outputs)
        return engine_core_outputs, model_executed

    def _lwd_publish_finished_usage(self, engine_core_outputs: Any) -> None:  # 【MR2】把完成记录转成 usage 事件发回协调器
        records = self.scheduler.lwd_finished_records
        event_queue = getattr(
            self, "cloud_control_event_queue", self.cloud_control_event_queue
        )
        from vllm.v1.lwd_control.lwd_comm.lwd_prefix import LwdUsage

        for request_id, (prompt_tokens, completion_tokens) in list(records.items()):  # 【MR2】遍历调度器记录的完成快照
            cached_tokens = self._lwd_prefix_hits.get(request_id, 0)  # 【MR2】cached_tokens 取该请求的前缀命中数
            usage = LwdUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
            )
            if event_queue is not None:  # 【MR2】有事件队列就投递 usage
                event_queue.put(
                    {
                        "type": "usage",
                        "request_id": request_id,
                        "usage": usage,
                        "engine_index": self._lwd_engine_index,
                    }
                )
            records.pop(request_id, None)  # 【MR2】发布后删除记录，避免重复上报
            self._lwd_prefix_hits.pop(request_id, None)  # 【MR2】命中记录同样清理

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
        """Merge every connection's done MQ into independent watermarks."""  # 【MR2】逐连接收割 done 队列，合并成各自的水位
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
            record_irecv_completions,
        )

        watermark_items: list[Any] = []
        for connection, done_mq in self._lwd_done_mqs.items():  # 【MR2】每条连接单独消费 done MQ
            while True:
                try:
                    item = done_mq.dequeue(timeout=0)  # 【MR2】非阻塞取，取空即换下一条连接
                except TimeoutError:
                    break
                except Exception:
                    lwd_logger.exception(  # 【MR2】单条连接队列异常只影响它自己
                        "[pre-recv] done MQ drain failed for %s", connection
                    )
                    break
                if (  # 【MR2】refill_done 是三元组（标记, 请求, nonce）
                    isinstance(item, tuple)
                    and len(item) == 3
                    and item[0] == "refill_done"
                ):
                    self.scheduler.lwd_refill_done(str(item[1]), int(item[2]))
                    continue
                if (  # 【MR2】error 是四元组（连接, “error”, seqno, 错误）
                    isinstance(item, tuple)
                    and len(item) == 4
                    and item[1] == "error"
                ):
                    if item[0] != connection:  # 【MR2】错误项必须属于当前连接，跨连接说明 worker 串了
                        raise ValueError(
                            f"[LWD] cross-connection early-recv error on "
                            f"{connection}: {item!r}"
                        )
                    self._lwd_fail_connection(connection, item[2], item[3])  # 【MR2】只让这一条连接失败，其它边继续
                    continue
                if (  # 【MR2】完成项是 ((连接, 通道, seqno), arrived)
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[0], tuple)
                ):
                    key, arrived = item
                    if len(key) != 3:  # 【MR2】完成 key 必须是三元组
                        raise ValueError(
                            "[LWD] early-recv completion key must be "
                            f"(connection, channel, seqno), got {key!r}"
                        )
                    if key[0] != connection:  # 【MR2】key 的第一个元素必须等于当前连接
                        raise ValueError(
                            f"[LWD] cross-connection early-recv completion on "
                            f"{connection}: {item!r}"
                        )
                    watermark_items.append(key)  # 【MR2】收集水位项，统一推进
                    self._lwd_note_arrived(key[0], key[2], arrived)  # 【MR2】按连接和 seqno 合并到达行
                    continue
                raise ValueError(  # 【MR2】未知格式直接报错，避免静默丢数据
                    f"[LWD] unsupported early-recv completion item: {item!r}"
                )
        if not watermark_items:
            return
        advanced = record_irecv_completions(watermark_items)
        for connection, _, seqno in advanced:  # 【MR2】水位推进后，按连接完成对应 seqno
            self._lwd_complete_seqno(connection, seqno)
        lwd_logger.info(
            "[Lwd][pre-recv] DRAIN n=%d advanced=%s ts=%.3f",  # 【MR2】DRAIN 日志：本轮完成项数量、推进后的水位、时间戳
            len(watermark_items),
            advanced,  # 【MR2】打印推进后的 (连接, 通道, seqno) 水位
            time.monotonic(),
        )

    def _lwd_fail_connection(  # 【MR2】连接级失败处理：只 abort 这条连接上的请求
        self, connection: LwdConnectionId, seqno: int, error: Any
    ) -> None:
        """Abort only the failed connection; other edges keep making progress."""
        if connection in self._lwd_failed_connections:  # 【MR2】同一条连接重复报错只处理一次
            return
        self._lwd_failed_connections.add(connection)  # 【MR2】标记失败，后续该连接的新消息直接拒绝
        lwd_logger.error(
            "[pre-recv] connection %s failed at seqno=%s: %s",
            connection,
            seqno,
            error,
        )
        request_ids = [  # 【MR2】找出路由到该连接的所有请求
            request_id
            for request_id, peer in self._lwd_request_routes.items()
            if self._lwd_connection_of(peer) == connection
        ]
        for request_id in request_ids:  # 【MR2】逐个退休并 abort
            self._lwd_retire_request(request_id, preserve_prompt=True)
            self.aborts_queue.put_nowait([request_id])
            self.input_queue.put_nowait((EngineCoreRequestType.ABORT, [request_id]))

    def _lwd_note_arrived(
        self,  # 【MR2】到达行合并现在带连接编号
        connection_id: LwdConnectionId,
        seqno: int,
        arrived: Iterable[tuple[str, int]] | None,
    ) -> None:
        """Merge arrivals only for requests in this connection's pending seqno."""  # 【MR2】只合并属于该连接、该 seqno 的请求
        pending = self._lwd_pending_seqnos.get(connection_id, {})
        request_ids = pending.get(seqno)
        if not arrived or request_ids is None:
            return
        touched: list[tuple[str, int, int]] = []
        for request_id, rows in arrived:
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
            if rows > previous:
                self.scheduler.lwd_recv_rows[request_id] = rows
                touched.append((request_id, previous, rows))
        if touched:
            lwd_logger.info("[cloud] arrival high watermarks updated: %s", touched)

    def _lwd_log_cloud_concurrency(self) -> None:  # 【MR2】云侧并发快照：运行中、等待、KV 占用
        """Debug-level snapshot of this engine's concurrency: how many
        requests it is running right now, plus the backlog and the KV
        cache occupancy they hold."""
        now = time.monotonic()
        if now - self._lwd_cloud_conc_last_log < _LWD_CLOUD_CONC_LOG_INTERVAL_S:
            return
        self._lwd_cloud_conc_last_log = now
        scheduler = self.scheduler
        block_pool = scheduler.kv_cache_manager.block_pool
        lwd_logger.debug(  # 【MR2】以 debug 级别输出
            "[Lwd][dbg] cloud concurrency cloud_id=%d engine=%d running=%d "
            "waiting=%d skipped=%d total=%d step=%d | kv_cache usage=%.1f%% "
            "free=%d total=%d",
            self.lwd_config.instance_id,
            self._lwd_engine_index,
            len(scheduler.running),
            len(scheduler.waiting),
            len(scheduler.skipped_waiting),
            len(scheduler.requests),
            scheduler.current_step,
            block_pool.get_usage() * 100.0,
            block_pool.get_num_free_blocks(),
            max(0, block_pool.num_gpu_blocks - 1),
        )

    def post_step(self, model_executed: bool) -> None:
        """Flush pending c2e to the edge at the end of the step (the
        scheduler only queues them; both native step paths reach this hook)."""
        self._lwd_log_cloud_concurrency()  # 【MR2】每个 post_step 尝试打并发日志（内部 10 秒节流）
        if self.scheduler.pending_c2e:
            finished_ids: list[str] = []  # 【MR2】记录本批真正结束的请求
            for notify in self.scheduler.pending_c2e:
                if not notify.req_ids:  # 【MR2】空通知直接跳过
                    continue
                peers = {  # 【MR2】一批 C2E 的请求必须都指向同一个来源边
                    self._lwd_request_routes.get(request_id)
                    or self._lwd_retired_routes.get(request_id)
                    for request_id in notify.req_ids
                }
                if len(peers) != 1 or None in peers:  # 【MR2】多于一个 peer 或有未知 peer 都报错，防止回包发错边
                    raise RuntimeError(
                        "[LWD] one C2E batch must have exactly one known source edge"
                    )
                peer = next(iter(peers))
                assert peer is not None
                restored = msgspec.structs.replace(  # 【MR2】回包前去掉 e{edge}- 前缀，还原边侧请求 id
                    notify,
                    req_ids=[unwrap_req_id(request_id) for request_id in notify.req_ids],
                )
                self._lwd_send_until_queued(peer, restored)  # 【MR2】把 C2E 发给该请求的来源 peer
                # Finished routes stay pinned until every pending C2E has
                # flushed; later zero-accepted/retirement notices are legal.
                for index, request_id in enumerate(notify.req_ids):
                    if (  # 【MR2】记录带 finish 的请求
                        notify.finish_reasons
                        and notify.finish_reasons[index] != -1
                    ):
                        finished_ids.append(request_id)
            self.scheduler.pending_c2e.clear()
            for request_id in finished_ids:  # 【MR2】C2E 冲刷完后清理退休路由
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
