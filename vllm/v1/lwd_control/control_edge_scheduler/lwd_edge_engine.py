# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Edge engine core: the control plane to every cloud DP rank, the check of
each cloud's scheduling limits and the request entry (edge admission +
announcement to the routed cloud DP rank; communication stays in the engine).

Stepping is the native step_with_batch_queue, adapted at construction: depth
2, is_pooling_model=True (no native sample_tokens) and an executor proxy (a
normally completed batch returning None becomes a placeholder).

Startup blocks until every cloud DP rank accepted the handshake and sent its
scheduling limits, within one handshake timeout; any failure stops the
engine. There is no self-healing: restart the whole group.
"""

from __future__ import annotations

import dataclasses  # 【MR2】引入 dataclasses：把结构化输出参数转成可序列化 dict
import json  # 【MR2】引入 json：结构化输出随 RequestNotify 传给云端
import threading
import time
from collections import deque
from typing import Any

from vllm.config.lwd import LwdConnectionId  # 【MR2】云侧参数、路由等统一用连接编号作 key
from vllm.lwd_logger import lwd_logger
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.lwd_control.control_edge_scheduler.lwd_coord_client import (
    LwdCoordClient,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_executor_proxy import (
    LwdEdgeExecutorProxy,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_prefix_control import (
    LwdEdgePrefixControl,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_probe import (  # 【MR2】引入边侧前缀探测管理器：探测结果异步回到引擎
    LwdEdgeProbeManager,  # 【MR2】探测管理器负责发起探测和取消
    LwdProbeCompletion,  # 【MR2】探测完成项：带首选云实例、engine 索引和命中结果
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler import (
    LwdEdgeScheduler,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_engine import (
    LwdBaseEngineCore,
)
from vllm.v1.lwd_control.lwd_comm.lwd_control_plane import LwdEdgeControlPlane
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import LwdPeer
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LWD_SAMPLING_DEFAULTS,
    LwdAbortNotify,
    LwdC2eNotify,
    LwdCloudParamsNotify,
    LwdEdgeToCloudMessage,
    LwdMessage,
    LwdPrefixReadyNotify,
    LwdRefillNotify,
    LwdRequestNotify,
    lwd_decode_cloud_to_edge,
)
from vllm.v1.request import Request

# Native batch queue depth: dispatched (embed + unembed) batches not yet
# reaped. 2 keeps the pipeline aligned; the reference branch's 4 is
# conservative, validate performance with 2 first.
LWD_EDGE_BATCH_QUEUE_DEPTH = 2
LWD_PREFIX_READY_TIMEOUT_S = 60.0

# A full outbound queue is retried with growing intervals (about 3 s in
# total) before failing: by then the cloud stopped reading or is gone.
_LWD_SEND_RETRY_STEPS = 5
_LWD_SEND_RETRY_INTERVAL_S = 0.2


class LwdEdgeEngineCore(LwdBaseEngineCore):
    """Edge PO engine: control plane to every cloud DP rank, cloud limits
    check and native stepping."""

    lwd_scheduler_cls = LwdEdgeScheduler
    scheduler: LwdEdgeScheduler

    def _lwd_assemble(self) -> None:
        self._lwd_prefix_enabled = bool(
            self.vllm_config.cache_config.enable_prefix_caching
        )
        self.lwd_config.validate_prefix_caching(self._lwd_prefix_enabled)  # 【MR2】启动时校验前缀缓存配置与云侧能力是否匹配
        self._lwd_prefix_control = (
            LwdEdgePrefixControl(self, LWD_PREFIX_READY_TIMEOUT_S)
            if self._lwd_prefix_enabled
            else None
        )
        self._lwd_cloud_params: dict[LwdConnectionId, LwdCloudParamsNotify] = {}  # 【MR2】云端调度参数按连接保存（每条边到每个云 DP 一份）
        self._lwd_probe_manager = LwdEdgeProbeManager(  # 【MR2】探测管理器：探测完成时通过 WAKEUP 唤醒引擎
            self.vllm_config,
            lambda: self.input_queue.put_nowait(
                (EngineCoreRequestType.WAKEUP, None)
            ),
        )
        self._lwd_probe_chains: dict[str, list[bytes]] = {}  # 【MR2】缓存每个请求探测时用的 block hash 链，保证通知云端时一致
        self.scheduler.lwd_on_finish = self._lwd_finish_request  # 【MR2】请求结束回调：清理探测链
        self.scheduler.lwd_has_pending = lambda: (  # 【MR2】引擎是否还有待处理工作：探测和前缀控制都算，避免提前休眠
            self._lwd_probe_manager.pending
            or bool(
                self._lwd_prefix_control and self._lwd_prefix_control.pending
            )
        )
        self._lwd_cloud_params_received = threading.Event()
        plane = self._lwd_start_control_plane()
        assert isinstance(plane, LwdEdgeControlPlane)
        handshake_timeout_s = plane.spec.config.handshake_timeout_s
        deadline = time.monotonic() + handshake_timeout_s
        self._lwd_clouds: dict[LwdConnectionId, LwdPeer] = plane.wait_until_ready(  # 【MR2】等所有目标云 DP 握手完成，返回“连接 -> peer”表
            handshake_timeout_s
        )
        expected = len(self._lwd_clouds)
        while len(self._lwd_cloud_params) < expected:
            if not self._lwd_cloud_params_received.wait(
                max(0.0, deadline - time.monotonic())
            ):
                missing = expected - len(self._lwd_cloud_params)
                raise RuntimeError(
                    f"[LWD] edge engine init failed: {missing} of "
                    f"{expected} cloud DP ranks accepted the handshake "
                    f"but sent no scheduling limits within "
                    f"{handshake_timeout_s}s"
                )
            self._lwd_cloud_params_received.clear()
        for dp_idx, params in self._lwd_cloud_params.items():
            self._lwd_check_cloud_params(dp_idx, params)
        self._lwd_configure_prefix_cache()
        self._lwd_coord_clients: dict[LwdConnectionId, LwdCoordClient] = {  # 【MR2】协调客户端也按连接创建，一个云 DP 一个
            dp_idx: LwdCoordClient(dp_idx.dp_idx, params.dp_coordinator_addr)  # 【MR2】客户端只依赖该 DP 的 engine 索引和协调地址
            for dp_idx, params in self._lwd_cloud_params.items()
            if params.dp_coordinator_addr
        }
        for dp_idx, params in self._lwd_cloud_params.items():  # 【MR2】把各云实例的前缀 hash 块大小告诉探测管理器
            if params.prefix_hash_block_size > 0:
                self._lwd_probe_manager._hash_sizes[dp_idx.cloud_id] = (  # 【MR2】按 cloud_id 记录 hash 块大小，探测时按同样粒度算 hash
                    params.prefix_hash_block_size
                )
        self.scheduler.lwd_send_to_cloud = self._lwd_send_to_cloud
        self.scheduler.lwd_refill_abort = self._lwd_abort_refill
        self.is_pooling_model = True
        self.batch_queue_size = LWD_EDGE_BATCH_QUEUE_DEPTH
        self.batch_queue = deque(maxlen=self.batch_queue_size)
        self.step_fn = self.step_with_batch_queue
        self.model_executor = LwdEdgeExecutorProxy(self.model_executor)
        lwd_logger.info(
            "[edge] engine assembled: %d cloud DP ranks, native batch-queue "
            "stepping depth=%d",
            len(self._lwd_clouds),
            self.batch_queue_size,
        )

    def _lwd_configure_prefix_cache(self) -> None:
        capabilities: set[tuple[int, int, bool, bool]] = set()
        for dp_idx, params in self._lwd_cloud_params.items():
            if self._lwd_prefix_enabled != params.prefix_caching:
                raise RuntimeError(
                    f"[LWD] prefix caching mismatch on dp{dp_idx}: "
                    f"edge={self._lwd_prefix_enabled} "
                    f"cloud={params.prefix_caching}"
                )
            capabilities.add(
                (
                    params.hash_block_size,
                    params.prefix_alignment_tokens,
                    params.prefix_alignment_required,
                    params.prefix_use_eagle,
                )
            )
        control = getattr(self, "_lwd_prefix_control", None)
        if control is None:
            return
        if len(capabilities) != 1:
            raise RuntimeError(
                f"[LWD] inconsistent prefix capabilities across cloud DP ranks: "
                f"{sorted(capabilities)}"
            )
        params = next(iter(self._lwd_cloud_params.values()))
        control.configure(params)
        self.scheduler._lwd_prefix_hash_block_size = control.hash_block_size

    def shutdown(self) -> None:
        control = getattr(self, "_lwd_prefix_control", None)
        if control is not None:
            control.shutdown()
        # Reached after a failed assembly too, before the clients exist.  # 【MR2】装配失败提前返回时也会走到这里，探测管理器必须能安全关闭
        self._lwd_probe_manager.close()  # 【MR2】关闭探测线程/连接池
        for client in getattr(self, "_lwd_coord_clients", {}).values():
            client.shutdown()
        super().shutdown()

    def _lwd_decode(self, payload: bytes) -> LwdMessage:
        return lwd_decode_cloud_to_edge(payload)

    def _lwd_connection_of(self, peer: LwdPeer) -> LwdConnectionId:  # 【MR2】从 peer 反推连接编号：用 (cloud_id, dp_rank) 匹配拓扑
        return next(  # 【MR2】拓扑里找不到对应连接说明对端身份不合法，直接抛错
            connection
            for connection in self.lwd_config.connections
            if (connection.cloud_id, connection.dp_idx)
            == (peer.instance_id, peer.dp_rank)
        ).connection_id

    def _lwd_dispatch_message(self, peer: LwdPeer, msg: LwdMessage) -> bool:
        """Cloud limits -> record and wake the startup wait; c2e -> scheduler
        notify queue + WAKEUP; anything else is unknown."""
        if isinstance(msg, LwdCloudParamsNotify):
            connection = self._lwd_connection_of(peer)  # 【MR2】收到云参数：先由 peer 推出连接
            self._lwd_cloud_params[connection] = msg  # 【MR2】按连接记录参数
            self.scheduler.lwd_update_cloud_params(connection, msg)  # 【MR2】同时下发给调度器（账户的前缀能力和负载参数来自这里）
            self._lwd_cloud_params_received.set()
            return True
        if isinstance(msg, (LwdPrefixReadyNotify, LwdRefillNotify)):
            self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, msg))
            return True
        if isinstance(msg, LwdC2eNotify):
            connection_id = self._lwd_connection_of(peer)  # 【MR2】C2E 也按来源 peer 推出连接编号
            lwd_logger.info(
                "[edge] c2e received from %s reqs=%d hidden_numel=%d",  # 【MR2】日志里带上连接编号，便于定位是哪条连接的回包
                connection_id,
                len(msg.req_ids),
                msg.hidden_num_elements,
            )
            self.scheduler.unembed_notify_queue.append((connection_id, msg))  # 【MR2】回包进入按连接区分的待解包队列，避免多条边串包
            self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))
            return True
        return False

    def _lwd_check_cloud_params(
        self, dp_idx: LwdConnectionId, params: LwdCloudParamsNotify  # 【MR2】参数校验的 key 从 dp_idx 改成完整连接编号
    ) -> None:
        """(max_num_batched_tokens, max_num_seqs) must match every cloud DP
        rank: a mismatch makes the cloud hold prefills behind decodes forever
        without any error."""
        sched = self.vllm_config.scheduler_config
        local = (sched.max_num_batched_tokens, sched.max_num_seqs)
        remote = (params.max_num_batched_tokens, params.max_num_seqs)
        if local != remote:
            raise RuntimeError(
                f"[LWD] edge/cloud scheduling config mismatch on {dp_idx}: "  # 【MR2】报错信息带上具体连接，方便定位是哪个云 DP 配错
                f"(max_num_batched_tokens, max_num_seqs) edge={local} != "
                f"cloud={remote}; a mismatch makes the cloud hold prefills "
                "behind decodes forever, so start both sides with the same values"
            )
        lwd_logger.info("[cloud-params] %s limits verified: %s", dp_idx, local)  # 【MR2】校验通过后打印该连接的调度上限

    def _lwd_send_to_cloud(  # 【MR2】发送入口改为按连接寻址（一条边到多个云 DP）
        self, dp_idx: LwdConnectionId, message: LwdEdgeToCloudMessage
    ) -> None:
        """Send to one cloud DP rank, retrying a full outbound queue; raises
        once the retries run out instead of losing the message."""
        cloud = self._lwd_clouds[dp_idx]
        for attempt in range(_LWD_SEND_RETRY_STEPS):
            if self._lwd_send(cloud, message):
                return
            time.sleep(_LWD_SEND_RETRY_INTERVAL_S * (attempt + 1))
        raise RuntimeError(
            f"[LWD] {type(message).__name__} to cloud {dp_idx} {cloud} not "  # 【MR2】发送失败信息带上连接编号，便于判断是哪条链路拥塞
            f"sent: outbound queue still full after {_LWD_SEND_RETRY_STEPS} "
            "retries (cloud stalled or gone?)"
        )

    # ------------------------------------------------------------------ #
    # Engine interface overrides                                         #
    # ------------------------------------------------------------------ #
    def add_request(self, request: Request, _request_wave: int = 0) -> None:
        lwd_logger.info(
            "[edge] request received req=%s tokens=%d (admit + announce)",
            request.request_id,
            request.num_prompt_tokens,
        )
        if self.scheduler.lwd_edge_request_active(request.request_id):
            raise AssertionError(f"duplicate active request id: {request.request_id}")
        if request.abort_immediately:  # 【MR2】已标记立即 abort 的请求跳过探测，直接准入后走 abort
            self._lwd_add_request_after_probe(request)
            return
        control = getattr(self, "_lwd_prefix_control", None)
        if control is not None:  # 【MR2】有前缀控制时交给它排队，等 READY 后再准入
            control.add_request(request)
            return
        if self._lwd_probe_manager.submit(request):  # 【MR2】探测任务提交成功就异步等待，不阻塞引擎
            return
        self._lwd_add_request_after_probe(request)  # 【MR2】没有探测能力时直接准入（fail-open）

    def _lwd_admit_and_notify(  # 【MR2】准入 + 通知云端的公共流程
        self,
        request: Request,
        preferred: LwdConnectionId | None = None,
    ) -> None:
        dp_idx = self.scheduler.lwd_admit_request(request, preferred)  # 【MR2】调度器在候选连接里选定路由（可带探测给的首选）
        client = self._lwd_coord_clients.get(dp_idx)  # 【MR2】拿到该连接的协调客户端（可能为空）
        try:
            if client is not None:
                client.send_first_req()  # 【MR2】首次请求唤醒云侧协调器
            self._lwd_notify_request(dp_idx, request)
        except Exception:
            self.scheduler.lwd_edge_cancel_admission(request.request_id)
            raise

    def _lwd_enqueue_request(self, request: Request) -> None:
        super().add_request(request)

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        control = getattr(self, "_lwd_prefix_control", None)
        if control is not None and control.handle_client_request(request_type, request):
            return
        if request_type == EngineCoreRequestType.WAKEUP and isinstance(
            request, LwdRefillNotify
        ):
            self._lwd_handle_refill(request)
            return
        super()._handle_client_request(request_type, request)

    def _lwd_handle_refill(self, notify: LwdRefillNotify) -> None:
        outputs = self.scheduler.lwd_edge_build_refill(notify)
        if outputs is None:
            lwd_logger.error("[edge-refill] req=%s cannot be served", notify.request_id)
            self._lwd_abort_refill(notify.request_id)
            return
        self.scheduler.lwd_edge_queue_refill(outputs)

    def _lwd_abort_refill(self, request_id: str) -> None:
        request = self.scheduler.requests.get(request_id)
        if request is None:
            lwd_logger.warning("[edge-refill] stale abort req=%s ignored", request_id)
            return
        self._lwd_abort_internally(request)

    def _lwd_abort_internally(self, request: Request) -> None:
        self.abort_requests([request.request_id])
        self._send_abort_outputs_to_client([request.request_id], request.client_index)

    def _lwd_add_request_after_probe(  # 【MR2】探测完成后的准入路径：先按结果算命中，再准入并通知
        self,
        request: Request,
        completion: LwdProbeCompletion | None = None,
    ) -> None:
        preferred = self._lwd_preferred_connection(completion)  # 【MR2】探测结果给出首选连接；没有结果则为 None
        request.lwd_prefix_probe_succeeded = False  # 【MR2】默认认为探测未成功（fail-open），命中按 0 处理
        if completion is not None and completion.chain is not None:  # 【MR2】缓存探测链，RequestNotify 里用它替代本地 block hashes
            self._lwd_probe_chains[request.request_id] = completion.chain
        if completion is not None and completion.result is not None:  # 【MR2】有探测结果时用结果修正 num_computed_tokens
            coordination = self.lwd_config.coordination
            placement_only = (  # 【MR2】placement_only：只探测位置，不使用命中
                coordination is not None and coordination.placement_only
            )
            request.lwd_prefix_probe_succeeded = not placement_only  # 【MR2】该模式视为探测未成功，避免建假的预留
            hit = 0 if placement_only else completion.result.hit_tokens  # 【MR2】命中数强制为 0，云端按完整 prompt 重算
            hit = min(max(hit, 0), max(request.num_prompt_tokens - 1, 0))  # 【MR2】命中数夹在 [0, prompt-1]：不允许把整个 prompt 都命中
            alignment = completion.result.alignment_tokens  # 【MR2】取云侧下发的命中对齐粒度
            if alignment > 0:
                hit -= hit % alignment  # 【MR2】向下取整到对齐边界，避免命中落在半个压缩状态上
            request.num_computed_tokens = hit  # 【MR2】写回已计算 token 数，后续 prefill 从命中行之后开始
        self._lwd_admit_and_notify(request, preferred)  # 【MR2】准入并通知云端（内部会选路由、发 RequestNotify）
        control = getattr(self, "_lwd_prefix_control", None)
        if control is None or request.request_id not in control.pending:  # 【MR2】前缀控制还在等 READY 时，由控制流程负责入队
            self._lwd_enqueue_request(request)

    def _lwd_preferred_connection(  # 【MR2】把探测结果映射成首选连接
        self, completion: LwdProbeCompletion | None
    ) -> LwdConnectionId | None:
        if completion is None or completion.result is None:  # 【MR2】没有命中结果就不指定首选，交给路由打分
            return None
        cloud_id = completion.cloud_id
        engine = completion.engine_index
        if engine is not None:  # 【MR2】探测给了 engine 索引就构造精确连接
            candidate = LwdConnectionId(
                self.lwd_config.instance_id, cloud_id, engine
            )
            if candidate in self._lwd_clouds:  # 【MR2】首选连接必须已完成握手，否则回退
                return candidate
        return next(  # 【MR2】回退到同云实例下编号最小的连接
            (
                connection_id
                for connection_id in sorted(self._lwd_clouds)
                if connection_id.cloud_id == cloud_id
            ),
            None,
        )

    def _lwd_drain_probe_completions(self) -> None:  # 【MR2】收割探测完成项，逐个走准入流程
        for completion in self._lwd_probe_manager.drain():
            self._lwd_add_request_after_probe(completion.request, completion)

    def step(self):  # 【MR2】普通 step 前先处理已完成的探测
        self._lwd_drain_probe_completions()
        return super().step()

    def step_with_batch_queue(self):  # 【MR2】批量调度入口同样先处理探测
        self._lwd_drain_probe_completions()
        return super().step_with_batch_queue()

    def abort_requests(self, request_ids: list[str]) -> None:
        """Notify each request's original cloud DP, then finish natively."""  # 【MR2】abort 时通知请求原来选定的云 DP，再走原生清理
        control = getattr(self, "_lwd_prefix_control", None)
        if control is not None:
            control.prepare_abort(request_ids)
        for request_id in request_ids:  # 【MR2】取消尚未完成的探测任务
            self._lwd_probe_manager.abort(request_id)  # 【MR2】清理探测链缓存
            self._lwd_probe_chains.pop(request_id, None)
        routed_aborts = self.scheduler.lwd_edge_abort(request_ids)
        for dp_idx, request_id in routed_aborts:
            request = self.scheduler.requests.get(request_id)  # 【MR2】取请求以读取它的 prefix nonce
            self._lwd_send_to_cloud(
                dp_idx,
                LwdAbortNotify(  # 【MR2】abort 消息带 nonce：云端据此忽略过期 abort
                    request_id, getattr(request, "lwd_prefix_nonce", 0)
                ),
            )
        super().abort_requests(request_ids)

    def _lwd_notify_request(  # 【MR2】向指定云 DP 发 RequestNotify
        self, dp_idx: LwdConnectionId, request: Request
    ) -> None:
        """Announce the request to one cloud DP rank, packing the sampling
        fields from the shared table."""
        sampling_params = request.sampling_params
        assert sampling_params is not None, "admission rejects pooling requests"
        values: dict[str, Any] = {
            f: getattr(sampling_params, f) for f in LWD_SAMPLING_DEFAULTS
        }
        if values["max_tokens"] is None:
            values["max_tokens"] = LWD_SAMPLING_DEFAULTS["max_tokens"]
        values["stop_token_ids"] = list(values["stop_token_ids"] or [])
        structured_outputs = sampling_params.structured_outputs  # 【MR2】结构化输出参数序列化成 JSON
        structured_outputs_payload = (  # 【MR2】云侧需要原始语法定义来编译约束，边侧不编译
            json.dumps(dataclasses.asdict(structured_outputs))
            if structured_outputs is not None
            else ""
        )
        self._lwd_send_to_cloud(
            dp_idx,
            LwdRequestNotify(
                request_id=request.request_id,
                num_prompt_tokens=request.num_prompt_tokens,
                block_hashes=list(  # 【MR2】block hashes 优先用探测链，保证云端命中和探测一致
                    self._lwd_probe_chains.get(
                        request.request_id, request.block_hashes
                    )
                ),
                eos_token_id=sampling_params.eos_token_id,
                prefix_nonce=getattr(request, "lwd_prefix_nonce", 0),  # 【MR2】prefix nonce 用于把 READY/refill 和本轮探测配对
                skip_reading_prefix_cache=(  # 【MR2】placement_only 时直接跳过云端读前缀缓存
                    request.skip_reading_prefix_cache
                    or (
                        self.lwd_config.coordination is not None
                        and self.lwd_config.coordination.placement_only
                    )
                ),
                prefix_hit_tokens=request.num_computed_tokens,  # 【MR2】把边侧算出的命中数一并告知云端
                prefix_probe_succeeded=getattr(  # 【MR2】告诉云端探测是否成功：失败时不要建前缀预留
                    request, "lwd_prefix_probe_succeeded", False
                ),
                structured_outputs=structured_outputs_payload,  # 【MR2】结构化输出语法随通知下发
                mrope_positions=self.scheduler.lwd_mrope_positions(request.request_id),  # 【MR2】mrope 位置随通知下发
                **values,
            ),
        )

    def _lwd_finish_request(self, request: Request) -> None:  # 【MR2】请求结束（正常/abort）时清理探测链，避免无界增长
        self._lwd_probe_chains.pop(request.request_id, None)
