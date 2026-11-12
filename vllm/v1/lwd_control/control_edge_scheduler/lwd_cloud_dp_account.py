# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-cloud-DP account of the edge scheduler.

One account holds everything the edge knows about one cloud DP rank: its
seqno stream (the UP pairing contract is per channel, so every engine has
its own), its pending prefill batches, its request load and its gate
state. The scheduler keeps one account per DP connection plus one routing
table (request id -> dp_idx) that is pinned at admission and never
changes: a request's KV and scheduler state live only in its target
engine, so every chunk, abort and result travels through the account of
the engine the request was routed to.

Load counts are maintained incrementally on the same transitions that
change the routing table, instead of rescanning every request on every
gate check.
"""

from __future__ import annotations

from vllm.config.lwd import LwdConnectionId  # 【MR2】账户身份从裸 dp_idx 改为完整连接编号

LWD_MAX_PENDING_CHUNKS = 10


class LwdCloudDpAccount:
    """The edge-side bookkeeping of one edge -> cloud DP connection."""  # 【MR2】每个“边 -> 云 DP”连接一份账户，下面所有计数都按连接隔离

    def __init__(
        self,
        connection_id: LwdConnectionId | int,  # 【MR2】构造参数改为连接编号，同时保留 int 兼容
        max_pending_chunks: int = LWD_MAX_PENDING_CHUNKS,
    ) -> None:
        if isinstance(connection_id, int):  # 【MR2】兼容旧调用：传 int 时补成 (edge=0, cloud=0, dp_idx)
            connection_id = LwdConnectionId(0, 0, connection_id)  # 【MR2】内部统一保存为 LwdConnectionId
        self.connection_id = connection_id
        self.max_pending_chunks = max_pending_chunks
        self.num_speculative_tokens = 0  # 【MR2】记录投机 token 数，云侧参数下发时填充
        self.prefix_use_eagle = False  # 【MR2】记录该连接是否使用 EAGLE 前缀
        self.prefix_alignment_tokens = 0  # 【MR2】记录前缀命中对齐粒度（tokens）
        self.prefix_alignment_required = False  # 【MR2】记录云侧是否强制要求按对齐边界命中
        self.seqno = 0
        """Next UP pairing number of this engine; taken only by a range
        notify that was sent."""
        self.pending_batches: dict[int, set[str]] = {}
        """seqno -> in-batch request ids awaiting UP send completion, request
        settlement, or a covering send watermark."""
        self.prefill_requests: set[str] = set()
        """Request ids whose prefill_load entry this account still holds.
        Every delivery of a request reaches the books (not only its first
        token), so the prefill->decode move must be gated on membership
        here, or each further token would decrement the load again."""
        self.prefill_load = 0
        """Requests routed here that still prefill (waiting, skipped or
        running)."""
        self.decode_load = 0
        """Requests routed here that finished embedding and await cloud
        results."""
        self.gate_held = False
        """Gate state of the last transition, for HOLD/OPEN logging only."""

    @property  # 【MR2】dp_idx 只作为诊断用的兼容视图
    def dp_idx(self) -> int:
        """Compatibility view for diagnostics; never use this as a map key."""  # 【MR2】不要再拿 dp_idx 当字典 key，连接编号才是唯一身份
        return self.connection_id.dp_idx

    # ------------------------------------------------------------------ load

    @property
    def load(self) -> int:
        return self.prefill_load + self.decode_load

    def score(self) -> int:
        """Routing score, lowest wins: prefill weighs 4, decode 1, the same
        weights as the native DP load balancer."""
        return 4 * self.prefill_load + self.decode_load

    def note_prefill_admitted(self, request_id: str) -> None:
        self.prefill_load += 1
        self.prefill_requests.add(request_id)

    def note_prefill_done(self, request_id: str) -> bool:
        """Move one request from prefill to decode. Returns False when the
        request was not on the prefill books (already moved, or settled):
        the caller delivers every token, so this gate keeps the move at
        exactly one."""
        if request_id not in self.prefill_requests:
            return False
        self.prefill_requests.discard(request_id)
        self.prefill_load -= 1
        self.decode_load += 1
        return True

    def note_request_settled(self, request_id: str, was_awaiting: bool) -> None:
        """A request left the account (finished, aborted or dropped)."""
        if was_awaiting:
            self.decode_load -= 1
        else:
            self.prefill_load -= 1
            self.prefill_requests.discard(request_id)

    # ----------------------------------------------------------------- batch

    def take_seqno(self) -> int:
        """Consume the next pairing number for a range notify."""
        seqno = self.seqno
        self.seqno = seqno + 1
        return seqno

    def register_batch(self, seqno: int, request_ids: set[str]) -> None:
        self.pending_batches[seqno] = set(request_ids)

    def pending_chunks(self) -> int:
        return len(self.pending_batches)

    def note_send_watermark(self, watermark: int) -> None:
        """Release this DP's batches whose sends completed in order."""
        for seqno in list(self.pending_batches):
            if seqno <= watermark:
                del self.pending_batches[seqno]

    def drop_request(self, request_id: str) -> None:
        """Remove a request from every batch entry (abort or finish);
        idempotent for requests in no batch."""
        for seqno in list(self.pending_batches):
            batch = self.pending_batches[seqno]
            batch.discard(request_id)
            if not batch:
                del self.pending_batches[seqno]

    # ------------------------------------------------------------------ gate

    def is_gated(self, max_num_running_reqs: int, has_resume_inflight: bool) -> bool:  # 【MR2】门控判断：请求槽满或待发块数达到上限时，暂停该连接的新 prefill
        """Whether new prefill admission to this engine must wait.

        A resume in flight is exempt (finish in-flight chunks before new
        ones; also prevents a max_num_seqs=1 config from deadlocking on
        itself). Otherwise the engine is gated when its request slots are
        full or its pending chunks reached the cap.
        """  # 【MR2】门控定义补充：请求槽满或待发块数达到上限即暂停新 prefill
        if has_resume_inflight:
            return False
        if self.load >= max_num_running_reqs:  # 【MR2】负载（运行中请求数）达到上限即门控
            return True
        return self.pending_chunks() >= self.max_pending_chunks
