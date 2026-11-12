# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Common base class for Lwd edge/cloud schedulers: phase predicates and
the visible-set scheduling primitive (three-queue remove - schedule -
reassemble).

Inherits AsyncScheduler; native scheduling behavior and isinstance checks
stay unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator
from enum import Enum, auto
from typing import Any

from vllm.lwd_logger import lwd_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.request import Request

# Bound of the finished-record ledger: entries are consumed by the engine
# usage settlement (consumed = removed); the bound evicts the residue of
# requests whose settlement never ran (an engine without coordination).
_LWD_FINISHED_RECORDS_CAP = 1024
_LWD_FINISHED_RECORDS_EVICT = 256


def lwd_spec_factor(vllm_config) -> int:
    """MTP budget factor: each decode request consumes 1+k tokens per step
    (draft tokens take budget and KV); constantly 1 without a spec config.
    The sole home of it (edge/cloud share one source; the spec factor
    follows the cloud-side LwdCloudParamsNotify); main's vllm/config/lwd is
    a topology snapshot without this derived value, so it lives in the
    base-class module alongside the scheduling primitive."""
    spec_config = getattr(vllm_config, "speculative_config", None)
    num_spec_tokens = (
        getattr(spec_config, "num_speculative_tokens", None)
        if spec_config is not None
        else None
    )
    return (num_spec_tokens or 0) + 1


class LwdReqPhase(Enum):
    """Working phase of a request (same criterion on edge and cloud):
    PREFILL = embedding not fully sent (num_computed < num_prompt), DECODE
    = waiting for the cloud-side sampling return."""

    PREFILL = auto()
    DECODE = auto()


class LwdBaseScheduler(AsyncScheduler):
    """Common base class of edge/cloud schedulers (an AsyncScheduler
    subclass)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Finished-request snapshots: request_id -> (prompt_tokens,
        # completion_tokens). _free_request is the single release choke
        # point (natural terminations call it directly from
        # update_from_output, aborts and errors through finish_requests),
        # while the LWD usage settlement runs later — after the request has
        # left self.requests — so the minimal snapshot is captured at the
        # choke point and popped by the consumer.
        self.lwd_finished_records: dict[str, tuple[int, int]] = {}

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        """Native-compatible scheduling entry: step_with_batch_queue passes
        throttle_prefills positionally; the signature converges here, and
        edge/cloud subclasses implement only one-step role scheduling. The
        throttle hint is not consumed -- each role's dispatch gate decides
        on its own (on par with mm)."""
        return self._lwd_schedule_step()

    def _lwd_schedule_step(self) -> SchedulerOutput:
        """One role-specific scheduling step; edge/cloud subclasses own it."""
        raise NotImplementedError

    @staticmethod
    def _lwd_req_phase(request: Request) -> LwdReqPhase:
        """The sole phase criterion, shared by all queue-by-phase scans on
        edge and cloud."""
        if request.num_computed_tokens < request.num_prompt_tokens:
            return LwdReqPhase.PREFILL
        return LwdReqPhase.DECODE

    def _lwd_reqs_in_phase(self, phase: LwdReqPhase, *queues) -> Iterator[Request]:
        """Scan multiple queues by phase (terminated requests skipped -- the
        books may still satisfy the criterion)."""
        for queue in queues:
            for request in queue:
                if not request.is_finished() and self._lwd_req_phase(request) is phase:
                    yield request

    def _lwd_new_queue(self, reqs: list[Request]) -> RequestQueue:
        """Create a new scheduling policy queue and load reqs into it."""
        queue = create_request_queue(self.policy)
        for req in reqs:
            queue.add_request(req)
        return queue

    def _lwd_schedule(
        self,
        req_ids: list[str],
        *,
        token_budget_cap: int | None = None,
        chunk_threshold: int | None = None,
    ) -> SchedulerOutput:
        """Isolated scheduling: schedule only the requests named by req_ids
        -- the rest are temporarily removed from the three queues and
        hidden, then reassembled into their original queues after the step
        (visible-set mechanism).

        waiting/skipped sources go through native admission
        (allocate/state transition/accounting all intact), running sources
        go through resume. Reassembly: still-scheduled requests are
        appended after the hidden running, non-preempted waiting requests
        are appended after the hidden waiting, and preempted requests are
        prepended in their native post-schedule order. Skipped requests go
        to the skipped queue head.

        token_budget_cap / chunk_threshold only tighten, never loosen, the
        native limits: the edge side passes a cap to reproduce the picker's
        chunk split, the cloud side passes the dual clamp to reproduce
        in-batch boundaries (threshold = max item in the batch + budget =
        decode exact consumption + Σchunk). Hidden running requests are
        invisible to native concurrent admission; slots are deducted by the
        hidden count to prevent over-issue (edge-side picker free_seq and
        cloud-side preflight are the first line of defense; this is defense
        in depth)."""
        picked = {self.requests[req_id] for req_id in req_ids}
        from_running = [req for req in self.running if req in picked]
        from_waiting = [req for req in self.waiting if req in picked]
        from_skipped = [req for req in self.skipped_waiting if req in picked]
        self.running = [req for req in self.running if req not in picked]
        self.waiting.remove_requests(from_waiting)
        self.skipped_waiting.remove_requests(from_skipped)
        saved = (self.running, self.waiting, self.skipped_waiting)
        # Put the visible set in place and schedule it alone
        self.running = from_running
        self.waiting = self._lwd_new_queue(from_waiting)
        self.skipped_waiting = self._lwd_new_queue(from_skipped)
        saved_cap = self.max_num_running_reqs
        self.max_num_running_reqs = max(0, saved_cap - len(saved[0]))
        saved_budget = self.max_num_scheduled_tokens
        if token_budget_cap is not None:
            self.max_num_scheduled_tokens = min(saved_budget, max(0, token_budget_cap))
        sched_cfg = self.scheduler_config
        saved_threshold = sched_cfg.long_prefill_token_threshold
        if chunk_threshold is not None:
            sched_cfg.long_prefill_token_threshold = chunk_threshold
        out = None
        try:
            out = super().schedule()
            reservations = getattr(self, "lwd_prefix_reservations", None)
            if reservations is not None and out is not None:
                for request_id, num_tokens in out.num_scheduled_tokens.items():
                    if num_tokens > 0:
                        # First successful allocation: ownership of the hit
                        # blocks passes to the request's own KV chain, and
                        # the hit is frozen into the snapshot ledger for
                        # the later usage settlement.
                        reservations.release(request_id, allocated=True)
        finally:
            self.max_num_running_reqs = saved_cap
            self.max_num_scheduled_tokens = saved_budget
            sched_cfg.long_prefill_token_threshold = saved_threshold
            post = (self.running, self.waiting, self.skipped_waiting)
            preempted_req_ids = (
                (out.preempted_req_ids or set()) if out is not None else set()
            )
            preempted: list[Request] = []
            rest: list[Request] = []
            for request in post[1]:
                if request.request_id in preempted_req_ids:
                    preempted.append(request)
                else:
                    rest.append(request)
            self.running, self.waiting, self.skipped_waiting = saved
            self.running += post[0]
            self.waiting.extend(rest)
            for request in reversed(preempted):
                self.waiting.prepend_request(request)
            self.skipped_waiting.prepend_requests(post[2])
        return out

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """The single release choke point of every terminal path.

        The finished snapshot lands here so the usage settlement can read
        it after the fact: natural terminations reach this method directly
        from update_from_output (never through finish_requests), while
        aborts and errors pass through finish_requests into it. At this
        moment the request object is alive with final lengths. The engine
        consumes the record when it settles usage; the bound evicts the
        residue of settlements that never run.

        The reservation release and the refill-gate cleanup fold into the
        same choke point: a reserve-before-first-allocation abort, a failed
        negotiation rollback and an in-flight refill abort all end here.
        """
        try:
            records = getattr(self, "lwd_finished_records", None)
            if records is None:
                records = self.lwd_finished_records = {}
            records[request.request_id] = (
                request.num_prompt_tokens,
                request.num_tokens - request.num_prompt_tokens,
            )
            if len(records) > _LWD_FINISHED_RECORDS_CAP:
                for stale in list(records)[:_LWD_FINISHED_RECORDS_EVICT]:
                    records.pop(stale, None)
                lwd_logger.warning(
                    "[sched] finished-record ledger full; evicted the oldest"
                )
        except Exception:  # noqa: BLE001  a failed snapshot must not block the release
            lwd_logger.warning(
                "[sched] finished snapshot failed (non-fatal)", exc_info=True
            )
        reservations = getattr(self, "lwd_prefix_reservations", None)
        if reservations is not None:
            reservations.release(request.request_id)
        refill_gate = getattr(self, "_lwd_refill_gate", None)
        if refill_gate is not None:
            refill_gate.discard(request.request_id)
        refill_pending = getattr(self, "_lwd_refill_pending", None)
        if refill_pending is not None:
            refill_pending.pop(request.request_id, None)
        return super()._free_request(request, delay_free_blocks)
