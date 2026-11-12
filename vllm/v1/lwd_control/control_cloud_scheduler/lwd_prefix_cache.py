# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Short-lived prefix reservations for LWD cloud scheduling."""

from __future__ import annotations

from math import lcm
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

# Bound of the cached-hit snapshot ledger: entries survive their
# reservation (released at first allocation) until the usage settlement
# reads them; the bound evicts the oldest residue.
_LWD_HIT_SNAPSHOT_CAP = 1024
_LWD_HIT_SNAPSHOT_EVICT = 256


class _LwdPrefixReservation(NamedTuple):
    groups: tuple[Sequence[KVCacheBlock], ...]
    hit_tokens: int
    pinned: list[KVCacheBlock]


class LwdPrefixReservations:
    """Pin negotiated prefix blocks until native allocation takes ownership."""

    def __init__(self, manager: KVCacheManager) -> None:
        self.manager = manager
        self._reservations: dict[str, _LwdPrefixReservation] = {}
        # Hit snapshots outliving their reservation: the usage settlement
        # happens after the release (first allocation hands the blocks to
        # the native request chain), so hit_of would report zero exactly
        # when a successful request needs its cached_tokens accounted.
        self._hit_snapshots: dict[str, int] = {}
        manager.lwd_prefix_reservations = self
        manager.lwd_prefix_caching = True

    def reserve(self, request: Request) -> int:
        """Reserve the negotiated hit and freeze its snapshot."""
        if request.request_id in self._reservations:
            raise ValueError(f"duplicate LWD prefix reservation: {request.request_id}")
        if request.skip_reading_prefix_cache:
            groups: tuple[Sequence[KVCacheBlock], ...] = ()
            hit = 0
        else:
            groups, hit, _ = self.manager.coordinator.find_longest_cache_hit(
                request.block_hashes,
                max_cache_hit_length=max(0, request.num_prompt_tokens - 1),
            )
        return self._touch_and_snapshot(request, groups, hit, set_floor=True)

    def reserve_hit(
        self,
        request: Request,
        groups: tuple[Sequence[KVCacheBlock], ...],
        hit: int,
    ) -> int:
        """Reserve a readmission hit without changing the staging floor."""
        if request.request_id in self._reservations:
            raise ValueError(f"duplicate LWD prefix reservation: {request.request_id}")
        return self._touch_and_snapshot(request, groups, hit, set_floor=False)

    def touch(self, groups: tuple[Sequence[KVCacheBlock], ...]) -> list[KVCacheBlock]:
        """Pin real blocks while preserving null slots only in the snapshot."""
        pinned = [
            block
            for group in groups
            for block in group
            if not getattr(block, "is_null", False)
        ]
        if pinned:
            self.manager.block_pool.touch(pinned)
        return pinned

    def snapshot(
        self, request: Request
    ) -> tuple[tuple[Sequence[KVCacheBlock], ...], int]:
        """Return the frozen hit, refusing to renegotiate a trimmed request."""
        entry = self._reservations.get(request.request_id)
        if entry is not None:
            return entry.groups, entry.hit_tokens
        if getattr(request, "lwd_resume_cap_tokens", None) is not None:
            raise RuntimeError(
                f"LWD reserved prefix for {request.request_id} is unavailable; "
                "cannot consume trimmed embeddings"
            )
        groups, hit, _ = self.manager.coordinator.find_longest_cache_hit(
            request.block_hashes, request.num_tokens - 1
        )
        return groups, hit

    def truncate(
        self,
        groups: tuple[Sequence[KVCacheBlock], ...],
        hit: int,
    ) -> tuple[Sequence[KVCacheBlock], ...]:
        """Truncate each cache group to the blocks covered by hit."""
        managers = self.manager.coordinator.single_type_managers
        return tuple(
            list(group[: hit // manager.block_size])
            for group, manager in zip(groups, managers, strict=True)
        )

    def release(self, request_id: str, *, allocated: bool = False) -> None:
        """Release a reservation and its pins; repeated calls are no-ops.

        ``allocated=True`` marks the first-successful-allocation release
        (the scheduler releases a request it just scheduled): the hit is
        frozen into the snapshot ledger for the later usage settlement.
        The remaining release sites (finish/abort cleanup, admission
        rollback) drop the reservation without a snapshot — a request that
        never consumed its blocks reports zero cached tokens.
        """
        entry = self._reservations.pop(request_id, None)
        if entry is None:
            return
        if allocated:
            self._hit_snapshots[request_id] = entry.hit_tokens
            if len(self._hit_snapshots) > _LWD_HIT_SNAPSHOT_CAP:
                for stale in list(self._hit_snapshots)[
                    :_LWD_HIT_SNAPSHOT_EVICT
                ]:
                    self._hit_snapshots.pop(stale, None)
                # Lazy import: this module stays free of runtime vllm
                # imports so CPU tests can load it directly (the eviction
                # path runs at most once per thousand requests).
                from vllm.lwd_logger import lwd_logger

                lwd_logger.warning(
                    "[prefix] hit snapshot ledger full; evicted the oldest"
                )
        if entry.pinned:
            self.manager.block_pool.free_blocks(entry.pinned)

    def cached_hit_of(self, request_id: str) -> int:
        """Frozen hit of a request whose reservation is already released.

        The usage settlement reads this after finish/abort: the reservation
        itself is gone by then (hit_of would report zero), but the blocks
        the request actually consumed from the cache stay accounted. A
        request never scheduled (aborted before its first allocation) has
        no snapshot and reports zero — an honest number, since its pinned
        blocks were never used for compute.
        """
        return self._hit_snapshots.get(request_id, 0)

    def has(self, request_id: str) -> bool:
        return request_id in self._reservations

    def hit_of(self, request_id: str) -> int:
        entry = self._reservations.get(request_id)
        return entry.hit_tokens if entry is not None else 0

    def _touch_and_snapshot(
        self,
        request: Request,
        groups: tuple[Sequence[KVCacheBlock], ...],
        hit: int,
        *,
        set_floor: bool,
    ) -> int:
        pinned = self.touch(groups)
        self._reservations[request.request_id] = _LwdPrefixReservation(
            groups=tuple(tuple(group) for group in groups),
            hit_tokens=hit,
            pinned=pinned,
        )
        if set_floor:
            request.lwd_resume_cap_tokens = hit
        return hit


def prefix_start(
    request: Request, reservations: LwdPrefixReservations | None = None
) -> int:
    """Return the prefill window start before and after native allocation."""
    if request.num_computed_tokens > 0:
        return request.num_computed_tokens
    if reservations is not None and reservations.has(request.request_id):
        return reservations.hit_of(request.request_id)
    return getattr(request, "lwd_resume_cap_tokens", 0)


class LwdPrefixCacheState(NamedTuple):
    enabled: bool
    hash_block_size: int
    alignment_tokens: int = 0
    alignment_required: bool = False
    use_eagle: bool = False
    reservations: LwdPrefixReservations | None = None


def init_lwd_prefix_cache(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    manager: KVCacheManager,
    *,
    use_eagle: bool = False,
) -> LwdPrefixCacheState:
    """Validate the cloud KV layout and create its reservation table."""
    from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        MambaSpec,
        SlidingWindowSpec,
        UniformTypeKVCacheSpecs,
    )

    enabled = bool(vllm_config.cache_config.enable_prefix_caching)
    hash_block_size = resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)[1]
    if not enabled:
        return LwdPrefixCacheState(enabled=False, hash_block_size=hash_block_size)
    if not manager.enable_caching:
        raise ValueError("LWD prefix caching requires --enable-prefix-caching on cloud")
    if vllm_config.parallel_config.prefill_context_parallel_size > 1:
        raise ValueError(
            "LWD prefix caching does not support PCP "
            "(prefill_context_parallel_size > 1)"
        )

    specs = []
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        specs.extend(
            spec.kv_cache_specs.values()
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else [spec]
        )
    if not specs or any(
        not isinstance(spec, (FullAttentionSpec, SlidingWindowSpec, MambaSpec))
        or (
            isinstance(spec, MambaSpec)
            and spec.mamba_cache_mode not in ("align", "all")
        )
        for spec in specs
    ):
        raise ValueError(
            "LWD prefix caching currently supports Full Attention/MLA, "
            "sliding-window and Mamba align/all groups only"
        )

    coordinator = manager.coordinator
    alignment_tokens = lcm(
        coordinator.scheduler_block_size,
        getattr(coordinator, "lcm_block_size", coordinator.scheduler_block_size),
    )
    alignment_required = any(
        isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "align"
        for spec in specs
    )
    return LwdPrefixCacheState(
        enabled=True,
        hash_block_size=hash_block_size,
        alignment_tokens=alignment_tokens,
        alignment_required=alignment_required,
        use_eagle=use_eagle,
        reservations=LwdPrefixReservations(manager),
    )
