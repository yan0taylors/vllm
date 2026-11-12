# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-value control_url direct routing: deterministic round-robin plus
per-cloud health cooldown.

Extracted from the edge scheduler's embedded multi-value logic. The
multi-value form (``control_url_map``, cloud_id -> URL) is the dev/debug and
emergency fallback of the gateway form: the edge picks a cloud itself by
round-robin and delivers the probe directly, while the single-value form
keeps "the gateway picks, the receipt decides" untouched. Health marking
only narrows the candidate set for *new* requests; in-flight requests keep
their bound cloud.

Contract (design ``lwd_control_url_multi_value.md``):

* Key-set validation against the known cloud domain: extra keys fail fast
  (a probed cloud without a data-plane channel would bind requests nowhere);
  missing keys pass with a one-shot warning (the cloud stays idle);
  no domain given means the map is inactive (no probe, single-value path).
* Deterministic rotation ``clouds[i % N]`` with read-position-then-advance
  order: exactly one advance per request, and it happens only at the
  submission point. ``peek_cloud`` is a pure read of the current position
  (the fail-open landing spot: the cloud after the last submission target)
  and never advances.
* Health marking: 3 consecutive probe failures cool a cloud down for 60s
  and exclude it from both the rotation domain and the peek domain; a
  successful probe resets it; expiry re-admits it (gentle retry: one more
  failure immediately re-cools); when every cloud is cooling the domain
  falls back to the full set (an empty domain is worse than dead clouds).
* Receipt authority: the submission target is authoritative in the
  multi-value form. A missing receipt cloud (legacy cloud control message)
  falls back to the target; a mismatched receipt warns once per
  (target, receipt) pair but still binds by the target, preventing the
  split of "reserved on cloud 1, data plane to cloud 0, granularity learned
  on cloud 0". The single-value form keeps the receipt as the decision.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

from vllm.lwd_logger import lwd_logger

# Health marking parameters (design lwd_control_url_multi_value.md): the
# consecutive-failure threshold and the cooldown window (seconds). Rotation
# guarantees each cloud is probed once per N requests, so a dead cloud is
# excluded after about threshold * N / request-rate (derivable and tunable).
# A successful probe resets the counter; after cooldown one more failure
# immediately re-cools (gentle retry). Tests monkeypatch these constants.
_LWD_MV_HEALTH_FAILS = 3
_LWD_MV_HEALTH_COOLDOWN_S = 60.0


class LwdMvRouter:
    """Round-robin and health state over one multi-value control_url map.

    ``url_map`` is the cloud_id -> URL mapping from the coordination config;
    ``known_clouds`` is the domain of cloud ids the data plane can reach (the
    topology's cloud ids). ``known_clouds=None`` leaves the map inactive, the
    equivalent of the deployment form with no reachable cloud set.
    """

    def __init__(
        self,
        url_map: Mapping[int, str] | None,
        known_clouds: Sequence[int] | None = None,
    ) -> None:
        self._url_map: dict[int, str] = dict(url_map or {})
        self._known_clouds: tuple[int, ...] | None = (
            tuple(sorted(known_clouds)) if known_clouds else None
        )
        self._cursor: int = 0
        self._health_fails: dict[int, int] = {}
        self._cooldown_until: dict[int, float] = {}
        self._checked: bool = False
        self._inactive_warned: bool = False
        self._rotation_warned: bool = False
        self._mismatch_warned: set[tuple[int, int]] = set()

    def url_map(self) -> dict[int, str]:
        """The validated multi-value map; empty means the form is off.

        Key-set validation is one-shot: extra keys fail fast; missing keys
        warn once. Without a known cloud domain the map has nowhere to
        deliver: announce once and treat it as unconfigured (no probe; the
        single-value control_url remains the working path)."""
        if not self._url_map:
            return {}
        if self._known_clouds is None or not self._known_clouds:
            if not self._inactive_warned:
                self._inactive_warned = True
                lwd_logger.warning(
                    "[edge-route] control_url_map inactive: no known cloud "
                    "domain (single-instance deployments should keep the "
                    "single-value control_url); probing disabled"
                )
            return {}
        if not self._checked:
            self._checked = True
            extra = sorted(set(self._url_map) - set(self._known_clouds))
            if extra:
                raise RuntimeError(
                    f"[edge-route] control_url_map keys {extra} not declared "
                    f"in the known clouds {list(self._known_clouds)}: a "
                    "probed cloud without a data-plane channel would bind "
                    "requests nowhere (fail-fast)"
                )
            missing = sorted(set(self._known_clouds) - set(self._url_map))
            if missing:
                lwd_logger.warning(
                    "[edge-route] known clouds %s absent from control_url_map: "
                    "never selected (idle clouds)",
                    missing,
                )
        return self._url_map

    def domain(self) -> list[int]:
        """The rotation domain: the map keys in ascending order (deterministic),
        with cooled-down clouds excluded; when every cloud is cooling the
        domain falls back to the full set (an empty domain is worse than dead
        clouds: keep probing, a success resets)."""
        keys = sorted(self.url_map())
        now = time.monotonic()
        live = [
            cloud for cloud in keys
            if self._cooldown_until.get(cloud, 0.0) <= now
        ]
        return live if live else keys

    def next_cloud(self) -> int | None:
        """Rotation pick at the submission point: read the position, then
        advance (exactly one advance per request). Returns None when the
        multi-value form is off (submission takes the single-value path)."""
        domain = self.domain()
        if not domain:
            return None
        if not self._rotation_warned:
            self._rotation_warned = True
            lwd_logger.warning(
                "[edge-route] multi-value control_url (dev/debug, no "
                "gateway): edge-side round-robin over clouds %s (no "
                "per-request gateway routing, no prefix affinity)",
                domain,
            )
        cloud = domain[self._cursor % len(domain)]
        self._cursor += 1
        return cloud

    def peek_cloud(self) -> int | None:
        """Pure read of the current rotation position; never advances.

        This is the fail-open landing spot: the next pending position, the
        cloud after the most recent submission target. The single advance
        point is ``next_cloud``; advancing here would let scheduler hot-path
        noise push the cursor and degrade the rotation."""
        domain = self.domain()
        if not domain:
            return None
        return domain[self._cursor % len(domain)]

    def health_fail(self, cloud_id: int) -> None:
        """Account a probe failure: consecutive failures reaching the
        threshold cool the cloud down (excluded from the rotation and peek
        domains)."""
        fails = self._health_fails.get(cloud_id, 0) + 1
        self._health_fails[cloud_id] = fails
        if fails >= _LWD_MV_HEALTH_FAILS:
            self._cooldown_until[cloud_id] = (
                time.monotonic() + _LWD_MV_HEALTH_COOLDOWN_S
            )
            lwd_logger.warning(
                "[edge-route] cloud %d unhealthy: %d consecutive probe "
                "failures, excluded from round-robin for %.0fs (gentle "
                "retry after cooldown)",
                cloud_id, fails, _LWD_MV_HEALTH_COOLDOWN_S,
            )

    def health_ok(self, cloud_id: int) -> None:
        """A successful probe resets the cloud (re-admitted; a cooled-down
        cloud probed healthy again resets the same way)."""
        if self._health_fails.get(cloud_id):
            self._health_fails[cloud_id] = 0
        if cloud_id in self._cooldown_until:
            del self._cooldown_until[cloud_id]
            lwd_logger.info(
                "[edge-route] cloud %d healthy again: back in round-robin",
                cloud_id,
            )

    def resolve_receipt(
        self, target_cloud: int | None, receipt_cloud: int | None
    ) -> int | None:
        """Resolve the routing receipt: the submission target is the
        authority in the multi-value form; the receipt only cross-checks.

        A missing receipt cloud (legacy cloud control message) falls back to
        the target; a mismatch (the cloud-side id drifting from the map keys)
        warns once per (target, receipt) pair but still binds by the target.
        The single-value form passes the receipt through unchanged: it is the
        gateway's routing decision."""
        if target_cloud is None:
            return receipt_cloud
        if receipt_cloud is not None and receipt_cloud != target_cloud:
            pair = (target_cloud, receipt_cloud)
            if pair not in self._mismatch_warned:
                self._mismatch_warned.add(pair)
                lwd_logger.warning(
                    "[edge-route] probe receipt cloud_id=%d != submission "
                    "target cloud=%d (check the cloud-side id vs the "
                    "topology cloud ids / control_url_map keys); binding by "
                    "submission target",
                    receipt_cloud, target_cloud,
                )
        return target_cloud
