# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Edge-side request-to-cloud binding table.

The single ledger that binds and unbinds a request with its target cloud
instance over the request lifecycle:

* ``bind`` registers the binding at request entry (before the admission
  announcement);
* range announcements and aborts are directed with ``cloud_of``;
* ``unbind`` releases the binding at finish/abort, so a late result no
  longer finds a request to claim.

The binding is driven by the probe receipt's cloud id (routed multi-cloud)
with a caller-provided default fallback. This table's interface is the
portable piece; the scheduler owns its lifetime.
"""

from __future__ import annotations


class LwdEdgeBindingTable:
    """req_id -> cloud_id binding table; a thin dict wrapper whose lifetime
    follows the scheduler."""

    def __init__(self) -> None:
        self._req_cloud: dict[str, int] = {}

    def bind(self, request_id: str, cloud_id: int) -> None:
        """Register the request's cloud binding (a repeated bind overwrites)."""
        self._req_cloud[request_id] = cloud_id

    def cloud_of(self, request_id: str, default: int | None = None) -> int | None:
        """Look up the request's bound cloud; unregistered returns ``default``
        (None when not given), so a late result after unbind is not claimed."""
        return self._req_cloud.get(request_id, default)

    def unbind(self, request_id: str) -> int | None:
        """Unbind and return the previously bound cloud; unregistered returns
        None (idempotent)."""
        return self._req_cloud.pop(request_id, None)

    def bound_cloud_ids(self) -> set[int]:
        """The set of cloud ids that currently hold in-flight requests."""
        return set(self._req_cloud.values())

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._req_cloud

    def __len__(self) -> int:
        return len(self._req_cloud)
