# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Edge-scoped request id namespace of the cloud engine.

Multiple edges attach to one cloud DP engine and their client-chosen request
ids can collide. Every request entering the engine is wrapped with its source
edge id ("<edge_id>:<request_id>", the target scheme corresponding to the
source branch's ``wrap_req_id`` "e<edge_id>-<request_id>") so the scheduler,
the arrival ledgers and the prefix reservations all share one collision-free
key space. Wire messages leaving the engine carry the edge's original id
again: the namespace is invisible on the edge.

Only the source edge is namespaced: one engine serves exactly one cloud DP
rank, so the (edge_id) prefix is already unique among its connections.
"""

from __future__ import annotations

_LWD_NAMESPACE_SEPARATOR = ":"


def lwd_wrap_request_id(edge_id: int, request_id: str) -> str:
    """Wrap an edge request id into the cloud-internal namespace."""
    return f"{edge_id}{_LWD_NAMESPACE_SEPARATOR}{request_id}"


def lwd_parse_request_edge(request_id: str) -> tuple[int | None, str]:
    """Return (source edge id, original request id); None when unnamespaced.

    The check is strictly decimal (no sign, spaces, underscores or non-ASCII
    digits) so an id like "-1:x" or "1_0:x" is a plain edge request id, never
    a misparsed namespace.
    """
    prefix, separator, rest = request_id.partition(_LWD_NAMESPACE_SEPARATOR)
    if separator and prefix and prefix.isascii() and prefix.isdecimal():
        return int(prefix), rest
    return None, request_id


def lwd_unwrap_request_id(request_id: str, *, multi_edge: bool = False) -> str:
    """Restore the edge's original request id for an outbound message.

    Unnamespaced ids pass through unchanged, which keeps single-edge
    deployments byte-identical to the pre-namespace wire format. With more
    than one attached edge an unnamespaced id cannot be attributed to a
    source and fails fast instead of leaking into a shared ledger.
    """
    edge_id, original = lwd_parse_request_edge(request_id)
    if edge_id is None and multi_edge:
        raise ValueError(
            f"[LWD] unnamespaced cloud request id {request_id!r} cannot be "
            "unwrapped in a multi-edge deployment"
        )
    return original
