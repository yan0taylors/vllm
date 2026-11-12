# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LWD edge-cloud prefix coordination: wire protocol, routing and probing.

``lwd_prefix`` is the frozen wire contract (hash chain, headers, probe
result); ``lwd_mm_identity`` derives the multimodal ABI fingerprint;
``lwd_binding`` holds the request-to-cloud ledger; ``lwd_mv_router`` picks
clouds for the multi-value direct form; ``lwd_edge_client`` speaks the
SSE gateway form and owns the probe runtime thread. All tunables come
from ``vllm.config.lwd.coordination``.
"""

from .lwd_binding import LwdEdgeBindingTable
from .lwd_edge_client import LwdEdgePrefixClient, LwdProbeRuntime
from .lwd_mm_identity import (
    MM_ABI_VERSION,
    compute_processor_fingerprint,
    mm_abi_header_value,
    validate_mm_abi_header,
)
from .lwd_mv_router import LwdMvRouter
from .lwd_prefix import (
    BLOCK_HASH_PREFIX,
    DIGEST_SIZE,
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_MM,
    TAIL_HASH_PREFIX,
    LwdPrefixHasher,
    LwdPrefixManifest,
    LwdProbeResult,
    LwdUsage,
)

__all__ = [
    "BLOCK_HASH_PREFIX",
    "DIGEST_SIZE",
    "MM_ABI_VERSION",
    "PROTOCOL_VERSION",
    "PROTOCOL_VERSION_MM",
    "TAIL_HASH_PREFIX",
    "LwdEdgeBindingTable",
    "LwdEdgePrefixClient",
    "LwdMvRouter",
    "LwdPrefixHasher",
    "LwdPrefixManifest",
    "LwdProbeResult",
    "LwdProbeRuntime",
    "LwdUsage",
    "compute_processor_fingerprint",
    "mm_abi_header_value",
    "validate_mm_abi_header",
]
