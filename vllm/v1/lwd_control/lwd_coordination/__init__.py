# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LWD edge-cloud prefix coordination: wire protocol, routing, probing and
the cloud-side HTTP probe/usage plane.

``lwd_prefix`` is the frozen wire contract (hash chain, headers, probe
result); ``lwd_mm_identity`` derives the multimodal ABI fingerprint;
``lwd_binding`` holds the request-to-cloud ledger; ``lwd_mv_router`` picks
clouds for the multi-value direct form; ``lwd_edge_client`` speaks the
SSE gateway form and owns the probe runtime thread; ``lwd_cloud_control``
is the cloud-side HTTP probe/usage server. All tunables come from
``vllm.config.lwd.coordination``.
"""

from .lwd_binding import LwdEdgeBindingTable
from .lwd_cloud_control import (
    LwdCloudControlBridge,
    LwdCloudControlProcessor,
    LwdProbeAnswer,
    LwdProbeCommand,
    LwdUsageEvent,
    create_lwd_cloud_control_app,
    lwd_start_cloud_control_server,
)
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
from .lwd_request_namespace import (
    lwd_parse_request_edge,
    lwd_unwrap_request_id,
    lwd_wrap_request_id,
)

__all__ = [
    "BLOCK_HASH_PREFIX",
    "DIGEST_SIZE",
    "MM_ABI_VERSION",
    "PROTOCOL_VERSION",
    "PROTOCOL_VERSION_MM",
    "TAIL_HASH_PREFIX",
    "LwdCloudControlBridge",
    "LwdCloudControlProcessor",
    "LwdEdgeBindingTable",
    "LwdEdgePrefixClient",
    "LwdMvRouter",
    "LwdPrefixHasher",
    "LwdPrefixManifest",
    "LwdProbeAnswer",
    "LwdProbeCommand",
    "LwdProbeResult",
    "LwdProbeRuntime",
    "LwdUsage",
    "LwdUsageEvent",
    "compute_processor_fingerprint",
    "create_lwd_cloud_control_app",
    "lwd_parse_request_edge",
    "lwd_start_cloud_control_server",
    "lwd_unwrap_request_id",
    "lwd_wrap_request_id",
    "mm_abi_header_value",
    "validate_mm_abi_header",
]
