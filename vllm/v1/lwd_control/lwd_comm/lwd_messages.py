# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Business messages of the LWD edge-cloud control plane.

Ported from the `prefill_only_chunked_prefill_mm` branch. Messages carry
scheduling metadata only; tensors travel on the data plane. The control link
treats them as opaque payloads.

Fields may only be appended with defaults. Any other change bumps
`LWD_MESSAGE_VERSION`, which the link handshake checks on both sides.
"""

import dataclasses
import json
from typing import Any, TypeAlias, TypeVar, cast

import msgspec

from vllm.config.lwd import LwdConnectionId

T = TypeVar("T")

# 6: every control notification carries its full (edge, cloud, DP) connection.
# Version 5 is reserved for the historical 1108 branch and is deliberately
# skipped; peers with missing or mismatched stamps fail fast outside 1E1C.
LWD_MESSAGE_VERSION = 6

LWD_NOT_FINISHED = -1
"""`LwdC2eNotify.finish_reasons` value of a request still running."""


class LwdRangeItem(msgspec.Struct, gc=False):  # type: ignore[call-arg]
    """One request's prompt chunk inside a prefill batch."""

    request_id: str
    offset: int
    """Chunk start in the prompt; the cloud requires it to equal the number
    of tokens it has already computed for the request."""
    num_tokens: int
    has_mrope: bool = False
    mrope_positions: bytes = b""
    """Legacy per-chunk positions; empty on the current protocol path. Full
    prompt positions travel in LwdRequestNotify."""
    is_refill: bool = False
    """Out-of-band embedding refill after cloud preemption."""
    prefix_nonce: int = 0
    """Generation of the request that owns this chunk."""


class LwdRangeNotify(msgspec.Struct, gc=False, tag=True):  # type: ignore[call-arg]
    """Edge -> cloud. Announces the chunks of one prefill batch.

    Duplicates are idempotent per (request_id, offset).
    """

    seqno: int
    """Batch number pairing the batch with its data-plane transfer."""
    items: list[LwdRangeItem] = []
    edge_id: int = 0
    cloud_id: int = 0
    dp_idx: int = 0
    """Connection stamp; all zero is the implicit 1E1C form."""


class LwdRequestNotify(msgspec.Struct, gc=False, tag=True):  # type: ignore[call-arg]
    """Edge -> cloud. Registers a request before its prefill chunks.

    Sampling fields are the ones that change cloud-side token selection;
    detokenizer-level parameters such as stop strings stay on the edge.
    """

    request_id: str
    num_prompt_tokens: int
    max_tokens: int = 16
    block_hashes: list[bytes] = []
    """Full-block hash chain of the prompt for cloud prefix cache hits; empty
    makes the cloud hash locally."""
    prefix_nonce: int = 0
    """Nonzero while the edge waits for a matching LwdPrefixReadyNotify."""
    skip_reading_prefix_cache: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    seed: int | None = None
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    ignore_eos: bool = False
    stop_token_ids: list[int] = []
    min_tokens: int = 0
    eos_token_id: int | None = None
    """Resolved by the edge tokenizer; the cloud cannot derive it. None leaves
    `max_tokens` as the only stop condition on the cloud."""
    mrope_positions: bytes = b""
    """Contiguous int64 [3, num_prompt_tokens] M-RoPE positions of the full
    prompt; empty when unused."""
    structured_outputs: str = ""
    """`lwd_encode_structured_outputs` wire JSON of the validated
    `StructuredOutputsParams`; empty for a plain request. The cloud rebuilds
    the params, compiles the grammar and applies the sampling bitmask; the
    edge compiles its own copy only for the picker's grammar-ready filter."""
    edge_id: int = 0
    cloud_id: int = 0
    dp_idx: int = 0


def lwd_encode_structured_outputs(params: Any) -> str:
    """JSON wire form of a validated `StructuredOutputsParams`.

    The private fields (`_backend*`, `init=False`) never ride the field
    dict: `params_cls(**fields)` rejects them as constructor keywords, so the
    edge-chosen backend travels next to the fields in the envelope and is
    re-attached after decode. This is the single implementation shared by the
    edge packer and the cloud rebuild.
    """
    data = dataclasses.asdict(params)
    backend = data.pop("_backend", None)
    data.pop("_backend_was_auto", None)
    return json.dumps({"fields": data, "backend": backend})


def lwd_decode_structured_outputs(payload: str, params_cls: type[T]) -> T | None:
    """Rebuild the constraint params from the wire JSON; None if plain.

    Args:
        payload: The `LwdRequestNotify.structured_outputs` string.
        params_cls: The dataclass to rebuild (its public, `init=True` fields
            must match the envelope's `fields` dict).

    Returns:
        The rebuilt params with `_backend` re-attached, or None when the
        payload is the plain-request empty string.

    Raises:
        ValueError: The payload is not a well-formed envelope.
    """
    if not payload:
        return None
    try:
        envelope = json.loads(payload)
        fields = envelope["fields"]
        backend = envelope["backend"]
        if not isinstance(fields, dict):
            raise TypeError(f"fields must be a dict, got {type(fields).__name__}")
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"malformed structured_outputs payload {payload!r}") from exc
    try:
        params = params_cls(**fields)
    except TypeError as exc:
        raise ValueError(
            f"structured_outputs fields do not fit {params_cls}: {exc}"
        ) from exc
    # init=False private field: cannot ride the constructor, reattach here.
    cast(Any, params)._backend = backend
    return params


LWD_SAMPLING_FIELDS: tuple[tuple[str, object], ...] = (
    ("max_tokens", 16),
    ("temperature", 1.0),
    ("top_p", 1.0),
    ("top_k", 0),
    ("min_p", 0.0),
    ("seed", None),
    ("repetition_penalty", 1.0),
    ("presence_penalty", 0.0),
    ("frequency_penalty", 0.0),
    ("ignore_eos", False),
    ("stop_token_ids", []),
    ("min_tokens", 0),
)
"""(field, default) of the `SamplingParams` arguments `LwdRequestNotify`
carries; the edge packs and the cloud rebuilds them from this one table.
`eos_token_id` is not a `SamplingParams` argument and travels on its own."""
LWD_SAMPLING_DEFAULTS = dict(LWD_SAMPLING_FIELDS)


class LwdAbortNotify(msgspec.Struct, gc=False, tag=True):  # type: ignore[call-arg]
    """Edge -> cloud. Aborts a request."""

    request_id: str
    prefix_nonce: int = 0
    """Generation to abort; zero is the legacy form."""
    edge_id: int = 0
    cloud_id: int = 0
    dp_idx: int = 0


class LwdCloudParamsNotify(msgspec.Struct, gc=False, tag=True):  # type: ignore[call-arg]
    """Cloud -> edge. Scheduling limits both sides must agree on.

    A mismatch does not break the link: it makes the cloud hold prefills and
    serialize them behind decodes without any error, so the edge must check
    these against its own configuration before serving.
    """

    max_num_batched_tokens: int
    max_num_seqs: int
    num_speculative_tokens: int
    dp_coordinator_addr: str = ""
    """Frontend stats endpoint of this DP rank's coordinator, for the edge to
    trigger the first DP wave; empty when the cloud runs no coordinator
    (single DP or non-MoE engines)."""
    prefix_caching: bool = False
    hash_block_size: int = 0
    prefix_alignment_tokens: int = 0
    prefix_alignment_required: bool = False
    prefix_use_eagle: bool = False
    edge_id: int = 0
    cloud_id: int = 0
    dp_idx: int = 0


class LwdPrefixReadyNotify(msgspec.Struct, gc=False, tag=True):  # type: ignore[call-arg]
    """Cloud -> edge. Result of one nonce-scoped prefix reservation."""

    request_id: str
    hit_tokens: int
    nonce: int
    error: str = ""
    edge_id: int = 0
    cloud_id: int = 0
    dp_idx: int = 0


class LwdRefillNotify(msgspec.Struct, gc=False, tag=True):  # type: ignore[call-arg]
    """Cloud -> edge. Re-embed an absolute prompt range after preemption."""

    request_id: str
    start_row: int
    end_row: int
    prefix_nonce: int = 0
    """Request generation that requested this refill; zero is the legacy form."""
    edge_id: int = 0
    cloud_id: int = 0
    dp_idx: int = 0


class LwdC2eNotify(msgspec.Struct, gc=False, tag=True):  # type: ignore[call-arg]
    """Cloud -> edge. Metadata of one cloud step.

    Arrives before the step's hidden states so the edge can post a receive of
    the exact size. `hidden_num_elements == 0` only finishes requests.
    """

    hidden_num_elements: int
    top_id_ths: list[list[int]]
    num_accepted_tokens: list[int]
    req_ids: list[str]
    finish_reasons: list[int] = []
    """Per request, aligned with `req_ids`: `LWD_NOT_FINISHED` or a
    `FinishReason` value."""
    down_seqno: int = -1
    """Channel sequence number of this step's DOWN hidden-state transfer; the
    edge pairs its receive with it."""
    token_ids: list[list[int]] = []
    """Diagnostics only: accepted token ids per request; empty unless the
    edge skips sampling."""
    prefix_nonces: list[int] = []
    """Generation of each request row, aligned with req_ids."""
    edge_id: int = 0
    cloud_id: int = 0
    dp_idx: int = 0
    """Source connection of this C2E carrier."""


LwdEdgeToCloudMessage: TypeAlias = LwdRequestNotify | LwdRangeNotify | LwdAbortNotify
LwdCloudToEdgeMessage: TypeAlias = (
    LwdCloudParamsNotify | LwdC2eNotify | LwdPrefixReadyNotify | LwdRefillNotify
)
LwdMessage: TypeAlias = LwdEdgeToCloudMessage | LwdCloudToEdgeMessage


def lwd_message_connection_id(message: LwdMessage) -> LwdConnectionId:
    """Read the connection envelope shared by every business notification."""
    return LwdConnectionId(message.edge_id, message.cloud_id, message.dp_idx)


def lwd_stamp_message(
    message: LwdMessage, connection_id: LwdConnectionId
) -> LwdMessage:
    """Stamp a notification immediately before encoding it for its peer."""
    message.edge_id = connection_id.edge_id
    message.cloud_id = connection_id.cloud_id
    message.dp_idx = connection_id.dp_idx
    return message


def lwd_validate_message_connection(
    message: LwdMessage, expected: LwdConnectionId
) -> None:
    """Reject missing or incorrect stamps instead of misrouting a notification."""
    actual = lwd_message_connection_id(message)
    if actual == expected:
        return
    reason = "missing" if actual == LwdConnectionId(0, 0, 0) else "mismatched"
    raise ValueError(
        f"[LWD] {reason} connection stamp on {type(message).__name__}: "
        f"got {actual}, expected {expected}"
    )


_LWD_EDGE_TO_CLOUD_DECODER = msgspec.msgpack.Decoder(LwdEdgeToCloudMessage)
_LWD_CLOUD_TO_EDGE_DECODER = msgspec.msgpack.Decoder(LwdCloudToEdgeMessage)


def lwd_encode_message(message: LwdMessage) -> bytes:
    return msgspec.msgpack.encode(message)


def lwd_decode_edge_to_cloud(data: bytes | memoryview) -> LwdEdgeToCloudMessage:
    """Decode a payload received by the cloud.

    Raises:
        msgspec.DecodeError: Not valid msgpack.
        msgspec.ValidationError: Not an edge -> cloud message.
    """
    return _LWD_EDGE_TO_CLOUD_DECODER.decode(data)


def lwd_decode_cloud_to_edge(data: bytes | memoryview) -> LwdCloudToEdgeMessage:
    """Decode a payload received by the edge.

    Raises:
        msgspec.DecodeError: Not valid msgpack.
        msgspec.ValidationError: Not a cloud -> edge message.
    """
    return _LWD_CLOUD_TO_EDGE_DECODER.decode(data)
