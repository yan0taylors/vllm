# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Processor fingerprint and MM-ABI identity: the shared multimodal chaining
material of the edge and the cloud.

The fingerprint is both HMAC material for media-covered blocks (the 0x04/0x05
hash domains) and the ``X-Edge-Cloud-MM-ABI: mm1:<base64url(fingerprint)>``
header riding v2 requests, which the cloud arbitrates. Edge-side chaining and
cloud-side comparison must be bit-identical, so both sides share this one
implementation (the on-wire digest encoding stays owned by the protocol layer
``lwd_prefix``).

**The fingerprint material deliberately excludes local model/tokenizer path
strings**: the same artifact is usually mounted at different paths on the edge
and the cloud, and mixing paths into the fingerprint would make one deployment
compute two fingerprints, so with ``enforce_mm_abi_match`` enabled every normal
image request would turn into a cloud-side 400. Stable identity is carried by
revision / code_revision / config hashes. Pure-text requests are unaffected:
the fingerprint only enters media-covered blocks, the text chain is unchanged
bit for bit.
"""

from __future__ import annotations

import copy
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from vllm import envs
from vllm.v1.lwd_control.lwd_coordination.lwd_prefix import (
    HEADER_MM_ABI,
    decode_digest,
    encode_digest,
)

MM_ABI_VERSION = "mm1"

_FILE_HASH_CHUNK_SIZE = 1024 * 1024
# Preprocessing package versions that decide "how media become placeholder
# tokens and content digests".
_PROCESSOR_SOFTWARE_PACKAGES = (
    "vllm",
    "vllm-ascend",
    "transformers",
    "pillow",
)
# MultiModalConfig fields that affect media preprocessing, placeholder layout
# and content digests.
_MM_CONFIG_FIELDS = (
    "mm_processor_kwargs",
    "media_io_kwargs",
    "limit_per_prompt",
    "interleave_mm_strings",
    "video_pruning_rate",
)
# These fields are node-local resource locations: they do not participate in
# the deployment identity.
_ARTIFACT_LOCATION_PLACEHOLDER = "<model-artifact>"


def mm_abi_header_value(fingerprint: bytes) -> str:
    """Render a processor fingerprint as the ``X-Edge-Cloud-MM-ABI`` value."""
    return f"{MM_ABI_VERSION}:{encode_digest(fingerprint)}"


def validate_mm_abi_header(value: str | None) -> str:
    """Validate an ``X-Edge-Cloud-MM-ABI`` header value, returned unchanged."""
    if value is None:
        raise ValueError(f"missing required header {HEADER_MM_ABI}")
    declared, separator, encoded = value.partition(":")
    if not separator or declared != MM_ABI_VERSION or not encoded:
        raise ValueError(
            f"{HEADER_MM_ABI} must be "
            f"'{MM_ABI_VERSION}:<base64url fingerprint>'"
        )
    # decode_digest also checks the 32-byte length.
    decode_digest(encoded)
    return value


def compute_processor_fingerprint(model_config: Any) -> bytes:
    """Return a 32-byte digest identifying the multimodal deployment ABI.

    Coverage: the resolved model/processor configuration + preprocessing
    package versions + the media digest algorithm. It detects common
    deployment drift (model version change, processor/media io configuration
    change, transformers upgrade) but does not prove that two media inputs
    execute equivalently (the pre-existing risk inherited from upstream
    mm_hash). Every field is read via ``getattr``: missing fields join the
    digest as None, so both sides share one reading convention."""
    mm_config = getattr(model_config, "multimodal_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    hf_text_config = getattr(model_config, "hf_text_config", None)
    payload: dict[str, Any] = {
        "revision": getattr(model_config, "revision", None),
        "code_revision": getattr(model_config, "code_revision", None),
        "tokenizer_revision": getattr(model_config, "tokenizer_revision", None),
        "hf_config_commit": getattr(hf_config, "_commit_hash", None),
        "model_type": getattr(hf_text_config, "model_type", None),
        "architecture": getattr(model_config, "_architecture", None),
        # vLLM's own config hash covers the graph choices that hand-listed
        # fields easily miss (dtype/quantization, MM encoder attention
        # backend/TP/FP8 and so on).
        "model_config_hash": _config_hash(
            model_config,
            neutralized_fields={
                "model": _ARTIFACT_LOCATION_PLACEHOLDER,
                "model_weights": _ARTIFACT_LOCATION_PLACEHOLDER,
                "tokenizer": _ARTIFACT_LOCATION_PLACEHOLDER,
            },
            scrub_hf_locations=True,
        ),
        "multimodal_config_hash": _config_hash(
            mm_config,
            neutralized_fields={
                # Weight calibration files join the identity by content, not
                # by local path.
                "mm_encoder_fp8_scale_path": _file_content_identity(
                    getattr(mm_config, "mm_encoder_fp8_scale_path", None)
                )
            },
        ),
        "hf_image_processor_config": getattr(
            model_config, "hf_image_processor_config", None
        ),
        "mm_hasher_algorithm": getattr(
            envs, "VLLM_MM_HASHER_ALGORITHM", None
        ),
        "software_versions": {
            package: _package_version(package)
            for package in _PROCESSOR_SOFTWARE_PACKAGES
        },
    }
    for field in _MM_CONFIG_FIELDS:
        value = getattr(mm_config, field, None)
        if value is None:
            # mm_processor_kwargs / media_io_kwargs are InitVars that may
            # only land on ModelConfig; fall back when MultiModalConfig is
            # missing so both sides share one reading convention.
            value = getattr(model_config, field, None)
        payload[field] = value
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           default=repr)
    return hashlib.sha256(canonical.encode("utf-8")).digest()


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _config_hash(
    config: Any,
    *,
    neutralized_fields: dict[str, Any] | None = None,
    scrub_hf_locations: bool = False,
) -> str | None:
    """Hash the configuration with node-local resource locations scrubbed.

    Returns None when the config has no ``compute_hash``."""
    if config is None:
        return None
    stable_config = copy.copy(config)
    for field, value in (neutralized_fields or {}).items():
        if hasattr(stable_config, field):
            setattr(stable_config, field, value)
    if scrub_hf_locations:
        for field in ("hf_config", "hf_text_config"):
            nested = getattr(stable_config, field, None)
            if nested is None:
                continue
            stable_nested = copy.deepcopy(nested)
            _scrub_name_or_path(stable_nested, set())
            setattr(stable_config, field, stable_nested)
    compute_hash = getattr(stable_config, "compute_hash", None)
    if not callable(compute_hash):
        return None
    return str(compute_hash())


def _scrub_name_or_path(value: Any, seen: set[int]) -> None:
    """Recursively replace ``_name_or_path`` in HF configs with a placeholder
    constant (removing local paths)."""
    if value is None or isinstance(
        value, (bool, int, float, str, bytes)
    ):
        return
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if isinstance(value, dict):
        if "_name_or_path" in value:
            value["_name_or_path"] = _ARTIFACT_LOCATION_PLACEHOLDER
        for nested in value.values():
            _scrub_name_or_path(nested, seen)
        return
    if isinstance(value, (list, tuple, set)):
        for nested in value:
            _scrub_name_or_path(nested, seen)
        return
    attributes = getattr(value, "__dict__", None)
    if not isinstance(attributes, dict):
        return
    if "_name_or_path" in attributes:
        attributes["_name_or_path"] = _ARTIFACT_LOCATION_PLACEHOLDER
    for nested in attributes.values():
        _scrub_name_or_path(nested, seen)


def _file_content_identity(path: Any) -> str | None:
    """Identify a graph input by its bytes rather than its local path.

    Returns None for an empty path."""
    if path is None:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(_FILE_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
