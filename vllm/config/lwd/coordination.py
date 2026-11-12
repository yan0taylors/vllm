# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache coordination metadata only; no key reads, clients or server startup."""

from typing import Annotated, Any
from urllib.parse import urlsplit

from pydantic import ConfigDict, Field, StrictBool, field_validator, model_validator
from pydantic.dataclasses import dataclass

from .topology import LwdPort, LwdPositiveSeconds, LwdString, parse_cloud_url_map

LWD_PROBE_HEADROOM_S = 5.0
"""Seconds the cloud-side deadline stays below probe_timeout so the cloud
answers first (200 or 503) and the edge only gives up afterwards; the
scheduler watchdog adds the same headroom on top of the edge deadline."""


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdCoordinationConfig:
    enabled: StrictBool = False
    control_url: LwdString | None = None
    """Single-value form (gateway/single cloud): the receipt is the routing
    decision, the production contract."""
    control_url_map: dict[int, str] = Field(default_factory=dict)
    """Multi-value direct form (cloud_id -> URL, dev/debug fallback): the
    edge picks a cloud itself by round-robin; mutually exclusive with the
    single-value form. Keys share the id space of the topology clouds."""
    placement_only: StrictBool = False
    """Probe only places the request on a cloud DP (pure load balancing, no
    HMAC chain); drops the tenant key requirement on the edge."""
    tenant_key_file: LwdString | None = None
    consumer_id: LwdString | None = None
    listen_host: LwdString = "0.0.0.0"
    listen_port: LwdPort = 8100
    instance_id: LwdString = "cloud-0"
    connect_timeout: Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)] = (
        10.0
    )
    probe_timeout: LwdPositiveSeconds = 25.0
    """End-to-end probe deadline: the cloud answers at
    probe_timeout - LWD_PROBE_HEADROOM_S and the edge gives up at
    probe_timeout, so the cloud answers first."""

    @field_validator("control_url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        if value is not None:
            url = urlsplit(value)
            if url.scheme not in ("http", "https") or not url.hostname:
                raise ValueError("[LWD] control_url must be an HTTP(S) endpoint")
        return value

    @field_validator("control_url_map", mode="before")
    @classmethod
    def validate_map(cls, value: object) -> object:
        if value is not None:
            return parse_cloud_url_map(value)
        return value

    @model_validator(mode="after")
    def validate_url_forms(self) -> "LwdCoordinationConfig":
        if self.control_url and self.control_url_map:
            raise ValueError(
                "[LWD] control_url (single, gateway form) and control_url_map "
                "(multi-value direct form) are mutually exclusive"
            )
        return self

    @classmethod
    def from_dict(
        cls, raw: dict[str, Any] | None, *, cloud_id: int | None = None
    ) -> "LwdCoordinationConfig":
        """Build the runtime config, normalizing the two wire spellings of
        the control endpoint: a dict-literal control_url and a field-name
        control_url_map both fold into control_url_map. Giving both is a
        config ambiguity and fails fast, as does a non-string non-map
        control_url."""
        raw = dict(raw) if raw else {}
        raw_url = raw.pop("control_url", None)
        raw_map = raw.pop("control_url_map", None)
        if raw_url is not None and raw_map is not None:
            raise ValueError(
                "[LWD] control_url (single/map form) and control_url_map are "
                "mutually exclusive"
            )
        if raw_map is not None:
            raw["control_url_map"] = parse_cloud_url_map(raw_map)
        elif isinstance(raw_url, dict):
            raw["control_url_map"] = parse_cloud_url_map(raw_url)
        elif isinstance(raw_url, str):
            raw["control_url"] = raw_url
        elif raw_url is not None:
            raise ValueError(
                "[LWD] control_url must be a string (gateway/single cloud) or "
                "a cloud_id -> URL map (multi-value direct form)"
            )
        defaults = {} if cloud_id is None else {"instance_id": f"cloud-{cloud_id}"}
        return cls(**(defaults | raw))

    def validate_role(self, role: str) -> None:
        if not self.enabled:
            return
        if self.probe_timeout <= LWD_PROBE_HEADROOM_S:
            raise ValueError(
                "[LWD] probe_timeout must be greater than the "
                f"{LWD_PROBE_HEADROOM_S:g}s cloud headroom (the cloud answers "
                "at probe_timeout minus the headroom; the edge gives up at "
                "probe_timeout)"
            )
        if role == "edge":
            if not (self.control_url or self.control_url_map):
                raise ValueError(
                    "[LWD] edge coordination requires control_url or "
                    "control_url_map"
                )
            if not self.tenant_key_file and not self.placement_only:
                raise ValueError(
                    "[LWD] edge coordination requires tenant_key_file "
                    "(the key is read edge-side only; the cloud must not "
                    "hold it)"
                )
        elif role == "cloud" and self.tenant_key_file:
            # The tenant key must be held edge-side only: it is the privacy
            # premise of the protocol, so a cloud-side setting is rejected
            # instead of silently ignored.
            raise ValueError(
                "[LWD] tenant_key_file is edge-side only; remove it from the "
                "cloud coordination block"
            )
