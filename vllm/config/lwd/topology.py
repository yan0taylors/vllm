# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""File schema for LWD, independent of devices and distributed initialization."""

from collections.abc import Hashable
from hashlib import sha256
from ipaddress import ip_address
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

import msgspec
import yaml
from pydantic import (
    AfterValidator,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    TypeAdapter,
    field_validator,
    model_validator,
)
from pydantic.dataclasses import dataclass

LwdId = Annotated[int, Field(strict=True, ge=0)]
LwdCount = Annotated[int, Field(strict=True, ge=1)]
LwdPort = Annotated[int, Field(strict=True, ge=1, le=65535)]
LwdString = Annotated[StrictStr, Field(pattern=r"\S")]
LwdPositiveSeconds = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]


def _address(value: str) -> str:
    address = ip_address(value)
    if address.is_unspecified or address.is_multicast:
        raise ValueError("[LWD] addr must be a unicast endpoint IP")
    return str(address)


def _http_endpoint(value: str, field: str) -> str:
    url = urlsplit(value)
    if url.scheme not in ("http", "https") or not url.hostname:
        raise ValueError(f"[LWD] {field} must be an HTTP(S) endpoint")
    return value


def parse_cloud_url_map(raw: object) -> dict[int, str]:
    """Parse the multi-value control_url mapping (cloud_id -> URL).

    Keys may arrive as strings (JSON) or ints (YAML) and are normalized to
    non-negative ints; values must be strings naming HTTP(S) endpoints.
    Parse failures fail fast, the same severity as schema errors: silently
    dropping the map would disable the whole probe chain and fall
    multi-cloud routing back to the first cloud without any error.
    """
    if not isinstance(raw, dict):
        raise ValueError("[LWD] control_url map must be an object of cloud_id -> URL")
    parsed: dict[int, str] = {}
    for key, value in raw.items():
        try:
            cloud_id = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"[LWD] control_url map keys must be cloud ids (integers): {key!r}"
            ) from exc
        if cloud_id < 0:
            raise ValueError(
                f"[LWD] control_url map keys must be non-negative: {key!r}"
            )
        if not isinstance(value, str):
            raise ValueError(f"[LWD] control_url map values must be strings: {key!r}")
        _http_endpoint(value, f"control_url_map[{cloud_id}]")
        parsed[cloud_id] = value
    return parsed


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate mapping keys instead of silently replacing values."""

    def construct_mapping(
        self, node: yaml.MappingNode, deep: bool = False
    ) -> dict[Hashable, Any]:
        result: dict[Hashable, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("[LWD] YAML mapping keys must be strings")
            if key in result:
                raise ValueError(f"[LWD] duplicate YAML key: {key!r}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdDeployment:
    mode: Annotated[int, Field(strict=True, ge=0, le=0)]
    scene: Literal["single_instance", "edge_share", "cloud_share", "lwd_cluster"]
    hccl_world_size: LwdCount
    edges_num: LwdCount
    clouds_num: LwdCount


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdScramble:
    enable: StrictBool = False
    device_id: LwdString | None = None
    custom_factor: LwdString | None = None


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdFeatures:
    enable_early_recv: StrictBool = False
    scramble: LwdScramble = Field(default_factory=LwdScramble)


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdControlPlaneConfig:
    """Edge-cloud control link tunables, shared by every instance."""

    send_queue_max: Annotated[int, Field(strict=True, ge=1)] = 1000
    """Outbound messages buffered per peer before sending reports backpressure."""
    hello_retry_interval_s: LwdPositiveSeconds = 5.0
    """Seconds between edge registration attempts until the cloud answers."""
    handshake_timeout_s: LwdPositiveSeconds = 1800.0
    """Seconds an edge waits for its cloud; must cover a full cloud engine
    startup (weights, KV cache, graph capture)."""


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdCoordination:
    """Prefix-cache coordination block, embedded per instance for cloud_share."""

    enabled: StrictBool = False
    # Edge-side fields
    control_url: LwdString | None = None
    """Single-value form (gateway/single cloud); a dict literal is the
    multi-value form, normalized into control_url_map below."""
    control_url_map: dict[int, str] | None = None
    """Multi-value direct form (cloud_id -> URL, dev/debug fallback): the
    edge picks a cloud itself by round-robin instead of the gateway."""
    placement_only: StrictBool = False
    """Probe only places the request on a cloud DP (pure load balancing,
    no HMAC chain); drops the tenant key and prefix caching requirements."""
    tenant_key_file: LwdString | None = None
    consumer_id: LwdString | None = None
    # Cloud-side fields
    listen_host: LwdString = "0.0.0.0"
    listen_port: LwdPort = 8100
    instance_id: LwdString | None = None
    # Shared
    connect_timeout: Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)] = (
        10.0
    )
    probe_timeout: LwdPositiveSeconds = 25.0
    """End-to-end probe deadline: the cloud answers at probe_timeout-5 and
    the edge gives up at probe_timeout, so the cloud answers first."""

    @model_validator(mode="before")
    @classmethod
    def normalize_control_url(cls, data: Any) -> Any:
        """Fold the dict spelling of control_url into control_url_map; the
        two spellings are mutually exclusive (both given is a config
        ambiguity)."""
        if not isinstance(data, dict):
            return data
        raw_url = data.get("control_url")
        raw_map = data.get("control_url_map")
        if raw_url is not None and raw_map is not None:
            raise ValueError(
                "[LWD] control_url (single/map form) and control_url_map are "
                "mutually exclusive"
            )
        if isinstance(raw_url, dict):
            data = dict(data)
            data["control_url_map"] = parse_cloud_url_map(raw_url)
            data.pop("control_url")
        return data

    @field_validator("control_url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        if value is not None:
            _http_endpoint(value, "control_url")
        return value

    @field_validator("control_url_map", mode="before")
    @classmethod
    def validate_map(cls, value: object) -> object:
        if value is not None:
            return parse_cloud_url_map(value)
        return value


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdDP:
    dp_idx: LwdId
    addr: Annotated[StrictStr, AfterValidator(_address)]
    ranks: Annotated[tuple[LwdId, ...], Field(min_length=1)]
    ctrl_port: LwdPort | None = None


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdInstance:
    id: LwdId
    dp: Annotated[tuple[LwdDP, ...], Field(min_length=1)]
    coordination: LwdCoordination | None = None

    @field_validator("dp")
    @classmethod
    def order_dps(cls, dps: tuple[LwdDP, ...]) -> tuple[LwdDP, ...]:
        dps = tuple(sorted(dps, key=lambda dp: dp.dp_idx))
        if [dp.dp_idx for dp in dps] != list(range(len(dps))):
            raise ValueError("[LWD] dp_idx must be unique and contiguous from 0")
        return dps


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdLink:
    edge: LwdId
    cloud: LwdId


class LwdConnectionId(
    msgspec.Struct, frozen=True, order=True, array_like=True, gc=False
):  # type: ignore[call-arg]
    """Stable identity of one edge-to-cloud DP connection."""

    edge_id: int
    cloud_id: int
    dp_idx: int


@dataclass(frozen=True)
class LwdDPConnection:
    """One explicit instance link expanded at matching DP indices."""

    edge_id: int
    cloud_id: int
    dp_idx: int
    edge: LwdDP
    cloud: LwdDP

    @property
    def connection_id(self) -> LwdConnectionId:
        return LwdConnectionId(self.edge_id, self.cloud_id, self.dp_idx)

    @property
    def edge_leader_rank(self) -> int:
        return self.edge.ranks[0]

    @property
    def cloud_leader_rank(self) -> int:
        return self.cloud.ranks[0]


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdTopology:
    deployment: LwdDeployment
    feature_ctrl: LwdFeatures
    edges: Annotated[tuple[LwdInstance, ...], Field(min_length=1)]
    clouds: Annotated[tuple[LwdInstance, ...], Field(min_length=1)]
    instance_links: Annotated[tuple[LwdLink, ...], Field(min_length=1)]
    control_plane: LwdControlPlaneConfig = Field(default_factory=LwdControlPlaneConfig)
    digest: str = Field(default="", init=False)
    """First 16 hex characters of the source file digest; empty for from_dict."""

    @classmethod
    def from_file(cls, path: str) -> "LwdTopology":
        try:
            file_bytes = Path(path).read_bytes()
            raw = yaml.load(file_bytes.decode("utf-8"), Loader=_UniqueKeyLoader)
            topology = cls.from_dict(raw)
            # The digest belongs to the loaded snapshot, not to YAML input.
            object.__setattr__(topology, "digest", sha256(file_bytes).hexdigest()[:16])
            return topology
        except (OSError, UnicodeError, yaml.YAMLError, ValueError) as exc:
            raise ValueError(f"[LWD] Invalid topology file {path!r}: {exc}") from exc

    @classmethod
    def from_dict(cls, raw: Any) -> "LwdTopology":
        topology = TypeAdapter(cls).validate_python(raw)
        topology.validate_layout()
        return topology

    @field_validator("edges", "clouds")
    @classmethod
    def order_instances(
        cls, instances: tuple[LwdInstance, ...]
    ) -> tuple[LwdInstance, ...]:
        instances = tuple(sorted(instances, key=lambda instance: instance.id))
        if [instance.id for instance in instances] != list(range(len(instances))):
            raise ValueError("[LWD] Instance IDs must be unique and contiguous from 0")
        return instances

    def validate_layout(self) -> None:
        """Validate instance links and placement without initializing communication."""
        if (self.deployment.edges_num, self.deployment.clouds_num) != (
            len(self.edges),
            len(self.clouds),
        ):
            raise ValueError(
                "[LWD] deployment instance counts do not match edges/clouds"
            )
        if len(set(self.instance_links)) != len(self.instance_links):
            raise ValueError("[LWD] duplicate instance link")
        if any(
            link.edge >= len(self.edges) or link.cloud >= len(self.clouds)
            for link in self.instance_links
        ):
            raise ValueError("[LWD] instance_links: unknown edge/cloud instance")
        for instances, cloud in ((self.edges, False), (self.clouds, True)):
            if any(
                (dp.ctrl_port is not None) != cloud for i in instances for dp in i.dp
            ):
                raise ValueError(
                    "[LWD] ctrl_port is required on clouds and omitted on edges"
                )
        self._validate_scene()
        if len({len(instance.dp) for instance in (*self.edges, *self.clouds)}) != 1:
            raise ValueError(
                "[LWD] Instances must have matching DP indices and the same DP count"
            )
        self._validate_ranks()
        self._validate_control_ports()
        self._validate_coordination_maps()

    def _validate_coordination_maps(self) -> None:
        """Edge-cloud consistency of the multi-value control_url form.

        Map keys are cloud ids the edge will probe directly; a key that is
        not a declared cloud would bind requests to a cloud without a
        data-plane channel, so it fails fast here at load time instead of on
        the first probe. Missing clouds stay legal (they are simply never
        selected); the router announces them once at runtime.
        """
        cloud_ids = [cloud.id for cloud in self.clouds]
        for instance in (*self.edges, *self.clouds):
            coordination = instance.coordination
            if coordination is None or not coordination.control_url_map:
                continue
            extra = sorted(set(coordination.control_url_map) - set(cloud_ids))
            if extra:
                raise ValueError(
                    f"[LWD] instance {instance.id} control_url_map keys {extra} "
                    "are not declared clouds: a probed cloud without a "
                    "data-plane channel would bind requests nowhere"
                )

    def _validate_ranks(self) -> None:
        for edge in self.edges:
            if len({(dp.addr, dp.ranks) for dp in edge.dp}) != 1:
                raise ValueError(
                    "[LWD] Edge DPs must share their instance's ranks/address"
                )
        for cloud in self.clouds:
            if len({len(dp.ranks) for dp in cloud.dp}) != 1:
                raise ValueError(
                    "[LWD] Cloud instance must split its ranks equally among DPs"
                )
        # Edge DPs reuse an allocation; edge_share also reuses it across instances.
        edges = self.edges
        if self.deployment.scene == "edge_share":
            if len({(edge.dp[0].addr, edge.dp[0].ranks) for edge in edges}) != 1:
                raise ValueError(
                    "[LWD] edge_share instances must share edge ranks/address"
                )
            edges = edges[:1]
        allocations = [edge.dp[0] for edge in edges]
        allocations.extend(dp for cloud in self.clouds for dp in cloud.dp)
        ranks: set[int] = set()
        for dp in allocations:
            if len(set(dp.ranks)) != len(dp.ranks) or ranks.intersection(dp.ranks):
                raise ValueError(
                    "[LWD] Separate allocations require distinct ranks; "
                    "duplicate or overlapping ranks"
                )
            ranks.update(dp.ranks)
        if ranks != set(range(self.deployment.hccl_world_size)):
            raise ValueError(
                "[LWD] Unique ranks must cover [0, hccl_world_size) without gaps"
            )

    def _validate_control_ports(self) -> None:
        dps = [dp for cloud in self.clouds for dp in cloud.dp]
        if self.deployment.scene == "edge_share":
            # These are shared per-DP channel references, not cloud bind addresses.
            ports: dict[int, int | None] = {}
            for dp in dps:
                if ports.setdefault(dp.dp_idx, dp.ctrl_port) != dp.ctrl_port:
                    raise ValueError(
                        "[LWD] edge_share requires the same ctrl_port for "
                        "the same DP across all cloud instances"
                    )
            if len(set(ports.values())) != len(ports):
                raise ValueError(
                    "[LWD] edge_share requires different ctrl_port values per DP"
                )
            return
        if len({(dp.addr, dp.ctrl_port) for dp in dps}) != len(dps):
            raise ValueError(
                "[LWD] Cloud DPs on the same address must use "
                "different ctrl_port values"
            )

    def _validate_scene(self) -> None:
        """Validate logical instance shape, independently of physical host count."""
        scene = self.deployment.scene
        edge_count, cloud_count = len(self.edges), len(self.clouds)
        if scene == "single_instance" and (edge_count, cloud_count) != (1, 1):
            raise ValueError("[LWD] single_instance requires one edge and one cloud")
        if scene == "cloud_share" and (edge_count < 2 or cloud_count != 1):
            raise ValueError("[LWD] cloud_share requires multiple edges and one cloud")
        if scene == "lwd_cluster" and (edge_count < 1 or cloud_count < 2):
            raise ValueError(
                "[LWD] lwd_cluster requires at least one edge and two clouds"
            )
        # Coordination is optional, but a coordinated scene must enable it on
        # every instance so no edge/cloud pair disagrees about the protocol.
        if scene in ("cloud_share", "lwd_cluster"):
            instances = (*self.edges, *self.clouds)
            enabled = [
                instance.coordination is not None and instance.coordination.enabled
                for instance in instances
            ]
            if any(enabled) and not all(enabled):
                raise ValueError(
                    "[LWD] coordination must be consistently enabled or disabled "
                    "on every edge and cloud instance"
                )
        else:
            for instance in (*self.edges, *self.clouds):
                if instance.coordination is not None:
                    raise ValueError(
                        "[LWD] coordination requires cloud_share/lwd_cluster, "
                        f"got scene={scene}"
                    )
        if scene == "edge_share":
            if edge_count != cloud_count or cloud_count < 2:
                raise ValueError(
                    "[LWD] edge_share requires equal multiple edge/cloud IDs"
                )
            if (
                len(self.instance_links) != edge_count
                or len({link.edge for link in self.instance_links}) != edge_count
                or len({link.cloud for link in self.instance_links}) != cloud_count
            ):
                raise ValueError("[LWD] edge_share requires one-to-one instance links")
        elif set(self.instance_links) != {
            LwdLink(edge.id, cloud.id) for edge in self.edges for cloud in self.clouds
        }:
            raise ValueError(
                "[LWD] Scene requires full-mesh instance_links; "
                "missing pair or isolated instance"
            )

    def connections(self, role: str, instance_id: int) -> tuple[LwdDPConnection, ...]:
        """Describe every DP connection for an instance without opening sockets."""
        self.instance(role, instance_id)
        result = []
        for link in self.instance_links:
            local_id = link.edge if role == "edge" else link.cloud
            if local_id != instance_id:
                continue
            for edge_dp, cloud_dp in zip(
                self.edges[link.edge].dp, self.clouds[link.cloud].dp
            ):
                result.append(
                    LwdDPConnection(
                        link.edge,
                        link.cloud,
                        edge_dp.dp_idx,
                        edge_dp,
                        cloud_dp,
                    )
                )
        return tuple(sorted(result, key=lambda connection: connection.connection_id))

    def instance(self, role: str, instance_id: int) -> LwdInstance:
        if role not in ("edge", "cloud"):
            raise ValueError("[LWD] role must be 'edge' or 'cloud'")
        instances = self.edges if role == "edge" else self.clouds
        for instance in instances:
            if instance.id == instance_id:
                return instance
        raise ValueError(
            f"[LWD] Instance not found: role={role}, instance_id={instance_id}"
        )
