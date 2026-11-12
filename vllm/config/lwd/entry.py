# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Original path/role/instance entry, without parallel or model projection."""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pydantic import ConfigDict, TypeAdapter, with_config
from typing_extensions import TypedDict

from .api import LwdAPIConfig
from .coordination import LwdCoordinationConfig
from .topology import LwdDPConnection, LwdId, LwdInstance, LwdString, LwdTopology


@with_config(ConfigDict(extra="forbid"))
class _LwdEntry(TypedDict):
    path: LwdString
    role: Literal["edge", "cloud"]
    instance_id: LwdId


@dataclass(frozen=True)
class LwdConfig:
    path: str
    role: str
    instance_id: int
    topology: LwdTopology
    coordination: LwdCoordinationConfig | None = None
    api: LwdAPIConfig = field(default_factory=LwdAPIConfig)

    def __post_init__(self) -> None:
        self.api.validate_topology(self)

    @classmethod
    def from_dict(cls, raw: Any, *, api: LwdAPIConfig | None = None) -> "LwdConfig":
        if isinstance(raw, dict) and "coordination" in raw:
            raise ValueError(
                "[LWD] coordination is retired from lwd_config; "
                "it is now generated inside the topology yaml by "
                "gen_lwd_topology.py. Remove the coordination block "
                "from your lwd_config.yaml."
            )
        entry = TypeAdapter(_LwdEntry).validate_python(raw)
        path, role, identity = entry["path"], entry["role"], entry["instance_id"]
        topology = LwdTopology.from_file(path)
        topology.instance(role, identity)
        coordination = None
        # coordination is embedded in each instance, not at the top level.
        instance = topology.instance(role, identity)
        if instance.coordination is not None:
            coord_dict = asdict(instance.coordination)
            if coord_dict["instance_id"] is None:
                del coord_dict["instance_id"]
            coordination = LwdCoordinationConfig.from_dict(
                coord_dict,
                cloud_id=identity if role == "cloud" else None,
            )
            coordination.validate_role(role)
        return cls(
            path,
            role,
            identity,
            topology,
            coordination,
            api=api if api is not None else LwdAPIConfig(),
        )

    @property
    def instance(self) -> LwdInstance:
        return self.topology.instance(self.role, self.instance_id)

    @property
    def connections(self) -> tuple[LwdDPConnection, ...]:
        return self.topology.connections(self.role, self.instance_id)
