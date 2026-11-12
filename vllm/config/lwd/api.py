# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LWD frontend-attach API metadata: CLI flags, endpoint validation and the
scene/instance rules the attach service (see
``vllm/entrypoints/openai/lwd/frontend_attach.py``) builds on."""

import argparse
from dataclasses import dataclass
from ipaddress import ip_address
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .entry import LwdConfig


def _endpoint(value: str | None, flag: str) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise ValueError(f"[LWD] {flag} requires IP:port")
    host, sep, port = value.rpartition(":")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    elif ":" in host:
        raise ValueError(f"[LWD] {flag}: IPv6 must use [IP]:port")
    if not sep or not port.isascii() or not port.isdecimal():
        raise ValueError(f"[LWD] {flag} requires IP:port")
    if not 1 <= int(port) <= 65535:
        raise ValueError(f"[LWD] {flag}: port must be in [1, 65535]")
    try:
        addr = ip_address(host)
    except ValueError as exc:
        raise ValueError(f"[LWD] {flag}: invalid IP {host!r}") from exc
    if addr.is_unspecified or addr.is_multicast:
        raise ValueError(f"[LWD] {flag}: use a concrete unicast IP")


@dataclass(frozen=True)
class LwdAPIConfig:
    """API endpoint metadata forwarded with the topology snapshot.

    ``client_index`` is the frontend grouping key of the attach service
    (outputs return to the frontend that stamped the request), never a
    topology identity: each attach frontend presents a distinct
    non-primary edge identity whose instance id doubles as its
    client_index, and frontend 0 belongs to the primary edge running the
    main command.
    """

    api_server_rpc_port: str | None = None
    """Primary endpoint as IP:port, despite the historical option name."""
    api_server_attach: str | None = None
    """Primary endpoint referenced by a non-primary edge frontend."""

    @property
    def requested(self) -> bool:
        return (
            self.api_server_rpc_port is not None or self.api_server_attach is not None
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("LWD API configuration")
        group.add_argument(
            "--api-server-rpc-port",
            metavar="IP:port",
            help="LWD multi-frontend attach (main command): bind the "
            "frontend registration server on this endpoint's port and "
            "release the engine handshake only once every attach frontend "
            "registered (count-to-K barrier).",
        )
        group.add_argument(
            "--api-server-attach",
            metavar="IP:port",
            help="LWD multi-frontend attach (attach command): endpoint of "
            "the main command's --api-server-rpc-port. This command runs "
            "exactly one ApiServer frontend with its own --host/--port "
            "listen and no engines.",
        )

    @staticmethod
    def validate_frontend(args: Any, *, headless: bool = False) -> None:
        requested = any(
            getattr(args, name, None) is not None
            for name in ("api_server_rpc_port", "api_server_attach")
        )
        if requested and (
            headless or getattr(args, "headless", False) or getattr(args, "grpc", False)
        ):
            raise ValueError("[LWD] API options require the HTTP frontend")

    @classmethod
    def from_namespace(cls, args: Any, *, headless: bool = False) -> "LwdAPIConfig":
        cls.validate_frontend(args, headless=headless)
        return cls(
            api_server_rpc_port=getattr(args, "api_server_rpc_port", None),
            api_server_attach=getattr(args, "api_server_attach", None),
        )

    def validate_topology(self, lwd: "LwdConfig | None") -> None:
        if not self.requested:
            return
        if self.api_server_rpc_port is not None and self.api_server_attach is not None:
            raise ValueError(
                "[LWD] --api-server-rpc-port and --api-server-attach "
                "are mutually exclusive"
            )
        _endpoint(self.api_server_rpc_port, "--api-server-rpc-port")
        _endpoint(self.api_server_attach, "--api-server-attach")
        if lwd is None:
            raise ValueError("[LWD] API options require lwd_config.path")
        if lwd.role != "edge":
            raise ValueError("[LWD] API options apply to edge instances only")
        scene = lwd.topology.deployment.scene
        if scene not in ("edge_share", "cloud_share", "lwd_cluster"):
            raise ValueError(
                "[LWD] API attachment metadata requires scene=edge_share, "
                "cloud_share or lwd_cluster"
            )
        if self.api_server_rpc_port is not None and lwd.instance_id != 0:
            raise ValueError("[LWD] --api-server-rpc-port requires edge instance_id=0")
        if self.api_server_attach is not None and lwd.instance_id == 0:
            raise ValueError("[LWD] --api-server-attach requires a non-primary edge")
        # Scene coupling of the frontend count K: legacy edge_share keeps
        # the one-to-one pairing (one frontend per edge instance, K equals
        # the edge count) with the instance identity as the client_index;
        # the decoupled cloud_share/lwd_cluster scenes declare K freely
        # (--api-server-count, no longer tied to the cloud or instance
        # count) and the client_index is a pure frontend grouping key,
        # never used for cloud selection. The count check itself lives in
        # the attach dispatch, which sees the CLI arg.
