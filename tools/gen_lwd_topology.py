# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate one LWD topology without importing the inference runtime."""

import argparse
import runpy
from ipaddress import ip_address
from pathlib import Path
from typing import Any

import yaml

SCENES = ("single_instance", "edge_share", "cloud_share", "lwd_cluster")


def _instances(value: str) -> list[tuple[int, str]]:
    items = [item.strip() for item in value.split(",")]
    explicit = "@" in value
    instances = []
    for index, item in enumerate(items):
        if explicit:
            if item.count("@") != 1:
                raise ValueError("Do not mix plain addresses and ID@address entries")
            identity, address = item.split("@")
            if not identity.isdecimal():
                raise ValueError("Instance IDs must be non-negative integers")
            index = int(identity)
        else:
            address = item
        instances.append((index, str(ip_address(address))))
    if not explicit and len({addr for _, addr in instances}) != len(instances):
        raise ValueError("Repeated addresses require explicit ID@address entries")
    return sorted(instances)


def _edge_instances(scene: str, value: str, cloud_count: int) -> list[tuple[int, str]]:
    if scene == "edge_share":
        if "@" in value or "," in value:
            raise ValueError(
                "edge_share requires one plain edge IP; edge IDs follow cloud IDs"
            )
        address = str(ip_address(value.strip()))
        return [(identity, address) for identity in range(cloud_count)]
    if scene == "cloud_share" and "@" not in value:
        raise ValueError("cloud_share requires --edge-machines ID@IP entries")
    return _instances(value)


def _instance(
    identity: int, address: str, ranks: list[list[int]], port_base: int | None = None
) -> dict[str, Any]:
    dps = []
    for index, group in enumerate(ranks):
        item: dict[str, Any] = {"dp_idx": index, "addr": address, "ranks": group}
        if port_base is not None:
            item["ctrl_port"] = port_base + index
        dps.append(item)
    return {"id": identity, "dp": dps}


def _cloud_layout(
    instances: list[tuple[int, str, int]],
    dp: int,
    edge_rank_count: int,
    port_base: int,
    *,
    shared_ports: bool = False,
) -> list[dict[str, Any]]:
    next_rank = edge_rank_count
    next_ports: dict[str, int] = {}
    clouds = []
    for identity, address, cards in instances:
        per_dp = cards // dp
        ranks = [
            list(range(start, start + per_dp))
            for start in range(next_rank, next_rank + cards, per_dp)
        ]
        port = next_ports.get(address, port_base)
        clouds.append(_instance(identity, address, ranks, port))
        next_rank += cards
        # edge_share reuses edge-side channels; other scenes bind per cloud DP.
        if not shared_ports:
            next_ports[address] = port + dp
    return clouds


def _cloud_instances(machines: str, cards: str, dp: int) -> list[tuple[int, str, int]]:
    instances = _instances(machines)
    allocations: dict[int, int] = {}
    for item in cards.split(","):
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 2 or not all(
            part.isascii() and part.isdecimal() for part in parts
        ):
            raise ValueError("--cloud-cards requires ID:count entries")
        identity, count = map(int, parts)
        if identity in allocations:
            raise ValueError("--cloud-cards contains a duplicate instance ID")
        if count < 1 or count % dp:
            raise ValueError(
                "Each cloud card count must be positive and divisible by DP"
            )
        allocations[identity] = count
    if set(allocations) != {identity for identity, _ in instances}:
        raise ValueError("--cloud-cards must specify every cloud instance exactly once")
    return [
        (identity, address, allocations[identity]) for identity, address in instances
    ]


def build_topology(
    scene: str,
    edge_machines: str,
    cloud_machines: str,
    dp: int,
    port_base: int,
    *,
    cloud_cards: str,
    tenant_key_file: str = "/run/secrets/edge-cloud-tenant-key",
    consumer_ids: str = "enterprise-a",
    connect_timeout: float = 5.0,
    listen_host: str = "0.0.0.0",
    listen_port: int = 8100,
    control_url: str | None = None,
    enable_coordination: bool = False,
) -> dict[str, Any]:
    """Allocate global ranks; host-local physical devices remain launch settings."""
    if scene not in SCENES:
        raise ValueError(f"Unknown scene: {scene}")
    if type(dp) is not int or dp < 1:
        raise ValueError("DP must be one positive integer")
    if not 1 <= port_base <= 65535 - dp + 1:
        raise ValueError("All generated control ports must be within 1..65535")
    if scene == "edge_share" and "@" not in cloud_machines:
        raise ValueError("edge_share requires --cloud-machines ID@IP entries")
    if enable_coordination and scene not in ("cloud_share", "lwd_cluster"):
        raise ValueError("coordination requires cloud_share/lwd_cluster")
    cloud_instances = _cloud_instances(cloud_machines, cloud_cards, dp)
    instances = _edge_instances(scene, edge_machines, len(cloud_instances))
    shared = scene == "edge_share"
    edges = [
        _instance(identity, address, [[0 if shared else identity] for _ in range(dp)])
        for identity, address in instances
    ]
    # Distinct cloud-reuse instances retain distinct ranks on the same host.
    edge_rank_count = 1 if shared else len(edges)
    clouds = _cloud_layout(
        cloud_instances, dp, edge_rank_count, port_base, shared_ports=shared
    )
    links = (
        [{"edge": edge["id"], "cloud": edge["id"]} for edge in edges]
        if shared
        else [
            {"edge": edge["id"], "cloud": cloud["id"]}
            for edge in edges
            for cloud in clouds
        ]
    )
    result: dict[str, Any] = {
        "deployment": {
            "mode": 0,
            "scene": scene,
            "hccl_world_size": edge_rank_count + sum(c for _, _, c in cloud_instances),
            "edges_num": len(edges),
            "clouds_num": len(clouds),
        },
        "feature_ctrl": {
            "enable_early_recv": False,
            "scramble": {"enable": False, "device_id": None, "custom_factor": None},
        },
        "edges": edges,
        "clouds": clouds,
        "instance_links": links,
    }
    if enable_coordination:
        consumer_id_list = [cid.strip() for cid in consumer_ids.split(",")]
        if len(consumer_id_list) != len(edges):
            raise ValueError(
                f"--consumer-ids count ({len(consumer_id_list)}) must match "
                f"edge instance count ({len(edges)})"
            )
        # control_url: use user-provided value, or auto-derive from cloud IP + port.
        if control_url is None:
            cloud_ip = cloud_instances[0][1]
            control_url = f"http://{cloud_ip}:{listen_port}/v1/chat/completions"
        # Embed coordination into each edge instance (edge-side fields).
        for i, edge in enumerate(edges):
            edge["coordination"] = {
                "enabled": True,
                "control_url": control_url,
                "tenant_key_file": tenant_key_file,
                "consumer_id": consumer_id_list[i],
                "connect_timeout": connect_timeout,
            }
        # Embed coordination into each cloud instance (cloud-side fields).
        for cloud in clouds:
            cloud["coordination"] = {
                "enabled": True,
                "listen_host": listen_host,
                "listen_port": listen_port,
            }
    return result


def _validate(raw: dict[str, Any]) -> None:
    # Use the runtime schema without importing torch, plugins or a device backend.
    path = Path(__file__).resolve().parents[1] / "vllm/config/lwd/topology.py"
    runpy.run_path(str(path))["LwdTopology"].from_dict(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, choices=SCENES)
    for name, help_text in (
        ("edge-machines", "Edge IPs; cloud_share requires ID@IP per instance"),
        ("cloud-machines", "Cloud IPs; edge_share requires ID@IP per instance"),
        ("cloud-cards", "Comma-separated ID:card_count for every cloud instance"),
    ):
        parser.add_argument(f"--{name}", required=True, help=help_text)
    parser.add_argument(
        "--dp", type=int, default=1, help="Must divide each cloud card count"
    )
    parser.add_argument("--port-base", type=int, default=5550)
    parser.add_argument("--enable-early-recv", action="store_true")
    parser.add_argument("--enable-scramble", action="store_true")
    parser.add_argument(
        "--enable-coordination",
        action="store_true",
        help="Embed enabled coordination blocks in cloud_share/lwd_cluster",
    )
    parser.add_argument(
        "--scramble-device-id",
        default=None,
        help="Payload scramble: shared device-pair fingerprint",
    )
    parser.add_argument(
        "--scramble-custom-factor",
        default=None,
        help="Payload scramble: business factor shared by both peers",
    )
    parser.add_argument(
        "--tenant-key-file",
        default="/run/secrets/edge-cloud-tenant-key",
        help="cloud_share/lwd_cluster: tenant key file path on edge side",
    )
    parser.add_argument(
        "--consumer-ids",
        default="enterprise-a",
        help=(
            "cloud_share/lwd_cluster: comma-separated consumer IDs, "
            "one per edge instance"
        ),
    )
    parser.add_argument(
        "--control-url",
        default=None,
        help=(
            "cloud_share/lwd_cluster: edge->cloud control service URL; "
            "auto-derived if omitted"
        ),
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=5.0,
        help=(
            "cloud_share/lwd_cluster: edge->cloud control connection timeout (seconds)"
        ),
    )
    parser.add_argument(
        "--listen-host",
        default="0.0.0.0",
        help="cloud_share/lwd_cluster: cloud-side coordination service listen host",
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=8100,
        help="cloud_share/lwd_cluster: cloud-side coordination service listen port",
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="One output YAML file"
    )
    args = parser.parse_args()
    try:
        raw = build_topology(
            args.scene,
            args.edge_machines,
            args.cloud_machines,
            args.dp,
            args.port_base,
            cloud_cards=args.cloud_cards,
            tenant_key_file=args.tenant_key_file,
            consumer_ids=args.consumer_ids,
            control_url=args.control_url,
            connect_timeout=args.connect_timeout,
            listen_host=args.listen_host,
            listen_port=args.listen_port,
            enable_coordination=args.enable_coordination,
        )
        raw["feature_ctrl"] = {
            "enable_early_recv": args.enable_early_recv,
            "scramble": {
                "enable": args.enable_scramble,
                "device_id": args.scramble_device_id,
                "custom_factor": args.scramble_custom_factor,
            },
        }
        _validate(raw)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
