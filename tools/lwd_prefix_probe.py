# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cloud prefix-negotiation HTTP probe (a scripted replacement for curl).

Sends a real HTTP probe to a running cloud (coordination enabled), parses
the ``X-Edge-Cloud-Prefix-Hit-Tokens`` receipt out of the SSE control
message, and exits non-zero when the hit differs from ``--expect-hit``
(default 0), so it can sit in scripts/CI.

``--control-url`` accepts comma-separated multi-value form (the
multi-cloud dev/debug direct form): each cloud is probed in turn and each
receipt is printed (hits and self-reported cloud id) for side-by-side
comparison. A transport failure on any cloud or a hit mismatch on any
cloud exits non-zero; the single-value behavior is unchanged.

Usage examples:

  python tools/lwd_prefix_probe.py \\
    --control-url http://127.0.0.1:8100/v1/chat/completions \\
    --tenant-key-file /path/tenant.key \\
    --edge-id 0 --block-size 16 --prompt-tokens 32 \\
    --request-id probe-1 --expect-hit 0

  # Multi-value: probe two clouds' direct endpoints in turn.
  python tools/lwd_prefix_probe.py \\
    --control-url "$CLOUD0_URL,$CLOUD1_URL" \\
    --tenant-key-file /path/tenant.key \\
    --edge-id 0 --request-id probe-multi --expect-hit 0
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import aiohttp

from vllm.config.lwd.coordination import LwdCoordinationConfig
from vllm.v1.lwd_control.lwd_coordination.lwd_edge_client import (
    LwdEdgePrefixClient,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-url", required=True,
                        help="cloud control-plane URL "
                             "(http://<cloud>:8100/v1/chat/completions); "
                             "comma-separated multi-value probes each cloud "
                             "in turn")
    parser.add_argument("--tenant-key-file", required=True,
                        help="tenant key file (>=16 bytes, edge-held only)")
    parser.add_argument("--edge-id", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=16,
                        help="manifest block granularity (must be a "
                             "multiple of the cloud hash block granularity)")
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--request-id", default="probe")
    parser.add_argument("--expect-hit", type=int, default=0,
                        help="expected hit_tokens; a mismatch exits "
                             "non-zero")
    return parser.parse_args()


async def _probe(args: argparse.Namespace) -> int:
    with open(args.tenant_key_file, "rb") as file:
        tenant_key = file.read()
    urls = [url.strip() for url in args.control_url.split(",") if url.strip()]
    if not urls:
        print("[lwd-probe] empty --control-url", file=sys.stderr)
        return 1
    # One request id reused across clouds: the cloud processes are
    # independent, so the reservation keys do not collide; with the same
    # tenant key and prompt the HMAC chain is byte-identical, which is
    # exactly what makes the per-cloud hits comparable.
    prompt_token_ids = list(range(args.prompt_tokens))
    coordination = LwdCoordinationConfig(enabled=True)
    failures = 0
    async with aiohttp.ClientSession() as session:
        for url in urls:
            client = LwdEdgePrefixClient(
                coordination,
                args.block_size,
                tenant_key,
                edge_id=args.edge_id,
                control_url=url,
            )
            try:
                result = await client.negotiate(
                    session, args.request_id, prompt_token_ids
                )
            except Exception as exc:  # noqa: BLE001  per-cloud isolation:
                # one cloud failing does not stop the others
                print(
                    f"[lwd-probe] url={url} request_id={args.request_id} "
                    f"ERROR {exc}",
                    file=sys.stderr,
                )
                failures += 1
                continue
            hit_ok = result.hit_tokens == args.expect_hit
            print(
                f"[lwd-probe] url={url} request_id={result.request_id} "
                f"instance={result.instance_id} cloud_id={result.cloud_id} "
                f"block_size={result.block_size} "
                f"hit_blocks={result.hit_blocks} "
                f"hit_tokens={result.hit_tokens}"
                f"{'' if hit_ok else ' (MISMATCH)'}"
            )
            if not hit_ok:
                failures += 1
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_probe(_parse_args())))
