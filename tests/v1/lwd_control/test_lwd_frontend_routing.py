# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MR5 regressions: multi-frontend frontend attach (registration and
output grouping).

Covers the behavioral seams of the frontend-attach plan:

* A - scene validation decoupling (``LwdAPIConfig.validate_topology``):
  legacy ``edge_share`` keeps the one-to-one primary/non-primary pairing
  and only that scene accepted attach metadata before; the decoupled
  ``cloud_share``/``lwd_cluster`` scenes legalize attach metadata while
  ``client_index`` stays a frontend grouping key (the attach frontend's
  non-primary edge instance id), never a topology identity;
* B - the edge scheduler's per-client output grouping in
  ``update_from_output``/``_lwd_deliver_unembed``: the request's
  ``client_index`` is the grouping key (unset requests fall back to the
  legacy group 0), late rows for finished requests are idempotently
  dropped.

Everything is CPU-only: scene validation runs against real typed
topologies (no sockets), the scheduler seams are exercised via AST
extraction onto a stub base class, and the serve wiring is checked with
source-scan guards (no process is started).
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from vllm.config.lwd import LwdAPIConfig, LwdConfig, LwdTopology

REPO_ROOT = Path(__file__).resolve().parents[3]
VLLM_DIR = REPO_ROOT / "vllm"
SERVE_PATH = VLLM_DIR / "entrypoints" / "cli" / "serve.py"
ATTACH_PATH = VLLM_DIR / "entrypoints" / "openai" / "lwd" / "frontend_attach.py"
SCHED_PATH = (
    VLLM_DIR
    / "v1"
    / "lwd_control"
    / "control_edge_scheduler"
    / "lwd_edge_scheduler.py"
)

# 1E1C edge_share: two one-to-one pairs (edge i -> cloud i).
EDGE_SHARE = ("edge_share", 2, 2)
# 2E1C cloud_share: two edges sharing one cloud, full mesh.
CLOUD_SHARE = ("cloud_share", 2, 1)
# 2E2C lwd_cluster: full mesh.
LWD_CLUSTER = ("lwd_cluster", 2, 2)
# 1E1C single instance.
SINGLE_INSTANCE = ("single_instance", 1, 1)


def _topology(shape: tuple[str, int, int]) -> LwdTopology:
    """Valid typed topology for a scene.

    edge_share instances share one edge allocation (one physical engine
    group); every other scene gives each instance its own rank. Cloud DPs
    always carry a ctrl_port, distinct per cloud: DPs on the same address
    must not share a port (topology _validate_control_ports).
    """
    scene, edges, clouds = shape
    shared_edge = scene == "edge_share"
    coordination = (
        {"coordination": {"enabled": True}}
        if scene in ("cloud_share", "lwd_cluster")
        else {}
    )
    edge_entries = [
        {
            "id": edge_id,
            "dp": [
                {
                    "dp_idx": 0,
                    "addr": "127.0.0.1",
                    "ranks": [0] if shared_edge else [edge_id],
                }
            ],
            **coordination,
        }
        for edge_id in range(edges)
    ]
    cloud_base = 1 if shared_edge else edges
    cloud_entries = [
        {
            "id": cloud_id,
            "dp": [
                {
                    "dp_idx": 0,
                    "addr": "127.0.0.1",
                    "ranks": [cloud_base + cloud_id],
                    "ctrl_port": 5550 + cloud_id,
                }
            ],
            **coordination,
        }
        for cloud_id in range(clouds)
    ]
    if scene == "edge_share":
        links = [{"edge": i, "cloud": i} for i in range(edges)]
    else:
        links = [
            {"edge": i, "cloud": j} for i in range(edges) for j in range(clouds)
        ]
    return LwdTopology.from_dict(
        {
            "deployment": {
                "mode": 0,
                "scene": scene,
                "hccl_world_size": cloud_base + clouds,
                "edges_num": edges,
                "clouds_num": clouds,
            },
            "feature_ctrl": {"enable_early_recv": False},
            "edges": edge_entries,
            "clouds": cloud_entries,
            "instance_links": links,
        }
    )


def _lwd(
    shape: tuple[str, int, int], role: str, instance_id: int, **api: str
) -> LwdConfig:
    """LWD identity with attach API metadata; construction itself runs
    ``LwdAPIConfig.validate_topology`` (raises on any violated rule)."""
    return LwdConfig(
        path="",
        role=role,
        instance_id=instance_id,
        topology=_topology(shape),
        api=LwdAPIConfig(**api),
    )


class TestAttachSceneValidation:
    """A: scene rules of the attach metadata validation."""

    ATTACH = "10.0.0.1:9700"
    RPC = "10.0.0.1:9701"

    def test_edge_share_attach_requires_non_primary_edge(self):
        # Legacy one-to-one pairing: frontend i is edge instance i's
        # frontend; instance 0's frontend belongs to the main command.
        _lwd(EDGE_SHARE, "edge", 1, api_server_attach=self.ATTACH)
        with pytest.raises(ValueError, match="non-primary"):
            _lwd(EDGE_SHARE, "edge", 0, api_server_attach=self.ATTACH)

    def test_edge_share_rpc_port_requires_primary_edge(self):
        _lwd(EDGE_SHARE, "edge", 0, api_server_rpc_port=self.RPC)
        with pytest.raises(ValueError, match="instance_id=0"):
            _lwd(EDGE_SHARE, "edge", 1, api_server_rpc_port=self.RPC)

    def test_decoupled_scenes_legalize_attach(self):
        # cloud_share/lwd_cluster: attach metadata accepted for a
        # non-primary edge; the instance id doubles as the client_index
        # (frontend grouping key), not a topology pairing.
        _lwd(CLOUD_SHARE, "edge", 1, api_server_attach=self.ATTACH)
        _lwd(LWD_CLUSTER, "edge", 1, api_server_attach=self.ATTACH)

    def test_decoupled_scenes_keep_primary_rules(self):
        with pytest.raises(ValueError, match="non-primary"):
            _lwd(CLOUD_SHARE, "edge", 0, api_server_attach=self.ATTACH)
        _lwd(CLOUD_SHARE, "edge", 0, api_server_rpc_port=self.RPC)
        with pytest.raises(ValueError, match="instance_id=0"):
            _lwd(LWD_CLUSTER, "edge", 1, api_server_rpc_port=self.RPC)

    def test_single_instance_scene_rejects_attach(self):
        # single_instance has no non-primary edge for --api-server-attach
        # to reach the scene check; the primary --api-server-rpc-port does.
        with pytest.raises(ValueError, match="scene=edge_share"):
            _lwd(SINGLE_INSTANCE, "edge", 0, api_server_rpc_port=self.RPC)

    def test_cloud_role_rejects_attach(self):
        with pytest.raises(ValueError, match="edge instances only"):
            _lwd(EDGE_SHARE, "cloud", 0, api_server_attach=self.ATTACH)

    def test_rpc_port_and_attach_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            _lwd(
                EDGE_SHARE,
                "edge",
                0,
                api_server_rpc_port=self.RPC,
                api_server_attach=self.ATTACH,
            )

    def test_unrequested_api_config_accepts_any_identity(self):
        # No API options: validation is inert (single-instance identity
        # without attach flags must construct cleanly).
        _lwd(SINGLE_INSTANCE, "edge", 0)


def _load_attach_fn(fn_name: str):
    """Extract one top-level function from the attach module and compile
    it in isolation (the module import chain pulls in zmq/vllm; the pure
    count resolver has no such dependencies)."""
    tree = ast.parse(ATTACH_PATH.read_text(encoding="utf-8"))
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn_name
    )
    module = ast.Module(body=[node], type_ignores=[])
    scope: dict = {}
    exec(compile(module, str(ATTACH_PATH), "exec"), scope)
    return scope[fn_name]


class TestFrontendCountDecoupling:
    """A: frontend count K resolution (main command).

    legacy edge_share: K is the edge instance count (one frontend per
    instance, explicit count must equal it). Decoupled scenes: K is
    declared freely - no longer tied to the cloud or instance count."""

    def _resolve(self, scene: str, edges: int, count: int | None) -> int:
        return _load_attach_fn("_resolve_frontend_count")(scene, edges, count)

    def test_edge_share_count_defaults_to_edge_instances(self):
        assert self._resolve("edge_share", 3, None) == 3
        assert self._resolve("edge_share", 2, 2) == 2

    def test_edge_share_explicit_count_must_pair_one_to_one(self):
        with pytest.raises(ValueError, match="edge_share"):
            self._resolve("edge_share", 3, 2)

    def test_decoupled_count_is_free_of_instance_count(self):
        # A single-edge decoupled group may declare any K: the frontend
        # count is a grouping concern, not a topology pairing.
        assert self._resolve("lwd_cluster", 1, 4) == 4
        assert self._resolve("cloud_share", 2, 5) == 5
        assert self._resolve("cloud_share", 2, None) == 1

    def test_decoupled_count_must_be_positive(self):
        with pytest.raises(ValueError, match="positive"):
            self._resolve("cloud_share", 2, 0)


def test_serve_wires_frontend_attach_dispatch():
    """Source-scan guard: the serve entrypoint dispatches attach commands
    before the generic frontend-count defaulting and runs the count-to-K
    barrier inside the engine-launch scope."""
    serve = SERVE_PATH.read_text(encoding="utf-8")
    assert "lwd_dispatch_attach_frontend" in serve
    assert "prepare_attach_mode" in serve
    assert "wait_for_attach_registrations" in serve
    assert "attach_scope" in serve
    assert "lwd_control" not in serve


def test_attach_module_homes_the_registration_protocol():
    """Source-scan guard: the protocol frame types and both entrypoints
    live in the entrypoints module, platform-agnostic (no engine-side
    rendezvous concepts)."""
    attach = ATTACH_PATH.read_text(encoding="utf-8")
    for frame in (
        '_MSG_ATTACH = "LWD_FRONTEND_ATTACH"',
        '_MSG_ACCEPT = "LWD_FRONTEND_ACCEPT"',
        '_MSG_ENDPOINTS = "LWD_FRONTEND_ENDPOINTS"',
        '_MSG_BARRIER = "LWD_FRONTEND_BARRIER"',
        '_MSG_ERROR = "LWD_FRONTEND_ERROR"',
    ):
        assert frame in attach
    assert "def _resolve_frontend_count(" in attach
    assert "def run_attach_api_server(" in attach
    assert "def create_frontend_registration_server(" in attach
    for retired in ("instance_parallel", "role_registry"):
        assert retired not in attach


# ------------------------------------------------------------------ #
# B: the edge scheduler's per-client output grouping                  #
# ------------------------------------------------------------------ #

_LWD_NOT_FINISHED = -1


class _StubLogger:
    """vllm.lwd_logger stand-in: the module import chain pulls in torch
    (unavailable on light hosts); the methods under test only use
    warning/info."""

    def warning(self, *a, **k):
        pass

    def info(self, *a, **k):
        pass


class _StubDiag:
    """LwdControlDiag stand-in: token traces only."""

    @staticmethod
    def edge_tokens_delivered(*a, **k):
        pass

    @staticmethod
    def trace_edge_kill_row(*a, **k):
        pass


class _FinishReason:
    """FinishReason stand-in: sentinel codes plus the callable used by
    _lwd_finish_code to re-materialize a code."""

    STOP = "stop"
    LENGTH = "length"
    ABORT = "abort"
    ERROR = "error"

    def __init__(self, code):
        self.code = code


class _RequestStatus:
    FINISHED_STOPPED = "finished_stopped"
    FINISHED_LENGTH_CAPPED = "finished_length_capped"
    FINISHED_ABORTED = "finished_aborted"


class _EngineCoreOutput:
    def __init__(self, request_id, new_token_ids, finish_reason=None):
        self.request_id = request_id
        self.new_token_ids = new_token_ids
        self.finish_reason = finish_reason


class _EngineCoreOutputs:
    def __init__(self, outputs=None, finished_requests=None):
        self.outputs = outputs
        self.finished_requests = finished_requests


class _LwdBatchType:
    LWD_UNEMBED = "unembed"


_FINISH_STATUS = {
    _FinishReason.STOP: _RequestStatus.FINISHED_STOPPED,
    _FinishReason.LENGTH: _RequestStatus.FINISHED_LENGTH_CAPPED,
    _FinishReason.ABORT: _RequestStatus.FINISHED_ABORTED,
    _FinishReason.ERROR: _RequestStatus.FINISHED_ABORTED,
}


class _Account:
    """Cloud-DP account bookkeeping stand-in: only the fields the settle
    path touches."""

    def __init__(self, connection_id):
        self.connection_id = connection_id
        self.pending_batches = {}
        self.prefill_requests = set()
        self.dropped = []

    def drop_request(self, request_id):
        self.prefill_requests.discard(request_id)
        self.dropped.append(request_id)

    def note_request_settled(self, request_id, was_awaiting):
        pass

    def note_prefill_done(self, request_id):
        pass


_CONNECTION = ("cloud", 0, 0)


class _SchedBase:
    """Attribute/method stubs for everything the extracted scheduler
    methods reference but do not need real logic for."""

    def __init__(self):
        self.requests = {}
        self.vllm_config = None
        self.lwd_on_tokens = None
        self.lwd_on_finish = None
        self._lwd_route = {}
        self._lwd_accounts = {_CONNECTION: _Account(_CONNECTION)}
        self._lwd_announced_refills = set()

    def _lwd_dp_of_settling(self, request_id):
        return self._lwd_route.get(request_id)

    def _lwd_account_of(self, request_id):
        connection_id = self._lwd_dp_of_settling(request_id)
        return None if connection_id is None else self._lwd_accounts[connection_id]

    def _lwd_purge_mrope_positions(self):
        pass

    def finish_requests(self, request_ids, status=None):
        for rid in request_ids:
            request = self.requests.get(rid)
            if request is not None:
                request.status = status
                request.finished = True


_SCHED_METHODS = (
    "lwd_edge_deliver_tokens",
    "_lwd_settle_request",
    "update_from_output",
    "_lwd_deliver_unembed",
    "_lwd_row_disposition",
    "_lwd_finish_code",
)


def _make_sched():
    """Compile the grouping methods out of the scheduler source into a
    class block inheriting the stub base (super() needs the class body's
    __class__ cell), with the module-level dependencies stubbed."""
    tree = ast.parse(SCHED_PATH.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "LwdEdgeScheduler"
    )
    fns = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name in _SCHED_METHODS
    ]
    assert {fn.name for fn in fns} == set(_SCHED_METHODS), "scheduler method drifted"
    class_def = ast.ClassDef(
        name="LwdEdgeSchedulerUnderTest",
        bases=[ast.Name(id="_SchedBase", ctx=ast.Load())],
        keywords=[],
        body=fns,
        decorator_list=[],
    )
    module = ast.Module(body=[class_def], type_ignores=[])
    ast.fix_missing_locations(module)
    scope: dict = {
        "_SchedBase": _SchedBase,
        "lwd_logger": _StubLogger(),
        "LwdControlDiag": _StubDiag(),
        "FinishReason": _FinishReason,
        "RequestStatus": _RequestStatus,
        "_LWD_FINISH_STATUS": _FINISH_STATUS,
        "EngineCoreOutput": _EngineCoreOutput,
        "EngineCoreOutputs": _EngineCoreOutputs,
        "LwdBatchType": _LwdBatchType,
        "LWD_NOT_FINISHED": _LWD_NOT_FINISHED,
    }
    exec(compile(module, str(SCHED_PATH), "exec"), scope)
    return scope["LwdEdgeSchedulerUnderTest"]()


def _request(rid: str, client_index=None, finished=False):
    """Request stand-in; ``client_index`` left unset mimics legacy
    single-frontend callers (grouping falls back to 0)."""
    request = NS(
        request_id=rid,
        lwd_prefix_nonce=0,
        status="finished" if finished else "waiting",
        finished=finished,
    )
    if client_index is not None:
        request.client_index = client_index
    request.is_finished = lambda: request.finished
    request.append_output_token_ids = lambda token_ids: None
    return request


def _notify(req_ids, finish_reasons=None):
    return NS(
        req_ids=list(req_ids),
        finish_reasons=list(finish_reasons or [_LWD_NOT_FINISHED] * len(req_ids)),
        num_accepted_tokens=None,
        prefix_nonces=None,
        down_seqno=7,
    )


def _unembed_step(sched, notify, sampled: dict):
    """Drive update_from_output with one UNEMBED batch."""
    scheduler_output = NS(
        lwd_batch=NS(batch_type=_LwdBatchType.LWD_UNEMBED),
        lwd_c2e_notify=[notify],
    )
    model_runner_output = NS(
        req_ids=list(sampled),
        sampled_token_ids=[sampled[rid] for rid in sampled],
    )
    return sched.update_from_output(scheduler_output, model_runner_output)


class TestOutputGrouping:
    """B: outputs group per client_index into per-client
    EngineCoreOutputs; late outputs neither drop nor cross-group."""

    def test_multi_client_grouping(self):
        sched = _make_sched()
        for rid, client in (("a", 0), ("b", 1), ("c", 2)):
            sched.requests[rid] = _request(rid, client_index=client)
        results = _unembed_step(
            sched,
            _notify(["a", "b", "c"]),
            {"a": [1], "b": [2], "c": [3]},
        )
        assert set(results) == {0, 1, 2}
        assert [out.request_id for out in results[1].outputs] == ["b"]
        assert [out.request_id for out in results[2].outputs] == ["c"]
        assert results[0].finished_requests is None

    def test_finished_requests_group_per_client(self):
        sched = _make_sched()
        for rid, client in (("a", 1), ("b", 2)):
            sched.requests[rid] = _request(rid, client_index=client)
        results = _unembed_step(
            sched,
            _notify(["a", "b"], [_FinishReason.STOP, _FinishReason.LENGTH]),
            {"a": [1], "b": [2]},
        )
        assert set(results[1].finished_requests) == {"a"}
        assert set(results[2].finished_requests) == {"b"}

    def test_single_frontend_path_unchanged(self):
        # K=1 regression: requests without client_index group under 0
        # exactly like the pre-attach hardcoded {0: EngineCoreOutputs(...)}.
        sched = _make_sched()
        for rid in ("a", "b"):
            sched.requests[rid] = _request(rid)
        results = _unembed_step(
            sched,
            _notify(["a", "b"], [_FinishReason.STOP, _LWD_NOT_FINISHED]),
            {"a": [1], "b": [2]},
        )
        assert set(results) == {0}
        assert [out.request_id for out in results[0].outputs] == ["a", "b"]
        assert set(results[0].finished_requests) == {"a"}

    def test_late_row_for_finished_request_is_dropped(self):
        # Late cloud payload after the terminal settle: the request is
        # finished, the row is idempotently dropped -- nothing lands in any
        # group (in particular not the 0 group).
        sched = _make_sched()
        sched.requests["r1"] = _request("r1", client_index=2)
        first = _unembed_step(
            sched, _notify(["r1"], [_FinishReason.STOP]), {"r1": [1]}
        )
        assert set(first) == {2}
        late = _unembed_step(
            sched, _notify(["r1"], [_FinishReason.STOP]), {"r1": [9]}
        )
        assert late == {}

    def test_late_row_for_inflight_request_keeps_group(self):
        # The attach window: a request already streaming (first token
        # delivered, not finished) stays live, so a late row still
        # reaches the originating frontend's group.
        sched = _make_sched()
        sched.requests["r1"] = _request("r1", client_index=2)
        first = _unembed_step(sched, _notify(["r1"]), {"r1": [1]})
        assert set(first) == {2}
        late = _unembed_step(sched, _notify(["r1"]), {"r1": [2]})
        assert set(late) == {2}
        assert [out.new_token_ids for out in late[2].outputs] == [[2]]

    def test_terminal_output_reaches_own_client_group(self):
        # The grouping key is read from the live request at row entry,
        # before the finished branch settles it: the terminal token and
        # finish signal of an attached frontend (client_index > 0) must
        # not fall back to the 0 group.
        sched = _make_sched()
        sched.requests["r1"] = _request("r1", client_index=3)
        results = _unembed_step(
            sched, _notify(["r1"], [_FinishReason.STOP]), {"r1": [42]}
        )
        (out,) = results[3].outputs
        assert out.new_token_ids == [42]
        assert results[3].finished_requests == {"r1"}


if __name__ == "__main__":
    raise SystemExit("run with pytest")
