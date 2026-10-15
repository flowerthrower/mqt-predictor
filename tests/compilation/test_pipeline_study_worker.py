# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Clean timing, resource limits, and recovery for the server study."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import psutil  # ty: ignore[unresolved-import] -- the compilation test environment includes the gnn extra.
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline

from mqt.predictor.rl.experiments import pipeline_study_worker
from mqt.predictor.rl.experiments.inputs import Inputs, load_target
from mqt.predictor.rl.experiments.metrics import observe
from mqt.predictor.rl.experiments.pipeline_learning import input_features, plan_features
from mqt.predictor.rl.experiments.pipeline_qiskit import run_qiskit_variant
from mqt.predictor.rl.experiments.pipeline_study_worker import StudyWorker

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "experiments/assets"
REQUEST: dict[str, Any] = {
    "circuit": "train/ghz_2_indep.qasm",
    "seed": 0,
    "plan": {"id": "qiskit3-native", "compiler": "qiskit", "level": 3, "native": True},
}


def stalled_worker(connection: Connection, assets: Path, settings: dict[str, Any]) -> None:
    """Keep reporting phase changes while a child process stalls."""
    del assets
    os.setsid()
    connection.send(("ready", None))
    request = connection.recv()
    if request.get("complete"):
        connection.send(("result", {"status": "ok"}))
        connection.recv()
    with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]) as child:
        Path(settings["pid_file"]).write_text(str(child.pid), encoding="utf-8")
        while True:
            connection.send(("phase", "stalled test pass"))
            time.sleep(0.01)


def test_qiskit_native_and_keep_preserve_exact_score_and_maps(tmp_path: Path) -> None:
    """Use the public worker entry point and compare to clean native Qiskit."""
    inputs = Inputs(ASSETS)
    _, target = load_target(ASSETS)
    expected = run_qiskit_variant(inputs.circuit(REQUEST["circuit"]), target, REQUEST["seed"])
    assert expected.layout is not None
    worker = StudyWorker(ASSETS, {"output": str(tmp_path)}, 0)
    try:
        native = worker.run(REQUEST)
        keep = worker.run({**REQUEST, "plan": {**REQUEST["plan"], "native": False, "id": "qiskit3-keep"}})
        for result in (native, keep):
            assert result["status"] == "ok", result
            assert result["score"] == observe(expected, target, physical=True)
            assert result["score"]["esp_kind"] == "exact"
            assert result["final_layout"] == expected.layout.final_index_layout()
            assert result["seconds"] > 0
            assert result["selection_seconds"] == 0
            assert result["features"]
            json.dumps(result, allow_nan=False)
        assert native["stage_trace"] == []
        assert keep["stage_trace"][0]["action"] == "KEEP"
        assert keep["worker_startup_seconds"] == 0
    finally:
        worker.close()


def test_aggregate_timeout_kills_children_and_worker_recovers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase messages cannot renew the deadline; the next real compilation works."""
    entry = pipeline_study_worker._worker_main  # ruff: ignore[private-member-access] -- replace only the spawned target.
    monkeypatch.setattr(pipeline_study_worker, "_worker_main", stalled_worker)
    pid_file = tmp_path / "child.pid"
    worker = StudyWorker(
        ASSETS,
        {"output": str(tmp_path), "pid_file": str(pid_file), "compilation_timeout_seconds": 1.0},
        1,
    )
    try:
        failed = worker.run(REQUEST)
        assert failed["status"] == "timeout", failed
        assert failed["offending_phase"] == "stalled test pass"
        assert failed["seconds"] is None
        assert failed["score"] is None
        assert worker.process is None
        assert worker.connection is None
        child_pid = int(pid_file.read_text())
        assert not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
        monkeypatch.setattr(pipeline_study_worker, "_worker_main", entry)
        worker.settings["compilation_timeout_seconds"] = 60
        result = worker.run(REQUEST)
        assert result["status"] == "ok", result
        assert result["score"]["esp_kind"] == "exact"
    finally:
        worker.close()


def test_saved_policy_selects_one_plan_and_accounts_for_inference(tmp_path: Path) -> None:
    """Load a fitted plan predictor in the child and score its single compilation."""
    inputs = Inputs(ASSETS)
    _, target = load_target(ASSETS)
    circuit = inputs.circuit(REQUEST["circuit"])
    features = input_features(circuit, target.num_qubits)
    plan = {
        **REQUEST["plan"],
        "id": "qiskit3-post-layout",
        "native": False,
        "optimization": "post_layout",
    }
    model = make_pipeline(DictVectorizer(), DummyRegressor(strategy="constant", constant=[0.01, 0.7]))
    model.fit([plan_features(features, plan)], [[0.01, 0.7]])
    path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "plans": [REQUEST["plan"], plan], "baseline_id": REQUEST["plan"]["id"]}, path)
    expected = run_qiskit_variant(circuit, target, 0, optimization="post_layout")
    worker = StudyWorker(ASSETS, {"output": str(tmp_path)}, 4)
    try:
        result = worker.run({"circuit": REQUEST["circuit"], "seed": 0, "policy_file": str(path), "weight": 10.0})
        assert result["status"] == "ok", result
        assert result["plan_id"] == plan["id"]
        assert result["features"] == features
        assert result["score"] == observe(expected, target, physical=True)
        assert 0 < result["selection_seconds"] < result["seconds"]
    finally:
        worker.close()


def test_memory_limit_is_unscored_and_disposes_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply the memory guard to an established process as well as startup."""
    monkeypatch.setattr(pipeline_study_worker, "_worker_main", stalled_worker)
    worker = StudyWorker(ASSETS, {"output": str(tmp_path), "pid_file": str(tmp_path / "child.pid")}, 2)
    try:
        assert worker.run({"complete": True})["status"] == "ok"
        worker.settings["memory_limit_bytes"] = 1
        result = worker.run(REQUEST)
        assert result["status"] == "memory_limit", result
        assert result["peak_rss_bytes"] > 1
        assert result["seconds"] is None
        assert result["score"] is None
        assert worker.process is None
        assert worker.connection is None
    finally:
        worker.close()


def test_worker_slots_have_separate_runtime_ports_and_logs(tmp_path: Path) -> None:
    """Concurrent compiler jobs must never share BQSKit endpoints or log paths."""
    workers = [StudyWorker(ASSETS, {"output": str(tmp_path), "bqskit_port_base": 49472}, slot) for slot in range(12)]
    ports = [worker.settings[key] for worker in workers for key in ("bqskit_port", "bqskit_worker_port")]
    assert len(set(ports)) == 24
    assert len({worker.settings["log_file"] for worker in workers}) == 12
    assert all(worker.settings["threads"] == 1 for worker in workers)


def test_concurrent_native_bqskit_workers_are_isolated_and_cleaned(tmp_path: Path) -> None:
    """Run two native runtimes together and leave none of their processes alive."""
    workers = [StudyWorker(ASSETS, {"output": str(tmp_path), "bqskit_port_base": 51472}, slot) for slot in range(2)]
    executor = ThreadPoolExecutor(max_workers=2)
    requests = [
        {
            "circuit": f"train/ghz_{width}_indep.qasm",
            "seed": 0,
            "plan": {"id": "bqskit1-native", "compiler": "bqskit", "level": 1, "native": True},
        }
        for width in (2, 3)
    ]
    futures = [executor.submit(worker.run, request) for worker, request in zip(workers, requests, strict=True)]
    owned = set()
    runtimes_seen = set()
    try:
        deadline = time.monotonic() + 120
        while not all(future.done() for future in futures):
            assert time.monotonic() < deadline
            for slot, worker in enumerate(workers):
                with contextlib.suppress(psutil.NoSuchProcess, ValueError):
                    if worker.process is not None and worker.process.pid is not None:
                        process = psutil.Process(worker.process.pid)
                        children = process.children(recursive=True)
                        owned.update([process, *children])
                        if children:
                            runtimes_seen.add(slot)
            time.sleep(0.05)
        for width, future in zip((2, 3), futures, strict=True):
            result = future.result()
            assert result["status"] == "ok", result
            assert result["score"]["esp_kind"] == "exact"
            assert 0 < result["score"]["esp"] <= 1
            assert len(set(result["final_layout"])) == width
        assert runtimes_seen == {0, 1}
        assert len({worker.settings[key] for worker in workers for key in ("bqskit_port", "bqskit_worker_port")}) == 4
        logs = {Path(worker.settings["log_file"]) for worker in workers}
        assert len(logs) == 2
        assert all(path.exists() for path in logs)
    finally:
        for worker in workers:
            worker.cancel()
        executor.shutdown(wait=True)
        for worker in workers:
            worker.close()
    assert all(not process.is_running() or process.status() == psutil.STATUS_ZOMBIE for process in owned)


def test_bqskit_timeout_recovers_when_group_signals_are_denied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Kill a real stalled runtime by owned PID when the host rejects killpg."""
    owned = []

    def deny_group(group: int, sig: int) -> None:
        del sig
        process = psutil.Process(group)
        owned.extend([process, *process.children(recursive=True)])
        msg = "Process-group signaling denied by host."
        raise PermissionError(msg)

    worker = StudyWorker(
        ASSETS,
        {"output": str(tmp_path), "compilation_timeout_seconds": 5, "bqskit_port_base": 52472},
        0,
    )
    try:
        with monkeypatch.context() as patch:
            patch.setattr(os, "killpg", deny_group)
            result = worker.run({
                "circuit": "train/cdkm_ripple_carry_adder_4_indep.qasm",
                "seed": 0,
                "plan": {"id": "bqskit3-native", "compiler": "bqskit", "level": 3, "native": True},
            })
        assert result["status"] == "timeout", result
        assert result["score"] is None
        assert len(owned) >= 3
        assert all(not process.is_running() or process.status() == psutil.STATUS_ZOMBIE for process in owned)
        assert worker.process is None
        assert worker.connection is None
        worker.settings["compilation_timeout_seconds"] = 60
        recovered = worker.run(REQUEST)
        assert recovered["status"] == "ok", recovered
        assert recovered["score"]["esp_kind"] == "exact"
    finally:
        worker.close()


def test_pool_cancellation_cleans_active_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The pool can cancel without closing process handles from another thread."""
    monkeypatch.setattr(pipeline_study_worker, "_worker_main", stalled_worker)
    pid_file = tmp_path / "child.pid"
    worker = StudyWorker(ASSETS, {"output": str(tmp_path), "pid_file": str(pid_file)}, 5)
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(worker.run, REQUEST)
    try:
        deadline = time.monotonic() + 30
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pid_file.exists()
        worker.cancel()
        result = future.result(timeout=5)
        assert result["status"] == "cancelled", result
        assert result["score"] is None
        assert worker.process is None
        child_pid = int(pid_file.read_text())
        assert not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
    finally:
        worker.cancel()
        executor.shutdown(wait=True)
        worker.close()


def test_interrupt_disposes_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Parent cancellation cleans the process before propagating the interrupt."""
    worker = StudyWorker(ASSETS, {"output": str(tmp_path)}, 3)
    try:
        assert worker.run(REQUEST)["status"] == "ok"
        assert worker.process is not None
        process_pid = worker.process.pid

        def interrupt(deadline: float) -> tuple[str, Any]:
            del deadline
            raise KeyboardInterrupt

        monkeypatch.setattr(worker, "_receive", interrupt)
        with pytest.raises(KeyboardInterrupt):
            worker.run(REQUEST)
        assert worker.process is None
        assert worker.connection is None
        assert not psutil.pid_exists(process_pid)
    finally:
        worker.close()
