# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Contracts needed for comparable, recoverable SCASIA runs."""

from __future__ import annotations

import asyncio
import ctypes
import gc
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from bqskit.compiler.passdata import PassData
from pytket.circuit import Node, Qubit
from pytket.extensions.qiskit import qiskit_to_tk
from pytket.passes import RenameQubitsPass
from pytket.predicates import CompilationUnit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator
from qiskit.transpiler import Layout, PassManager, TransformationPass, TranspileLayout
from stable_baselines3.common.vec_env import DummyVecEnv

from mqt.predictor.rl.actions import bqskit_actions
from mqt.predictor.rl.experiments.inputs import Inputs, load_target
from mqt.predictor.rl.experiments.metrics import efficiency, observe
from mqt.predictor.rl.experiments.worker import (
    CompilerWorker,
    PassObserver,
    observe_qiskit_passes,
    tket_physical_circuit,
)

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from gymnasium.spaces import Dict, Discrete
    from qiskit.dagcircuit import DAGCircuit

    from mqt.predictor.rl.experiments.environment import ExperimentEnv

pytest.importorskip("torch_geometric")
scasia = import_module("mqt.predictor.rl.experiments.scasia")

pytestmark = pytest.mark.filterwarnings(
    "ignore:Failing to pass a value to the 'type_params' parameter:DeprecationWarning:torch_geometric.inspector"
)
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def inputs() -> Inputs:
    """Use the bundled data, never generate replacement benchmark circuits."""
    return Inputs(ROOT / "experiments/assets")


@pytest.fixture
def config(tmp_path: Path) -> dict[str, Any]:
    """Use the human-readable defaults."""
    config = scasia.resolve_config(ROOT / "experiments/scasia.toml")
    config["output"] = str(tmp_path)
    return config


def test_frozen_inputs(inputs: Inputs, config: dict[str, Any]) -> None:
    """Check hashes, split, calibration date and identical physical targets."""
    assert len(inputs.names("train")) == 321
    assert len(inputs.names("test")) == 41
    assert inputs.circuit(inputs.names("train")[0]).num_qubits > 0
    backend, target = load_target(inputs.path)
    assert backend.properties().last_update_date == datetime.fromisoformat("2026-04-17T12:19:24+02:00")
    assert target.num_qubits == 156
    for mode in ("qiskit", "tket", "original", "paper"):
        env = scasia.make_env(mode, config, inputs)
        assert [(q.t1, q.t2, q.frequency) for q in env.device.qubit_properties] == [
            (q.t1, q.t2, q.frequency) for q in target.qubit_properties
        ]
        assert set(env.device.operation_names) == set(target.operation_names)
        for operation in target.operation_names:
            assert {q: (p.error, p.duration) if p else None for q, p in env.device[operation].items()} == {
                q: (p.error, p.duration) if p else None for q, p in target[operation].items()
            }


def test_distinct_methods_shared_actions(inputs: Inputs, config: dict[str, Any]) -> None:
    """Legacy observations stay discrete while paper observations include budget."""
    old = scasia.make_env("original", config, inputs)
    paper = scasia.make_env("paper", config, inputs)
    assert [action.name for action in old.action_set.values()] == [action.name for action in paper.action_set.values()]
    circuit = QuantumCircuit(2)
    circuit.h(0)
    old_obs, _ = old.reset(circuit, seed=0)
    paper_obs, _ = paper.reset(circuit, seed=0)
    assert len(old_obs) == 7
    assert old_obs["num_qubits"] == 2
    assert old_obs["depth"] == 1
    assert cast("Discrete", cast("Dict", old.observation_space)["num_qubits"]).n == 157
    assert "remaining_steps" not in old_obs
    assert paper_obs["remaining_steps"].item() == 1
    assert paper_obs["num_qubits"].item() == pytest.approx(2 / 156)
    assert old.valid_actions == old.actions_synthesis_indices + old.actions_opt_indices
    assert set(paper.actions_layout_indices).issubset(paper.valid_actions)
    assert len(old.action_masks()) == len(paper.action_masks())
    assert old._valid_actions_v2(False, True, False) == old.actions_synthesis_indices + old.actions_opt_indices  # ruff: ignore[private-member-access]


def ready_env(mode: str, inputs: Inputs, config: dict[str, Any]) -> ExperimentEnv:
    """Prepare a valid physical two-qubit circuit without compilation."""
    env = scasia.make_env(mode, config, inputs)
    circuit = QuantumCircuit(2)
    circuit.cz(0, 1)
    env.reset(circuit, seed=0)
    env.layout = TranspileLayout(
        initial_layout=Layout({q: i for i, q in enumerate(circuit.qubits)}),
        input_qubit_mapping={q: i for i, q in enumerate(circuit.qubits)},
        _output_qubit_list=circuit.qubits,
        _input_qubit_count=2,
    )
    env.valid_actions = env.determine_valid_actions_for_state()
    return env


@pytest.mark.parametrize("mode", ["original", "paper"])
def test_episode_boundary(mode: str, inputs: Inputs, config: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """At action 32, legacy bootstraps with zero reward; paper terminates with ESP."""
    env = ready_env(mode, inputs, config)
    monkeypatch.setattr(env, "apply_action", lambda _: env.state)
    monkeypatch.setattr(env, "calculate_reward", lambda: 0.7)
    env.num_steps = 31
    _, reward, terminated, truncated, _ = env.step(env.actions_structure_preserving_indices[0])
    assert env.num_steps == 32
    assert terminated is (mode == "paper")
    assert truncated is (mode == "original")
    assert reward == pytest.approx(0.7 if mode == "paper" else 0.0)


@pytest.mark.parametrize("mode", ["original", "paper"])
def test_explicit_termination(
    mode: str, inputs: Inputs, config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both methods receive the shared objective exactly once on explicit stop."""
    env = ready_env(mode, inputs, config)
    monkeypatch.setattr(env, "apply_action", lambda _: env.state)
    monkeypatch.setattr(env, "calculate_reward", lambda: 0.7)
    _, reward, terminated, truncated, _ = env.step(env.action_terminate_index)
    assert (reward, terminated, truncated) == (0.7, True, False)


@pytest.mark.parametrize("mode", ["original", "paper"])
@pytest.mark.parametrize("failure", [RuntimeError, TimeoutError])
def test_terminal_failures(
    mode: str, failure: type[Exception], inputs: Inputs, config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed and timed-out passes do not bootstrap in either method."""
    env = ready_env(mode, inputs, config)

    def fail(_: int) -> None:
        msg = "stalled pass"
        raise failure(msg)

    monkeypatch.setattr(env, "apply_action", fail)
    _, reward, terminated, truncated, info = env.step(env.actions_opt_indices[0])
    assert (reward, terminated, truncated) == (0.0 if mode == "original" else -0.001, True, False)
    assert info["termination_reason"] == "pass_error"


def test_legacy_bootstrap_flag(inputs: Inputs, config: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose Gymnasium's external-truncation contract to SB3."""
    config["experiment"]["episode_actions"] = 1
    env = scasia.make_env("original", config, inputs)
    monkeypatch.setattr(env, "apply_action", lambda _: env.state)
    vector = DummyVecEnv([lambda: env])
    vector.reset()
    _, rewards, dones, infos = vector.step(np.array([env.actions_opt_indices[0]]))
    assert dones[0]
    assert rewards[0] == 0
    assert infos[0]["TimeLimit.truncated"]
    assert "terminal_observation" in infos[0]


def test_invalid_paper_horizon(inputs: Inputs, config: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """An unfinished paper compilation at the horizon receives the terminal penalty."""
    env = scasia.make_env("paper", config, inputs)
    circuit = QuantumCircuit(2)
    circuit.h(0)
    env.reset(circuit)
    env.num_steps = 31
    monkeypatch.setattr(env, "apply_action", lambda _: circuit)
    assert env.step(env.actions_opt_indices[0])[1:4] == (-0.001, True, False)


class TimedPass(TransformationPass):
    """Use native SDK pass dispatch to exercise the watchdog."""

    def __init__(self, index: int, hang: bool, child_path: str) -> None:
        """Select a bounded pass or a native GIL-holding stall."""
        super().__init__()
        self.index = index
        self.hang = hang
        self.child_path = child_path

    def name(self) -> str:
        """Identify the exact offending invocation."""
        return f"pass-{self.index}"

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Start a child on the stalled path to verify process-group cleanup."""
        if self.hang:
            child = subprocess.Popen(["/bin/sleep", "10"])
            Path(self.child_path).write_text(str(child.pid), encoding="utf-8")
            ctypes.PyDLL(None).sleep(5)
        time.sleep(0.04)
        return dag


def _timed_worker(connection: Connection, assets: Path, _settings: dict[str, Any]) -> None:
    """Run real Qiskit pass-manager dispatch inside the disposable worker."""
    os.setsid()
    _, target = load_target(assets)
    connection.send(("ready", None))
    while True:
        request = connection.recv()
        circuit = QuantumCircuit(2)
        circuit.h(0)
        observer = PassObserver(connection, target)
        with observe_qiskit_passes(observer):
            PassManager([TimedPass(index, request["hang"], request.get("child_path", "")) for index in range(3)]).run(
                circuit
            )
        connection.send(("result", {"status": "ok", "pid": os.getpid()}))


def test_pass_timeout_and_recovery(
    inputs: Inputs, config: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Total pipeline time may exceed a pass budget; a native stall is killed."""
    monkeypatch.setattr("mqt.predictor.rl.experiments.worker._worker_main", _timed_worker)
    settings = scasia.worker_settings(config, "original")
    settings["pass_timeout_seconds"] = 0.1
    worker = CompilerWorker(inputs.path, settings)
    try:
        first = worker.run({"hang": False})
        assert first["status"] == "ok"
        assert first["runtime_seconds"] > 0.1
        child_path = tmp_path / "child.pid"
        stalled = worker.run({"hang": True, "child_path": str(child_path)})
        assert stalled["status"] == "timeout"
        assert stalled["offending_pass"] == "pass-0"
        assert stalled["traces"][0]["status"] == "timeout"
        child_state = subprocess.run(
            ["ps", "-o", "stat=", "-p", child_path.read_text()], capture_output=True, text=True, check=False
        ).stdout.strip()
        assert not child_state or child_state.startswith("Z")
        recovered = worker.run({"hang": False})
        assert recovered["status"] == "ok"
        assert recovered["pid"] != first["pid"]
    finally:
        worker.close()


def test_row_isolation(config: dict[str, Any]) -> None:
    """Independent RL jobs have disjoint BQSKit ports and output directories."""
    original = scasia.worker_settings(config, "original")
    paper = scasia.worker_settings(config, "paper")
    assert {original["bqskit_port"], original["bqskit_worker_port"]}.isdisjoint({
        paper["bqskit_port"],
        paper["bqskit_worker_port"],
    })
    assert len({Path(config["output"]) / mode for mode in ("qiskit", "tket", "original", "paper")}) == 4


def test_efficiency_formulas() -> None:
    """Use hand-computed deltas and count repeated invocations separately."""

    def score(value: float, kind: str = "exact", synthesized: bool = True) -> dict[str, Any]:
        return {"esp": value, "esp_kind": kind, "state": {"synthesis": synthesized}}

    traces = [
        {"name": "opt", "category": "optimization", "status": "ok", "before": score(0.4), "after": score(0.6)},
        {"name": "opt", "category": "optimization", "status": "ok", "before": score(0.6), "after": score(0.5)},
        {
            "name": "synth",
            "category": "synthesis",
            "status": "ok",
            "before": score(0.4, "approximate", False),
            "after": score(0.5),
        },
        {"name": "synth", "category": "synthesis", "status": "timeout"},
        {"name": "analysis", "category": "analysis", "status": "ok"},
        {"name": "outer", "category": "optimization", "status": "ok", "container": True},
    ]
    metrics = efficiency(traces)
    assert metrics["optimization_efficiency"] == pytest.approx(2 / 3)
    assert metrics["structural_success_fraction"] == pytest.approx(0.5)
    assert metrics["overall_efficiency"] == pytest.approx(7 / 12)
    assert metrics["ppe"] == pytest.approx(0.05)
    assert metrics["coverage"] == pytest.approx(0.5)
    assert metrics["proxy_exact_transitions"] == 1
    assert metrics["timed_out_passes"] == 1
    assert metrics["analysis_invocations"] == 1
    assert efficiency([])["ppe"] is None


@pytest.mark.parametrize("multiple_registers", [False, True])
@pytest.mark.parametrize("num_qubits", [2, 3])
def test_tket_physical_indices_and_permutation(inputs: Inputs, multiple_registers: bool, num_qubits: int) -> None:
    """Sparse Boston nodes retain their physical numbers and logical outputs."""
    original = (
        QuantumCircuit(*(QuantumRegister(1, name) for name in ("c", "b", "a")[:num_qubits]))
        if multiple_registers
        else QuantumCircuit(num_qubits)
    )
    original.x(0)
    for index in range(num_qubits - 1):
        original.swap(index, index + 1)
    circuit = qiskit_to_tk(original)
    circuit.replace_SWAPs()
    unit = CompilationUnit(circuit)
    RenameQubitsPass({
        Qubit(register.name, index): Node(66 + original.find_bit(qubit).index)
        for register in original.qregs
        for index, qubit in enumerate(register)
    }).apply(unit)
    converted = tket_physical_circuit(unit, original)
    assert converted.num_qubits == 66 + num_qubits
    assert converted.find_bit(converted.data[0].qubits[0]).index == 66
    assert converted.layout is not None
    assert converted.layout.final_index_layout() == [*range(67, 66 + num_qubits), 66]
    _, target = load_target(inputs.path)
    assert observe(converted, target, physical=True)["esp_kind"] == "exact"


@pytest.mark.parametrize("compiler", ["qiskit", "tket"])
def test_native_baselines(compiler: str, inputs: Inputs, config: dict[str, Any]) -> None:
    """Smoke-test native offline compilation, physical scoring and pass traces."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    worker = CompilerWorker(inputs.path, scasia.worker_settings(config, compiler))
    try:
        result = worker.run({"compiler": compiler, "circuit": circuit, "seed": 0})
        assert result["status"] == "ok", result
        assert result["score"]["esp_kind"] == "exact"
        assert 0 < result["score"]["esp"] <= 1
        assert 0 < result["score"]["expected_fidelity"] <= 1
        assert len(result["traces"]) > 1
        assert all(entry["status"] == "ok" for entry in result["traces"])
        json.dumps(efficiency(result["traces"]), allow_nan=False)
    finally:
        worker.close()


def test_concurrent_bqskit_workers(inputs: Inputs, config: dict[str, Any]) -> None:
    """Run actual BQSKit runtimes concurrently on the two configured port pairs."""
    environments = [scasia.make_env(mode, config, inputs) for mode in ("original", "paper")]

    def compile_one(env: ExperimentEnv) -> dict[str, Any]:
        circuit = QuantumCircuit(2)
        circuit.cz(0, 1)
        env.reset(circuit, seed=0)
        action = next(index for index, action in env.action_set.items() if action.name == "TrivialPlacementPass")
        env.apply_action(action)
        return env.last_result

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(compile_one, environments))
        assert all(result["status"] == "ok" for result in results)
        assert all(result["circuit"].num_qubits == 156 for result in results)
    finally:
        for env in environments:
            env.close()


def test_synthesis_accuracy_check() -> None:
    """Accept an equivalent block and reject a shorter, inequivalent block."""
    circuit = QuantumCircuit(3)
    circuit.ccx(0, 1, 2)
    block = bqskit_actions.qiskit_to_bqskit(circuit)
    data = PassData(block)
    check = bqskit_actions._CheckSynthesisPass()  # ruff: ignore[private-member-access]
    asyncio.run(check.run(block, data))
    wrong = bqskit_actions.qiskit_to_bqskit(QuantumCircuit(3))
    with pytest.raises(ValueError, match=r"BQSKit synthesis cost .* exceeds tolerance"):
        asyncio.run(check.run(wrong, data))


@pytest.mark.parametrize("mode", ["original", "paper"])
@pytest.mark.parametrize("action_name", ["QSearchSynthesisPass", "LEAPSynthesisPass"])
def test_inaccurate_synthesis_fails_and_recovers(
    mode: str, action_name: str, inputs: Inputs, config: dict[str, Any]
) -> None:
    """An exhausted search cannot replace Toffoli with an inequivalent circuit."""
    env = scasia.make_env(mode, config, inputs)
    circuit = QuantumCircuit(3)
    circuit.ccx(0, 1, 2)
    try:
        env.reset(circuit, seed=0)
        action = next(index for index, action in env.action_set.items() if action.name == action_name)
        _, reward, terminated, truncated, info = env.step(action)
        assert (reward, terminated, truncated) == (0.0 if mode == "original" else -0.001, True, False)
        assert info["termination_reason"] == "pass_error"
        assert env.last_result["status"] == "error"
        assert env.last_result["offending_pass"] == action_name
        assert env.state == circuit

        circuit = QuantumCircuit(2)
        circuit.h(0)
        env.reset(circuit, seed=0)
        compiled = env.apply_action(action)
        assert env.last_result["status"] == "ok"
        assert env.is_circuit_synthesized(compiled)
        assert Operator(compiled).equiv(Operator(circuit))
    finally:
        env.close()


@pytest.mark.model_training
@pytest.mark.parametrize("compiler", ["original", "paper"])
def test_training_save_load_resume(
    compiler: str, inputs: Inputs, config: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interrupt after an updated checkpoint, resume, load and evaluate real policies."""
    config["experiment"].update(training_timesteps=4, episode_actions=2, checkpoint_steps=2, evaluation_repetitions=1)
    config["worker"]["pass_timeout_seconds"] = 3.0
    config["original"]["ppo"].update(n_steps=2, batch_size=2, n_epochs=1)
    config["paper"]["gnn"].update(n_steps=2, batch_size=2, n_epochs=1)
    monkeypatch.setattr(inputs, "names", lambda split: [f"{split}/{'ghz' if split == 'train' else 'bv'}_2_indep.qasm"])
    env = scasia.make_env(compiler, config, inputs)
    manifest = {"identity": scasia.run_identity(config, inputs, env, compiler), "restarts": [], "checkpoints": {}}
    on_start = scasia.RollingCheckpoint._on_rollout_start  # ruff: ignore[private-member-access]

    def interrupt_after_checkpoint(callback: scasia.RollingCheckpoint) -> None:
        on_start(callback)
        if callback.model.num_timesteps == 2:
            msg = "smoke interruption"
            raise InterruptedError(msg)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(scasia.RollingCheckpoint, "_on_rollout_start", interrupt_after_checkpoint)
            with pytest.raises(InterruptedError, match="smoke interruption"):
                scasia.train(compiler, config, env, tmp_path, manifest, resume=False)
        gc.collect()
        assert (tmp_path / "checkpoint.zip").is_file()
        assert manifest["actual_training_timesteps"] == 2
        scasia.train(compiler, config, env, tmp_path, manifest, resume=True)
        gc.collect()
        assert manifest["actual_training_timesteps"] == 4
        assert manifest["restarts"][0]["timesteps"] == 2
        assert (tmp_path / "final.zip").is_file()
        scasia.evaluate(compiler, config, env, tmp_path, manifest, resume=False)
        with (tmp_path / "evaluation.jsonl").open("a") as stream:
            stream.write('{"partial":')
        scasia.evaluate(compiler, config, env, tmp_path, manifest, resume=True)
        records = [json.loads(line) for line in (tmp_path / "evaluation.jsonl").read_text().splitlines()]
        assert len(records) == 1
        assert records[0]["circuit"] == "test/bv_2_indep.qasm"
        assert records[0]["actions"]
        assert records[0]["traces"]
        altered = {**manifest, "identity": {**manifest["identity"], "wrong_dataset": True}}
        with pytest.raises(ValueError, match="identity differs"):
            scasia.train(compiler, config, env, tmp_path, altered, resume=True)
    finally:
        env.close()
        # These full legacy one-hot policies are large; retain no disposable model copies.
        for checkpoint in tmp_path.glob("*.zip"):
            checkpoint.unlink()
        gc.collect()
