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
import runpy
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
import torch
from bqskit.compiler.passdata import PassData
from pytket.circuit import Node, Qubit
from pytket.extensions.qiskit import qiskit_to_tk
from pytket.passes import RenameQubitsPass
from pytket.predicates import CompilationUnit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Operator
from qiskit.transpiler import Layout, PassManager, TransformationPass, TranspileLayout
from qiskit.transpiler.passes import VF2PostLayout
from qiskit.transpiler.preset_passmanagers import common, generate_preset_pass_manager
from stable_baselines3.common import save_util
from stable_baselines3.common.vec_env import DummyVecEnv

from mqt.predictor.reward import estimated_success_probability
from mqt.predictor.rl.actions import bqskit_actions
from mqt.predictor.rl.actions.base import CompilationOrigin, DeferredDeviceAction, PassType
from mqt.predictor.rl.actions.qiskit_actions import run_qiskit_action
from mqt.predictor.rl.actions.registry import get_actions_by_pass_type
from mqt.predictor.rl.experiments.inputs import Inputs, load_target
from mqt.predictor.rl.experiments.metrics import efficiency, observe
from mqt.predictor.rl.experiments.qiskit_teacher import collect_demonstrations
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
    from mqt.predictor.rl.gnn import GNNMaskableMultiInputActorCriticPolicy

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


@pytest.fixture
def pretrained_checkpoint(inputs: Inputs, config: dict[str, Any], tmp_path: Path) -> Path:
    """Save an untrained checkpoint with the current action and observation shapes."""
    env = scasia.make_env("paper", config, inputs)
    wrapped = scasia.PreviousActionObservationWrapper(env)
    try:
        model = scasia.create_gnn_model(
            wrapped, scasia.GNNConfig(**config["paper"]["gnn"]), verbose=0, tensorboard_log=str(tmp_path), seed=0
        )
        scasia.add_previous_action_inputs(model, len(env.action_set))
        path = tmp_path / "pretrained.zip"
        model.save(path)
        metadata = json.loads((ROOT / "experiments/assets/gnn-context20.json").read_text())
        metadata.update(
            sha256=scasia.digest(path.read_bytes()),
            actions=[action.name for action in env.action_set.values()],
            gnn=config["paper"]["gnn"],
            source_commit="test fixture",
            training={"kind": "untrained checkpoint for import tests"},
        )
        path.with_suffix(".json").write_text(json.dumps(metadata))
        return path
    finally:
        wrapped.close()


@pytest.fixture
def comparison_results(tmp_path: Path) -> Path:
    """Two circuits with unequal matched coverage, a timeout and a missing row."""
    for compiler in ("qiskit", "tket", "original", "paper"):
        output = tmp_path / compiler
        output.mkdir()
        identity = {
            "compiler": compiler,
            "inputs": {"circuits": {"test/a.qasm": "a", "test/b.qasm": "b"}},
            "target": "boston",
            "lock_sha256": "lock",
            "source_sha256": "source",
            "dependencies": {},
            "python": "3.12",
            "actions": ["same actions"],
            "commit": "test-commit",
            "settings": {"experiment": {"evaluation_circuits": 0, "evaluation_repetitions": 2}},
        }
        (output / "manifest.json").write_text(json.dumps({"identity": identity, "actual_training_timesteps": 0}))
        records = []
        for index, esp in enumerate((0.2, 0.4, 0.8, 0.99)):
            if compiler == "original" and index == 3:
                continue
            records.append({
                "compiler": compiler,
                "circuit": "test/a.qasm" if index < 2 else "test/b.qasm",
                "repetition": index % 2,
                "seed": index,
                "status": "timeout" if compiler == "tket" and index == 3 else "ok",
                "final_esp": esp + 0.01 if compiler == "paper" else esp,
                "runtime_seconds": 100 if index == 3 else 2,
            })
        (output / "evaluation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    return tmp_path


def test_comparison_pairs_repetitions_and_counts_failures(comparison_results: Path) -> None:
    """Give circuits equal weight and exclude failed or missing pairs in every row."""
    compare = runpy.run_path(str(ROOT / "experiments/compare.py"))["compare"]
    report = compare(comparison_results)
    rows = {row["compiler"]: row for row in report["summary"]}
    assert report["matched"] == 3
    assert [row["matched_repetitions"] for row in report["paired"]] == [2, 1]
    assert rows["qiskit"]["paired_mean_esp"] == pytest.approx(0.55)
    assert rows["paper"]["paired_mean_esp"] == pytest.approx(0.56)
    assert rows["tket"]["timeout"] == 1
    assert rows["original"]["missing"] == 1
    assert rows["original"]["completed"] == 3
    assert report["runtimes"]["tket"] == [2, 2, 2, 100]


def test_comparison_rejects_incompatible_inputs_and_duplicates(comparison_results: Path) -> None:
    """Refuse a different calibration or duplicate observations instead of pooling them."""
    compare = runpy.run_path(str(ROOT / "experiments/compare.py"))["compare"]
    path = comparison_results / "paper/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["identity"]["target"] = "different calibration"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incompatible target"):
        compare(comparison_results)
    manifest["identity"]["target"] = "boston"
    path.write_text(json.dumps(manifest))
    path = comparison_results / "paper/evaluation.jsonl"
    text = path.read_text()
    path.write_text(text + text.splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="duplicate repetition"):
        compare(comparison_results)


def test_comparison_selects_named_run_variants(comparison_results: Path) -> None:
    """Compare two paper runs separately while flagging method differences."""
    previous = comparison_results / "paper"
    current = comparison_results / "warmstart/paper"
    current.mkdir(parents=True)
    manifest = json.loads((previous / "manifest.json").read_text())
    manifest["identity"]["settings"]["paper"] = {"warmstart": {"epochs": 5}}
    manifest["identity"]["actions"] = ["updated actions"]
    (current / "manifest.json").write_text(json.dumps(manifest))
    records = [
        dict(json.loads(line), final_esp=0.9) for line in (previous / "evaluation.jsonl").read_text().splitlines()
    ]
    (current / "evaluation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    compare = runpy.run_path(str(ROOT / "experiments/compare.py"))["compare"]
    report = compare(
        comparison_results,
        run_paths={"qiskit": comparison_results / "qiskit", "GNN earlier": previous, "GNN warm start": current},
    )
    rows = {row["compiler"]: row for row in report["summary"]}
    assert report["matched"] == 4
    assert rows["GNN earlier"]["paired_mean_esp"] == pytest.approx(0.6075)
    assert rows["GNN warm start"]["paired_mean_esp"] == pytest.approx(0.9)
    assert rows["GNN warm start"]["method"] == "paper"
    assert rows["GNN warm start"]["directory"] == str(current)
    assert any("method/training settings differ" in message for message in report["warnings"])
    assert any("action registries differ" in message for message in report["warnings"])
    (previous / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incompatible settings"):
        compare(comparison_results)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [("experiment", "evaluation_seed", 7), ("worker", "pass_timeout_seconds", 30)],
)
def test_comparison_variants_require_same_evaluation(
    comparison_results: Path, section: str, field: str, value: int
) -> None:
    """Selecting run variants does not relax evaluation seed or timeout checks."""
    path = comparison_results / "paper/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["identity"]["settings"].setdefault(section, {})[field] = value
    path.write_text(json.dumps(manifest))
    compare = runpy.run_path(str(ROOT / "experiments/compare.py"))["compare"]
    with pytest.raises(ValueError, match=f"incompatible {section}.{field}"):
        compare(
            comparison_results,
            run_paths={"qiskit": comparison_results / "qiskit", "GNN": comparison_results / "paper"},
        )


def test_comparison_without_valid_results(comparison_results: Path) -> None:
    """An all-failed evaluation has no quality estimate, rather than a zero ESP."""
    for path in comparison_results.glob("*/evaluation.jsonl"):
        rows = [dict(json.loads(line), status="error", final_esp=None) for line in path.read_text().splitlines()]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    compare = runpy.run_path(str(ROOT / "experiments/compare.py"))["compare"]
    report = compare(comparison_results)
    assert report["matched"] == 0
    assert report["paired"] == []
    assert all(row["paired_mean_esp"] is None and row["error"] == row["completed"] for row in report["summary"])


@pytest.mark.parametrize(
    ("compiler", "filename"),
    [("qiskit", "manifest.json"), ("paper", "manifest.json"), ("original", "evaluation.jsonl"), ("tket", "empty")],
)
def test_comparison_excludes_unavailable_runs(comparison_results: Path, compiler: str, filename: str) -> None:
    """Compare available rows even when another row has not started evaluation."""
    if filename == "empty":
        (comparison_results / compiler / "evaluation.jsonl").write_text("")
    else:
        (comparison_results / compiler / filename).unlink()
    compare = runpy.run_path(str(ROOT / "experiments/compare.py"))["compare"]
    report = compare(comparison_results)
    included = {"qiskit", "tket", "original", "paper"} - {compiler}
    assert {row["compiler"] for row in report["summary"]} == included
    assert set(report["runtimes"]) == included
    assert report["matched"] == 3
    assert all(
        row["paired_mean_esp"] == pytest.approx(0.56 if row["compiler"] == "paper" else 0.55)
        for row in report["summary"]
    )
    assert any(message.startswith(f"{compiler}: excluded") for message in report["warnings"])


def test_comparison_reads_only_complete_records(comparison_results: Path) -> None:
    """A live writer's unfinished final record is excluded without modifying it."""
    path = comparison_results / "paper/evaluation.jsonl"
    lines = path.read_text().splitlines(keepends=True)
    partial = "".join(lines[:-1]) + lines[-1][:20]
    path.write_text(partial)
    compare = runpy.run_path(str(ROOT / "experiments/compare.py"))["compare"]
    report = compare(comparison_results)
    paper = next(row for row in report["summary"] if row["compiler"] == "paper")
    assert paper["completed"] == 3
    assert paper["missing"] == 1
    assert report["matched"] == 3
    assert any("paper: ignored unfinished" in message for message in report["warnings"])
    assert path.read_text() == partial
    path.write_text("".join(lines) + "{invalid}\n")
    with pytest.raises(json.JSONDecodeError):
        compare(comparison_results)


def test_comparison_before_evaluation(comparison_results: Path) -> None:
    """There is no comparison yet when all rows are still waiting for evaluation."""
    for path in comparison_results.glob("*/evaluation.jsonl"):
        path.write_text('{"compiler":')
    compare = runpy.run_path(str(ROOT / "experiments/compare.py"))["compare"]
    report = compare(comparison_results)
    assert report["summary"] == []
    assert report["paired"] == []
    assert report["matched"] == 0
    assert sum("excluded" in message for message in report["warnings"]) == 4


@pytest.mark.parametrize("compilers", [("qiskit", "paper"), ("qiskit",)])
def test_comparison_groups_algorithm_circuit_means(compilers: tuple[str, ...]) -> None:
    """Keep underscores in algorithm names and weight circuits equally across run counts."""
    grouped_esp = runpy.run_path(str(ROOT / "experiments/compare.py"))["grouped_esp"]
    report = {
        "summary": [{"compiler": name} for name in compilers],
        "paired": [
            {"circuit": "test/vqe_real_amp_10_indep.qasm", "matched_repetitions": 10, "qiskit": 0.2, "paper": 0.4},
            {"circuit": "test/vqe_real_amp_20_indep.qasm", "matched_repetitions": 1, "qiskit": 0.8, "paper": 0.8},
            {"circuit": "test/qft_5_indep.qasm", "matched_repetitions": 10, "qiskit": 0.9, "paper": 0.1},
        ],
    }
    groups = grouped_esp(report)
    assert [row["algorithm"] for row in groups] == (
        ["qft", "vqe_real_amp"] if "paper" in compilers else ["vqe_real_amp", "qft"]
    )
    vqe = next(row for row in groups if row["algorithm"] == "vqe_real_amp")
    assert vqe["qiskit"] == pytest.approx(0.5)
    if "paper" in compilers:
        assert vqe["paper"] == pytest.approx(0.6)
    else:
        assert "paper" not in vqe
    assert grouped_esp({**report, "paired": []}) == []


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
    registry = {kind: [action.name for action in actions] for kind, actions in get_actions_by_pass_type().items()}
    old = scasia.make_env("original", config, inputs)
    paper = scasia.make_env("paper", config, inputs)
    assert config["original"]["ppo"]["gamma"] == pytest.approx(0.98)
    assert config["paper"]["gnn"]["gamma"] == pytest.approx(1.0)
    assert not old.intermediate_reward
    assert not paper.intermediate_reward
    assert registry == {
        kind: [action.name for action in actions] for kind, actions in get_actions_by_pass_type().items()
    }
    assert [action.name for action in old.action_set.values()] == [action.name for action in paper.action_set.values()]
    names = {action.name for action in paper.action_set.values()}
    assert names.isdisjoint({"QiskitO3", "MGDPass", "AIRouting", "AIRouting_opt"})
    assert {"Optimize1qGatesDecomposition_preserve", "Opt2qBlocks_preserve"}.issubset(names)
    assert "ConsolidateBlocks" not in names
    assert cast("Discrete", paper.action_space).n == len(paper.action_set)
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


def test_paper_graph_preserves_normalized_sizes(inputs: Inputs, config: dict[str, Any]) -> None:
    """Raw physical width and depth must not overwrite the normalized GNN input."""
    env = scasia.make_env("paper", config, inputs)
    wrapped = scasia.NormalizedGNNObservationWrapper(env)
    circuit = QuantumCircuit(156)
    for _ in range(100):
        circuit.sx(0)
    observation, _ = wrapped.reset(circuit, seed=0)
    assert wrapped.graph_observation["global_features"][0, :2].tolist() == pytest.approx([
        observation["num_qubits"].item(),
        observation["depth"].item(),
    ])
    assert wrapped.graph_observation["global_features"][0, 1] < 1
    assert wrapped.graph_observation.num_nodes == 100


def test_previous_action_observation_after_failed_layout(inputs: Inputs, config: dict[str, Any]) -> None:
    """A failed VF2 probe remains visible to the policy until the next reset."""
    env = scasia.make_env("paper", config, inputs)
    wrapped = scasia.PreviousActionObservationWrapper(env)
    circuit = QuantumCircuit(5)
    for target in range(1, 5):
        circuit.cx(0, target)
    action = next(index for index, candidate in env.action_set.items() if candidate.name == "VF2Layout")
    try:
        wrapped.reset(circuit, seed=0)
        assert wrapped.graph_observation["global_features"].shape == (1, 37 + len(env.action_set))
        assert not wrapped.graph_observation["global_features"][0, 37:].any()
        _, _, terminated, truncated, _ = wrapped.step(action)
        assert not terminated
        assert not truncated
        assert env.last_result["status"] == "ok"
        assert env.layout is None
        assert env.state == circuit
        expected = torch.zeros(len(env.action_set))
        expected[action] = 1
        torch.testing.assert_close(wrapped.graph_observation["global_features"][0, 37:], expected)
        wrapped.reset(circuit, seed=0)
        assert not wrapped.graph_observation["global_features"][0, 37:].any()
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["original", "paper"])
def test_block_synthesis_action_observation(mode: str, inputs: Inputs, config: dict[str, Any]) -> None:
    """Expose synthesized gates to the policy only after the whole block action completes."""
    env = scasia.make_env(mode, config, inputs)
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.rx(0.3, 1)
    circuit.cx(0, 1)
    env.reset(circuit, seed=0)
    action = next(index for index, candidate in env.action_set.items() if candidate.name == "Opt2qBlocks")
    try:
        observation, _, terminated, truncated, _ = env.step(action)
        assert env.last_result["status"] == "ok", env.last_result.get("error")
        assert not terminated
        assert not truncated
        assert "unitary" not in env.state.count_ops()
        assert env.layout is None
        assert Operator(env.state).equiv(Operator(circuit))
        assert all(np.isfinite(value).all() for value in observation.values())
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["original", "paper"])
@pytest.mark.parametrize("action_name", ["Optimize1qGatesDecomposition_preserve", "Opt2qBlocks_preserve"])
def test_native_optimizations_preserve_compilation(
    mode: str, action_name: str, inputs: Inputs, config: dict[str, Any]
) -> None:
    """Select canonical optimizations in a compiled state and execute the same action in the worker."""
    env = ready_env(mode, inputs, config)
    env.state.cz(0, 1)
    env.state.sx(0)
    env.state.sx(0)
    original = env.state.copy()
    action = next(index for index, candidate in env.action_set.items() if candidate.name == action_name)
    assert env.action_masks()[action]
    try:
        _, reward, terminated, truncated, _ = env.step(action)
        assert env.last_result["status"] == "ok"
        assert env.trace[0]["name"] == action_name
        assert reward == 0
        assert not terminated
        assert not truncated
        assert env.action_masks()[env.action_terminate_index]
        assert env.is_circuit_synthesized(env.state)
        assert env.state.size() < original.size()
        assert Operator(env.state).equiv(Operator(original))
        assert env.layout is not None
        assert env.layout.final_index_layout() == [0, 1]
        assert env.step(env.action_terminate_index)[1:4] == (env.calculate_reward(), True, False)
        assert env.trace[-1]["name"] == "terminate"
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["original", "paper"])
def test_native_no_effect_action_mask(mode: str, inputs: Inputs, config: dict[str, Any]) -> None:
    """Mask a native no-op only in paper mode, until a circuit change or reset."""
    env = ready_env(mode, inputs, config)
    env.state.sx(0)
    env.state.sx(0)
    actions = {action.name: index for index, action in env.action_set.items()}
    no_op = actions["RemoveIdentityEquivalent"]
    optimization = actions["Optimize1qGatesDecomposition_preserve"]
    original = env.state.copy()
    properties = env.qiskit_properties.copy()
    try:
        env.step(no_op)
        assert env.last_result["status"] == "ok"
        assert env.state == original
        assert env.qiskit_properties != properties
        assert env.action_masks()[no_op]
        env.step(no_op)
        assert env.last_result["status"] == "ok"
        assert env.state == original
        assert env.action_masks()[no_op] is (mode == "original")
        assert env.action_masks()[env.action_terminate_index]
        env.step(optimization)
        assert env.last_result["status"] == "ok"
        assert env.state != original
        assert env.action_masks()[no_op]
        env.step(no_op)
        assert env.action_masks()[no_op] is (mode == "original")
        if mode == "paper":
            compiled = env.state.copy()
            env.step(actions["RemoveRedundancies"])
            assert env.last_result["status"] == "ok"
            assert env.state == compiled
            assert env.action_masks()[no_op]
            env.step(no_op)
            assert not env.action_masks()[no_op]
        env.reset(original, seed=0)
        assert env.action_masks()[no_op]
    finally:
        env.close()


def test_layout_only_progress_is_not_no_effect(inputs: Inputs, config: dict[str, Any]) -> None:
    """Assigning physical wires is progress even when an empty circuit stays unchanged."""
    env = scasia.make_env("paper", config, inputs)
    circuit = QuantumCircuit(env.device.num_qubits)
    env.reset(circuit, seed=0)
    action = next(index for index, candidate in env.action_set.items() if candidate.name == "VF2Layout")
    try:
        env.step(action)
        assert env.last_result["status"] == "ok"
        assert env.state == circuit
        assert env.layout is not None
        assert action not in env.no_effect_actions
        assert env.action_masks()[env.action_terminate_index]
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["original", "paper"])
def test_tket_preserving_optimization_after_layout(mode: str, inputs: Inputs, config: dict[str, Any]) -> None:
    """Paper mode can remove redundancies without changing physical wires or output permutations."""
    env = scasia.make_env(mode, config, inputs)
    circuit = QuantumCircuit(3)
    circuit.x(0)
    circuit.x(0)
    circuit.cz(0, 1)
    circuit.rz(0.2, 1)
    circuit.rz(0.3, 1)
    circuit.x(2)
    env.reset(circuit, seed=0)
    env.layout = TranspileLayout(
        initial_layout=Layout({circuit.qubits[0]: 2, circuit.qubits[1]: 1, circuit.qubits[2]: 0}),
        input_qubit_mapping={qubit: index for index, qubit in enumerate(circuit.qubits)},
        final_layout=Layout({circuit.qubits[0]: 1, circuit.qubits[1]: 0, circuit.qubits[2]: 2}),
        _output_qubit_list=circuit.qubits,
        _input_qubit_count=3,
    )
    env.valid_actions = env.determine_valid_actions_for_state()
    action = next(index for index, candidate in env.action_set.items() if candidate.name == "RemoveRedundancies")
    assert env.action_masks()[action] == (mode == "paper")
    try:
        if mode == "original":
            return
        _, _, terminated, truncated, _ = env.step(action)
        assert env.last_result["status"] == "ok"
        assert not terminated
        assert not truncated
        assert env.state.count_ops() == {"cz": 1, "x": 1, "rz": 1}
        assert Operator(env.state).equiv(Operator(circuit))
        assert env.state.qubits == circuit.qubits
        assert env.is_circuit_synthesized(env.state)
        assert env.layout is not None
        assert env.layout.final_index_layout() == [2, 0, 1]
        assert env.action_masks()[env.action_terminate_index]
        compiled = env.state.copy()
        env.step(action)
        assert env.last_result["status"] == "ok"
        assert env.state == compiled
        assert env.action_masks()[action]
    finally:
        env.close()


@pytest.mark.parametrize(("name", "seed"), [("ghz_19", 0), ("qpeinexact_17", 32)])
def test_standard_vf2_preserves_outputs(name: str, seed: int, inputs: Inputs, config: dict[str, Any]) -> None:
    """Use the shared canonical VF2 action and retain physical/output mappings."""
    env = scasia.make_env("paper", config, inputs)
    original = inputs.circuit(f"train/{name}_indep.qasm")
    circuit = generate_preset_pass_manager(3, target=env.device, seed_transpiler=seed).run(original)
    env.reset(circuit, seed=seed)
    env.layout = circuit.layout
    env.num_qubits_uncompiled_circuit = original.num_qubits
    env.valid_actions = env.determine_valid_actions_for_state()
    action = next(index for index, action in env.action_set.items() if action.name == "VF2PostLayout")
    assert env.action_masks()[action]
    call_limit, max_trials = common.get_vf2_limits(3, None, None, exact_match=True)
    standard = DeferredDeviceAction(
        "VF2PostLayout",
        CompilationOrigin.QISKIT,
        PassType.FINAL_OPT,
        lambda target: [VF2PostLayout(target=target, seed=-1, call_limit=call_limit, max_trials=max_trials)],
    )
    expected, expected_layout = run_qiskit_action(
        standard,
        circuit,
        env.device,
        circuit.layout,
        input_qubit_count=original.num_qubits,
        seed=-1,
    )
    try:
        env.step(action)
        assert env.last_result["status"] == "ok"
        after = estimated_success_probability(env.state, env.device)
        assert circuit_to_dag(env.state) == circuit_to_dag(expected)
        assert expected_layout is not None
        assert env.layout is not None
        assert env.layout.final_index_layout() == expected_layout.final_index_layout()
        assert env.trace[-1]["after"]["esp"] == after
        assert env.action_masks()[env.action_terminate_index]
        assert circuit.layout is not None
        assert env.layout is not None
        permutation = dict(
            zip(
                circuit.layout.final_index_layout(filter_ancillas=False),
                env.layout.final_index_layout(filter_ancillas=False),
                strict=True,
            )
        )
        equivalent = env.state.copy_empty_like()
        equivalent.global_phase = circuit.global_phase
        for instruction in circuit:
            equivalent.append(
                instruction.operation,
                [env.state.qubits[permutation[circuit.find_bit(qubit).index]] for qubit in instruction.qubits],
                [env.state.clbits[circuit.find_bit(bit).index] for bit in instruction.clbits],
            )
        assert circuit_to_dag(equivalent) == circuit_to_dag(env.state)
    finally:
        env.close()


@pytest.mark.parametrize("name", ["ghz_19", "bv_13", "qpeinexact_17", "ae_7", "qaoa_19"])
def test_o3_teacher_replay(name: str, inputs: Inputs, config: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Replay the atomic teacher exactly without exposing unitary blocks to the policy."""
    monkeypatch.setattr(inputs, "names", lambda split: [f"{split}/{name}_indep.qasm"])
    env = scasia.make_env("paper", config, inputs)
    try:
        samples, report = collect_demonstrations(
            scasia.NormalizedGNNObservationWrapper(env), 0, config["paper"]["gnn"]["gamma"]
        )
        assert len(report["circuits"]) == 1
        assert report["pipeline"] == "O3 with atomic block synthesis before layout"
        assert report["transitions"] == len(samples) <= 32
        assert report["circuits"][0]["actions"][-1] == "terminate"
        assert all(value == pytest.approx(report["circuits"][0]["esp"]) for _, _, _, value in samples)
        assert all(mask[action] for _, mask, action, _ in samples)
        assert all(not trace["after"]["gate_counts"].get("unitary") for trace in env.trace)
        if name == "ae_7":
            assert "Opt2qBlocks" in report["circuits"][0]["actions"]
    finally:
        env.close()


@pytest.mark.parametrize(("name", "failed_layout"), [("ghz_2", False), ("ae_3", True)])
def test_o3_teacher_layout_feedback(
    name: str, failed_layout: bool, inputs: Inputs, config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Expose real failed layout attempts in the teacher's previous-action context."""
    monkeypatch.setattr(inputs, "names", lambda split: [f"{split}/{name}_indep.qasm"])
    env = scasia.make_env("paper", config, inputs)
    try:
        samples, report = collect_demonstrations(
            scasia.PreviousActionObservationWrapper(env), 0, 1.0, retain_failed_layouts=True
        )
        actions = report["circuits"][0]["actions"]
        assert report["retain_failed_layouts"]
        assert actions.count("VF2Layout") == 1
        index = actions.index("VF2Layout")
        trace = env.trace[index]
        assert trace["after"]["state"]["layout"] == (not failed_layout)
        if failed_layout:
            assert actions[index + 1] == "QiskitSabreMapping"
        graph, mask, action, _ = samples[index + 1]
        previous_action = graph["global_features"][0, -len(env.action_set) :]
        vf2_index = next(i for i, candidate in env.action_set.items() if candidate.name == "VF2Layout")
        assert previous_action.sum().item() == 1
        assert previous_action[vf2_index].item() == 1
        assert mask[action]
    finally:
        env.close()


def test_teacher_rejects_short_budget(inputs: Inputs, config: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Never silently truncate a teacher demonstration to fit the learning budget."""
    monkeypatch.setattr(inputs, "names", lambda split: [f"{split}/ghz_2_indep.qasm"])
    config["experiment"]["episode_actions"] = 1
    env = scasia.make_env("paper", config, inputs)
    try:
        with pytest.raises(ValueError, match="episode limit is 1"):
            collect_demonstrations(scasia.NormalizedGNNObservationWrapper(env), 0, 0.98)
    finally:
        env.close()


@pytest.mark.parametrize("mapping", ["TrivialLayout", "TrivialPlacementPass"])
@pytest.mark.parametrize("preparation", [("BasisTranslator",), ("CliffordSimp", "BasisTranslator"), ("BlockZXZPass",)])
def test_virtual_permutation_survives_mapping(
    mapping: str, preparation: tuple[str, ...], inputs: Inputs, config: dict[str, Any]
) -> None:
    """Canonical virtual swaps retain their outputs through Qiskit and BQSKit mapping."""
    env = scasia.make_env("paper", config, inputs)
    circuit = QuantumCircuit(QuantumRegister(1, "b"), QuantumRegister(2, "a"))
    circuit.x(0)
    circuit.swap(0, 2)
    circuit.cx(0, 1)
    env.reset(circuit, seed=0)
    try:
        for name in ("ElidePermutations", *preparation, mapping):
            action = next(index for index, item in env.action_set.items() if item.name == name)
            assert env.action_masks()[action]
            env.step(action)
            assert env.last_result["status"] == "ok", env.last_result.get("error")
        assert env.layout is not None
        assert env.layout.final_index_layout() == [2, 1, 0]
        compact = QuantumCircuit(3)
        compact.global_phase = env.state.global_phase
        for item in env.state:
            compact.append(item.operation, [env.state.find_bit(qubit).index for qubit in item.qubits])
        compact.swap(0, 2)
        assert Operator(compact).equiv(Operator(circuit))
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["original", "paper"])
@pytest.mark.parametrize("action_cost", [0.0, 0.0001])
def test_episode_boundary(
    mode: str, action_cost: float, inputs: Inputs, config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """At action 32, legacy bootstraps with zero reward; paper terminates with ESP."""
    config["paper"]["refinement"] = {"action_cost": action_cost}
    env = ready_env(mode, inputs, config)
    monkeypatch.setattr(env, "apply_action", lambda _: env.state)
    monkeypatch.setattr(env, "calculate_reward", lambda: 0.7)
    env.num_steps = 31
    _, reward, terminated, truncated, _ = env.step(env.actions_structure_preserving_indices[0])
    assert env.num_steps == 32
    assert terminated is (mode == "paper")
    assert truncated is (mode == "original")
    assert reward == pytest.approx(0.7 - action_cost if mode == "paper" else 0.0)


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
@pytest.mark.parametrize("action_cost", [0.0, 0.0001])
def test_terminal_failures(
    mode: str,
    failure: type[Exception],
    action_cost: float,
    inputs: Inputs,
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed and timed-out passes do not bootstrap in either method."""
    config["paper"]["refinement"] = {"action_cost": action_cost}
    env = ready_env(mode, inputs, config)

    def fail(_: int) -> None:
        msg = "stalled pass"
        raise failure(msg)

    monkeypatch.setattr(env, "apply_action", fail)
    _, reward, terminated, truncated, info = env.step(env.actions_opt_indices[0])
    assert reward == pytest.approx(0.0 if mode == "original" else -0.001 - action_cost)
    assert terminated
    assert not truncated
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


def test_physical_barrier_does_not_require_coupling(inputs: Inputs) -> None:
    """A barrier across nonadjacent wires stays exact; a real two-qubit gate does not."""
    _, target = load_target(inputs.path)
    assert not target.instruction_supported("cz", (0, 2))
    circuit = QuantumCircuit(3, 1)
    circuit.x(0)
    circuit.barrier(0, 2)
    circuit.measure(0, 0)
    score = observe(circuit, target, physical=True)
    assert score["state"]["routing"]
    assert score["esp_kind"] == "exact"
    assert score["esp"] == estimated_success_probability(circuit, target)
    circuit.cz(0, 2)
    score = observe(circuit, target, physical=True)
    assert not score["state"]["routing"]
    assert score["esp_kind"] != "exact"


def test_unitary_block_observation(inputs: Inputs, config: dict[str, Any]) -> None:
    """Keep unitary intermediates observable without aborting native compilation."""
    _, target = load_target(inputs.path)
    block = QuantumCircuit(2)
    block.h(0)
    block.cx(0, 1)
    circuit = QuantumCircuit(3)
    circuit.unitary(Operator(block), [0, 1])
    circuit.swap(1, 2)
    circuit.measure_all()
    score = observe(circuit, target, physical=False)
    assert score["state"] == {"synthesis": False, "layout": False, "routing": False}
    assert score["depth"] == circuit.depth()
    assert score["gate_counts"] == dict(circuit.count_ops())
    assert score["esp"] is None
    assert score["expected_fidelity"] is None
    assert score["esp_kind"] == "unavailable"
    assert score["score_error"] == "ESP proxy does not support unitary blocks."

    worker = CompilerWorker(inputs.path, scasia.worker_settings(config, "qiskit"))
    try:
        result = worker.run({"compiler": "qiskit", "circuit": circuit, "seed": 0})
        assert result["status"] == "ok", result
        assert result["traces"][0]["before"] == score
        assert all(entry["status"] == "ok" for entry in result["traces"])
        assert result["score"]["esp_kind"] == "exact"
        reference = generate_preset_pass_manager(optimization_level=3, target=target, seed_transpiler=0).run(circuit)
        assert result["score"]["esp"] == estimated_success_probability(reference, target)
    finally:
        worker.close()


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
@pytest.mark.parametrize(
    ("compiler", "warmstart_epochs", "pretrained", "refined"),
    [
        ("original", 0, False, False),
        ("paper", 0, False, False),
        ("paper", 2, False, False),
        ("paper", 0, True, False),
        ("paper", 0, True, True),
    ],
)
def test_training_save_load_resume(
    compiler: str,
    warmstart_epochs: int,
    pretrained: bool,
    refined: bool,
    inputs: Inputs,
    config: dict[str, Any],
    pretrained_checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interrupt after an updated checkpoint, resume, load and evaluate real policies."""
    config["experiment"].update(training_timesteps=4, episode_actions=2, checkpoint_steps=2, evaluation_repetitions=1)
    config["worker"]["pass_timeout_seconds"] = 3.0
    config["original"]["ppo"].update(n_steps=2, batch_size=2, n_epochs=1)
    config["paper"]["gnn"].update(n_steps=2, batch_size=2, n_epochs=1)
    config["paper"]["warmstart"].update(epochs=warmstart_epochs, batch_size=2)
    if warmstart_epochs or pretrained:
        config["experiment"]["episode_actions"] = 32
    if pretrained:
        config["paper"]["warmstart"]["checkpoint"] = str(pretrained_checkpoint)
        if refined:
            config["paper"]["refinement"] = {
                "quality_features": True,
                "action_cost": 0.0001,
                "teacher_kl_coefficient": 0.02,
            }

        def reject_teacher(*_args: object, **_kwargs: object) -> None:
            pytest.fail("Pretrained runs must not replay the teacher.")

        monkeypatch.setattr(scasia, "collect_demonstrations", reject_teacher)
    monkeypatch.setattr(inputs, "names", lambda split: [f"{split}/{'ghz' if split == 'train' else 'bv'}_2_indep.qasm"])
    env = scasia.make_env(compiler, config, inputs)
    manifest: dict[str, Any] = {
        "identity": scasia.run_identity(config, inputs, env, compiler),
        "restarts": [],
        "checkpoints": {},
    }
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
        if warmstart_epochs:
            assert (tmp_path / "warmstart.zip").is_file()
            assert len(manifest["warmstart"]["epochs"]) == warmstart_epochs
        if pretrained:
            assert manifest["checkpoints"]["warmstart"]["timesteps"] == 0
            assert manifest["checkpoints"]["warmstart"]["warmstart_epochs"] == 20
            assert manifest["pretrained"]["warmstart_epochs"] == 20
        scasia.train(compiler, config, env, tmp_path, manifest, resume=True)
        gc.collect()
        assert manifest["actual_training_timesteps"] == 4
        if warmstart_epochs:
            assert len(manifest["warmstart"]["epochs"]) == warmstart_epochs
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


def test_refinement_observations_rewards_and_teacher(
    inputs: Inputs, config: dict[str, Any], pretrained_checkpoint: Path, tmp_path: Path
) -> None:
    """Expose exact ESP, preserve the initial policy, and penalize teacher drift."""
    config["paper"]["warmstart"].update(checkpoint=str(pretrained_checkpoint), epochs=0)
    config["paper"]["refinement"] = {"quality_features": True, "action_cost": 0.0001, "teacher_kl_coefficient": 0.02}
    env = scasia.make_env("paper", config, inputs)
    wrapped = scasia.PreviousActionObservationWrapper(env, quality_features=True)
    model = scasia.create_gnn_model(
        wrapped, scasia.GNNConfig(**config["paper"]["gnn"]), verbose=0, tensorboard_log=str(tmp_path), seed=0
    )
    manifest = {"identity": scasia.run_identity(config, inputs, env, "paper")}
    try:
        scasia.import_pretrained(model, config, env, manifest)
        wrapped.reset(inputs.circuit("train/ghz_2_indep.qasm"), seed=0)
        graph = wrapped.graph_observation
        assert graph["global_features"].shape == (1, 83)
        assert graph["global_features"][0, -2:].tolist() == [0.0, 0.0]
        old_graph = cast("Any", graph).clone()
        old_graph.global_features = old_graph.global_features[:, :-2]
        masks = np.asarray(env.action_masks())
        policy = cast("GNNMaskableMultiInputActorCriticPolicy", model.policy)
        policy.set_training_mode(False)
        actions = torch.arange(len(env.action_set))
        with torch.no_grad():
            old_obs, _ = policy.obs_to_tensor(old_graph)
            old_log_probs = policy.get_distribution(old_obs, action_masks=masks).log_prob(actions)
        rng_state = torch.get_rng_state()
        scasia.add_quality_inputs(model)
        assert torch.equal(torch.get_rng_state(), rng_state)
        with torch.no_grad():
            obs, _ = policy.obs_to_tensor(graph)
            log_probs = policy.get_distribution(obs, action_masks=masks).log_prob(actions)
        torch.testing.assert_close(log_probs, old_log_probs, rtol=0, atol=0)

        penalty = scasia.TeacherPenalty(config["paper"]["warmstart"]["checkpoint"], 0.02)
        penalty.init_callback(model)
        assert all(not parameter.requires_grad for parameter in penalty.teacher.parameters())
        action = int(np.flatnonzero(masks)[0])
        with torch.no_grad():
            reference = penalty.teacher.get_distribution(old_obs, action_masks=masks).log_prob(torch.tensor([action]))
            cast("torch.nn.Linear", policy.action_net).bias[action] += 30
            current = policy.get_distribution(obs, action_masks=masks).log_prob(torch.tensor([action]))
        rewards = np.array([0.5], dtype=np.float32)
        penalty.update_locals({
            "graph_observations": [graph],
            "action_masks": masks.reshape(1, -1),
            "actions": np.array([[action]]),
            "log_probs": current,
            "rewards": rewards,
        })
        assert penalty.on_step()
        assert rewards[0] == pytest.approx(0.5 - 0.02 * (current - reference).item())
        assert graph["global_features"].shape == (1, 83)

        for name in ("VF2Layout", "BasisTranslator"):
            index = next(i for i, candidate in env.action_set.items() if candidate.name == name)
            _, reward, terminated, truncated, _ = wrapped.step(index)
            assert reward == pytest.approx(-0.0001)
            assert not terminated
            assert not truncated
        score = env.last_result["score"]["esp"]
        assert wrapped.graph_observation["global_features"][0, -2:].tolist() == pytest.approx([score, 1.0])
        _, reward, terminated, truncated, _ = wrapped.step(env.action_terminate_index)
        assert reward == score
        assert terminated
        assert not truncated
        wrapped.reset(inputs.circuit("train/ghz_2_indep.qasm"), seed=0)
        assert wrapped.graph_observation["global_features"][0, -2:].tolist() == [0.0, 0.0]
    finally:
        wrapped.close()


def test_pretrained_import_preserves_parameters_and_fresh_schedule(
    inputs: Inputs, pretrained_checkpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Import tensors without old Python metadata, then load a portable native checkpoint."""
    config = scasia.resolve_config(ROOT / "experiments/scasia-pretrained.toml")
    config["paper"]["warmstart"]["checkpoint"] = str(pretrained_checkpoint)
    config["output"] = str(tmp_path)
    config["experiment"]["training_timesteps"] = 0
    checkpoint = Path(config["paper"]["warmstart"]["checkpoint"])
    _, parameters, _ = save_util.load_from_zip_file(checkpoint, load_data=False, device="cpu")
    env = scasia.make_env("paper", config, inputs)
    manifest: dict[str, Any] = {
        "identity": scasia.run_identity(config, inputs, env, "paper"),
        "restarts": [],
        "checkpoints": {},
    }

    def reject_metadata(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Pretrained import must not deserialize Python-specific checkpoint metadata.")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(save_util, "json_to_data", reject_metadata)
            patch.setattr(scasia, "collect_demonstrations", reject_metadata)
            scasia.train("paper", config, env, tmp_path, manifest, resume=False)
        model = scasia.GNNMaskablePPO.load(tmp_path / "final.zip")
        torch.testing.assert_close(model.get_parameters(), parameters, rtol=0, atol=0)
        assert model.num_timesteps == 0
        assert model.__dict__["scasia_warmstart_epochs"] == 20
        assert model.tensorboard_log == str(tmp_path / "logs")
        assert model.seed == config["experiment"]["training_seed"]
        assert [model.lr_schedule(progress) for progress in (1, 0.5, 0)] == pytest.approx([0.001, 0.00055, 0.0001])
        assert type(model.policy.features_extractor).__module__ == "mqt.predictor.rl.experiments.observations"
        assert manifest["actual_training_timesteps"] == 0
        assert manifest["pretrained"]["sha256"] == scasia.digest(checkpoint.read_bytes())
        assert manifest["checkpoints"]["checkpoint"]["warmstart_epochs"] == 20
    finally:
        env.close()
        for path in tmp_path.glob("*.zip"):
            path.unlink()
        gc.collect()


@pytest.mark.parametrize("field", ["sha256", "inputs", "actions"])
def test_pretrained_import_rejects_incompatible_source(
    field: str, inputs: Inputs, config: dict[str, Any], pretrained_checkpoint: Path, tmp_path: Path
) -> None:
    """Reject a changed model, dataset or action order before starting training."""
    source = pretrained_checkpoint
    checkpoint = tmp_path / "source.zip"
    checkpoint.symlink_to(source)
    metadata = json.loads(source.with_suffix(".json").read_text())
    metadata[field] = "changed"
    checkpoint.with_suffix(".json").write_text(json.dumps(metadata))
    config["paper"]["warmstart"].update(checkpoint=str(checkpoint), epochs=0)
    config["experiment"]["training_timesteps"] = 0
    env = scasia.make_env("paper", config, inputs)
    manifest: dict[str, Any] = {
        "identity": scasia.run_identity(config, inputs, env, "paper"),
        "restarts": [],
        "checkpoints": {},
    }
    try:
        with pytest.raises(ValueError, match="Pretrained checkpoint differs"):
            scasia.train("paper", config, env, tmp_path, manifest, resume=False)
        assert not (tmp_path / "checkpoint.zip").exists()
    finally:
        env.close()


def test_pretrained_import_rejects_previous_action_registry(
    inputs: Inputs, config: dict[str, Any], tmp_path: Path
) -> None:
    """The bundled 45-action model cannot initialize the atomic-action registry."""
    config["paper"]["warmstart"].update(checkpoint=str(ROOT / "experiments/assets/gnn-context20.zip"), epochs=0)
    config["experiment"]["training_timesteps"] = 0
    env = scasia.make_env("paper", config, inputs)
    manifest = {"identity": scasia.run_identity(config, inputs, env, "paper")}
    try:
        with pytest.raises(ValueError, match="Pretrained checkpoint differs"):
            scasia.train("paper", config, env, tmp_path, manifest, resume=False)
        assert not (tmp_path / "checkpoint.zip").exists()
    finally:
        env.close()


@pytest.mark.model_training
@pytest.mark.parametrize("previous_action", [False, True])
def test_imitation_only_resume(
    previous_action: bool, inputs: Inputs, config: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recover an interrupted imitation phase without counting its updates as PPO steps."""
    config["experiment"].update(training_timesteps=0, evaluation_repetitions=1)
    config["paper"]["previous_action"] = previous_action
    config["paper"]["warmstart"].update(epochs=10, batch_size=4)
    monkeypatch.setattr(inputs, "names", lambda split: [f"{split}/{'ghz' if split == 'train' else 'bv'}_2_indep.qasm"])
    env = scasia.make_env("paper", config, inputs)
    manifest: dict[str, Any] = {
        "identity": scasia.run_identity(config, inputs, env, "paper"),
        "restarts": [],
        "checkpoints": {},
    }
    save = scasia.RollingCheckpoint.save

    def interrupt_after_epoch(callback: scasia.RollingCheckpoint, name: str) -> None:
        save(callback, name)
        if name == "checkpoint" and callback.model.__dict__.get("scasia_warmstart_epochs") == 1:
            msg = "imitation interruption"
            raise InterruptedError(msg)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(scasia.RollingCheckpoint, "save", interrupt_after_epoch)
            with pytest.raises(InterruptedError, match="imitation interruption"):
                scasia.train("paper", config, env, tmp_path, manifest, resume=False)
        scasia.train("paper", config, env, tmp_path, manifest, resume=True)
        assert manifest["actual_training_timesteps"] == 0
        assert [row["epoch"] for row in manifest["warmstart"]["epochs"]] == list(range(1, 11))
        assert manifest["warmstart"]["epochs"][-1]["loss"] < manifest["warmstart"]["epochs"][0]["loss"]
        assert manifest["checkpoints"]["warmstart"]["warmstart_epochs"] == 10
        assert (tmp_path / "final.zip").is_file()
        model = scasia.GNNMaskablePPO.load(tmp_path / "final.zip")
        assert getattr(model.policy.features_extractor, "action_count", None) == (
            len(env.action_set) if previous_action else None
        )
        scasia.evaluate("paper", config, env, tmp_path, manifest, resume=False)
        records = [json.loads(line) for line in (tmp_path / "evaluation.jsonl").read_text().splitlines()]
        assert len(records) == 1
        assert records[0]["status"] == "ok"
    finally:
        env.close()
        for checkpoint in tmp_path.glob("*.zip"):
            checkpoint.unlink()
