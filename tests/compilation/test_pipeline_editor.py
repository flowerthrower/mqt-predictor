# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Contracts for the native pipeline editor's worker and cluster entry point."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

from mqt.predictor.reward import estimated_success_probability
from mqt.predictor.rl.experiments.inputs import Inputs, load_target
from mqt.predictor.rl.experiments.pipeline_actions import PipelineActions
from mqt.predictor.rl.experiments.pipeline_editor import EditPolicy, main
from mqt.predictor.rl.experiments.worker import CompilerWorker

ROOT = Path(__file__).resolve().parents[2]


def test_negative_edit_budget_is_rejected() -> None:
    """All adapters reject a negative edit budget through the shared action rules."""
    with pytest.raises(ValueError, match="nonnegative"):
        PipelineActions(None, max_edits=-1)


def test_policy_history_contains_only_completed_results() -> None:
    """Expose signed pass effects without leaking future results or mutable traces."""
    observations = []

    def policy(state: dict[str, Any]) -> str:
        observations.append(state)
        return "KEEP"

    actions = PipelineActions(policy)
    state = {"stage_index": 4, "stage_name": "KAKDecomposition"}
    trace: list[dict[str, Any]] = []
    actions.decide(state.copy(), trace=trace, optional=True, insertions={}, can_end=True)
    assert observations[0]["history"] == []
    trace.append({
        **state,
        "stage_name": "SynthesiseTket",
        "inserted_stage_index": 6,
        "action": "INSERT:6",
        "occurrence": 0,
        "automatic": False,
        "runtime_seconds": 0.25,
        "before": {"qubits": 3, "depth": 8, "two_qubit_gates": 2, "gate_counts": {"CX": 2, "H": 1}},
        "after": {"qubits": 3, "depth": 6, "two_qubit_gates": 1, "gate_counts": {"CX": 1, "Rz": 3}},
    })
    actions.decide(state.copy(), trace=trace, optional=True, insertions={}, can_end=True)
    assert observations[0]["history"] == []
    expected = {
        "stage_index": 4,
        "stage_name": "SynthesiseTket",
        "inserted_stage_index": 6,
        "action": "INSERT:6",
        "executed": True,
        "automatic": False,
        "runtime_seconds": 0.25,
        "changes": {"qubits": 0, "depth": -2, "two_qubit_gates": -1, "gate_counts": {"CX": -1, "H": -1, "Rz": 3}},
    }
    assert observations[1]["history"] == [expected]
    observations[1]["history"][0]["changes"]["gate_counts"]["CX"] = 999
    actions.decide(state.copy(), trace=trace, optional=True, insertions={}, can_end=True)
    assert observations[2]["history"][0]["changes"]["gate_counts"]["CX"] == -1
    assert trace[0]["after"]["gate_counts"]["CX"] == 1
    json.dumps(observations, allow_nan=False)


@pytest.mark.parametrize(("max_edits", "edits"), [(-1, []), (0, ["4:SKIP"]), (2, ["4:INSERT:4"] * 3)])
def test_cli_rejects_invalid_budget_before_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_edits: int, edits: list[str]
) -> None:
    """Reject negative budgets and excess edits before starting a run."""
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pipeline_editor",
            "--compiler",
            "qiskit",
            "--output",
            str(output),
            "--circuit",
            "train/ghz_2_indep.qasm",
            "--max-edits",
            str(max_edits),
            *[item for edit in edits for item in ("--edit", edit)],
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not output.exists()


def test_edit_policy_rejects_mandatory_and_unknown_edits() -> None:
    """Keep mandatory stages and reject unused stage indices."""
    state = {"stage_index": 2, "stage_name": "Mapping", "allowed_actions": ["KEEP"]}
    with pytest.raises(ValueError, match=r"Invalid 'SKIP'.*Mapping"):
        EditPolicy({2: ["SKIP"]})(state)
    policy = EditPolicy({99: ["SKIP"]})
    assert policy(state) == "KEEP"
    with pytest.raises(ValueError, match=r"Unknown editable stage indices:.*99"):
        policy.validate()
    policy = EditPolicy({2: ["INSERT:2", "END"]})
    state = {**state, "allowed_actions": ["KEEP", "INSERT:2", "END"]}
    assert policy(state) == "INSERT:2"
    with pytest.raises(ValueError, match="Unconsumed edits"):
        policy.validate()
    assert policy(state) == "END"
    policy.validate()


@pytest.mark.parametrize("compiler", ["qiskit", "tket", "bqskit"])
def test_pipeline_worker_scores_frozen_circuit(compiler: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Compile the same frozen two-qubit input, preserving physical maps and traces."""
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        monkeypatch.setenv(variable, "1")
    inputs = Inputs(ROOT / "experiments/assets")
    _, target = load_target(inputs.path)
    circuit = inputs.circuit("train/ghz_2_indep.qasm")
    config = tomllib.loads((ROOT / "experiments/scasia.toml").read_text())
    settings = {
        **config["worker"],
        "bqskit_port": 39472,
        "bqskit_worker_port": 39474,
        "log_file": str(tmp_path / "worker.log"),
    }
    worker = CompilerWorker(inputs.path, settings)
    try:
        result = worker.run({"compiler": "pipeline_editor", "pipeline": compiler, "circuit": circuit, "seed": 0})
        assert result["status"] == "ok", result
        assert result["score"]["esp_kind"] == "exact"
        assert 0 < result["score"]["esp"] <= 1
        assert result["score"]["esp"] == estimated_success_probability(result["circuit"], target)
        assert result["final_layout"] == result["circuit"].layout.final_index_layout()
        assert len(set(result["final_layout"])) == circuit.num_qubits
        assert all(0 <= qubit < target.num_qubits for qubit in result["final_layout"])
        assert result["stage_trace"]
        assert all(entry["action"] == "KEEP" and entry["status"] == "ok" for entry in result["stage_trace"])
        assert all(
            "gate_counts" in entry["before"] and "gate_counts" in entry["after"] for entry in result["stage_trace"]
        )
        assert result["compilation_seconds"] > 0
        json.dumps(result["stage_trace"], allow_nan=False)
    finally:
        worker.close()


@pytest.mark.parametrize(
    ("compiler", "level_args", "native_stage", "edit_index"),
    [
        ("qiskit", [], "optimization", 4),
        ("tket", ["--tket-level", "3"], "GreedyPauliSimp", 6),
        ("bqskit", ["--bqskit-level", "3"], "Iterative Resynthesis Optimization", 7),
    ],
)
@pytest.mark.parametrize("edited", [False, True])
def test_pipeline_cli_writes_manifest_and_result(
    tmp_path: Path, compiler: str, level_args: list[str], native_stage: str, edit_index: int, edited: bool
) -> None:
    """Use the installed module command and inspect its completed run artifacts."""
    output = tmp_path / "run"
    edits = [f"INSERT:{edit_index}"] * 3 + ["END"] if edited else []
    edit_args = [item for action in edits for item in ("--edit", f"{edit_index}:{action}")]
    if edited:
        edit_args.extend(["--max-edits", "4"])
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mqt.predictor.rl.experiments.pipeline_editor",
            "--compiler",
            compiler,
            *level_args,
            *edit_args,
            "--config",
            str(ROOT / "experiments/scasia.toml"),
            "--output",
            str(output),
            "--circuit",
            "train/ghz_2_indep.qasm",
        ],
        cwd=ROOT,
        check=True,
        timeout=120,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    rows: list[dict[str, Any]] = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert manifest["compiler"] == compiler
    if level_args:
        assert manifest[f"{compiler}_level"] == 3
    assert manifest["circuits"] == ["train/ghz_2_indep.qasm"]
    assert manifest["seeds"] == [0]
    assert manifest["edits"] == ({str(edit_index): edits} if edited else {})
    assert manifest["max_edits"] == (4 if edited else 2)
    assert manifest["inputs"] == Inputs(ROOT / "experiments/assets").manifest
    assert manifest["dependencies"]["qiskit"]
    assert len(manifest["commit"]) == 40
    assert len(manifest["lock_sha256"]) == 64
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert rows[0]["score"]["esp_kind"] == "exact"
    assert rows[0]["circuit_name"] == "train/ghz_2_indep.qasm"
    assert rows[0]["seed"] == 0
    assert rows[0]["stage_trace"]
    assert native_stage in {stage["stage_name"] for stage in rows[0]["stage_trace"]}
    if edited:
        decisions = [entry for entry in rows[0]["stage_trace"] if entry["stage_index"] == edit_index]
        assert [entry["action"] for entry in decisions] == edits
        assert [entry["status"] for entry in decisions] == ["ok", "ok", "ok", "skipped"]
        assert [entry["edits_remaining"] for entry in decisions] == [4, 3, 2, 1]
        assert all(entry["inserted_stage_index"] == edit_index for entry in decisions[:-1])
