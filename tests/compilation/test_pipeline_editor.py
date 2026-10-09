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
from mqt.predictor.rl.experiments.pipeline_editor import EditPolicy
from mqt.predictor.rl.experiments.worker import CompilerWorker

ROOT = Path(__file__).resolve().parents[2]


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
    edit_args = ["--edit", f"{edit_index}:INSERT:{edit_index}", "--edit", f"{edit_index}:END"] if edited else []
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
    assert manifest["edits"] == ({str(edit_index): [f"INSERT:{edit_index}", "END"]} if edited else {})
    assert manifest["max_edits"] == 2
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
        assert [entry["action"] for entry in decisions] == [f"INSERT:{edit_index}", "END"]
        assert decisions[0]["status"] == "ok"
        assert decisions[1]["status"] == "skipped"
        assert [entry["edits_remaining"] for entry in decisions] == [2, 1]
        assert decisions[0]["inserted_stage_index"] == edit_index
