# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Bounded candidates, immutable study inputs, and restart-safe measurements."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from mqt.predictor.rl.experiments import pipeline_study as study
from mqt.predictor.rl.experiments.inputs import Inputs

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "experiments/assets"


@pytest.fixture(scope="module")
def candidates() -> list[dict[str, Any]]:
    """Construct native workflows without executing their compilation passes."""
    return study.candidate_plans(Inputs(ASSETS), limit=32, max_edits=6, seed=0)


@pytest.fixture
def manifest(tmp_path: Path, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep orchestration tests small without replacing compiler configuration."""
    return {
        "config": {
            "output": str(tmp_path),
            "assets": str(ASSETS),
            "study": {
                "max_edits": 6,
                "compiler_seeds": [0, 1, 2],
                "model_seeds": [0, 1],
                "tradeoff_weights": [10],
                "timing_circuits": 1,
                "timing_repetitions": 2,
            },
            "worker": {"memory_gib": 0.001, "compilation_timeout_seconds": 5},
        },
        "plans": candidates,
        "training": ["train/ghz_2_indep.qasm"],
        "evaluation": ["test/qft_5_indep.qasm"],
        "timing": ["test/qft_5_indep.qasm"],
        "inputs": {"circuits": {}},
        "protocol": {"failures": "Exclude incomplete results from paired outcomes."},
    }


def test_candidate_controls_and_edit_budget(candidates: list[dict[str, Any]]) -> None:
    """Include native controls and bounded multi-edit alternatives with stable IDs."""
    assert len({plan["id"] for plan in candidates}) == len(candidates)
    assert {plan["id"] for plan in candidates if plan["native"]} == {
        "qiskit1-native",
        "qiskit2-native",
        "qiskit3-native",
        "tket2-native",
        "tket3-native",
        "bqskit1-native",
        "bqskit2-native",
        "bqskit3-native",
    }
    assert {plan["optimization"] for plan in candidates if plan["compiler"] == "qiskit"} == {
        "native",
        "post_layout",
        "one_sweep",
        "peephole_once",
        "skip_optimization",
    }
    counts = Counter((plan["compiler"], plan["level"]) for plan in candidates if not plan["native"])
    assert all(count <= 32 for count in counts.values())
    edit_counts = [sum(len(actions) for actions in plan["edits"].values()) for plan in candidates]
    assert max(edit_counts) <= 6
    assert all(
        any(
            sum(len(actions) for actions in plan["edits"].values()) > 2
            for plan in candidates
            if plan["compiler"] == compiler
        )
        for compiler in ("qiskit", "tket", "bqskit")
    )
    assert any(count == 1 for count in edit_counts)
    assert any(
        action.startswith("INSERT:") for plan in candidates for actions in plan["edits"].values() for action in actions
    )
    assert candidates == study.candidate_plans(Inputs(ASSETS), limit=32, max_edits=6, seed=0)
    complete = study.candidate_plans(Inputs(ASSETS), limit=512, max_edits=6, seed=0)
    assert {plan["id"] for plan in complete if plan["priority"] <= 2} <= {plan["id"] for plan in candidates}
    for level in (2, 3):
        exploratory = [plan for plan in candidates if plan["compiler"] == "tket" and plan["level"] == level]
        assert {plan["priority"] for plan in exploratory} >= {3, 4}
        assert any(plan["priority"] == 4 and len(plan["edits"]) > 2 for plan in exploratory)
    small = study.candidate_plans(Inputs(ASSETS), limit=2, max_edits=1, seed=0)
    assert all(count <= 2 for count in Counter((p["compiler"], p["level"]) for p in small if not p["native"]).values())
    assert all(sum(len(actions) for actions in plan["edits"].values()) <= 1 for plan in small)


def test_collection_covers_edits_early_without_losing_jobs(manifest: dict[str, Any]) -> None:
    """Each small circuit batch gets matched edits before later circuits and seeds."""
    manifest["training"] = study.balanced(Inputs(ASSETS).names("train"))[:11]
    manifest["config"]["study"]["compiler_seeds"] = [2, 0, 1]
    jobs = list(study.collection_jobs(manifest))
    expected = {
        (name, seed, plan["id"])
        for name in manifest["training"]
        for seed in (0, 1, 2)
        for plan in manifest["plans"]
        if plan["compiler"] != "tket" or seed == 0
    }
    assert len(jobs) == len(expected)
    assert {(job["circuit"], job["seed"], job["variant"]) for job in jobs} == expected
    assert jobs == list(study.collection_jobs(manifest))
    first_batch = set(manifest["training"][:8])
    next_batch = next(index for index, job in enumerate(jobs) if job["circuit"] not in first_batch)
    early = jobs[:next_batch]
    assert {job["seed"] for job in early} == {0}
    assert len(early) == 8 * len(manifest["plans"])
    assert any(job["plan"]["priority"] == 4 for job in early)
    assert all(
        any(
            job["compiler"] == compiler and sum(len(actions) for actions in job["plan"]["edits"].values()) > 2
            for job in early
        )
        for compiler in ("qiskit", "tket", "bqskit")
    )
    first_extra_seed = next(index for index, job in enumerate(jobs) if job["seed"] != 0)
    assert first_extra_seed == len(manifest["training"]) * len(manifest["plans"])
    references = {}
    for index, job in enumerate(jobs):
        key = (job["compiler"], job["circuit"], job["seed"])
        if job["variant"] == f"{job['compiler']}3-native":
            references[key] = index
        elif not job["plan"]["native"]:
            assert references[key] < index


def test_tket_uses_only_its_native_seed_and_timing_has_one_warmup(manifest: dict[str, Any]) -> None:
    """Do not represent deterministic TKET duplicates as independent compiler seeds."""
    for jobs in (list(study.collection_jobs(manifest)), list(study.evaluation_jobs(manifest, [], timing=False))):
        assert {job["seed"] for job in jobs if job["compiler"] == "tket"} == {0}
        assert {job["seed"] for job in jobs if job["compiler"] == "qiskit"} == {0, 1, 2}
    timed = list(study.evaluation_jobs(manifest, [], timing=True))
    assert {job["seed"] for job in timed} == {0}
    for variant in {job["variant"] for job in timed}:
        entries = [job for job in timed if job["variant"] == variant]
        assert {job["repeat"] for job in entries} == {-1, 0, 1}
        assert sum(job["warmup"] for job in entries) == 1


def test_prepare_preserves_historical_split_and_requires_same_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, candidates: list[dict[str, Any]]
) -> None:
    """Resume the exact frozen 321/41 inputs and reject a changed protocol."""
    config = tmp_path / "study.toml"
    config.write_text(
        (ROOT / "experiments/pipeline-weekend.toml").read_text().replace('assets = "assets"', f'assets = "{ASSETS}"')
    )
    output = tmp_path / "run"
    monkeypatch.setattr(study, "candidate_plans", lambda *_: candidates)
    created, inputs = study.prepare(config, output, resume=False, smoke=False)
    assert len(created["training"]) == 321
    assert len(created["evaluation"]) == 41
    assert len(created["timing"]) == len(set(created["timing"])) == 12
    assert set(created["timing"]) <= set(created["evaluation"])
    assert any("/qpe" in name or "/vqe" in name for name in created["timing"])
    assert set(created["training"]) == set(inputs.names("train"))
    assert set(created["evaluation"]) == set(inputs.names("test"))
    assert created["inputs"] == inputs.manifest
    assert "src/mqt/predictor/_version.py" not in created["sources"]
    assert "src/mqt/predictor/rl/experiments/pipeline_study.py" in created["sources"]
    assert {inputs.manifest["circuits"][name] for name in created["training"]}.isdisjoint(
        inputs.manifest["circuits"][name] for name in created["evaluation"]
    )
    resumed, _ = study.prepare(config, output, resume=True, smoke=False)
    assert resumed == created
    assert [row["resume"] for row in study.read_rows(output / "restarts.jsonl")] == [False, True]
    with pytest.raises(FileExistsError, match="resume"):
        study.prepare(config, output, resume=False, smoke=False)
    small, _ = study.prepare(config, tmp_path / "smoke", resume=False, smoke=True)
    assert len(small["training"]) == 8
    assert len(small["evaluation"]) == 2
    assert all(inputs.circuit(name).num_qubits <= 3 for name in small["training"] + small["evaluation"])
    config.write_text(config.read_text().replace("max_edits = 6", "max_edits = 5"))
    with pytest.raises(ValueError, match="changed"):
        study.prepare(config, output, resume=True, smoke=False)


def test_interrupted_tail_is_preserved_but_middle_corruption_is_rejected(tmp_path: Path) -> None:
    """Only a final incomplete JSON record can be repaired on restart."""
    path = tmp_path / "collect.jsonl"
    complete = b'{"status": "ok"}\n'
    partial = b'{"status":'
    path.write_bytes(complete + partial)
    assert study.read_rows(path, repair_tail=True) == [{"status": "ok"}]
    assert path.read_bytes() == complete
    backups = list(tmp_path.glob("collect.partial-*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == partial
    corrupted = complete + partial + b"\n" + complete
    path.write_bytes(corrupted)
    with pytest.raises(json.JSONDecodeError):
        study.read_rows(path, repair_tail=True)
    assert path.read_bytes() == corrupted


def test_resume_does_not_repeat_completed_compilations(
    manifest: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restart the public scheduler with completed and new requests exactly once."""
    calls = []
    closed = []

    class Worker:
        def __init__(self, assets: Path, settings: dict[str, Any], slot: int) -> None:
            assert assets == ASSETS
            self.settings = settings
            self.slot = slot

        def run(self, request: dict[str, Any]) -> dict[str, Any]:
            calls.append(request["variant"])
            return {"status": "ok", "seconds": 1, "score": {"esp": 0.8}}

        def close(self) -> None:
            closed.append(self.slot)

        def cancel(self) -> None:
            pass

    monkeypatch.setattr(study, "StudyWorker", Worker)
    jobs = [
        {"compiler": "qiskit", "circuit": "train/ghz_2_indep.qasm", "variant": variant, "seed": 0}
        for variant in ("a", "b", "c")
    ]
    study.run_jobs(manifest, "collect", jobs[:2], workers=1, hours=0.1)
    path = Path(manifest["config"]["output"]) / "collect.jsonl"
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    study.run_jobs(manifest, "collect", jobs, workers=1, hours=0.1)
    assert calls == ["a", "b", "c"]
    assert closed == [0, 0]
    rows = study.read_rows(path)
    assert len(rows) == len({row["job_id"] for row in rows}) == 3


def test_frozen_models_are_reused_and_reject_changed_data_or_model(
    manifest: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evaluation and resume use frozen models without fitting on new outcomes."""
    output = Path(manifest["config"]["output"])
    collection = output / "collect.jsonl"
    training_rows = [
        {
            "plan_id": f"{compiler}3-native",
            "status": "ok",
            "circuit": "train/ghz_2_indep.qasm",
            "seed": 0,
            "features": {"qubits": 2},
            "score": {"esp": 0.8},
            "seconds": 1,
        }
        for compiler in ("qiskit", "tket", "bqskit")
    ]
    training_text = "".join(json.dumps(row) + "\n" for row in training_rows)
    collection.write_text(training_text)
    fitted = []

    def train(
        rows: list[dict[str, Any]], plans: list[dict[str, Any]], compiler: str, **_: object
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        assert rows == training_rows
        assert plans == manifest["plans"]
        fitted.append(compiler)
        return [{"weight": 10, "model_seed": 0, "model": compiler}], {"compiler": compiler}

    monkeypatch.setattr(study, "train_models", train)
    policies = study.fit_models(manifest)
    (output / "evaluate.jsonl").write_text('{"test_outcome": 1}\n')
    assert study.fit_models(manifest) == policies
    assert fitted == ["qiskit", "tket", "bqskit"]
    collection.write_text(training_text + "\n")
    with pytest.raises(ValueError, match="data changed"):
        study.fit_models(manifest)
    collection.write_text(training_text)
    (output / policies[0]["file"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="model identity changed"):
        study.fit_models(manifest)


def test_report_excludes_timeouts_unmatched_results_and_warmups(manifest: dict[str, Any]) -> None:
    """Quality and latency use only successful paired measured compilations."""
    output = Path(manifest["config"]["output"])

    def row(circuit: str, variant: str, esp: float | None, seconds: float | None, repeat: int = 0) -> dict[str, Any]:
        return {
            "compiler": "qiskit",
            "circuit": circuit,
            "variant": variant,
            "seed": 0,
            "repeat": repeat,
            "warmup": repeat < 0,
            "status": "ok" if esp is not None else "timeout",
            "score": {"esp": esp} if esp is not None else None,
            "seconds": seconds,
        }

    rows = [
        row("a", "qiskit3-native", 0.8, 2),
        row("a", "edit", 0.9, 3),
        row("b", "qiskit3-native", None, None),
        row("b", "edit", 0.7, 1),
        row("c", "qiskit3-native", 0.8, 2),
        row("c", "edit", None, None),
        row("a", "qiskit3-native", 0.2, 100, -1),
        row("a", "edit", 1.0, 100, -1),
    ]
    (output / "timing.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in rows))
    report = study.write_report(manifest)
    timing = report["phases"]["timing"]
    assert timing["records"] == 6
    assert timing["latency_claims_supported"]
    assert not report["phases"]["collect"]["latency_claims_supported"]
    edited = timing["variants"]["edit"]
    assert edited["statuses"] == {"ok": 2, "timeout": 1}
    assert edited["matched_compilations"] == edited["matched_circuits"] == 1
    assert edited["mean_delta_esp_pp"] == pytest.approx(10)
    assert edited["runtime_ratio"] == pytest.approx(1.5)
    assert json.loads((output / "report.json").read_text()) == report


def test_timing_uses_frozen_subset_instead_of_evaluation_prefix(manifest: dict[str, Any]) -> None:
    """Timing follows the sampled manifest names, independent of list position."""
    manifest["evaluation"].append("test/ae_8_indep.qasm")
    manifest["timing"] = ["test/ae_8_indep.qasm"]
    assert {job["circuit"] for job in study.evaluation_jobs(manifest, [], timing=True)} == {"test/ae_8_indep.qasm"}
    assert {job["circuit"] for job in study.evaluation_jobs(manifest, [], timing=False)} == set(manifest["evaluation"])
