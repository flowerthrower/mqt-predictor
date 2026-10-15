# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Check causal plan selection, signed objectives, and incomplete measurements."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pytest
from qiskit import QuantumCircuit

from mqt.predictor.rl.experiments import pipeline_learning as learning
from mqt.predictor.rl.experiments.pipeline_learning import (
    choose_plan,
    family_folds,
    input_features,
    paired_rows,
    plan_features,
    train_models,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def measurements() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Use a one-point ESP improvement that costs 20% extra runtime."""
    plans = [
        {"id": "qiskit3-native", "compiler": "qiskit", "level": 3, "edits": {}},
        {"id": "qiskit3-edit", "compiler": "qiskit", "level": 3, "edits": {"4": ["INSERT:4"]}},
    ]
    rows = []
    for family in range(8):
        for size in (2, 3):
            name = f"train/family{family}_{size}_indep.qasm"
            for plan in plans:
                edited = plan["id"] == "qiskit3-edit"
                rows.append({
                    "circuit": name,
                    "seed": 0,
                    "plan_id": plan["id"],
                    "status": "ok",
                    "features": {"qubits": float(size)},
                    "score": {"esp": 0.51 if edited else 0.5},
                    "seconds": 1.2 if edited else 1.0,
                })
    return rows, plans


def test_input_and_plan_features_do_not_use_compiled_outcomes() -> None:
    """Features reuse the existing manual observations and proposed edits only."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    actual = input_features(circuit)
    assert actual["qubits"] == 2
    assert actual["rl.num_qubits"] == pytest.approx(2 / 156)
    assert actual["frequency.cx"] == pytest.approx(0.5)
    plan = {"id": "qiskit3-native", "compiler": "qiskit", "level": 3, "edits": {}}
    before = plan_features(actual, plan)
    plan.update(score={"esp": 0.999}, seconds=1e9, future_actions=["SKIP"])
    assert plan_features(actual, plan) == before
    assert before["native"] == 1
    assert plan_features(actual, {**plan, "id": "qiskit3-keep"})["native"] == 0


def test_family_folds_merge_identical_circuits_across_families() -> None:
    """The duplicate VQE inputs cannot leak across a family-held-out fold."""
    names = [
        f"train/{family}_{size}_indep.qasm"
        for family in ("vqe_real_amp", "vqe_two_local", "qft", "ghz")
        for size in (2, 3)
    ]
    hashes = dict.fromkeys(["train/vqe_real_amp_2_indep.qasm", "train/vqe_two_local_2_indep.qasm"], "same")
    folds = family_folds(names, hashes)
    assert len({folds[name] for name in names if "vqe_" in name}) == 1
    assert folds["train/qft_2_indep.qasm"] == folds["train/qft_3_indep.qasm"]
    assert len(set(folds.values())) == 3


def test_matching_keeps_failures_out_of_regression_targets(
    measurements: tuple[list[dict[str, Any]], list[dict[str, Any]]],
) -> None:
    """A timeout contributes coverage only, and cannot become zero ESP."""
    rows, plans = measurements
    rows[1].update(status="timeout", score={"esp": 0}, seconds=600)
    rows.append({**rows[3], "seed": 1})
    paired, report = paired_rows(rows, plans, "qiskit3-native")
    assert report["exclusions"] == {"timeout": 1, "unmatched_baseline": 1}
    assert len(paired) == len(rows) - 2
    edited = [row for row in paired if row["plan_id"] == "qiskit3-edit"]
    assert all(row["delta_esp"] == pytest.approx(0.01) and row["time_ratio"] == pytest.approx(1.2) for row in edited)
    assert report["circuits_per_plan"]["qiskit3-edit"] == 15
    rows[0]["features"] = {"qubits": 999.0}
    rows[1]["status"] = "ok"
    with pytest.raises(ValueError, match="features differ"):
        paired_rows(rows, plans, "qiskit3-native")


def test_signed_tradeoff_changes_one_predicted_plan_and_survives_save_load(
    measurements: tuple[list[dict[str, Any]], list[dict[str, Any]]], tmp_path: Path
) -> None:
    """The same learned outcomes prefer speed or ESP without trial compilation."""
    rows, plans = measurements
    bundles, report = train_models(rows, plans, "qiskit", weights=(10, 50), model_seeds=(0, 1, 2))
    assert len(bundles) == 6
    assert {bundle["model_seed"] for bundle in bundles} == {0, 1, 2}
    assert report["weights"]["10"]["common_circuits"] == 16
    assert all(
        value["unavailable_selected_outcomes"] == 0
        for result in report["weights"].values()
        for value in result["models"].values()
    )
    for bundle in bundles:
        assert choose_plan(bundle, {"qubits": 4.0}, 10)["id"] == "qiskit3-native"
        assert choose_plan(bundle, {"qubits": 4.0}, 50)["id"] == "qiskit3-edit"
    path = tmp_path / "selector.joblib"
    joblib.dump(bundles[0], path)
    restored = joblib.load(path)
    vocabulary = deepcopy(restored["model"].regressor_.named_steps["vectorizer"].vocabulary_)
    assert choose_plan(restored, {"qubits": 4.0, "unseen_input_feature": 1.0}, 50)["id"] == "qiskit3-edit"
    assert restored["model"].regressor_.named_steps["vectorizer"].vocabulary_ == vocabulary


def test_missing_chosen_outcomes_are_reported_without_an_oracle_fallback(
    measurements: tuple[list[dict[str, Any]], list[dict[str, Any]]],
) -> None:
    """A candidate missing on one held-out input stays selected and unavailable."""
    rows, plans = measurements
    missing = rows.pop(1)
    _, report = train_models(rows, plans, "qiskit", weights=(50,), model_seeds=(0,))
    for result in report["weights"]["50"]["models"].values():
        selected = next(row for row in result["choices"] if row["circuit"] == missing["circuit"])
        assert selected["plan_id"] == "qiskit3-edit"
        assert selected["available"] is False
        assert result["unavailable_selected_outcomes"] == 1
    assert report["weights"]["50"]["common_circuits"] == 15


def test_native_cost_is_zero_even_if_the_model_predicts_otherwise(
    measurements: tuple[list[dict[str, Any]], list[dict[str, Any]]],
) -> None:
    """Native-relative predictions cannot invent a gain for the reference itself."""
    _, plans = measurements

    class Predictions:
        def predict(self, features: list[dict[str, Any]]) -> np.ndarray:
            assert len(features) == 2
            return np.array([[1.0, 0.0], [0.01, 1.2]])

    bundle = {"model": Predictions(), "plans": plans, "baseline_id": "qiskit3-native"}
    assert choose_plan(bundle, {}, 50)["id"] == "qiskit3-edit"
    assert choose_plan(bundle, {}, 10)["id"] == "qiskit3-native"


def test_model_selection_charges_prediction_latency(
    measurements: tuple[list[dict[str, Any]], list[dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identical compilation outcomes prefer the faster selector, not a tie rule."""
    rows, plans = measurements
    clock = [0.0]

    def choose(bundle: dict[str, Any], features: dict[str, float], weight: float) -> dict[str, Any]:
        assert features
        assert weight == 50
        clock[0] += 0.01 if bundle["model_kind"] == "tree" else 0.001
        return bundle["plans"][1]

    monkeypatch.setattr(learning, "choose_plan", choose)
    monkeypatch.setattr(learning.time, "perf_counter", lambda: clock[0])
    bundles, report = train_models(rows, plans, "qiskit", weights=(50,), model_seeds=(0,))
    result = report["weights"]["50"]
    assert result["selected_kind"] == bundles[0]["model_kind"] == "forest"
    assert result["models"]["tree"]["mean_cost_common"] == pytest.approx(-0.29)
    assert result["models"]["forest"]["mean_cost_common"] == pytest.approx(-0.299)
    assert all(entry["selection_seconds"] == pytest.approx(0.001) for entry in result["models"]["forest"]["choices"])
