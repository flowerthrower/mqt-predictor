# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Select one native edit plan from predicted ESP and compilation cost."""

from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from mqt.predictor.rl.helper import create_feature_dict

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


def input_features(circuit: QuantumCircuit, target_width: int = 156) -> dict[str, float]:
    """Reuse manual RL observations and input counts before any compilation."""
    counts = {gate: count for gate, count in circuit.count_ops().items() if gate != "barrier"}
    total = sum(counts.values())
    return {
        **{f"rl.{name}": float(value[0]) for name, value in create_feature_dict(circuit, target_width).items()},
        "qubits": float(circuit.num_qubits),
        "depth": float(circuit.depth()),
        "operations": float(total),
        **{f"count.{gate}": float(count) for gate, count in counts.items()},
        **{f"frequency.{gate}": count / max(total, 1) for gate, count in counts.items()},
    }


def plan_features(features: dict[str, float], plan: dict[str, Any]) -> dict[str, float | str]:
    """Describe the input and proposed edits without their compiled outcomes."""
    edits = plan.get("edits", {})
    return {
        **{f"input.{name}": value for name, value in features.items()},
        "compiler": plan["compiler"],
        "level": float(plan["level"]),
        "optimization": plan.get("optimization", "native"),
        "native": float(plan.get("native", plan["id"].endswith("-native"))),
        "edits": float(sum(action != "KEEP" for actions in edits.values() for action in actions)),
        **{
            f"edit.{stage}.{position}": action
            for stage, actions in edits.items()
            for position, action in enumerate(actions)
        },
    }


def family_folds(names: list[str], circuit_hashes: dict[str, str]) -> dict[str, int]:
    """Keep families and identical circuits together, including cross-family duplicates."""
    families = {name: re.sub(r"_\d+_indep\.qasm$", "", Path(name).name) for name in names}
    parents = {family: family for family in families.values()}

    def root(family: str) -> str:
        while parents[family] != family:
            family = parents[family]
        return family

    hashes: dict[str, str] = {}
    for name in names:
        if name in circuit_hashes:
            previous = hashes.setdefault(circuit_hashes[name], families[name])
            parents[root(families[name])] = root(previous)
    groups = [root(families[name]) for name in names]
    count = min(4, len(set(groups)))
    if count < 2:
        msg = "At least two independent circuit families are required for model selection."
        raise ValueError(msg)
    result = {}
    for fold, (_, held_out) in enumerate(GroupKFold(count).split(names, groups=groups)):
        result.update({names[index]: fold for index in held_out})
    return result


def paired_rows(
    rows: list[dict[str, Any]], plans: list[dict[str, Any]], baseline_id: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use successful native-matched measurements; retain exclusion counts."""
    allowed = {plan["id"] for plan in plans}
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    exclusions: Counter[str] = Counter()
    for row in rows:
        if row["plan_id"] not in allowed:
            continue
        if row["status"] != "ok":
            exclusions[row["status"]] += 1
        elif not (
            isinstance(row.get("score", {}).get("esp"), (int, float))
            and math.isfinite(row["score"]["esp"])
            and 0 <= row["score"]["esp"] <= 1
            and math.isfinite(row["seconds"])
            and row["seconds"] > 0
        ):
            exclusions["invalid_measurement"] += 1
        else:
            grouped[row["circuit"], row["seed"], row["plan_id"]].append(row)
    measured: dict[tuple[str, int, str], dict[str, Any]] = {}
    for key, repeats in grouped.items():
        if any(row["features"] != repeats[0]["features"] for row in repeats):
            msg = f"Input features changed between repetitions: {key}"
            raise ValueError(msg)
        measured[key] = {
            "circuit": key[0],
            "seed": key[1],
            "plan_id": key[2],
            "features": repeats[0]["features"],
            "esp": float(np.mean([row["score"]["esp"] for row in repeats])),
            "seconds": float(np.mean([row["seconds"] for row in repeats])),
        }
    matched: list[dict[str, Any]] = []
    for (name, seed, _), row in measured.items():
        baseline = measured.get((name, seed, baseline_id))
        if baseline is None:
            exclusions["unmatched_baseline"] += 1
            continue
        if row["features"] != baseline["features"]:
            msg = f"Plan features differ from native input features: {name}, seed {seed}"
            raise ValueError(msg)
        matched.append({
            **row,
            "delta_esp": row["esp"] - baseline["esp"],
            "time_ratio": row["seconds"] / baseline["seconds"],
            "native_seconds": baseline["seconds"],
        })
    return matched, {
        "exclusions": dict(exclusions),
        "matched_measurements": len(matched),
        "circuits_per_plan": {
            plan: len({r["circuit"] for r in matched if r["plan_id"] == plan}) for plan in sorted(allowed)
        },
    }


def _fit(
    rows: list[dict[str, Any]], plans: list[dict[str, Any]], baseline_id: str, kind: str, seed: int
) -> dict[str, Any]:
    counts = {plan["id"]: len({row["circuit"] for row in rows if row["plan_id"] == plan["id"]}) for plan in plans}
    supported = [plan for plan in plans if plan["id"] == baseline_id or counts[plan["id"]] >= 4]
    by_id = {plan["id"]: plan for plan in supported}
    selected = [row for row in rows if row["plan_id"] in by_id]
    per_circuit = Counter(row["circuit"] for row in selected)
    regressor = (
        DecisionTreeRegressor(max_depth=6, min_samples_leaf=4, random_state=seed)
        if kind == "tree"
        else RandomForestRegressor(n_estimators=64, max_depth=12, min_samples_leaf=4, random_state=seed, n_jobs=1)
    )
    model = TransformedTargetRegressor(
        regressor=Pipeline([("vectorizer", DictVectorizer()), ("regressor", regressor)]),
        transformer=StandardScaler(),
    )
    model.fit(
        [plan_features(row["features"], by_id[row["plan_id"]]) for row in selected],
        [[row["delta_esp"], row["time_ratio"]] for row in selected],
        regressor__sample_weight=[1 / per_circuit[row["circuit"]] for row in selected],
    )
    return {"model": model, "plans": supported, "baseline_id": baseline_id, "model_kind": kind, "model_seed": seed}


def choose_plan(bundle: dict[str, Any], features: dict[str, float], weight: float) -> dict[str, Any]:
    """Choose one plan using predictions; native's relative cost is exactly zero."""
    plans = bundle["plans"]
    outcomes = bundle["model"].predict([plan_features(features, plan) for plan in plans])
    costs = [
        float(time_ratio - 1 - weight * delta_esp) if plan["id"] != bundle["baseline_id"] else 0.0
        for plan, (delta_esp, time_ratio) in zip(plans, outcomes, strict=True)
    ]
    return plans[
        min(
            range(len(plans)),
            key=lambda index: (costs[index], plans[index]["id"] != bundle["baseline_id"], plans[index]["id"]),
        )
    ]


def train_models(
    rows: list[dict[str, Any]],
    plans: list[dict[str, Any]],
    compiler: str,
    *,
    weights: tuple[float, ...] = (10, 50, 100),
    model_seeds: tuple[int, ...] = (0, 1, 2),
    circuit_hashes: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select tree or forest within training families, then fit each requested seed.

    Each model predicts ESP change and relative time separately. A one-point
    ESP gain offsets 10%, 50%, or 100% more runtime at weights 10, 50, or 100.
    Missing selected outcomes are reported, never replaced with an available
    alternative. Tree/forest selection uses their common measured predictions
    and selection latency. Shared input-feature cost cancels between both kinds.
    """
    candidates = [plan for plan in plans if plan["compiler"] == compiler]
    baseline_id = f"{compiler}3-native"
    if baseline_id not in {plan["id"] for plan in candidates}:
        msg = f"Missing native reference plan: {baseline_id}"
        raise ValueError(msg)
    paired, coverage = paired_rows(rows, candidates, baseline_id)
    names = sorted({row["circuit"] for row in paired})
    folds = family_folds(names, circuit_hashes or {})
    outcomes = {(row["circuit"], row["seed"], row["plan_id"]): row for row in paired}
    inputs = {(row["circuit"], row["seed"]): row["features"] for row in paired}
    predictions: dict[str, dict[float, dict[tuple[str, int], dict[str, Any]]]] = {
        kind: {weight: {} for weight in weights} for kind in ("tree", "forest")
    }
    for fold in sorted(set(folds.values())):
        training = [row for row in paired if folds[row["circuit"]] != fold]
        held_out = {key: features for key, features in inputs.items() if folds[key[0]] == fold}
        for kind in predictions:
            bundle = _fit(training, candidates, baseline_id, kind, 0)
            choose_plan(bundle, next(iter(held_out.values())), weights[0])
            for weight in weights:
                for key, features in held_out.items():
                    started = time.perf_counter()
                    plan = choose_plan(bundle, features, weight)
                    selection_seconds = time.perf_counter() - started
                    result = outcomes.get((*key, plan["id"]))
                    predictions[kind][weight][key] = {
                        "plan_id": plan["id"],
                        "outcome": result,
                        "selection_seconds": selection_seconds,
                    }
    report: dict[str, Any] = {
        "compiler": compiler,
        **coverage,
        "folds": folds,
        "minimum_candidate_circuits": 4,
        "weights": {},
        "objective": "time/native_time - 1 - weight * (ESP - native_ESP)",
        "timing": "CV cost adds measured plan-selection latency to collected compilation time. Shared feature cost cancels between model kinds. Final serial evaluation includes all deployed overhead and is authoritative.",
    }
    bundles: list[dict[str, Any]] = []
    fits: dict[tuple[str, int], dict[str, Any]] = {}
    for weight in weights:
        common = {
            key for key in inputs if all(predictions[kind][weight][key]["outcome"] is not None for kind in predictions)
        }
        metrics = {}
        for kind, choices in predictions.items():
            by_circuit: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for key in common:
                entry = choices[weight][key]
                by_circuit[key[0]].append({**entry["outcome"], "selection_seconds": entry["selection_seconds"]})
            costs = [
                float(
                    np.mean([
                        row["time_ratio"]
                        + row["selection_seconds"] / row["native_seconds"]
                        - 1
                        - weight * row["delta_esp"]
                        for row in entries
                    ])
                )
                for entries in by_circuit.values()
            ]
            metrics[kind] = {
                "mean_cost_common": float(np.mean(costs)) if costs else None,
                "unavailable_selected_outcomes": sum(entry["outcome"] is None for entry in choices[weight].values()),
                "choices": [
                    {
                        "circuit": key[0],
                        "seed": key[1],
                        "plan_id": entry["plan_id"],
                        "available": entry["outcome"] is not None,
                        "selection_seconds": entry["selection_seconds"],
                    }
                    for key, entry in sorted(choices[weight].items())
                ],
            }
        eligible = [kind for kind in metrics if metrics[kind]["mean_cost_common"] is not None]
        if not eligible:
            msg = "No common measured validation outcomes for model selection."
            raise ValueError(msg)
        selected_kind = min(eligible, key=lambda kind: (metrics[kind]["mean_cost_common"], kind != "tree"))
        report["weights"][str(weight)] = {
            "selected_kind": selected_kind,
            "common_measurements": len(common),
            "common_circuits": len({key[0] for key in common}),
            "models": metrics,
        }
        for seed in model_seeds:
            if (selected_kind, seed) not in fits:
                fits[selected_kind, seed] = _fit(paired, candidates, baseline_id, selected_kind, seed)
            bundles.append({**fits[selected_kind, seed], "compiler": compiler, "weight": weight})
    report["supported_plan_ids"] = sorted({plan["id"] for bundle in bundles for plan in bundle["plans"]})
    return bundles, report
