# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Pass observations and manuscript efficiency metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiskit.exceptions import QiskitError

from mqt.predictor.reward import estimated_success_probability, expected_fidelity
from mqt.predictor.rl.approx_reward import (
    approximate_estimated_success_probability,
    average_target_calibration,
)
from mqt.predictor.utils import calc_supermarq_features

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.transpiler import Target

STRUCTURAL = {"synthesis", "layout", "routing", "mapping"}
OPTIMIZATION = {"optimization", "final_optimization"}


def observe(circuit: QuantumCircuit, target: Target, *, physical: bool) -> dict[str, Any]:
    """Score exact physical circuits and label pre-mapping ESP proxies."""
    native = all(item.operation.name in target.operation_names or item.operation.name == "barrier" for item in circuit)
    edges = set(target.build_coupling_map().get_edges())
    routed = physical and all(
        len(item.qubits) != 2 or tuple(circuit.find_bit(q).index for q in item.qubits) in edges for item in circuit
    )
    result: dict[str, Any] = {
        "state": {"synthesis": native, "layout": physical, "routing": routed},
        "esp": None,
        "esp_kind": "unavailable",
        "expected_fidelity": None,
        "depth": circuit.depth(),
        "gate_counts": dict(circuit.count_ops()),
    }
    try:  # ruff: ignore[too-many-statements-in-try-clause] -- unavailable calibration is recorded, not a failed job.
        if native and physical and routed:
            result["expected_fidelity"] = expected_fidelity(circuit, target)
            result["esp"] = estimated_success_probability(circuit, target)
            result["esp_kind"] = "exact"
        else:
            errors, durations, coherence = average_target_calibration(target)
            features = calc_supermarq_features(circuit)
            result["esp"] = approximate_estimated_success_probability(
                circuit,
                device=target,
                error_rates=errors,
                gate_durations=durations,
                coherence_time=coherence,
                parallelism=float(features.parallelism),
                liveness=float(features.liveness),
            )
            result["esp_kind"] = "approximate"
    except (KeyError, ValueError, TypeError, RuntimeError, QiskitError) as error:
        result["score_error"] = f"{type(error).__name__}: {error}"
    return result


def efficiency(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute signed PPE, optimization ratio and structural success fraction."""
    passes = [entry for entry in traces if not entry.get("container")]
    structural = [entry for entry in passes if entry["category"] in STRUCTURAL]
    optimization = [entry for entry in passes if entry["category"] in OPTIMIZATION]
    deltas: list[float] = []
    opt_deltas: list[float] = []
    transitions = 0
    exact = 0
    for entry in structural + optimization:
        before, after = entry.get("before", {}), entry.get("after", {})
        a, b = before.get("esp"), after.get("esp")
        kinds = before.get("esp_kind"), after.get("esp_kind")
        comparable = entry.get("status") == "ok" and a is not None and b is not None and kinds[0] == kinds[1]
        entry["comparable"] = comparable
        if a is not None and b is not None and kinds[0] != kinds[1]:
            transitions += 1
        if comparable:
            deltas.append(b - a)
            exact += kinds[0] == "exact"
            if entry["category"] in OPTIMIZATION:
                opt_deltas.append(b - a)
    successes = 0
    for entry in structural:
        before = entry.get("before", {}).get("state", {})
        after = entry.get("after", {}).get("state", {})
        keys = ("synthesis", "layout", "routing") if entry["category"] == "mapping" else (entry["category"],)
        successes += (
            entry.get("status") == "ok"
            and all(after.get(key, False) for key in keys)
            and any(not before.get(key, False) for key in keys)
        )
    movement = sum(abs(delta) for delta in opt_deltas)
    eta_opt = sum(max(0, delta) for delta in opt_deltas) / movement if movement else 0.0
    eta_struct = successes / len(structural) if structural else 0.0
    count = len(structural) + len(optimization)
    return {
        "optimization_efficiency": eta_opt,
        "structural_success_fraction": eta_struct,
        "overall_efficiency": (len(structural) * eta_struct + len(optimization) * eta_opt) / count if count else 0.0,
        "ppe": sum(deltas) / len(deltas) if deltas else None,
        "optimization_invocations": len(optimization),
        "structural_invocations": len(structural),
        "structural_successes": successes,
        "comparable_observations": len(deltas),
        "comparable_optimization_observations": len(opt_deltas),
        "exact_observations": exact,
        "proxy_observations": len(deltas) - exact,
        "proxy_exact_transitions": transitions,
        "coverage": len(deltas) / count if count else None,
        "administrative_invocations": sum(entry["category"] == "administrative" for entry in passes),
        "analysis_invocations": sum(entry["category"] == "analysis" for entry in passes),
        "failed_passes": sum(entry.get("status") == "error" for entry in passes),
        "timed_out_passes": sum(entry.get("status") == "timeout" for entry in passes),
    }
