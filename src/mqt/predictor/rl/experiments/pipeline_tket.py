# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Edit optional stages of TKET's native offline IBM O2 pipeline."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pytket.extensions.qiskit import IBMQBackend, qiskit_to_tk
from pytket.passes import SequencePass
from pytket.predicates import CompilationUnit

from mqt.predictor.rl.experiments.worker import tket_physical_circuit

if TYPE_CHECKING:
    from pytket import Circuit
    from qiskit import QuantumCircuit
    from qiskit_ibm_runtime.fake_provider import FakeBoston

OPTIONAL_STAGES = frozenset({
    "FullPeepholeOptimise",
    "KAKDecomposition",
    "CliffordSimp",
    "SynthesiseTket",
    "AutoSquash",
    "RemoveRedundancies",
})
Policy = Callable[[dict[str, Any]], str]


def features(circuit: Circuit) -> dict[str, Any]:
    """Return plain circuit statistics without converting or scoring the circuit."""
    commands = circuit.get_commands()
    return {
        "qubits": circuit.n_qubits,
        "depth": circuit.depth(),
        "gate_counts": dict(Counter(command.op.type.name for command in commands)),
        "two_qubit_gates": circuit.n_2qb_gates(),
    }


def run_tket(
    unit: CompilationUnit,
    pipeline: SequencePass,
    policy: Policy | None = None,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Apply native stage objects to one unit; REPEAT adds one invocation."""
    trace = []
    for index, stage in enumerate(pipeline.get_sequence()):
        config = stage.to_dict()["StandardPass"]
        name = config["name"]
        optional = name in OPTIONAL_STAGES
        before = features(unit.circuit)
        state = {
            "compiler": "tket",
            "stage_index": index,
            "stage_name": name,
            "stage_config": config,
            "allowed_actions": ["KEEP", "SKIP", "REPEAT"] if optional else ["KEEP"],
            "before": before,
        }
        action = "KEEP" if policy is None else policy(state)
        if action not in state["allowed_actions"]:
            msg = f"Invalid {action!r} for TKET stage {index}:{name}."
            raise ValueError(msg)
        invocations = {"SKIP": 0, "KEEP": 1, "REPEAT": 2}[action]
        for occurrence in range(max(1, invocations)):
            entry = {
                **state,
                "action": action,
                "occurrence": occurrence if invocations else None,
                "before": before,
            }
            if on_event is not None:
                on_event({**entry, "event": "before"})
            if invocations:
                stage.apply(unit)
            entry["after"] = features(unit.circuit)
            trace.append(entry)
            if on_event is not None:
                on_event({**entry, "event": "after"})
            before = entry["after"]
    return trace


def compile_tket(
    circuit: QuantumCircuit,
    backend: FakeBoston,
    policy: Policy | None = None,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[QuantumCircuit, CompilationUnit, list[dict[str, Any]]]:
    """Keep native LightSABRE defaults (seed 0, 50 attempts) and final maps."""
    pipeline = IBMQBackend.default_compilation_pass_offline(
        backend.configuration(), backend.properties(), optimisation_level=2
    )
    assert isinstance(pipeline, SequencePass)
    unit = CompilationUnit(qiskit_to_tk(circuit))
    trace = run_tket(unit, pipeline, policy, on_event=on_event)
    return tket_physical_circuit(unit, circuit), unit, trace
