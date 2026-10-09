# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Edit optional stages of TKET's native offline IBM pipelines."""

from __future__ import annotations

import time
from collections import Counter
from typing import TYPE_CHECKING, Any

from pytket.extensions.qiskit import IBMQBackend, qiskit_to_tk
from pytket.passes import SequencePass
from pytket.predicates import CompilationUnit

from mqt.predictor.rl.experiments.pipeline_actions import MAX_EDITS, PipelineActions
from mqt.predictor.rl.experiments.worker import tket_physical_circuit

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytket import Circuit
    from qiskit import QuantumCircuit
    from qiskit_ibm_runtime.fake_provider import FakeBoston

    from mqt.predictor.rl.experiments.pipeline_actions import Policy

OPTIONAL_STAGES = frozenset({
    "FullPeepholeOptimise",
    "GreedyPauliSimp",
    "KAKDecomposition",
    "CliffordSimp",
    "SynthesiseTket",
    "AutoSquash",
    "RemoveRedundancies",
})


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
    max_edits: int = MAX_EDITS,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Edit native optimization blocks while retaining routing and rebasing."""
    trace = []
    actions = PipelineActions(policy, max_edits=max_edits)
    stages = pipeline.get_sequence()
    configs = [stage.to_dict()["StandardPass"] for stage in stages]
    optional = [config["name"] in OPTIONAL_STAGES for config in configs]
    for index, stage in enumerate(stages):
        config = configs[index]
        name = config["name"]
        # Keep swap-enabled preprocessing before mapping and synthesis before
        # the final rebase. Insertions cannot cross required native stages.
        start = stop = index
        while start > 0 and optional[start - 1]:
            start -= 1
        while stop < len(stages) and optional[stop]:
            stop += 1
        occurrence = 0
        while True:
            state = {
                "compiler": "tket",
                "stage_index": index,
                "stage_name": name,
                "stage_config": config,
                "before": features(unit.circuit),
            }
            insertions = {
                candidate: {"stage_name": configs[candidate]["name"], "stage_config": configs[candidate]}
                for candidate in range(start, stop)
                if all(predicate.verify(unit.circuit) for predicate in stages[candidate].get_preconditions())
            }
            action = actions.decide(
                state, trace=trace, optional=optional[index], insertions=insertions, can_end=any(optional[index:])
            )
            inserted = int(action.split(":")[1]) if action.startswith("INSERT:") else None
            selected = stage if inserted is None else stages[inserted]
            execute = action != "SKIP" and (action != "END" or not optional[index])
            entry = {
                **{key: value for key, value in state.items() if key != "history"},
                "action": action,
                "occurrence": occurrence if execute else None,
            }
            if inserted is not None:
                entry.update(inserted_stage_index=inserted, **insertions[inserted])
            if on_event is not None:
                on_event({**entry, "event": "before"})
            started = time.monotonic()
            if execute:
                selected.apply(unit)
            entry["runtime_seconds"] = time.monotonic() - started if execute else 0.0
            entry["after"] = features(unit.circuit)
            trace.append(entry)
            if on_event is not None:
                on_event({**entry, "event": "after"})
            if inserted is None:
                break
            occurrence += 1
    return trace


def compile_tket(
    circuit: QuantumCircuit,
    backend: FakeBoston,
    policy: Policy | None = None,
    *,
    optimisation_level: int = 2,
    max_edits: int = MAX_EDITS,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[QuantumCircuit, CompilationUnit, list[dict[str, Any]]]:
    """Keep native LightSABRE defaults (seed 0, 50 attempts) and final maps."""
    pipeline = IBMQBackend.default_compilation_pass_offline(
        backend.configuration(), backend.properties(), optimisation_level=optimisation_level
    )
    assert isinstance(pipeline, SequencePass)
    unit = CompilationUnit(qiskit_to_tk(circuit))
    trace = run_tket(unit, pipeline, policy, max_edits=max_edits, on_event=on_event)
    return tket_physical_circuit(unit, circuit), unit, trace
