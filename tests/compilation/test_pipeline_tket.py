# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Focused parity and validity checks for native TKET pipeline edits."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from pytket import Circuit
from pytket.extensions.qiskit import IBMQBackend, qiskit_to_tk
from pytket.passes import SequencePass
from pytket.predicates import CompilationUnit
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator, Statevector

from mqt.predictor.rl.experiments.inputs import load_target
from mqt.predictor.rl.experiments.metrics import observe
from mqt.predictor.rl.experiments.pipeline_tket import compile_tket, features, run_tket
from mqt.predictor.rl.experiments.worker import tket_physical_circuit

if TYPE_CHECKING:
    from pytket.circuit import UnitID
    from qiskit.transpiler import Target

    from mqt.predictor.rl.experiments.inputs import FrozenBoston

ASSETS = Path(__file__).resolve().parents[2] / "experiments/assets"


@pytest.fixture(scope="module")
def backend_target() -> tuple[FrozenBoston, Target]:
    """Use the shared offline Boston target and calibration."""
    return load_target(ASSETS)


def measured_circuit(multiple_registers: bool) -> QuantumCircuit:
    """Exercise output permutations and non-alphabetical register order."""
    circuit = (
        QuantumCircuit(
            QuantumRegister(1, "z"),
            QuantumRegister(2, "a"),
            ClassicalRegister(3, "out"),
        )
        if multiple_registers
        else QuantumCircuit(3, 3)
    )
    circuit.h(0)
    circuit.cx(0, 2)
    circuit.ry(0.31, 1)
    circuit.cx(2, 1)
    circuit.swap(0, 1)
    circuit.measure(range(3), [2, 0, 1])
    return circuit


def classical_distribution(circuit: QuantumCircuit) -> dict[tuple[int, ...], float]:
    """Sum exact measurement branches, including gates after measurements."""
    wires = sorted({circuit.find_bit(qubit).index for instruction in circuit.data for qubit in instruction.qubits})
    compact = {physical: index for index, physical in enumerate(wires)}
    branches = [(Statevector.from_int(0, 2 ** len(wires)), (0,) * circuit.num_clbits)]
    for instruction in circuit.data:
        qubits = [compact[circuit.find_bit(qubit).index] for qubit in instruction.qubits]
        if instruction.operation.name != "measure":
            branches = [(state.evolve(instruction.operation, qubits), bits) for state, bits in branches]
            continue
        bit = circuit.find_bit(instruction.clbits[0]).index
        measured = []
        for state, bits in branches:
            for outcome in (0, 1):
                amplitudes = state.data.copy()
                amplitudes[((np.arange(len(amplitudes)) >> qubits[0]) & 1) != outcome] = 0
                measured.append((Statevector(amplitudes), (*bits[:bit], outcome, *bits[bit + 1 :])))
        branches = measured
    result = defaultdict(float)
    for state, bits in branches:
        result[bits] += float(np.vdot(state.data, state.data).real)
    return dict(result)


def test_features_count_gates_without_barriers() -> None:
    """Policy features must not count two-wire barriers as entangling gates."""
    circuit = Circuit(2).CX(0, 1).add_barrier([0, 1])
    assert features(circuit)["two_qubit_gates"] == 1


@pytest.mark.parametrize("multiple_registers", [False, True])
@pytest.mark.parametrize("optimisation_level", [2, 3])
def test_keep_matches_native_circuit_maps_and_measurements(
    backend_target: tuple[FrozenBoston, Target], multiple_registers: bool, optimisation_level: int
) -> None:
    """KEEP must preserve the native preset's circuit, maps and score."""
    backend, target = backend_target
    circuit = measured_circuit(multiple_registers)
    expected = CompilationUnit(qiskit_to_tk(circuit))
    pipeline = IBMQBackend.default_compilation_pass_offline(
        backend.configuration(), backend.properties(), optimisation_level=optimisation_level
    )
    pipeline.apply(expected)
    actual, unit, trace = compile_tket(circuit, backend, optimisation_level=optimisation_level)
    reference = tket_physical_circuit(expected, circuit)

    assert unit.circuit == expected.circuit
    assert unit.initial_map == expected.initial_map
    assert unit.final_map == expected.final_map
    assert actual == reference
    assert actual.layout is not None
    assert reference.layout is not None
    assert actual.layout.initial_index_layout() == reference.layout.initial_index_layout()
    assert actual.layout.final_index_layout() == reference.layout.final_index_layout()
    assert observe(actual, target, physical=True) == observe(reference, target, physical=True)
    assert len(trace) == 10
    assert all(entry["action"] == "KEEP" for entry in trace)
    assert classical_distribution(actual) == pytest.approx(classical_distribution(circuit), abs=1e-10)
    if optimisation_level == 3:
        assert trace[3]["stage_name"] == "GreedyPauliSimp"
        assert "SKIP" in trace[3]["allowed_actions"]
    json.dumps(trace, allow_nan=False)


@pytest.mark.parametrize("action", ["SKIP", "INSERT", "END"])
@pytest.mark.parametrize("optimisation_level", [2, 3])
def test_optional_edits_preserve_native_validity_and_measurement_semantics(
    backend_target: tuple[FrozenBoston, Target], action: str, optimisation_level: int
) -> None:
    """Edits retain connectivity, basis and measurement outcomes within budget."""
    backend, target = backend_target
    circuit = measured_circuit(True)
    seen = []

    def policy(state: dict[str, Any]) -> str:
        seen.append(state)
        selected = f"INSERT:{state['stage_index']}" if action == "INSERT" else action
        return selected if "SKIP" in state["allowed_actions"] and selected in state["allowed_actions"] else "KEEP"

    events = []
    actual, unit, trace = compile_tket(
        circuit, backend, policy, optimisation_level=optimisation_level, on_event=events.append
    )
    edited = [entry for entry in trace if entry["action"] != "KEEP" and not entry["automatic"]]
    assert len(edited) == (1 if action == "END" else 2)
    expected_invocations = {"SKIP": 8, "INSERT": 12, "END": 4 if optimisation_level == 2 else 6}[action]
    assert sum(entry["occurrence"] is not None for entry in trace) == expected_invocations
    if action == "END":
        assert len(seen) == (2 if optimisation_level == 2 else 4)
        assert all(entry["automatic"] for entry in trace[len(seen) :])
    else:
        assert trace[-1]["edits_remaining"] == 0
    assert len(events) == 2 * len(trace)
    assert [len(state["history"]) for state in seen] == list(range(len(seen)))
    assert all("history" not in entry for entry in trace)
    for state in seen[1:]:
        previous = state["history"][-1]
        entry = trace[len(state["history"]) - 1]
        assert previous["action"] == entry["action"]
        assert previous["executed"] == (entry["occurrence"] is not None)
        assert previous["runtime_seconds"] == entry["runtime_seconds"]
        if not previous["executed"]:
            assert previous["runtime_seconds"] == 0
            assert previous["changes"] == {"qubits": 0, "depth": 0, "two_qubit_gates": 0, "gate_counts": {}}
    for entry, before_event, after_event in zip(trace, events[::2], events[1::2], strict=True):
        assert before_event == {
            **{key: value for key, value in entry.items() if key not in {"after", "runtime_seconds"}},
            "event": "before",
        }
        assert after_event == {**entry, "event": "after"}
    assert observe(actual, target, physical=True)["esp_kind"] == "exact"
    assert all(
        target.instruction_supported(
            operation_name=instruction.operation.name,
            qargs=tuple(actual.find_bit(qubit).index for qubit in instruction.qubits),
        )
        for instruction in actual.data
    )

    assert classical_distribution(actual) == pytest.approx(classical_distribution(circuit), abs=1e-10)
    assert unit.final_map

    unmeasured = circuit.remove_final_measurements(inplace=False)
    assert unmeasured is not None
    source = qiskit_to_tk(unmeasured)
    _, compiled, _ = compile_tket(unmeasured, backend, policy, optimisation_level=optimisation_level)
    n_qubits = source.n_qubits
    positions: dict[UnitID, int] = {qubit: index for index, qubit in enumerate(compiled.circuit.qubits)}
    axes = [
        offset + positions[mapping[qubit]]
        for offset, mapping in ((0, compiled.final_map), (n_qubits, compiled.initial_map))
        for qubit in source.qubits
    ]
    logical_unitary = (
        compiled.circuit
        .get_unitary()
        .reshape((2,) * (2 * n_qubits))
        .transpose(axes)
        .reshape((2**n_qubits, 2**n_qubits))
    )
    assert Operator(logical_unitary).equiv(Operator(source.get_unitary()))


def test_insert_reuses_native_stage_before_continuing_same_cursor(
    backend_target: tuple[FrozenBoston, Target],
) -> None:
    """Inserted native stages preserve their configuration, ordering and maps."""
    backend, target = backend_target
    circuit = measured_circuit(True)
    expected = CompilationUnit(qiskit_to_tk(circuit))
    pipeline = IBMQBackend.default_compilation_pass_offline(
        backend.configuration(), backend.properties(), optimisation_level=2
    )
    assert isinstance(pipeline, SequencePass)
    stages = pipeline.get_sequence()
    insertions = {4: 6, 5: 4}
    for index, stage in enumerate(stages):
        if index in insertions:
            stages[insertions[index]].apply(expected)
        stage.apply(expected)

    seen = []

    def policy(state: dict[str, Any]) -> str:
        seen.append(state)
        if (source := insertions.pop(state["stage_index"], None)) is not None:
            return f"INSERT:{source}"
        return "KEEP"

    actual, unit, trace = compile_tket(circuit, backend, policy)
    assert unit.circuit == expected.circuit
    assert unit.initial_map == expected.initial_map
    assert unit.final_map == expected.final_map
    assert actual == tket_physical_circuit(expected, circuit)
    assert observe(actual, target, physical=True)["esp_kind"] == "exact"
    assert classical_distribution(actual) == pytest.approx(classical_distribution(circuit), abs=1e-10)
    assert [(entry["stage_index"], entry["stage_name"]) for entry in trace[4:8]] == [
        (4, "SynthesiseTket"),
        (4, "KAKDecomposition"),
        (5, "KAKDecomposition"),
        (5, "CliffordSimp"),
    ]
    assert trace[4]["stage_config"] == stages[6].to_dict()["StandardPass"]
    assert trace[4]["inserted_stage_index"] == 6
    assert trace[4]["occurrence"] == 0
    assert trace[5]["occurrence"] == 1
    assert seen[5]["before"] == trace[4]["after"]
    assert seen[5]["history"][-1]["stage_index"] == 4
    assert seen[5]["history"][-1]["inserted_stage_index"] == 6
    assert seen[5]["history"][-1]["stage_name"] == "SynthesiseTket"


@pytest.mark.parametrize(("cursor", "source"), [(4, 1), (8, 6)])
def test_insert_cannot_cross_native_mapping_or_final_basis_boundary(
    backend_target: tuple[FrozenBoston, Target], cursor: int, source: int
) -> None:
    """Swap-enabled preprocessing and CX synthesis retain their native phases."""
    backend, _ = backend_target
    with pytest.raises(ValueError, match="Invalid"):
        compile_tket(
            measured_circuit(False),
            backend,
            lambda state: f"INSERT:{source}" if state["stage_index"] == cursor else "KEEP",
        )


def test_end_at_required_stage_still_executes_native_finalization(
    backend_target: tuple[FrozenBoston, Target],
) -> None:
    """END before preprocessing retains conversion, mapping and native basis."""
    backend, target = backend_target
    circuit = measured_circuit(True)
    actual, unit, trace = compile_tket(circuit, backend, lambda _: "END", optimisation_level=3)
    assert trace[0]["action"] == "END"
    assert trace[0]["occurrence"] == 0
    assert [entry["stage_name"] for entry in trace if entry["occurrence"] is not None] == [
        "DecomposeBoxes",
        "RemoveBarriers",
        "AutoRebase",
        "AutoRebase",
        "CustomPassMap",
        "AutoRebase",
    ]
    assert all(entry["automatic"] for entry in trace[1:])
    assert observe(actual, target, physical=True)["esp_kind"] == "exact"
    assert all(
        target.instruction_supported(
            operation_name=instruction.operation.name,
            qargs=tuple(actual.find_bit(qubit).index for qubit in instruction.qubits),
        )
        for instruction in actual.data
    )
    assert classical_distribution(actual) == pytest.approx(classical_distribution(circuit), abs=1e-10)
    assert unit.final_map


def test_mapping_stage_cannot_be_skipped(backend_target: tuple[FrozenBoston, Target]) -> None:
    """Reject a policy decision that removes required mapping."""
    backend, _ = backend_target
    pipeline = IBMQBackend.default_compilation_pass_offline(
        backend.configuration(), backend.properties(), optimisation_level=2
    )
    assert isinstance(pipeline, SequencePass)
    unit = CompilationUnit(qiskit_to_tk(measured_circuit(False)))
    with pytest.raises(ValueError, match="CustomPassMap"):
        run_tket(
            unit,
            pipeline,
            lambda state: "SKIP" if state["stage_name"] == "CustomPassMap" else "KEEP",
        )
