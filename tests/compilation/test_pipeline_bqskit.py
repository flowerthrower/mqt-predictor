# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Native KEEP parity and small BQSKit stage-edit semantic checks."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from bqskit import compile as native_compile
from bqskit.compiler.compile import build_workflow
from bqskit.compiler.machine import MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate, U3Gate
from bqskit.ir.gates.measure import MeasurementPlaceholder

from mqt.predictor.rl.experiments.pipeline_bqskit import compile_bqskit, run_bqskit
from mqt.predictor.rl.experiments.worker import RuntimeCompiler

if TYPE_CHECKING:
    from collections.abc import Iterator

    from bqskit.compiler.passdata import PassData


@pytest.fixture(scope="module")
def compiler() -> Iterator[RuntimeCompiler]:
    """Use one isolated native runtime and always close it."""
    instance = RuntimeCompiler(num_workers=1, port=39472, worker_port=39474, num_blas_threads=1)
    try:
        yield instance
    finally:
        instance.close()


def measured_circuit(width: int) -> Circuit:
    """Include cancelled gates, routing, and reversed measurement outputs."""
    circuit = Circuit(width)
    circuit.append_gate(U3Gate(), 0, [0.31, -0.21, 0.12])
    circuit.append_gate(U3Gate(), width - 1, [0.62, 0.1, 0.7])
    circuit.append_gate(CNOTGate(), [0, width - 1])
    circuit.append_gate(CNOTGate(), [0, width - 1])
    circuit.append_gate(CNOTGate(), [width - 1, 0])
    if width == 3:
        circuit.append_gate(CNOTGate(), [2, 1])
    measurements = {qubit: ("out", width - 1 - qubit) for qubit in range(width)}
    circuit.append_gate(MeasurementPlaceholder([("out", width)], measurements), list(range(width)))
    return circuit


def model(width: int) -> MachineModel:
    """Use a small connected native target."""
    return MachineModel(width, [(qubit, qubit + 1) for qubit in range(width - 1)], {CNOTGate(), U3Gate()})


def signature(circuit: Circuit) -> list[tuple[Any, ...]]:
    """Keep cycles, gates, wires, and all numerical parameters."""
    return [
        (cycle, operation.gate, tuple(operation.location), tuple(operation.params))
        for cycle, operation in circuit.operations_with_cycles()
    ]


def assert_native_equal(actual: Circuit, expected: Circuit) -> None:
    """Match exact structure and per-gate unitaries despite native angle variance."""
    actual_ops = signature(actual)
    expected_ops = signature(expected)
    assert len(actual_ops) == len(expected_ops)
    for got, reference in zip(actual_ops, expected_ops, strict=True):
        assert got[:3] == reference[:3]
        if got[3] != reference[3]:
            # Repeated native compiles also choose equivalent Euler angles.
            assert got[1].get_unitary(got[3]).get_distance_from(reference[1].get_unitary(reference[3])) < 1e-12


@pytest.mark.parametrize("width", [2, 3])
@pytest.mark.parametrize("level", [1, 2, 3])
def test_staged_keep_matches_native(compiler: RuntimeCompiler, width: int, level: int) -> None:
    """Retain native gates, physical maps, measurement wires, and semantics."""
    circuit = measured_circuit(width)
    target = model(width)
    expected, initial, final = native_compile(
        circuit.copy(), target, compiler=compiler, optimization_level=level, seed=0, with_mapping=True
    )
    events = []
    actual, data, trace = compile_bqskit(circuit, target, compiler, optimization_level=level, on_event=events.append)

    if level == 3:
        # Native resynthesis can redistribute rotations across gates on reruns.
        assert [op[:3] for op in signature(actual)] == [op[:3] for op in signature(expected)]
    else:
        assert_native_equal(actual, expected)
    assert data.initial_mapping == initial
    assert data.final_mapping == final
    assert [entry["stage_name"] for entry in trace] == [
        stage.name for stage in build_workflow(circuit, target, optimization_level=level, seed=0)
    ]
    assert len(events) == 2 * len(trace)
    assert all(entry["action"] == "KEEP" for entry in trace)
    candidates = {1: {}, 2: {8: {8}, 9: {8}}, 3: {7: {7}, 8: {7}, 9: {9}, 10: {9}}}[level]
    for entry in trace:
        assert set(entry["insertion_candidates"]) == candidates.get(entry["stage_index"], set())
        assert entry["edits_remaining"] == 2
    assert all(events[index]["event"] == ("before" if index % 2 == 0 else "after") for index in range(len(events)))
    json.dumps(trace, allow_nan=False)
    assert_measurements_and_unitary(circuit, actual, data)
    if level == 3:
        assert_measurements_and_unitary(circuit, expected, data)


def assert_measurements_and_unitary(circuit: Circuit, compiled: Circuit, data: PassData) -> None:
    """Compare full unitaries with initial and final physical permutations."""
    source_measurements = next(op.gate for op in circuit if isinstance(op.gate, MeasurementPlaceholder))
    measurements = next(op.gate for op in compiled if isinstance(op.gate, MeasurementPlaceholder))
    assert measurements.classical_regs == source_measurements.classical_regs
    assert measurements.measurements == {
        data.final_mapping[q]: bit for q, bit in source_measurements.measurements.items()
    }
    source = circuit.copy()
    source.remove_all_measurements()
    output = compiled.copy()
    output.remove_all_measurements()
    width = source.num_qudits

    def permutation(mapping: list[int]) -> list[int]:
        return [
            sum(((basis >> (width - 1 - q)) & 1) << (width - 1 - mapping[q]) for q in range(width))
            for basis in range(2**width)
        ]

    expected = np.asarray(source.get_unitary())
    actual = np.asarray(output.get_unitary())[
        np.ix_(permutation(data.final_mapping), permutation(data.initial_mapping))
    ]
    phase = np.vdot(expected, actual)
    phase /= abs(phase)
    np.testing.assert_allclose(actual, expected * phase, atol=2e-7, rtol=0)


@pytest.mark.parametrize(
    ("level", "edits"),
    [
        (2, {8: ["SKIP"]}),
        (2, {9: ["INSERT:8", "INSERT:8"]}),
        (3, {7: ["SKIP"], 9: ["SKIP"]}),
        (3, {8: ["INSERT:7", "END"]}),
        (3, {0: ["END"]}),
    ],
)
def test_optional_edit_preserves_unitary_maps_and_measurements(
    compiler: RuntimeCompiler, level: int, edits: dict[int, list[str]]
) -> None:
    """Skip, insert, or end while retaining native finalization and semantics."""
    circuit = measured_circuit(3)
    seen = []
    commands = {index: iter(actions) for index, actions in edits.items()}

    def policy(state: dict[str, Any]) -> str:
        seen.append(state)
        return next(commands[state["stage_index"]], "KEEP") if state["stage_index"] in commands else "KEEP"

    actual, data, trace = compile_bqskit(circuit, model(3), compiler, policy, optimization_level=level)
    assert [len(state["history"]) for state in seen] == list(range(len(seen)))
    assert all("history" not in entry for entry in trace)
    for state in seen[1:]:
        previous = state["history"][-1]
        entry = trace[len(state["history"]) - 1]
        assert previous["stage_name"] == entry["stage_name"]
        assert previous["inserted_stage_index"] == entry.get("inserted_stage_index")
        assert previous["executed"] == (entry["occurrence"] is not None)
        assert previous["runtime_seconds"] == entry["runtime_seconds"]
        if not previous["executed"]:
            assert previous["runtime_seconds"] == 0
            assert previous["changes"] == {"qubits": 0, "depth": 0, "two_qubit_gates": 0, "gate_counts": {}}
    assert_measurements_and_unitary(circuit, actual, data)
    assert [entry["action"] for entry in trace if entry["action"] != "KEEP" and not entry["automatic"]] == [
        action for actions in edits.values() for action in actions
    ]
    assert all(next(remaining, None) is None for remaining in commands.values())
    assert trace[-1]["edits_remaining"] == 2 - sum(len(actions) for actions in edits.values())
    assert trace[-1]["stage_name"] == "RestoreMeasurements"
    assert trace[-1]["occurrence"] == 0
    inserted = [entry for entry in trace if "inserted_stage_index" in entry]
    for entry in inserted:
        source = next(item for item in trace if item["stage_index"] == entry["inserted_stage_index"])
        assert entry["stage_name"] == source["stage_name"]
        assert entry["occurrence"] is not None
    if 0 in edits:
        assert len(seen) == 1
        assert all(entry["automatic"] for entry in trace[1:])
        assert [entry["stage_index"] for entry in trace if entry["occurrence"] is None] == [7, 9]
    if edits.get(9) == ["INSERT:8", "INSERT:8"]:
        decisions = [entry for entry in trace if entry["stage_index"] == 9]
        assert [entry["occurrence"] for entry in decisions] == [0, 1, 2]
        assert decisions[-1]["allowed_actions"] == ["KEEP"]
        assert [entry["before"] for entry in decisions[1:]] == [entry["after"] for entry in decisions[:-1]]
    native_gates = set(model(3).gate_set)
    assert all(
        operation.gate in native_gates or isinstance(operation.gate, MeasurementPlaceholder) for operation in actual
    )
    without_measurements = actual.copy()
    without_measurements.remove_all_measurements()
    assert set(without_measurements.coupling_graph) <= set(model(3).coupling_graph)


def test_required_stage_cannot_be_skipped(compiler: RuntimeCompiler) -> None:
    """Protect native setup, mapping, retargeting, and measurement passes."""
    circuit = measured_circuit(2)
    workflow = build_workflow(circuit, model(2), optimization_level=2, seed=0)
    with pytest.raises(ValueError, match="SetRandomSeedPass"):
        run_bqskit(circuit, workflow, compiler, lambda _: "SKIP")
