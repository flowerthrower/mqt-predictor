# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Check native KEEP parity and valid optional-stage edits."""

from __future__ import annotations

from copy import deepcopy
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.passmanager import ConditionalController, DoWhileController
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from mqt.predictor.rl.experiments import pipeline_qiskit as editor
from mqt.predictor.rl.experiments.inputs import Inputs, load_target
from mqt.predictor.rl.experiments.qiskit_teacher import demonstration

if TYPE_CHECKING:
    from collections.abc import Callable

    from qiskit.dagcircuit import DAGCircuit
    from qiskit.passmanager import PropertySet
    from qiskit.transpiler import StagedPassManager, Target

ASSETS = Path(__file__).resolve().parents[2] / "experiments/assets"


@pytest.fixture(scope="module")
def frozen_inputs() -> tuple[Inputs, Target]:
    """Use the shared circuit archive and offline Boston calibration."""
    return Inputs(ASSETS), load_target(ASSETS)[1]


@pytest.mark.parametrize("seed", [0, 7])
@pytest.mark.parametrize("name", ["test/ae_8_indep.qasm", "test/qft_5_indep.qasm"])
def test_keep_matches_native_frozen_target(frozen_inputs: tuple[Inputs, Target], seed: int, name: str) -> None:
    """KEEP matches every executed pass and resulting DAG, including layouts."""
    inputs, target = frozen_inputs
    circuit = inputs.circuit(name)
    native_trace: list[tuple[str, DAGCircuit]] = []
    edited_trace: list[tuple[str, DAGCircuit]] = []

    def record(entries: list[tuple[str, DAGCircuit]]) -> Callable:
        return lambda **event: entries.append((event["pass_"].name(), deepcopy(event["dag"])))

    expected = generate_preset_pass_manager(3, target=target, seed_transpiler=seed).run(
        circuit, callback=record(native_trace)
    )
    actual, decisions = editor.run_native_qiskit(circuit, target, seed, callback=record(edited_trace))
    assert native_trace == edited_trace
    assert actual == expected
    assert actual.layout is not None
    assert expected.layout is not None
    assert actual.layout.final_index_layout() == expected.layout.final_index_layout()
    assert actual.layout.initial_index_layout() == expected.layout.initial_index_layout()
    assert len(decisions) == 1
    assert decisions[0]["action"] == "KEEP"
    assert decisions[0]["stage_config"] == {"optimization_level": 3, "seed_transpiler": seed}


def test_keep_retains_native_pass_objects_and_seed(
    monkeypatch: pytest.MonkeyPatch, frozen_inputs: tuple[Inputs, Target]
) -> None:
    """Use the exact pass objects and seed supplied by the native preset."""
    inputs, target = frozen_inputs
    preset = generate_preset_pass_manager(3, target=target, seed_transpiler=7)
    original_init = preset.init
    assert preset.optimization is not None
    original_tasks = preset.optimization.to_flow_controller().tasks

    def factory(level: int, *, target: Target, seed_transpiler: int) -> StagedPassManager:
        assert level == 3
        assert seed_transpiler == 7
        assert target is frozen_inputs[1]
        return preset

    monkeypatch.setattr(editor, "generate_preset_pass_manager", factory)
    editor.run_native_qiskit(inputs.circuit("test/qft_5_indep.qasm"), target, 7)
    wrapped = preset.optimization.to_flow_controller().tasks[0]
    assert preset.init is original_init
    assert all(
        actual is expected for actual, expected in zip(cast("Any", wrapped).native.tasks, original_tasks, strict=True)
    )


def test_ae8_native_keep_does_not_use_modified_atomic_teacher(frozen_inputs: tuple[Inputs, Target]) -> None:
    """AE8 detects the previous teacher's substituted initial synthesis."""
    inputs, target = frozen_inputs
    circuit = inputs.circuit("test/ae_8_indep.qasm")
    actual, _ = editor.run_native_qiskit(circuit, target, 0)
    modified_teacher, _ = demonstration(circuit, target, 0)
    assert circuit_to_dag(actual) != circuit_to_dag(modified_teacher)


@pytest.mark.parametrize(
    ("decisions", "max_edits"),
    [
        (("KEEP",), 0),
        (("KEEP",), 2),
        (("SKIP",), 2),
        (("END",), 2),
        (("INSERT:4", "KEEP"), 2),
        (("INSERT:4", "INSERT:4", "KEEP"), 2),
        (("INSERT:4", "INSERT:4", "INSERT:4", "INSERT:4", "KEEP"), 4),
    ],
)
def test_edits_preserve_unitary_and_measurement_wiring(decisions: tuple[str, ...], max_edits: int) -> None:
    """Optional edits preserve semantics and physical output indices."""
    target = GenericBackendV2(
        4,
        basis_gates=["cx", "rz", "sx", "x"],
        coupling_map=[[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]],
        seed=123,
    ).target
    circuit = QuantumCircuit(4, 4)
    circuit.h(0)
    circuit.cx(0, 3)
    circuit.rz(0.31, 1)
    circuit.cx(1, 2)
    circuit.swap(0, 2)
    circuit.measure([0, 1, 2, 3], [2, 0, 3, 1])
    calls = []
    actions = iter(decisions)

    def policy(state: dict[str, Any]) -> str:
        calls.append(state)
        return next(actions)

    events = []
    compiled, trace = editor.run_native_qiskit(circuit, target, 7, policy, max_edits=max_edits, on_event=events.append)
    assert len(calls) == len(decisions)
    assert calls[0]["compiler"] == "qiskit"
    assert calls[0]["stage_name"] == "optimization"
    assert set(calls[0]["allowed_actions"]) == ({"KEEP", "SKIP", "INSERT:4", "END"} if max_edits else {"KEEP"})
    assert calls[0]["insertion_candidates"][4]["stage_name"] == "optimization"
    assert len(trace) == len(decisions)
    assert [len(state["history"]) for state in calls] == list(range(len(decisions)))
    assert all("history" not in entry for entry in trace)
    assert [event["event"] for event in events] == ["before", "after"] * len(trace)
    for index, entry in enumerate(trace):
        assert entry["action"] == decisions[index]
        assert entry["occurrence"] == (None if decisions[index] in {"SKIP", "END"} else index)
        assert entry["edits_remaining"] == max_edits - index
        if decisions[index] == "INSERT:4":
            assert entry["inserted_stage_index"] == 4
        if index:
            assert entry["before"] == trace[index - 1]["after"]
            previous = calls[index]["history"][-1]
            assert previous["action"] == decisions[index - 1]
            assert previous["inserted_stage_index"] == 4
            assert previous["runtime_seconds"] == trace[index - 1]["runtime_seconds"]
            assert previous["executed"]
        if decisions[index] in {"SKIP", "END"}:
            assert entry["before"] == entry["after"]
            assert entry["runtime_seconds"] == 0
    if len(decisions) == max_edits + 1:
        assert calls[-1]["allowed_actions"] == ["KEEP"]
    compiled_unitary = compiled.remove_final_measurements(inplace=False)
    original_unitary = circuit.remove_final_measurements(inplace=False)
    assert compiled_unitary is not None
    assert original_unitary is not None
    assert Operator.from_circuit(compiled_unitary).equiv(Operator(original_unitary))
    assert compiled.layout is not None
    final_layout = compiled.layout.final_index_layout()
    measurements = {
        compiled.find_bit(item.clbits[0]).index: compiled.find_bit(item.qubits[0]).index
        for item in compiled.data
        if item.operation.name == "measure"
    }
    assert measurements == {2: final_layout[0], 0: final_layout[1], 3: final_layout[2], 1: final_layout[3]}
    for item in compiled.data:
        assert target.instruction_supported(
            operation_name=item.operation.name,
            qargs=tuple(compiled.find_bit(q).index for q in item.qubits),
        )


def test_insert_keeps_controller_property_set_and_reruns_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Insertion reruns existing analyses on the same shared state."""
    properties: list[PropertySet] = []

    class Initialize(AnalysisPass):
        def run(self, dag: DAGCircuit) -> None:
            del dag
            self.property_set["visits"] = 0
            self.property_set["enabled"] = True
            properties.append(self.property_set)

    class Increment(AnalysisPass):
        def run(self, dag: DAGCircuit) -> None:
            del dag
            assert self.property_set is properties[0]
            self.property_set["visits"] += 1

    increment = Increment()
    loop = DoWhileController(increment, do_while=lambda props: props["visits"] < 2)
    native = ConditionalController(loop, condition=itemgetter("enabled"))
    target = GenericBackendV2(2, seed=123).target
    preset = generate_preset_pass_manager(3, target=target, seed_transpiler=0)
    preset.pre_optimization = PassManager(Initialize())
    preset.optimization = PassManager(native)
    monkeypatch.setattr(editor, "generate_preset_pass_manager", lambda *_, **__: preset)
    actions = iter(["INSERT:4", "KEEP"])
    _, trace = editor.run_native_qiskit(QuantumCircuit(2), target, 0, lambda _: next(actions))
    wrapped = preset.optimization.to_flow_controller().tasks[0]
    assert cast("Any", wrapped).native.tasks[0] is native
    assert native.tasks[0] is loop
    assert loop.tasks[0] is increment
    assert len(properties) == 1
    assert properties[0]["visits"] == 3
    assert [entry["occurrence"] for entry in trace] == [0, 1]


@pytest.mark.parametrize("decision", ["STOP", "REPEAT", "INSERT:0", "INSERT:tket"])
def test_invalid_decision_is_rejected(decision: str) -> None:
    """Reject unsupported decisions before executing the native stage."""
    target = GenericBackendV2(2, seed=123).target
    with pytest.raises(ValueError, match="Invalid"):
        editor.run_native_qiskit(QuantumCircuit(2), target, 0, lambda _: decision)


@pytest.mark.parametrize("max_edits", [0, 2, 4])
def test_insert_budget_is_enforced(max_edits: int) -> None:
    """A policy cannot keep inserting after using its configured budget."""
    target = GenericBackendV2(2, seed=123).target
    calls = []

    def policy(state: dict[str, Any]) -> str:
        calls.append(state)
        return "INSERT:4"

    with pytest.raises(ValueError, match="Invalid"):
        editor.run_native_qiskit(QuantumCircuit(2), target, 0, policy, max_edits=max_edits)
    assert len(calls) == max_edits + 1
    assert calls[-1]["allowed_actions"] == ["KEEP"]


def test_end_preserves_native_finalization(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ending optional work still executes the native post-optimization stage."""
    calls = []

    class Record(AnalysisPass):
        def __init__(self, label: str) -> None:
            super().__init__()
            self.label = label

        def run(self, dag: DAGCircuit) -> None:
            del dag
            calls.append(self.label)

    target = GenericBackendV2(2, seed=123).target
    preset = generate_preset_pass_manager(3, target=target, seed_transpiler=0)
    preset.optimization = PassManager(Record("optimization"))
    preset.post_optimization = PassManager(Record("finalization"))
    monkeypatch.setattr(editor, "generate_preset_pass_manager", lambda *_, **__: preset)
    _, trace = editor.run_native_qiskit(QuantumCircuit(2), target, 0, lambda _: "END")
    assert calls == ["finalization"]
    assert len(trace) == 1
    assert trace[0]["action"] == "END"
    assert trace[0]["occurrence"] is None
