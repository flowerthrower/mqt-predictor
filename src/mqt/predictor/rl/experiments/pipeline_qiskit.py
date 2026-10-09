# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Edit the optional optimization stage of an otherwise native Qiskit O3 run."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from qiskit.passmanager.base_tasks import Task
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.dagcircuit import DAGCircuit
    from qiskit.passmanager.compilation_status import PassManagerState
    from qiskit.transpiler import Target


Policy = Callable[[dict[str, Any]], str]


def features(dag: DAGCircuit) -> dict[str, Any]:
    """Read plain counts from the current DAG without changing its representation."""
    nodes = dag.gate_nodes()
    return {
        "qubits": dag.num_qubits(),
        "active_qubits": len({q for node in nodes for q in node.qargs}),
        "depth": dag.depth(),
        "gates": len(nodes),
        "gate_counts": dict(dag.count_ops()),
        "one_qubit_gates": sum(len(node.qargs) == 1 for node in nodes),
        "two_qubit_gates": sum(len(node.qargs) == 2 for node in nodes),
    }


class _OptimizationDecision(Task):
    """Keep the native controller, DAG, and property set at the stage boundary."""

    def __init__(
        self,
        native: Task,
        policy: Policy | None,
        trace: list[dict[str, Any]],
        *,
        seed: int = 0,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Retain the native stage and its policy and observation hooks."""
        self.native = native
        self.policy = policy
        self.trace = trace
        self.seed = seed
        self.on_event = on_event

    def execute(
        self,
        passmanager_ir: DAGCircuit,
        state: PassManagerState,
        callback: Callable | None = None,
    ) -> tuple[DAGCircuit, PassManagerState]:
        """Choose the number of executions without rebuilding the native controller."""
        before = features(passmanager_ir)
        policy_state = {
            "compiler": "qiskit",
            "stage_index": 4,
            "stage_name": "optimization",
            "stage_config": {"optimization_level": 3, "seed_transpiler": self.seed},
            "allowed_actions": ["KEEP", "SKIP", "REPEAT"],
            "before": before,
        }
        decision = "KEEP" if self.policy is None else self.policy(policy_state)
        executions = {"SKIP": 0, "KEEP": 1, "REPEAT": 2}
        if decision not in executions:
            msg = f"Unknown optimization decision: {decision!r}"
            raise ValueError(msg)
        for index in range(max(1, executions[decision])):
            entry = {
                **policy_state,
                "action": decision,
                "occurrence": index if executions[decision] else None,
                "before": before,
            }
            if self.on_event is not None:
                self.on_event({**entry, "event": "before"})
            if index:
                # A requested repeat must rerun analyses as well as transformations.
                state.workflow_status.completed_passes.clear()
            if executions[decision]:
                passmanager_ir, state = self.native.execute(passmanager_ir, state, callback)
            entry["after"] = features(passmanager_ir)
            self.trace.append(entry)
            if self.on_event is not None:
                self.on_event({**entry, "event": "after"})
            before = entry["after"]
        return passmanager_ir, state


def run_native_qiskit(
    circuit: QuantumCircuit,
    target: Target,
    seed: int,
    policy: Policy | None = None,
    *,
    callback: Callable | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[QuantumCircuit, list[dict[str, Any]]]:
    """Run native O3 with one KEEP/SKIP/REPEAT decision before optimization.

    KEEP executes the original stage once. REPEAT executes that same controller
    twice on the evolving DAG and property set, invalidating the completed-pass
    cache between executions. Neither choice rewrites native passes or loops.
    """
    preset = generate_preset_pass_manager(3, target=target, seed_transpiler=seed)
    trace: list[dict[str, Any]] = []
    if preset.optimization is not None:
        native = preset.optimization.to_flow_controller()
        preset.optimization = PassManager([_OptimizationDecision(native, policy, trace, seed=seed, on_event=on_event)])
    return preset.run(circuit, callback=callback), trace
