# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Edit the optional optimization stage of an otherwise native Qiskit O3 run."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiskit.passmanager.base_tasks import Task
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from .pipeline_actions import PipelineActions

if TYPE_CHECKING:
    from collections.abc import Callable

    from qiskit import QuantumCircuit
    from qiskit.dagcircuit import DAGCircuit
    from qiskit.passmanager.compilation_status import PassManagerState
    from qiskit.transpiler import Target

    from .pipeline_actions import Policy


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
        self.actions = PipelineActions(policy)
        self.trace = trace
        self.seed = seed
        self.on_event = on_event

    def execute(
        self,
        passmanager_ir: DAGCircuit,
        state: PassManagerState,
        callback: Callable | None = None,
    ) -> tuple[DAGCircuit, PassManagerState]:
        """Apply bounded edits without rebuilding the native controller."""
        invocation = 0
        stage = {
            "stage_name": "optimization",
            "stage_config": {"optimization_level": 3, "seed_transpiler": self.seed},
        }
        while True:
            policy_state = {"compiler": "qiskit", "stage_index": 4, **stage, "before": features(passmanager_ir)}
            decision = self.actions.decide(policy_state, optional=True, insertions={4: stage}, can_end=True)
            execute = decision not in {"SKIP", "END"}
            entry = {
                **policy_state,
                "action": decision,
                "occurrence": invocation if execute else None,
            }
            if decision.startswith("INSERT:"):
                entry["inserted_stage_index"] = 4
            if self.on_event is not None:
                self.on_event({**entry, "event": "before"})
            if execute:
                if invocation:
                    # Each invocation must rerun analyses as well as transformations.
                    state.workflow_status.completed_passes.clear()
                passmanager_ir, state = self.native.execute(passmanager_ir, state, callback)
                invocation += 1
            entry["after"] = features(passmanager_ir)
            self.trace.append(entry)
            if self.on_event is not None:
                self.on_event({**entry, "event": "after"})
            if not decision.startswith("INSERT:"):
                break
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
    """Run native O3 with KEEP/SKIP/INSERT/END at the optimization boundary.

    INSERT:4 runs the original optimization controller before the next decision.
    END skips optimization while retaining native finalization. The shared DAG,
    property set, native passes, and conditional loops remain intact.
    """
    preset = generate_preset_pass_manager(3, target=target, seed_transpiler=seed)
    trace: list[dict[str, Any]] = []
    if preset.optimization is not None:
        native = preset.optimization.to_flow_controller()
        preset.optimization = PassManager([_OptimizationDecision(native, policy, trace, seed=seed, on_event=on_event)])
    return preset.run(circuit, callback=callback), trace
