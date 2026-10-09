# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Edit optional top-level stages of BQSKit's native circuit workflow."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from bqskit.compiler.compile import build_workflow

from .pipeline_actions import PipelineActions

if TYPE_CHECKING:
    from bqskit.compiler.compiler import Compiler
    from bqskit.compiler.machine import MachineModel
    from bqskit.compiler.passdata import PassData
    from bqskit.compiler.workflow import Workflow
    from bqskit.ir.circuit import Circuit

    from .pipeline_actions import Policy

OPTIONAL_STAGES = frozenset({
    "Gate Deletion Optimization",
    "Iterative Resynthesis Optimization",
    "Iterative Gate Deletion Optimization",
})
EventCallback = Callable[[dict[str, Any]], None]


def features(circuit: Circuit, data: PassData | None = None) -> dict[str, Any]:
    """Read counts and maps without scoring or converting the circuit."""
    result: dict[str, Any] = {
        "qubits": circuit.num_qudits,
        "depth": circuit.num_cycles,
        "operations": circuit.num_operations,
        "gate_counts": {gate.name: count for gate, count in circuit.gate_counts.items()},
        "two_qubit_gates": sum(count for gate, count in circuit.gate_counts.items() if gate.num_qudits == 2),
    }
    if data is not None:
        result.update(
            placement=list(data.placement),
            initial_mapping=list(data.initial_mapping),
            final_mapping=list(data.final_mapping),
        )
    return result


def run_bqskit(
    circuit: Circuit,
    workflow: Workflow,
    compiler: Compiler,
    policy: Policy | None = None,
    on_event: EventCallback | None = None,
) -> tuple[Circuit, PassData, list[dict[str, Any]]]:
    """Keep native nested workflows and carry their full PassData across stages.

    INSERT runs a configured native stage without advancing the cursor.
    END skips optional work but preserves required native finalization.
    The client receives every stage boundary;
    policy functions and watchdog connections never enter distributed workers.
    Native synthesis can return equivalent Euler angles across repeated runs;
    KEEP preserves its gate structure and numerical behavior, not byte identity.
    """
    current = circuit.copy()
    data: PassData | None = None
    trace: list[dict[str, Any]] = []
    actions = PipelineActions(policy)
    optional = {index: stage for index, stage in enumerate(workflow) if stage.name in OPTIONAL_STAGES}
    single_retarget = next(index for index, stage in enumerate(workflow) if stage.name == "Single Qudit Retargeting")
    finalization = next(index for index, stage in enumerate(workflow) if stage.name == "LogErrorPass")
    for index, stage in enumerate(workflow):
        # Resynthesis can introduce non-native single-qudit gates. Deletion
        # retains the native gate set; both run before placement/measurements.
        insertions = {
            candidate: {"stage_name": insertion.name}
            for candidate, insertion in optional.items()
            if (
                candidate <= index <= single_retarget
                if insertion.name == "Iterative Resynthesis Optimization"
                else single_retarget < index <= finalization
            )
        }
        occurrence = 0
        while True:
            before = features(current, data)
            state = {
                "compiler": "bqskit",
                "stage_index": index,
                "stage_name": stage.name,
                "before": before,
            }
            action = actions.decide(
                state, optional=index in optional, insertions=insertions, can_end=any(i >= index for i in optional)
            )
            insertion = int(action.split(":")[1]) if action.startswith("INSERT:") else None
            executed_stage = workflow[insertion] if insertion is not None else stage
            skip = action == "SKIP" or (action == "END" and index in optional)
            entry = {
                **state,
                "stage_name": executed_stage.name,
                "action": action,
                "occurrence": None if skip else occurrence,
            }
            if insertion is not None:
                entry["inserted_stage_index"] = insertion
            if on_event is not None:
                on_event({**entry, "event": "before"})
            started = time.monotonic()
            if not skip:
                # The list preserves the outer Workflow's per-stage RNG reset,
                # including when this stage is itself a nested Workflow.
                current, data = compiler.compile(current, [executed_stage], request_data=True, data=data)
            after = features(current, data)
            entry.update(after=after, runtime_seconds=0.0 if skip else time.monotonic() - started)
            trace.append(entry)
            if on_event is not None:
                on_event({**entry, "event": "after"})
            if insertion is None:
                break
            occurrence += 1
    if data is None:
        msg = "The native workflow must contain a required stage."
        raise ValueError(msg)
    return current, data, trace


def compile_bqskit(
    circuit: Circuit,
    model: MachineModel,
    compiler: Compiler,
    policy: Policy | None = None,
    *,
    optimization_level: int = 1,
    max_synthesis_size: int = 3,
    synthesis_epsilon: float = 1e-8,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
    seed: int = 0,
    on_event: EventCallback | None = None,
) -> tuple[Circuit, PassData, list[dict[str, Any]]]:
    """Use native level 1/2/3 defaults, including measurement and mapping passes."""
    if optimization_level not in (1, 2, 3):
        msg = "This prototype supports native BQSKit levels 1, 2, and 3."
        raise ValueError(msg)
    workflow = build_workflow(
        circuit,
        model,
        optimization_level=optimization_level,
        max_synthesis_size=max_synthesis_size,
        synthesis_epsilon=synthesis_epsilon,
        error_threshold=error_threshold,
        error_sim_size=error_sim_size,
        seed=seed,
    )
    return run_bqskit(circuit, workflow, compiler, policy, on_event)
