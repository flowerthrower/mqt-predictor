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

if TYPE_CHECKING:
    from bqskit.compiler.compiler import Compiler
    from bqskit.compiler.machine import MachineModel
    from bqskit.compiler.passdata import PassData
    from bqskit.compiler.workflow import Workflow
    from bqskit.ir.circuit import Circuit

OPTIONAL_STAGES = frozenset({"Gate Deletion Optimization"})
Policy = Callable[[dict[str, Any]], str]
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

    REPEAT adds one stage invocation. The client receives every stage boundary;
    policy functions and watchdog connections never enter distributed workers.
    Native synthesis can return equivalent Euler angles across repeated runs;
    KEEP preserves its gate structure and numerical behavior, not byte identity.
    """
    current = circuit.copy()
    data: PassData | None = None
    trace: list[dict[str, Any]] = []
    for index, stage in enumerate(workflow):
        before = features(current, data)
        state = {
            "compiler": "bqskit",
            "stage_index": index,
            "stage_name": stage.name,
            "allowed_actions": ["KEEP", "SKIP", "REPEAT"] if stage.name in OPTIONAL_STAGES else ["KEEP"],
            "before": before,
        }
        action = "KEEP" if policy is None else policy(state)
        if action not in state["allowed_actions"]:
            msg = f"Invalid {action!r} for BQSKit stage {index}:{stage.name}."
            raise ValueError(msg)
        invocations = {"SKIP": 0, "KEEP": 1, "REPEAT": 2}[action]
        for occurrence in range(invocations):
            entry = {**state, "action": action, "occurrence": occurrence, "before": before}
            if on_event is not None:
                on_event({**entry, "event": "before"})
            started = time.monotonic()
            # The list preserves the outer Workflow's per-stage RNG reset,
            # including when this stage is itself a nested Workflow.
            current, data = compiler.compile(current, [stage], request_data=True, data=data)
            after = features(current, data)
            entry.update(after=after, runtime_seconds=time.monotonic() - started)
            trace.append(entry)
            if on_event is not None:
                on_event({**entry, "event": "after"})
            before = after
        if not invocations:
            entry = {
                **state,
                "action": action,
                "occurrence": None,
                "before": before,
                "after": before,
                "runtime_seconds": 0.0,
            }
            trace.append(entry)
            if on_event is not None:
                on_event({**entry, "event": "before"})
                on_event({**entry, "event": "after"})
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
