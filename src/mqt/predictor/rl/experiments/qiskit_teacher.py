# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Canonical Qiskit actions and accepted O3 demonstrations for SCASIA."""

from __future__ import annotations

import time
from copy import deepcopy
from io import BytesIO
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from qiskit import qpy
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.passmanager import ConditionalController, DoWhileController, FlowControllerLinear, PropertySet
from qiskit.passmanager.compilation_status import PassManagerState, WorkflowStatus
from qiskit.transpiler import PassManager, TranspileLayout
from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.passes import VF2PostLayout
from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
from qiskit.transpiler.preset_passmanagers import common, generate_preset_pass_manager
from torch.nn.functional import mse_loss

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from qiskit import QuantumCircuit
    from qiskit.dagcircuit import DAGCircuit
    from qiskit.passmanager.base_tasks import Task
    from qiskit.transpiler import Target

    from mqt.predictor.rl.gnn import (
        GNNMaskableMultiInputActorCriticPolicy,
        GNNMaskablePPO,
        GNNObservationWrapper,
        GraphData,
    )

    from .environment import ExperimentEnv


NATIVE_ACTIONS = frozenset({
    "ElidePermutations",
    "ConsolidateBlocks",
    "TwoQubitPeepholeOptimization",
    "VF2Layout",
    "QiskitSabreMapping",
    "VF2PostLayout",
    "VF2PostLayout_2q",
    "BasisTranslator",
    "Optimize1qGatesDecomposition_preserve",
    "CommutativeCancellation",
    "RemoveIdentityEquivalent",
    "InverseCancellation",
    "RemoveDiagonalGatesBeforeMeasure",
})
LAYOUT_PROPERTIES = (
    "layout",
    "final_layout",
    "original_qubit_indices",
    "num_input_qubits",
    "original_layout",
    "virtual_permutation_layout",
)


def passes(task: Task) -> Iterator[BasePass]:
    """Read configured SDK passes through its public flow-controller tree."""
    if isinstance(task, BasePass):
        yield task
    else:
        assert isinstance(task, (FlowControllerLinear, ConditionalController, DoWhileController))
        for child in task.tasks:
            yield from passes(child)


def native_passes(name: str, target: Target, seed: int, *, physical: bool) -> list[Task]:
    """Use O3's pass parameters without executing its selection or optimization loop."""
    preset = generate_preset_pass_manager(3, target=target, seed_transpiler=seed)

    def get(stage: str, pass_name: str) -> BasePass:
        return next(p for p in passes(getattr(preset, stage).to_flow_controller()) if p.name() == pass_name)

    if name == "BasisTranslator":
        return [get("routing", "FilterOpNodes"), *preset.translation.to_flow_controller().tasks]
    if name == "QiskitSabreMapping":
        return [
            get("layout", "BarrierBeforeFinalMeasurements"),
            get("layout", "SabreLayout"),
        ]
    if name == "VF2Layout":
        return [
            get("layout", name),
            ConditionalController(
                common.generate_embed_passmanager(target).to_flow_controller(),
                condition=lambda props: props["VF2Layout_stop_reason"] == VF2LayoutStopReason.SOLUTION_FOUND,
            ),
        ]
    if name in {"VF2PostLayout", "VF2PostLayout_2q"}:
        return [
            get("routing" if name == "VF2PostLayout_2q" else "optimization", "VF2PostLayout"),
            ConditionalController(
                get("layout", "ApplyLayout"), condition=lambda props: props["post_layout"] is not None
            ),
            get("routing", "FilterOpNodes"),
        ]
    if name == "Optimize1qGatesDecomposition_preserve":
        return [get("optimization", "Optimize1qGatesDecomposition")]
    stage = (
        "optimization"
        if physical
        and name
        in {
            "TwoQubitPeepholeOptimization",
            "CommutativeCancellation",
            "RemoveIdentityEquivalent",
        }
        else "init"
    )
    return [get(stage, name)]


def apply_native_action(
    name: str,
    circuit: QuantumCircuit,
    target: Target,
    seed: int,
    properties: dict[str, Any],
    dag: DAGCircuit | None,
    *,
    physical: bool,
) -> tuple[QuantumCircuit, TranspileLayout | None, dict[str, Any], DAGCircuit]:
    """Carry virtual permutations between separate canonical pass invocations."""
    state = PassManagerState(WorkflowStatus(), PropertySet(properties))
    controller = PassManager(native_passes(name, target, seed, physical=physical)).to_flow_controller()
    dag, state = controller.execute(dag if dag is not None else circuit_to_dag(circuit), state)
    output = dag_to_circuit(dag)
    layout = TranspileLayout.from_property_set(dag, state.property_set) if state.property_set["layout"] else None
    output._layout = layout  # ruff: ignore[private-member-access]
    return (
        output,
        layout,
        {key: state.property_set[key] for key in LAYOUT_PROPERTIES if state.property_set[key] is not None},
        dag,
    )


def demonstration(circuit: QuantumCircuit, target: Target, seed: int) -> tuple[QuantumCircuit, list[str]]:
    """Retain O3's accepted changing passes, removing its no-ops and rolled-back suffixes."""
    actions: list[str] = []
    layout_action = "VF2Layout"

    def exact(dag: DAGCircuit) -> bytes:
        stream = BytesIO()
        qpy.dump(dag_to_circuit(dag), stream)
        return stream.getvalue()

    keys = [exact(circuit_to_dag(circuit))]

    def record(pass_: BasePass, dag: DAGCircuit, **_: object) -> None:
        nonlocal layout_action
        name = pass_.name()
        if isinstance(pass_, VF2PostLayout):
            layout_action = "VF2PostLayout_2q" if not pass_.strict_direction else name
        elif name == "VF2Layout":
            layout_action = name
        current = deepcopy(dag)
        for node in current.op_nodes():
            if node.op.label == "qiskit.transpiler.internal.routing.protection.barrier":
                current.remove_op_node(node)
        key = exact(current)
        if name == "MinimumPoint" and key != keys[-1]:
            index = keys.index(key)
            del actions[index:]
            del keys[index + 1 :]
            return
        if name == "EnlargeWithAncilla":
            return
        if name == "BasisTranslator" and actions and actions[-1] == "BasisTranslator":
            keys[-1] = key
            return
        if key == keys[-1]:
            return
        action = {
            "ApplyLayout": layout_action,
            "SabreLayout": "QiskitSabreMapping",
            "UnitarySynthesis": "BasisTranslator",
            "HighLevelSynthesis": "BasisTranslator",
            "Optimize1qGatesDecomposition": "Optimize1qGatesDecomposition_preserve",
        }.get(name, name)
        if action not in NATIVE_ACTIONS:
            msg = f"Unrepresented O3 transformation: {name}"
            raise ValueError(msg)
        actions.append(action)
        keys.append(key)

    output = generate_preset_pass_manager(3, target=target, seed_transpiler=seed).run(circuit, callback=record)
    return output, [*actions, "terminate"]


def collect_demonstrations(
    wrapped: GNNObservationWrapper,
    seed: int,
    gamma: float,
) -> tuple[list[tuple[GraphData, list[bool], int, float]], dict[str, Any]]:
    """Replay every training demonstration through the real environment before fitting."""
    env = cast("ExperimentEnv", wrapped.unwrapped)
    assert env.max_steps is not None
    samples = []
    rows = []
    started = time.monotonic()
    for name in env.inputs.names("train"):
        circuit = env.inputs.circuit(name)
        reference = env.worker.run({"compiler": "teacher", "circuit": circuit, "seed": seed})
        if reference["status"] != "ok":
            msg = f"O3 teacher failed for {name}: {reference.get('error')}"
            raise RuntimeError(msg)
        actions = reference["teacher_actions"]
        if len(actions) > env.max_steps:
            msg = f"O3 teacher needs {len(actions)} actions for {name}; episode limit is {env.max_steps}"
            raise ValueError(msg)
        wrapped.reset(circuit, seed=seed)
        episode = []
        for action_name in actions:
            action = next(i for i, entry in env.action_set.items() if entry.name == action_name)
            masks = wrapped.action_masks()
            if not masks[action]:
                msg = f"O3 teacher action {action_name} is masked for {name}"
                raise ValueError(msg)
            observation = wrapped.graph_observation
            _, reward, terminated, truncated, info = wrapped.step(action)
            if env.last_result["status"] != "ok" or truncated or terminated != (action_name == "terminate"):
                msg = f"O3 replay failed for {name} at {action_name}: {info}"
                raise RuntimeError(msg)
            episode.append((observation, masks, action, reward))
        expected = reference["circuit"]
        assert env.layout is not None
        if (
            circuit_to_dag(env.state) != circuit_to_dag(expected)
            or env.layout.final_index_layout() != expected.layout.final_index_layout()
            or env.last_result["score"]["esp"] != reference["score"]["esp"]
        ):
            msg = f"O3 replay differs from the native pipeline for {name}"
            raise ValueError(msg)
        total = 0.0
        returns = []
        for _, _, _, reward in reversed(episode):
            total = reward + gamma * total
            returns.append(total)
        samples.extend(
            (graph, mask, action, value)
            for (graph, mask, action, _), value in zip(episode, reversed(returns), strict=True)
        )
        rows.append({
            "circuit": name,
            "seed": seed,
            "actions": actions,
            "esp": reference["score"]["esp"],
            "native_pass_invocations": len(reference["traces"]),
            "native_runtime_seconds": reference["runtime_seconds"],
        })
        print(f"O3 teacher replay {len(rows)}/{len(env.inputs.names('train'))}: {name}", flush=True)
    return samples, {"circuits": rows, "transitions": len(samples), "runtime_seconds": time.monotonic() - started}


def fit_demonstrations(
    model: GNNMaskablePPO,
    samples: list[tuple[GraphData, list[bool], int, float]],
    settings: dict[str, int],
    on_epoch: Callable[[dict[str, Any]], None],
) -> None:
    """Fit masked action labels and demonstration returns, then hand the policy to PPO."""
    completed = model.__dict__.get("scasia_warmstart_epochs", 0)
    assert model.seed is not None
    policy = cast("GNNMaskableMultiInputActorCriticPolicy", model.policy)
    policy.set_training_mode(True)
    for epoch in range(completed, settings["epochs"]):
        indices = np.random.default_rng(model.seed + epoch).permutation(len(samples))
        loss_sum = 0.0
        correct = 0
        for start in range(0, len(indices), settings["batch_size"]):
            batch = [samples[index] for index in indices[start : start + settings["batch_size"]]]
            observations, _ = policy.obs_to_tensor([sample[0] for sample in batch])
            masks = np.asarray([sample[1] for sample in batch], dtype=bool)
            actions = torch.as_tensor([sample[2] for sample in batch], device=model.device)
            returns = torch.as_tensor([sample[3] for sample in batch], dtype=torch.float32, device=model.device)
            distribution = policy.get_distribution(observations, action_masks=masks)
            log_prob = distribution.log_prob(actions)
            values = policy.predict_values(observations)
            loss = -log_prob.mean() + model.vf_coef * mse_loss(values.flatten(), returns)
            model.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
            model.policy.optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
            with torch.no_grad():
                predicted = distribution.get_actions(deterministic=True)
                correct += int((predicted == actions).sum())
        model.__dict__["scasia_warmstart_epochs"] = epoch + 1
        on_epoch({"epoch": epoch + 1, "loss": loss_sum / len(samples), "action_accuracy": correct / len(samples)})
