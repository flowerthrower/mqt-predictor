# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Experiment environments with common inputs/actions and distinct learning semantics."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Any

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete
from qiskit.transpiler.passes import (
    Collect2qBlocks,
    ConsolidateBlocks,
    Optimize1qGatesDecomposition,
    UnitarySynthesis,
)

from mqt.predictor.rl.actions.base import CompilationOrigin, DeferredDeviceAction, PassType
from mqt.predictor.rl.predictorenv import PredictorEnv
from mqt.predictor.utils import calc_supermarq_features

from .qiskit_teacher import NATIVE_ACTIONS, native_passes

if TYPE_CHECKING:
    from pathlib import Path

    from qiskit import QuantumCircuit
    from qiskit.transpiler import Target

    from .inputs import Inputs
    from .worker import CompilerWorker

LEGACY_FEATURES = ("program_communication", "critical_depth", "entanglement_ratio", "parallelism", "liveness")


def configure_actions(env: PredictorEnv) -> None:
    """Expose individual canonical SDK actions with the settings needed to replay O3."""
    env.action_set = {index: copy(action) for index, action in env.action_set.items()}
    index = next(index for index, action in env.action_set.items() if action.name == "QiskitO3")
    env.action_set[index] = DeferredDeviceAction(
        "Optimize1qGatesDecomposition_preserve",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        lambda device: [Optimize1qGatesDecomposition(basis=device.operation_names)],
        preserves_layout=True,
        preserves_routing=True,
        preserves_synthesis=True,
    )
    index = env.action_terminate_index
    env.action_set[index + 1] = env.action_set[index]
    env.action_set[index] = DeferredDeviceAction(
        "Opt2qBlocks_preserve",
        CompilationOrigin.QISKIT,
        PassType.OPT,
        lambda device: [
            Collect2qBlocks(),
            ConsolidateBlocks(basis_gates=device.operation_names),
            UnitarySynthesis(basis_gates=device.operation_names, approximation_degree=1.0),
        ],
        preserves_layout=True,
        preserves_routing=True,
        preserves_synthesis=True,
    )
    env.action_terminate_index = index + 1
    for name in ("ConsolidateBlocks", "TwoQubitPeepholeOptimization", "VF2PostLayout_2q"):
        env.action_set[env.action_terminate_index + 1] = env.action_set[env.action_terminate_index]
        env.action_set[env.action_terminate_index] = DeferredDeviceAction(
            name,
            CompilationOrigin.QISKIT,
            PassType.FINAL_OPT if name == "VF2PostLayout_2q" else PassType.OPT,
            None,
            preserves_layout=True,
            preserves_routing=True,
            preserves_synthesis=name == "TwoQubitPeepholeOptimization",
        )
        env.action_terminate_index += 1
    for action in env.action_set.values():
        if action.name in NATIVE_ACTIONS:
            action.transpile_pass = lambda device, name=action.name: native_passes(
                name, device, 0, physical=name == "TwoQubitPeepholeOptimization"
            )
        if action.name == "ElidePermutations":
            action.pass_type = PassType.OPT
    for pass_type, attribute in (
        (PassType.OPT, "actions_opt_indices"),
        (PassType.LAYOUT, "actions_layout_indices"),
        (PassType.FINAL_OPT, "actions_final_optimization_indices"),
    ):
        setattr(env, attribute, [i for i, action in env.action_set.items() if action.pass_type == pass_type])
    env.actions_structure_preserving_indices = [
        i
        for i, action in env.action_set.items()
        if action.pass_type == PassType.OPT
        and action.preserves_layout
        and action.preserves_routing
        and action.preserves_synthesis
    ]
    env.action_space = Discrete(len(env.action_set))


class ExperimentEnv(PredictorEnv):
    """Reuse current paper semantics and execute the shared registry in a worker."""

    def __init__(
        self, device: Target, inputs: Inputs, worker: CompilerWorker, settings: dict[str, Any], mode: str
    ) -> None:
        """Bind shared inputs and worker to the selected learning method."""
        super().__init__(
            device,
            reward_function=settings["objective"],
            max_steps=settings["episode_actions"],
            mdp="v2" if mode == "original" else "v3",
            intermediate_reward=False,
            reward_scale=settings["reward_scale"],
            no_effect_penalty=settings["no_effect_penalty"],
        )
        configure_actions(self)
        self.inputs = inputs
        self.worker = worker
        self.mode = mode
        self.trace: list[dict[str, Any]] = []
        self.last_result: dict[str, Any] = {}
        self.worker_startup_seconds = 0.0
        self.compiler_seed = 0
        self.qiskit_properties: dict[str, Any] = {}
        self.no_effect_actions: set[int] = set()
        self.episode = 0

    def reset(
        self,
        qc: Path | str | QuantumCircuit | None = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load historical files explicitly and sample reproducibly."""
        Env.reset(self, seed=seed)
        if qc is None:
            names = self.inputs.names("train")
            qc = self.inputs.circuit(names[int(self.np_random.integers(len(names)))])
        self.trace = []
        self.last_result = {}
        self.no_effect_actions.clear()
        self.worker_startup_seconds = 0.0
        self.compiler_seed = seed if seed is not None else int(self.np_random.integers(0, np.iinfo(np.int32).max))
        observation, info = super().reset(qc, seed=None, options=options)
        self.qiskit_properties = {
            "original_qubit_indices": {qubit: i for i, qubit in enumerate(self.state.qubits)},
            "num_input_qubits": self.state.num_qubits,
        }
        self.episode += 1
        return observation, info

    def action_masks(self) -> list[bool]:
        """Apply SDK preconditions and suppress observed canonical no-ops in paper mode."""
        masks = super().action_masks()
        for index, action in self.action_set.items():
            if action.name == "ElidePermutations":
                masks[index] = masks[index] and self.layout is None
            elif action.name == "TwoQubitPeepholeOptimization":
                masks[index] = masks[index] and self.layout is not None and self.is_circuit_synthesized(self.state)
            elif action.name == "VF2PostLayout_2q" and self.mode == "paper":
                masks[index] = self._current_laid_out and self._current_routed
            elif self.mode == "paper" and action.origin == CompilationOrigin.TKET:
                masks[index] |= index in self.valid_actions and index in self.actions_structure_preserving_indices
            if self.mode == "paper" and index in self.no_effect_actions:
                masks[index] = False
        return masks

    def _get_stepwise_reward(self) -> tuple[float, str]:
        """The existing basis-translation proxy cannot score consolidated unitary blocks."""
        if self.state.count_ops().get("unitary"):
            return 0.0, "unavailable"
        return super()._get_stepwise_reward()

    def apply_action(self, action_index: int) -> QuantumCircuit:
        """Preserve current action implementations and layout bookkeeping."""
        result = self.worker.run({
            "compiler": self.mode,
            "circuit": self.state,
            "layout": self.layout,
            "input_qubits": self.num_qubits_uncompiled_circuit,
            "action": action_index,
            "seed": self.compiler_seed,
            "qiskit_properties": self.qiskit_properties,
            "episode": self.episode,
        })
        self.last_result = result
        self.worker_startup_seconds += result.get("worker_startup_seconds", 0.0)
        for entry in result["traces"]:
            entry["index"] = len(self.trace)
            self.trace.append(entry)
        if result["status"] != "ok":
            raise RuntimeError(result["error"])
        properties = result.pop("qiskit_properties", self.qiskit_properties)
        if self.mode == "paper":
            # Native Qiskit actions reuse one compilation seed throughout the episode.
            if (
                self.action_set[action_index].name not in NATIVE_ACTIONS
                or self.state != result["circuit"]
                or self.layout != result["circuit"].layout
                or self.qiskit_properties != properties
            ):
                self.no_effect_actions.clear()
            else:
                self.no_effect_actions.add(action_index)
        self.layout = result["circuit"].layout
        self.qiskit_properties = properties
        return result["circuit"]

    def close(self) -> None:
        """Release compiler processes."""
        self.worker.close()


class OriginalEnv(ExperimentEnv):
    """Port v2.0.0's seven features, masks, sparse rewards and failure endings."""

    def __init__(
        self, device: Target, inputs: Inputs, worker: CompilerWorker, settings: dict[str, Any], mode: str
    ) -> None:
        """Restore the original discrete observation encoding."""
        super().__init__(device, inputs, worker, settings, mode)
        self.observation_space = Dict({
            "num_qubits": Discrete(max(128, self.device.num_qubits + 1)),
            "depth": Discrete(1_000_000),
            **{name: Box(0, 1, shape=(1,), dtype=np.float32) for name in LEGACY_FEATURES},
        })

    def _get_observation(self) -> dict[str, Any]:
        features = calc_supermarq_features(self.state)
        return {
            "num_qubits": self.state.num_qubits,
            "depth": self.state.depth(),
            **{name: np.array([getattr(features, name)], dtype=np.float32) for name in LEGACY_FEATURES},
        }

    def _valid_actions_v2(self, synthesized: bool, laid_out: bool, routed: bool) -> list[int]:
        # These priorities are from v2.0.0, before the expanded v2 transition table.
        if not synthesized:
            return self.actions_synthesis_indices + self.actions_opt_indices
        if laid_out and routed:
            return [self.action_terminate_index, *self.actions_opt_indices]
        if laid_out:
            return self.actions_routing_indices
        return self.actions_mapping_indices + self.actions_layout_indices + self.actions_opt_indices

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Only explicit termination earns ESP; the new cap is external truncation."""
        self.num_steps += 1
        self.used_actions.append(self.action_set[action].name)
        terminated = action == self.action_terminate_index
        try:
            self.state = self.apply_action(action)
            self.state._layout = self.layout  # ruff: ignore[private-member-access]
            self._current_foms = {}
            self.valid_actions = self.determine_valid_actions_for_state()
            reward = self.calculate_reward() if terminated else 0.0
        except Exception as error:  # ruff: ignore[blind-except]
            self.error_occurred = True
            return self._get_observation(), 0.0, True, False, {"termination_reason": "pass_error", "error": str(error)}
        truncated = not terminated and self.max_steps is not None and self.num_steps >= self.max_steps
        info = {"termination_reason": "max_steps_exceeded"} if truncated else {}
        return self._get_observation(), reward, terminated, truncated, info
