# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Experiment environments with common inputs/actions and distinct learning semantics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete

from mqt.predictor.rl.predictorenv import PredictorEnv
from mqt.predictor.utils import calc_supermarq_features

if TYPE_CHECKING:
    from pathlib import Path

    from qiskit import QuantumCircuit
    from qiskit.transpiler import Target

    from .inputs import Inputs
    from .worker import CompilerWorker

LEGACY_FEATURES = ("program_communication", "critical_depth", "entanglement_ratio", "parallelism", "liveness")


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
            intermediate_reward=mode == "paper",
            reward_scale=settings["reward_scale"],
            no_effect_penalty=settings["no_effect_penalty"],
        )
        self.inputs = inputs
        self.worker = worker
        self.mode = mode
        self.trace: list[dict[str, Any]] = []
        self.last_result: dict[str, Any] = {}
        self.worker_startup_seconds = 0.0

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
        self.worker_startup_seconds = 0.0
        return super().reset(qc, seed=None, options=options)

    def apply_action(self, action_index: int) -> QuantumCircuit:
        """Preserve current action implementations and layout bookkeeping."""
        result = self.worker.run({
            "compiler": self.mode,
            "circuit": self.state,
            "layout": self.layout,
            "input_qubits": self.num_qubits_uncompiled_circuit,
            "action": action_index,
            "seed": int(self.np_random.integers(0, np.iinfo(np.int32).max)),
        })
        self.last_result = result
        self.worker_startup_seconds += result.get("worker_startup_seconds", 0.0)
        for entry in result["traces"]:
            entry["index"] = len(self.trace)
            self.trace.append(entry)
        if result["status"] != "ok":
            raise RuntimeError(result["error"])
        self.layout = result["circuit"].layout
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
