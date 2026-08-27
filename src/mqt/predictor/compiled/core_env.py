# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Core-only reinforcement-learning environment for pass ordering."""

from __future__ import annotations

import hashlib
import importlib
import math
from dataclasses import asdict
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

from .native_rl import PASS_PENALTY, CompileMetrics
from .policy import ACTION_NAMES, FEATURE_NAMES, MAX_PASSES

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from qiskit import QuantumCircuit


NUM_TRANSFORM_ACTIONS = len(ACTION_NAMES) - 1
TERMINATE_ACTION = len(ACTION_NAMES) - 1
DEPTH_NORMALIZATION_MAX = 1_000_000.0


class _QCProgram(Protocol):
    """Subset of the Core QC program API used by the environment."""

    def to_qco(self) -> _QCOProgram: ...

    def to_qiskit(self, *, target: object | None = None) -> QuantumCircuit: ...


class _QCOProgram(Protocol):
    """Subset of the Core QCO program API used by the environment."""

    @property
    def ir(self) -> str: ...

    def copy(self) -> _QCOProgram: ...

    def cleanup(self) -> None: ...

    def merge_single_qubit_rotation_gates(self) -> None: ...

    def fuse_single_qubit_unitary_runs(self, *, basis: str = "zyz") -> None: ...

    def decompose_multi_controlled(self) -> None: ...

    def lift_hadamards(self) -> None: ...

    def compile_for_target(self, target: object) -> None: ...

    def to_qc(self) -> _QCProgram: ...


class _QCProgramFactory(Protocol):
    """Subset of the Core QC program factory used by the environment."""

    @staticmethod
    def from_qiskit(circuit: QuantumCircuit) -> _QCProgram: ...


class _CoreCompiler(Protocol):
    """Subset of the lazy-loaded ``mqt.core.mlir`` module."""

    QCProgram: _QCProgramFactory


class _CompilerTarget(Protocol):
    """Subset of the Core compiler target used by the environment."""

    @property
    def num_qubits(self) -> int: ...


def _load_core_compiler() -> _CoreCompiler:
    """Load the optional Core MLIR bindings only when the environment is used."""
    try:
        module = importlib.import_module("mqt.core.mlir")
    except ImportError as error:
        msg = "CorePredictorEnv requires the MLIR Python bindings from MQT Core."
        raise ImportError(msg) from error
    return cast("_CoreCompiler", module)


def _circuit_metrics(circuit: QuantumCircuit) -> CompileMetrics:
    """Measure the structural objective on one target-compiled circuit."""
    two_qubit_depth = circuit.depth(lambda instruction: instruction.operation.num_qubits == 2)
    two_qubit = sum(instruction.operation.num_qubits == 2 for instruction in circuit.data)
    gates = sum(
        instruction.operation.num_qubits > 0
        and instruction.operation.name not in {"barrier", "delay", "measure", "reset"}
        for instruction in circuit.data
    )
    return CompileMetrics(
        two_qubit_depth=two_qubit_depth,
        two_qubit=two_qubit,
        depth=circuit.depth(),
        gates=gates,
    )


def _score(metrics: CompileMetrics, baseline: CompileMetrics, pass_count: int) -> float:
    """Score target-compiled quality relative to Core's canonical pipeline."""
    weighted_improvement = 0.0
    for weight, candidate, reference in zip(
        (1.0, 0.25, 0.1, 0.05),
        (metrics.two_qubit_depth, metrics.two_qubit, metrics.depth, metrics.gates),
        (baseline.two_qubit_depth, baseline.two_qubit, baseline.depth, baseline.gates),
        strict=True,
    ):
        weighted_improvement += weight * (reference - candidate) / max(reference, 1)
    return weighted_improvement - PASS_PENALTY * pass_count


def _structural_features(circuit: QuantumCircuit) -> tuple[int, float, float, float, float, float, bool]:
    """Reproduce the straight-line feature scan used by the C++ runtime."""
    states = [(0, 0)] * circuit.num_qubits
    interactions: set[tuple[int, int]] = set()
    num_gates = 0
    num_two_qubit_gates = 0
    activity = 0
    maximum = (0, 0)
    has_wide_unitary = False

    for instruction in circuit.data:
        operation = instruction.operation
        qubits = tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
        if operation.name == "barrier" or not qubits:
            continue
        if getattr(operation, "blocks", ()):
            msg = "compiled policy features require straight-line circuits"
            raise ValueError(msg)

        prior = max((states[qubit] for qubit in qubits), default=(0, 0))
        if operation.name in {"measure", "reset"}:
            next_state = (prior[0] + 1, prior[1])
            activity += len(qubits)
        else:
            is_two_qubit = len(qubits) == 2
            has_wide_unitary |= len(qubits) > 2
            next_state = (prior[0] + 1, prior[1] + int(is_two_qubit))
            num_gates += 1
            num_two_qubit_gates += int(is_two_qubit)
            activity += len(qubits)
            if is_two_qubit:
                interactions.add(tuple(sorted(qubits)))
        for qubit in qubits:
            states[qubit] = next_state
        maximum = max(maximum, next_state)

    depth, two_qubit_critical_depth = maximum
    num_qubits = circuit.num_qubits
    communication = 2 * len(interactions) / (num_qubits * (num_qubits - 1)) if num_qubits > 1 else 0.0
    critical_depth = two_qubit_critical_depth / num_two_qubit_gates if num_two_qubit_gates else 0.0
    entanglement_ratio = num_two_qubit_gates / num_gates if num_gates else 0.0
    parallelism = max(((num_gates / depth) - 1) / (num_qubits - 1), 0.0) if num_qubits > 1 and depth else 0.0
    liveness = activity / (num_qubits * depth) if num_qubits and depth else 0.0
    return depth, communication, critical_depth, entanglement_ratio, parallelism, liveness, has_wide_unitary


class CorePredictorEnv(Env):
    """Order exposed Core QCO passes on one persistent compiler program."""

    def __init__(
        self,
        circuits: Sequence[QuantumCircuit],
        target: _CompilerTarget,
        *,
        max_passes: int = 100,
    ) -> None:
        """Initialize the Core-only pass-ordering environment.

        Args:
            circuits: Qiskit circuits sampled when an episode is reset.
            target: Core compiler target used for terminal compilation.
            max_passes: Maximum number of transformation actions per episode.

        Raises:
            ValueError: If the circuit corpus, target, or pass budget is invalid.
            ImportError: If the optional Core MLIR Python bindings are unavailable.
        """
        if not circuits:
            msg = "CorePredictorEnv requires at least one circuit."
            raise ValueError(msg)
        if target.num_qubits <= 0:
            msg = "The Core compiler target must contain at least one qubit."
            raise ValueError(msg)
        if not 0 < max_passes <= MAX_PASSES:
            msg = f"max_passes must be between 1 and {MAX_PASSES}."
            raise ValueError(msg)

        self._compiler = _load_core_compiler()
        self._circuits = tuple(circuit.copy() for circuit in circuits)
        self.target = target
        self.max_passes = max_passes
        self.action_space = Discrete(len(ACTION_NAMES))
        self.observation_space = Box(low=0.0, high=1.0, shape=(len(FEATURE_NAMES),), dtype=np.float32)

        self._program: _QCOProgram | None = None
        self._compiled = False
        self._has_wide_unitary = False
        self._circuit_index = 0
        self._num_passes = 0
        self._action_counts = np.zeros(NUM_TRANSFORM_ACTIONS, dtype=np.int64)
        self._attempted: set[tuple[str, int]] = set()
        self._baseline_cache: dict[int, CompileMetrics] = {}
        self.used_actions: list[str] = []

    @property
    def program(self) -> _QCOProgram:
        """The authoritative persistent QCO program."""
        if self._program is None:
            msg = "CorePredictorEnv must be reset before its program is used."
            raise RuntimeError(msg)
        return self._program

    @property
    def num_passes(self) -> int:
        """The number of transformation actions in this episode."""
        return self._num_passes

    def _fingerprint(self) -> str:
        return hashlib.sha256(self.program.ir.encode()).hexdigest()

    def _as_qiskit(self, program: _QCOProgram, *, compiled: bool) -> QuantumCircuit:
        qc_program = program.copy().to_qc()
        return qc_program.to_qiskit(target=self.target if compiled else None)

    def _metrics(self, program: _QCOProgram) -> CompileMetrics:
        return _circuit_metrics(self._as_qiskit(program, compiled=True))

    def _baseline(self) -> CompileMetrics:
        if self._circuit_index not in self._baseline_cache:
            candidate = self.program.copy()
            candidate.compile_for_target(self.target)
            self._baseline_cache[self._circuit_index] = self._metrics(candidate)
        return self._baseline_cache[self._circuit_index]

    def _observation_for(
        self,
        program: _QCOProgram,
        *,
        compiled: bool,
        num_passes: int,
        action_counts: NDArray[np.int64],
    ) -> tuple[NDArray[np.float32], bool]:
        circuit = self._as_qiskit(program, compiled=compiled)
        depth, communication, critical_depth, entanglement_ratio, parallelism, liveness, has_wide_unitary = (
            _structural_features(circuit)
        )
        values = [
            circuit.num_qubits / self.target.num_qubits,
            math.log1p(depth) / math.log1p(DEPTH_NORMALIZATION_MAX),
            communication,
            critical_depth,
            entanglement_ratio,
            parallelism,
            liveness,
            num_passes / MAX_PASSES,
            *(count / MAX_PASSES for count in action_counts),
        ]
        return np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0), has_wide_unitary

    def _observation(self) -> NDArray[np.float32]:
        observation, _ = self._observation_for(
            self.program,
            compiled=self._compiled,
            num_passes=self._num_passes,
            action_counts=self._action_counts,
        )
        return observation

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, object]]:
        """Start an episode from one circuit in the configured corpus."""
        super().reset(seed=seed)
        requested_index = None if options is None else options.get("circuit_index")
        if requested_index is None:
            self._circuit_index = int(self.np_random.integers(len(self._circuits)))
        elif type(requested_index) is not int or not 0 <= requested_index < len(self._circuits):
            msg = "options['circuit_index'] must select a configured circuit."
            raise ValueError(msg)
        else:
            self._circuit_index = requested_index

        circuit = self._circuits[self._circuit_index]
        if circuit.num_qubits > self.target.num_qubits:
            msg = "The selected circuit contains more qubits than the compiler target."
            raise ValueError(msg)
        self._program = self._compiler.QCProgram.from_qiskit(circuit).to_qco()
        self._program.cleanup()
        self._compiled = False
        self._num_passes = 0
        self._action_counts.fill(0)
        self._attempted.clear()
        self.used_actions = []
        baseline = self._baseline()
        observation, self._has_wide_unitary = self._observation_for(
            self.program,
            compiled=False,
            num_passes=0,
            action_counts=self._action_counts,
        )
        return observation, {"baseline": asdict(baseline), "circuit_index": self._circuit_index}

    def action_masks(self) -> list[bool]:
        """Return legal actions, suppressing no-op retries on identical IR."""
        if self._num_passes >= self.max_passes:
            return [False] * NUM_TRANSFORM_ACTIONS + [True]
        fingerprint = self._fingerprint()
        transforms = [(fingerprint, action) not in self._attempted for action in range(NUM_TRANSFORM_ACTIONS)]
        transforms[1] &= not self._has_wide_unitary
        return [*transforms, True]

    @staticmethod
    def _apply_transform(program: _QCOProgram, action: int) -> None:
        if action == 0:
            program.merge_single_qubit_rotation_gates()
        elif action == 1:
            program.fuse_single_qubit_unitary_runs(basis="u")
        elif action == 2:
            program.decompose_multi_controlled()
        elif action == 3:
            program.lift_hadamards()

    def _failed_step(
        self,
        error: Exception,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, object]]:
        reward = -2.0 - PASS_PENALTY * self._num_passes
        return self._observation(), reward, False, True, {"error": f"{type(error).__name__}: {error}"}

    def step(self, action: int) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, object]]:
        """Apply one pass or compile the persistent program for the target."""
        if not self.action_space.contains(action):
            msg = f"Action {action} is not supported."
            raise ValueError(msg)
        action = int(action)
        if not self.action_masks()[action]:
            msg = f"Action {ACTION_NAMES[action]} is not legal in the current state."
            raise ValueError(msg)

        if action == TERMINATE_ACTION:
            candidate = self.program.copy()
            try:
                candidate.compile_for_target(self.target)
                metrics = self._metrics(candidate)
                observation, has_wide_unitary = self._observation_for(
                    candidate,
                    compiled=True,
                    num_passes=self._num_passes,
                    action_counts=self._action_counts,
                )
            except Exception as error:  # ruff:ignore[blind-except]
                return self._failed_step(error)
            self._program = candidate
            self._compiled = True
            self._has_wide_unitary = has_wide_unitary
            self.used_actions.append(ACTION_NAMES[action])
            reward = _score(metrics, self._baseline(), self._num_passes)
            return (
                observation,
                reward,
                True,
                False,
                {"baseline": asdict(self._baseline()), "metrics": asdict(metrics), "passes": self._num_passes},
            )

        fingerprint = self._fingerprint()
        before = self.program.ir
        candidate = self.program.copy()
        candidate_counts = self._action_counts.copy()
        candidate_counts[action] += 1
        try:
            self._apply_transform(candidate, action)
            observation, has_wide_unitary = self._observation_for(
                candidate,
                compiled=False,
                num_passes=self._num_passes + 1,
                action_counts=candidate_counts,
            )
        except Exception as error:  # ruff:ignore[blind-except]
            return self._failed_step(error)
        self._program = candidate
        self._compiled = False
        self._has_wide_unitary = has_wide_unitary
        self._attempted.add((fingerprint, action))
        self._num_passes += 1
        self._action_counts = candidate_counts
        self.used_actions.append(ACTION_NAMES[action])
        return observation, 0.0, False, False, {"changed": before != candidate.ir}


__all__ = ["CorePredictorEnv"]
