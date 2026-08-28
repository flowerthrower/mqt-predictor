# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Core-only reinforcement-learning environment for pass ordering."""

from __future__ import annotations

import contextlib
import importlib
import math
import signal
import threading
import time
import warnings
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete

from .policy import ACTION_NAMES, FEATURE_NAMES, MAX_STEPS, V3_OPERATION_NAMES

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from types import FrameType

    from numpy.typing import NDArray
    from qiskit import QuantumCircuit


NUM_TRANSFORM_ACTIONS = len(ACTION_NAMES) - 1
TERMINATE_ACTION = len(ACTION_NAMES) - 1
PLACE_AND_ROUTE_ACTION = 3
SYNTHESIZE_ACTION = 4
DEPTH_NORMALIZATION_MAX = 999_999


class _QCProgram(Protocol):
    """Subset of the Core QC program API used by the environment."""

    def to_qco(self) -> _QCOProgram: ...


class _QCOProgram(Protocol):
    """Subset of the Core QCO program API used by the environment."""

    @property
    def ir(self) -> str: ...

    def copy(self) -> _QCOProgram: ...

    def cleanup(self) -> None: ...

    def merge_single_qubit_rotation_gates(self) -> None: ...

    def fuse_single_qubit_unitary_runs(self, *, basis: str = "zyz") -> None: ...

    def fuse_two_qubit_gates(self) -> None: ...

    def decompose_multi_controlled(self) -> None: ...

    def place_and_route(self, target: object) -> None: ...

    def synthesize_for_target(self, target: object) -> None: ...

    def verify_target_conformance(self, target: object) -> None: ...

    def analyze_for_target(self, target: object) -> _QCOCircuitMetrics: ...

    def expected_fidelity(self, target: object) -> float: ...


class _QCOCircuitMetrics(Protocol):
    """Owned Core analysis result consumed by the observation adapter."""

    @property
    def operation_counts(self) -> dict[str, int]: ...

    @property
    def total_operations(self) -> int: ...

    @property
    def num_qubits(self) -> int: ...

    @property
    def depth(self) -> int: ...

    @property
    def critical_depth(self) -> float: ...

    @property
    def entanglement_ratio(self) -> float: ...

    @property
    def parallelism(self) -> float: ...

    @property
    def liveness(self) -> float: ...

    @property
    def program_communication(self) -> float: ...

    @property
    def mapped(self) -> bool: ...

    @property
    def routed(self) -> bool: ...

    @property
    def synthesized(self) -> bool: ...


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


@contextlib.contextmanager
def _enforce_pass_timeout(pass_timeout: float | None) -> Iterator[None]:
    """Apply the same best-effort POSIX pass timeout as Predictor v3.

    Python signal delivery cannot preempt a native pass that holds the GIL. A
    hard deadline for such passes therefore still requires process isolation.
    """
    if pass_timeout is None:
        yield
        return

    required = ("SIGALRM", "ITIMER_REAL", "getitimer", "setitimer")
    if not all(hasattr(signal, attribute) for attribute in required):
        warnings.warn("Pass timeouts are not supported on this platform.", RuntimeWarning, stacklevel=2)
        yield
        return
    if threading.current_thread() is not threading.main_thread():
        warnings.warn("Pass timeouts are only supported on the main thread.", RuntimeWarning, stacklevel=2)
        yield
        return

    def timeout_handler(_signum: int, _frame: FrameType | None) -> None:
        msg = f"Compilation pass exceeded the timeout of {pass_timeout:g} seconds."
        raise TimeoutError(msg)

    previous_delay, previous_interval = signal.getitimer(signal.ITIMER_REAL)
    if 0 < previous_delay <= pass_timeout:
        yield
        return

    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    start_time = time.monotonic()
    try:
        signal.setitimer(signal.ITIMER_REAL, pass_timeout)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_delay > 0:
            remaining_delay = max(previous_delay - (time.monotonic() - start_time), 1e-6)
            signal.setitimer(signal.ITIMER_REAL, remaining_delay, previous_interval)


class CorePredictorEnv(Env):
    """Order exposed Core QCO passes on one persistent compiler program."""

    def __init__(
        self,
        circuits: Sequence[QuantumCircuit],
        target: _CompilerTarget,
        *,
        max_steps: int = MAX_STEPS,
        pass_timeout: float | None = None,
    ) -> None:
        """Initialize the Core-only pass-ordering environment.

        Args:
            circuits: Qiskit circuits sampled when an episode is reset.
            target: Core compiler target used for terminal compilation.
            max_steps: Maximum number of actions, including termination, per episode.
            pass_timeout: Best-effort timeout in seconds for one Core pass.

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
        if not 0 < max_steps <= MAX_STEPS:
            msg = f"max_steps must be between 1 and {MAX_STEPS}."
            raise ValueError(msg)

        self._compiler = _load_core_compiler()
        self._circuits = tuple(circuit.copy() for circuit in circuits)
        self.target = target
        self.max_steps = max_steps
        self.pass_timeout = pass_timeout
        self.action_space = Discrete(len(ACTION_NAMES))
        self.observation_space = Dict({
            name: Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32) for name in FEATURE_NAMES
        })

        self._program: _QCOProgram | None = None
        self._mapped = False
        self._routed = False
        self._synthesized = False
        self._circuit_index = 0
        self._num_steps = 0
        self._episode_ended = False
        self._noop_actions: set[int] = set()
        self._last_observation = {name: np.zeros(1, dtype=np.float32) for name in FEATURE_NAMES}
        self.used_actions: list[str] = []

    @property
    def program(self) -> _QCOProgram:
        """The authoritative persistent QCO program."""
        if self._program is None:
            msg = "CorePredictorEnv must be reset before its program is used."
            raise RuntimeError(msg)
        return self._program

    @property
    def num_steps(self) -> int:
        """The number of successful actions in this episode, including termination."""
        return self._num_steps

    @property
    def pass_timeout(self) -> float | None:
        """The current best-effort per-pass timeout in seconds."""
        return self._pass_timeout

    @pass_timeout.setter
    def pass_timeout(self, pass_timeout: float | None) -> None:
        if pass_timeout is not None and pass_timeout <= 0:
            msg = "pass_timeout must be positive."
            raise ValueError(msg)
        self._pass_timeout = pass_timeout

    def _analysis_for(
        self,
        program: _QCOProgram,
    ) -> tuple[dict[str, NDArray[np.float32]], bool, bool, bool]:
        analysis = program.analyze_for_target(self.target)
        operation_counts = analysis.operation_counts
        denominator = max(analysis.total_operations, 1)
        v3_features = {
            **{name: operation_counts.get(name, 0) / denominator for name in V3_OPERATION_NAMES},
            "critical_depth": analysis.critical_depth,
            "depth": math.log1p(min(analysis.depth, DEPTH_NORMALIZATION_MAX)) / math.log1p(DEPTH_NORMALIZATION_MAX),
            "entanglement_ratio": analysis.entanglement_ratio,
            "liveness": analysis.liveness,
            "num_qubits": analysis.num_qubits / self.target.num_qubits,
            "parallelism": analysis.parallelism,
            "program_communication": analysis.program_communication,
        }
        observation = {
            name: np.clip(np.asarray([v3_features[name]], dtype=np.float32), 0.0, 1.0) for name in FEATURE_NAMES
        }
        return observation, analysis.mapped, analysis.routed, analysis.synthesized

    def _is_conformant(self) -> bool:
        return self._mapped and self._routed and self._synthesized

    def _state_info(self) -> dict[str, bool]:
        return {
            "mapped": self._mapped,
            "routed": self._routed,
            "synthesized": self._synthesized,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[dict[str, NDArray[np.float32]], dict[str, object]]:
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
        program = self._compiler.QCProgram.from_qiskit(circuit).to_qco()
        program.cleanup()
        program.decompose_multi_controlled()
        observation, mapped, routed, synthesized = self._analysis_for(program)

        self._program = program
        self._mapped = mapped
        self._routed = routed
        self._synthesized = synthesized
        self._num_steps = 0
        self._episode_ended = False
        self._noop_actions.clear()
        self._last_observation = observation
        self.used_actions = []
        return observation, {
            "circuit_index": self._circuit_index,
            **self._state_info(),
        }

    def action_masks(self) -> list[bool]:
        """Return actions permitted by the current compilation phase."""
        if self._num_steps == 0:
            return [True] * NUM_TRANSFORM_ACTIONS + [False]
        transforms = [action not in self._noop_actions for action in range(NUM_TRANSFORM_ACTIONS)]
        transforms[PLACE_AND_ROUTE_ACTION] &= not self._mapped
        transforms[SYNTHESIZE_ACTION] &= not self._synthesized
        return [*transforms, self._is_conformant()]

    @staticmethod
    def _apply_transform(program: _QCOProgram, action: int) -> None:
        if action == 0:
            program.merge_single_qubit_rotation_gates()
        elif action == 1:
            program.fuse_single_qubit_unitary_runs(basis="u")
        elif action == 2:
            program.fuse_two_qubit_gates()
        elif action == 3:
            msg = "place-and-route requires a compiler target"
            raise AssertionError(msg)
        elif action == 4:
            msg = "target synthesis requires a compiler target"
            raise AssertionError(msg)

    def _apply_action(self, program: _QCOProgram, action: int) -> None:
        if action == PLACE_AND_ROUTE_ACTION:
            program.place_and_route(self.target)
        elif action == SYNTHESIZE_ACTION:
            program.synthesize_for_target(self.target)
        else:
            self._apply_transform(program, action)

    def _failed_step(
        self,
        error: Exception,
    ) -> tuple[dict[str, NDArray[np.float32]], float, bool, bool, dict[str, object]]:
        self._episode_ended = True
        observation = {name: value.copy() for name, value in self._last_observation.items()}
        return observation, 0.0, False, True, {"Truncated because of error": f"{type(error).__name__}: {error}"}

    def _evaluate_action(
        self,
        action: int,
    ) -> tuple[_QCOProgram, dict[str, NDArray[np.float32]], bool, bool, bool, bool]:
        """Apply and evaluate one action without changing episode state."""
        before = self.program.ir
        candidate = self.program.copy()
        with _enforce_pass_timeout(self.pass_timeout):
            self._apply_action(candidate, action)
        changed = before != candidate.ir
        observation, candidate_mapped, candidate_routed, candidate_synthesized = self._analysis_for(candidate)
        return (
            candidate,
            observation,
            changed,
            candidate_mapped,
            candidate_routed,
            candidate_synthesized,
        )

    def step(self, action: int) -> tuple[dict[str, NDArray[np.float32]], float, bool, bool, dict[str, object]]:
        """Apply one Core stage or verify an already conformant program."""
        if self._episode_ended:
            msg = "This episode has ended; call reset() before selecting another action."
            raise RuntimeError(msg)
        if not self.action_space.contains(action):
            msg = f"Action {action} is not supported."
            raise ValueError(msg)
        action = int(action)
        if not self.action_masks()[action]:
            msg = f"Action {ACTION_NAMES[action]} is not legal in the current state."
            raise ValueError(msg)
        self.used_actions.append(ACTION_NAMES[action])

        if action == TERMINATE_ACTION:
            candidate = self.program.copy()
            try:
                with _enforce_pass_timeout(self.pass_timeout):
                    candidate.verify_target_conformance(self.target)
                reward = candidate.expected_fidelity(self.target)
                observation, candidate_mapped, candidate_routed, candidate_synthesized = self._analysis_for(candidate)
            except Exception as error:  # ruff:ignore[blind-except]
                return self._failed_step(error)
            self._program = candidate
            self._mapped = candidate_mapped
            self._routed = candidate_routed
            self._synthesized = candidate_synthesized
            self._num_steps += 1
            self._episode_ended = True
            self._last_observation = observation
            return (
                observation,
                reward,
                True,
                False,
                {
                    "steps": self._num_steps,
                    **self._state_info(),
                },
            )

        try:
            (
                candidate,
                observation,
                changed,
                candidate_mapped,
                candidate_routed,
                candidate_synthesized,
            ) = self._evaluate_action(action)
        except Exception as error:  # ruff:ignore[blind-except]
            return self._failed_step(error)
        self._program = candidate
        self._mapped = candidate_mapped
        self._routed = candidate_routed
        self._synthesized = candidate_synthesized
        if changed:
            self._noop_actions.clear()
        else:
            self._noop_actions.add(action)
        self._num_steps += 1
        self._last_observation = observation
        terminated = self._num_steps >= self.max_steps
        self._episode_ended = terminated
        return (
            observation,
            0.0,
            terminated,
            False,
            {
                "changed": changed,
                **({"termination_reason": "max_steps_exceeded"} if terminated else {}),
                **self._state_info(),
            },
        )


__all__ = ["CorePredictorEnv"]
