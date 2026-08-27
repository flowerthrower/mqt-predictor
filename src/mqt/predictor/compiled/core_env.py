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
import copy
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

from mqt.predictor.reward import expected_fidelity

from .policy import ACTION_NAMES, FEATURE_NAMES, MAX_STEPS, V3_OPERATION_NAMES

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from types import FrameType

    from numpy.typing import NDArray
    from qiskit import QuantumCircuit
    from qiskit.transpiler import Target


NUM_TRANSFORM_ACTIONS = len(ACTION_NAMES) - 1
TERMINATE_ACTION = len(ACTION_NAMES) - 1
PLACE_AND_ROUTE_ACTION = 3
SYNTHESIZE_ACTION = 4
DEPTH_NORMALIZATION_MAX = 999_999


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

    def fuse_two_qubit_gates(self) -> None: ...

    def decompose_multi_controlled(self) -> None: ...

    def place_and_route(self, target: object) -> None: ...

    def synthesize_for_target(self, target: object) -> None: ...

    def verify_target_conformance(self, target: object) -> None: ...

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


def _prepare_reward_target(target: Target) -> Target:
    """Copy a Qiskit target and complete only symmetric CZ calibrations.

    Core can reverse the operands of a CZ without changing the physical gate.
    Qiskit's expected-fidelity lookup is ordered, so mirror an existing CZ
    property only when the opposite ordering is absent. No other operation is
    assumed to be swap invariant.
    """
    prepared = copy.deepcopy(target)
    try:
        properties = prepared["cz"]
    except KeyError:
        return prepared
    for qargs, instruction_properties in tuple(properties.items()):
        if qargs is None or len(qargs) != 2:
            continue
        reverse = (qargs[1], qargs[0])
        if reverse not in properties:
            properties[reverse] = instruction_properties
    return prepared


def _structural_features(
    circuit: QuantumCircuit,
) -> tuple[int, float, float, float, float, float]:
    """Reproduce the straight-line feature scan used by the C++ runtime."""
    num_qubits = circuit.num_qubits
    states = [(0, 0)] * circuit.num_qubits
    interactions: set[tuple[int, int]] = set()
    num_gates = 0
    num_two_qubit_gates = 0
    activity = 0
    maximum = (0, 0)

    for instruction in circuit.data:
        operation = instruction.operation
        qubits = tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
        if operation.name == "barrier" or not qubits:
            continue
        if getattr(operation, "blocks", ()):
            msg = "compiled policy features require straight-line circuits"
            raise ValueError(msg)

        prior = states[qubits[0]]
        for qubit in qubits[1:]:
            if states[qubit][0] > prior[0]:
                prior = states[qubit]
        if operation.name in {"measure", "reset"}:
            next_state = (prior[0] + 1, prior[1])
            activity += len(qubits)
        else:
            is_two_qubit = len(qubits) == 2
            next_state = (prior[0] + 1, prior[1] + int(is_two_qubit))
            num_gates += 1
            num_two_qubit_gates += int(is_two_qubit)
            activity += len(qubits)
            if is_two_qubit:
                interactions.add(tuple(sorted(qubits)))
        for qubit in qubits:
            states[qubit] = next_state
        if next_state[0] >= maximum[0]:
            maximum = next_state

    depth, two_qubit_critical_depth = maximum
    communication = 2 * len(interactions) / (num_qubits * (num_qubits - 1)) if num_qubits > 1 else 0.0
    critical_depth = two_qubit_critical_depth / num_two_qubit_gates if num_two_qubit_gates else 0.0
    entanglement_ratio = num_two_qubit_gates / num_gates if num_gates else 0.0
    parallelism = max(((num_gates / depth) - 1) / (num_qubits - 1), 0.0) if num_qubits > 1 and depth else 0.0
    liveness = activity / (num_qubits * depth) if num_qubits and depth else 0.0
    return depth, communication, critical_depth, entanglement_ratio, parallelism, liveness


class CorePredictorEnv(Env):
    """Order exposed Core QCO passes on one persistent compiler program."""

    def __init__(
        self,
        circuits: Sequence[QuantumCircuit],
        target: _CompilerTarget,
        reward_target: Target,
        *,
        max_steps: int = MAX_STEPS,
        pass_timeout: float | None = None,
    ) -> None:
        """Initialize the Core-only pass-ordering environment.

        Args:
            circuits: Qiskit circuits sampled when an episode is reset.
            target: Core compiler target used for terminal compilation.
            reward_target: Qiskit target carrying the same device calibrations.
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
        if reward_target.num_qubits != target.num_qubits:
            msg = "The Core compiler target and Qiskit reward target must describe the same number of qubits."
            raise ValueError(msg)
        if not 0 < max_steps <= MAX_STEPS:
            msg = f"max_steps must be between 1 and {MAX_STEPS}."
            raise ValueError(msg)

        self._compiler = _load_core_compiler()
        self._circuits = tuple(circuit.copy() for circuit in circuits)
        self.target = target
        self.reward_target = _prepare_reward_target(reward_target)
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

    def _as_qiskit(self, program: _QCOProgram, *, mapped: bool) -> QuantumCircuit:
        qc_program = program.copy().to_qc()
        return qc_program.to_qiskit(target=self.target if mapped else None)

    def _observation_for(
        self,
        program: _QCOProgram,
        *,
        mapped: bool,
    ) -> dict[str, NDArray[np.float32]]:
        circuit = self._as_qiskit(program, mapped=mapped)
        num_qubits = circuit.num_qubits
        depth, communication, critical_depth, entanglement_ratio, parallelism, liveness = _structural_features(circuit)
        operation_counts = dict.fromkeys(V3_OPERATION_NAMES, 0)
        total_operations = 0
        for instruction in circuit.data:
            operation_name = instruction.operation.name
            if operation_name == "barrier":
                continue
            total_operations += 1
            if operation_name in operation_counts:
                operation_counts[operation_name] += 1
        denominator = max(total_operations, 1)
        v3_features = {
            **{name: count / denominator for name, count in operation_counts.items()},
            "critical_depth": critical_depth,
            "depth": math.log1p(min(depth, DEPTH_NORMALIZATION_MAX)) / math.log1p(DEPTH_NORMALIZATION_MAX),
            "entanglement_ratio": entanglement_ratio,
            "liveness": liveness,
            "num_qubits": num_qubits / self.target.num_qubits,
            "parallelism": parallelism,
            "program_communication": communication,
        }
        return {name: np.clip(np.asarray([v3_features[name]], dtype=np.float32), 0.0, 1.0) for name in FEATURE_NAMES}

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
        observation = self._observation_for(program, mapped=False)

        self._program = program
        self._mapped = False
        self._routed = False
        self._synthesized = False
        self._num_steps = 0
        self._episode_ended = False
        self._last_observation = observation
        self.used_actions = []
        return observation, {
            "circuit_index": self._circuit_index,
            **self._state_info(),
        }

    def action_masks(self) -> list[bool]:
        """Return actions permitted by the current compilation phase."""
        transforms = [True] * NUM_TRANSFORM_ACTIONS
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
        candidate_mapped = self._mapped
        candidate_routed = self._routed
        candidate_synthesized = self._synthesized
        if action < PLACE_AND_ROUTE_ACTION and changed:
            candidate_synthesized = False
        elif action == PLACE_AND_ROUTE_ACTION:
            candidate_mapped = True
            candidate_routed = True
            candidate_synthesized = False
        elif action == SYNTHESIZE_ACTION:
            candidate_synthesized = True
        observation = self._observation_for(
            candidate,
            mapped=candidate_mapped,
        )
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
                circuit = self._as_qiskit(candidate, mapped=True)
                reward = expected_fidelity(circuit, self.reward_target)
                observation = self._observation_for(candidate, mapped=True)
            except Exception as error:  # ruff:ignore[blind-except]
                return self._failed_step(error)
            self._program = candidate
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
        self._num_steps += 1
        self._last_observation = observation
        truncated = self._num_steps >= self.max_steps
        self._episode_ended = truncated
        return (
            observation,
            0.0,
            False,
            truncated,
            {
                "changed": changed,
                **({"truncation_reason": "max_steps_exceeded"} if truncated else {}),
                **self._state_info(),
            },
        )


__all__ = ["CorePredictorEnv"]
