# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the Core-only pass-ordering environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from gymnasium.spaces import Discrete
from mqt.bench import BenchmarkLevel, get_benchmark
from qiskit import QuantumCircuit, qasm3

from mqt.predictor.compiled import ACTION_NAMES, FEATURE_NAMES, CorePredictorEnv, core_env

if TYPE_CHECKING:
    from collections.abc import Sequence

INPUTS = Path(__file__).parents[2] / "cpp/test/Inputs"


@dataclass(frozen=True)
class _FakeTarget:
    """Minimal compiler target for environment tests."""

    num_qubits: int = 4


class _FakeQCProgram:
    """Minimal Core QC program carrying a Qiskit circuit."""

    def __init__(self, compiler: _FakeCompiler, circuit: QuantumCircuit, ir: str) -> None:
        self.compiler = compiler
        self.circuit = circuit.copy()
        self.ir = ir

    def to_qco(self) -> _FakeQCOProgram:
        return _FakeQCOProgram(self.compiler, self.circuit, self.ir)

    def to_qiskit(self, *, target: object | None = None) -> QuantumCircuit:
        self.compiler.exports.append(target)
        if any(marker in self.ir for marker in self.compiler.failing_export_markers):
            msg = f"failed export of {self.ir}"
            raise RuntimeError(msg)
        return self.circuit.copy()


class _FakeQCOProgram(_FakeQCProgram):
    """Transactional fake of the bound Core QCO program."""

    def copy(self) -> _FakeQCOProgram:
        return _FakeQCOProgram(self.compiler, self.circuit, self.ir)

    def cleanup(self) -> None:
        self.compiler.cleanups += 1
        self.ir += "|cleanup"

    def _apply(self, name: str) -> None:
        self.compiler.applied.append((name, self.ir))
        if name in self.compiler.failing_actions:
            self.ir += f"|partial-{name}"
            msg = f"failed {name}"
            raise RuntimeError(msg)
        if name not in self.compiler.noop_actions:
            self.ir += f"|{name}"

    def merge_single_qubit_rotation_gates(self) -> None:
        self._apply(ACTION_NAMES[0])

    def fuse_single_qubit_unitary_runs(self, *, basis: str = "zyz") -> None:
        self.compiler.fuse_bases.append(basis)
        self._apply(ACTION_NAMES[1])

    def decompose_multi_controlled(self) -> None:
        self._apply(ACTION_NAMES[2])

    def lift_hadamards(self) -> None:
        self._apply(ACTION_NAMES[3])

    def compile_for_target(self, _target: object) -> None:
        self.compiler.compilations += 1
        self.compiler.compilation_inputs.append(self.ir)
        if self.compiler.fail_compilation:
            self.ir += "|partial-compilation"
            msg = "failed target compilation"
            raise RuntimeError(msg)
        self.ir += "|compiled"

    def to_qc(self) -> _FakeQCProgram:
        return _FakeQCProgram(self.compiler, self.circuit, self.ir)


class _FakeQCProgramFactory:
    """Factory matching Core's static ``QCProgram.from_qiskit`` API."""

    def __init__(self, compiler: _FakeCompiler) -> None:
        self.compiler = compiler

    def from_qiskit(self, circuit: QuantumCircuit) -> _FakeQCProgram:
        self.compiler.imports += 1
        return _FakeQCProgram(self.compiler, circuit, f"raw-{self.compiler.imports}")


class _FakeCompiler:
    """Configurable fake of the lazy Core compiler module."""

    def __init__(self) -> None:
        self.QCProgram = _FakeQCProgramFactory(self)
        self.imports = 0
        self.compilations = 0
        self.cleanups = 0
        self.applied: list[tuple[str, str]] = []
        self.fuse_bases: list[str] = []
        self.compilation_inputs: list[str] = []
        self.exports: list[object | None] = []
        self.noop_actions: set[str] = set()
        self.failing_actions: set[str] = set()
        self.failing_export_markers: set[str] = set()
        self.fail_compilation = False


@pytest.fixture
def bell() -> QuantumCircuit:
    """Return a small observed circuit with nontrivial features."""
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure(range(2), range(2))
    return circuit


@pytest.fixture
def compiler(monkeypatch: pytest.MonkeyPatch) -> _FakeCompiler:
    """Replace the optional Core module with a lightweight fake."""
    fake = _FakeCompiler()
    monkeypatch.setattr(core_env, "_load_core_compiler", lambda: fake)
    return fake


def _environment(circuits: Sequence[QuantumCircuit], *, max_passes: int = 100) -> CorePredictorEnv:
    return CorePredictorEnv(circuits, _FakeTarget(), max_passes=max_passes)


def test_core_environment_has_compact_stable_abi(bell: QuantumCircuit, compiler: _FakeCompiler) -> None:
    """The Core environment exposes five actions and twelve bounded floats."""
    env = _environment([bell])

    observation, info = env.reset(seed=7)

    assert ACTION_NAMES == (
        "merge-single-qubit-rotation-gates",
        "fuse-single-qubit-unitary-runs",
        "decompose-multi-controlled",
        "hadamard-lifting",
        "terminate",
    )
    assert FEATURE_NAMES == (
        "relative_qubits",
        "log_depth",
        "program_communication",
        "critical_depth",
        "entanglement_ratio",
        "parallelism",
        "liveness",
        "step_fraction",
        "merge-single-qubit-rotation-gates_count",
        "fuse-single-qubit-unitary-runs_count",
        "decompose-multi-controlled_count",
        "hadamard-lifting_count",
    )
    assert isinstance(env.action_space, Discrete)
    assert env.action_space.n == len(ACTION_NAMES)
    assert env.observation_space.contains(observation)
    assert observation.dtype == np.float32
    assert env.action_masks() == [True] * 5
    assert info["circuit_index"] == 0
    assert compiler.imports == 1
    assert compiler.cleanups == 1


def test_actions_keep_one_persistent_qco_state(bell: QuantumCircuit, compiler: _FakeCompiler) -> None:
    """Passes consume the prior QCO result without reimporting Qiskit."""
    env = _environment([bell])
    env.reset(options={"circuit_index": 0})

    first_before = env.program.ir
    first_observation, _, _, _, first_info = env.step(0)
    second_before = env.program.ir
    second_observation, _, _, _, _ = env.step(1)

    assert compiler.imports == 1
    assert compiler.applied == [
        (ACTION_NAMES[0], first_before),
        (ACTION_NAMES[1], second_before),
    ]
    assert compiler.fuse_bases == ["u"]
    assert first_info == {"changed": True}
    assert first_observation[7] == pytest.approx(0.01)
    assert first_observation[8] == pytest.approx(0.01)
    assert second_observation[7] == pytest.approx(0.02)
    assert second_observation[9] == pytest.approx(0.01)


@pytest.mark.parametrize("action", range(4))
def test_each_core_action_calls_its_bound_pass(
    action: int,
    bell: QuantumCircuit,
    compiler: _FakeCompiler,
) -> None:
    """Each policy output invokes exactly its corresponding Core method."""
    env = _environment([bell])
    env.reset()
    before = env.program.ir

    env.step(action)

    assert compiler.applied == [(ACTION_NAMES[action], before)]


def test_noop_suppression_still_permits_repeated_passes_after_change(
    bell: QuantumCircuit,
    compiler: _FakeCompiler,
) -> None:
    """Only an exact action retry on identical IR is masked."""
    compiler.noop_actions.add(ACTION_NAMES[0])
    env = _environment([bell])
    env.reset()

    _, _, _, _, info = env.step(0)

    assert info == {"changed": False}
    assert not env.action_masks()[0]
    with pytest.raises(ValueError, match="not legal"):
        env.step(0)

    env.step(1)
    assert env.action_masks()[0]
    env.step(0)
    assert env.used_actions == [ACTION_NAMES[0], ACTION_NAMES[1], ACTION_NAMES[0]]


def test_transform_failure_does_not_commit_partial_program(bell: QuantumCircuit, compiler: _FakeCompiler) -> None:
    """A failed Core pass truncates without replacing the authoritative QCO."""
    env = _environment([bell])
    env.reset()
    original = env.program.ir
    compiler.failing_actions.add(ACTION_NAMES[2])

    _, reward, terminated, truncated, info = env.step(2)

    assert env.program.ir == original
    assert env.num_passes == 0
    assert reward == pytest.approx(-2.0)
    assert not terminated
    assert truncated
    assert "failed decompose-multi-controlled" in str(info["error"])


def test_terminal_failure_does_not_commit_partial_program(bell: QuantumCircuit, compiler: _FakeCompiler) -> None:
    """Failed target compilation leaves the pre-terminal QCO available."""
    env = _environment([bell])
    env.reset()
    original = env.program.ir
    compiler.fail_compilation = True

    _, reward, terminated, truncated, _ = env.step(4)

    assert env.program.ir == original
    assert reward == pytest.approx(-2.0)
    assert not terminated
    assert truncated


@pytest.mark.parametrize(("action", "marker"), [(0, ACTION_NAMES[0]), (4, "|compiled")])
def test_export_failure_does_not_commit_candidate(
    bell: QuantumCircuit,
    compiler: _FakeCompiler,
    action: int,
    marker: str,
) -> None:
    """Observation export is part of the same transaction as the Core action."""
    env = _environment([bell])
    env.reset()
    original = env.program.ir
    compiler.failing_export_markers.add(marker)

    _, reward, terminated, truncated, info = env.step(action)

    assert env.program.ir == original
    assert env.num_passes == 0
    assert reward == pytest.approx(-2.0)
    assert not terminated
    assert truncated
    assert "failed export" in str(info["error"])


def test_termination_compiles_copy_and_penalizes_pass_count(bell: QuantumCircuit, compiler: _FakeCompiler) -> None:
    """Terminate promotes a target-compiled copy and returns terminal quality."""
    compiler.noop_actions.add(ACTION_NAMES[0])
    env = _environment([bell])
    env.reset()
    env.step(0)
    before = env.program.ir

    observation, reward, terminated, truncated, info = env.step(4)

    assert compiler.compilation_inputs[-1] == before
    assert env.program.ir == f"{before}|compiled"
    assert env.observation_space.contains(observation)
    assert reward == pytest.approx(-0.001)
    assert terminated
    assert not truncated
    assert info["passes"] == 1
    assert compiler.exports[-1] is env.target


def test_exact_pass_cap_forces_termination_without_early_truncation(
    bell: QuantumCircuit,
    compiler: _FakeCompiler,
) -> None:
    """The policy may run exactly its pass budget before terminate is forced."""
    env = _environment([bell], max_passes=100)
    env.reset()
    assert compiler.imports == 1

    for _ in range(100):
        _, reward, terminated, truncated, _ = env.step(0)
        assert reward == pytest.approx(0.0)
        assert not terminated
        assert not truncated

    assert env.num_passes == 100
    assert env.action_masks() == [False, False, False, False, True]
    with pytest.raises(ValueError, match="not legal"):
        env.step(1)

    _, _, terminated, truncated, info = env.step(4)
    assert terminated
    assert not truncated
    assert info["passes"] == 100


def test_baseline_is_cached_per_circuit(bell: QuantumCircuit, compiler: _FakeCompiler) -> None:
    """Repeated resets do not recompile an unchanged canonical baseline."""
    other = bell.copy()
    other.x(0)
    env = _environment([bell, other])

    env.reset(options={"circuit_index": 0})
    env.reset(options={"circuit_index": 0})
    assert compiler.compilations == 1

    env.reset(options={"circuit_index": 1})
    assert compiler.compilations == 2


def test_reset_rejects_circuit_larger_than_target(bell: QuantumCircuit, compiler: _FakeCompiler) -> None:
    """Oversized training circuits fail before entering Core."""
    circuit = QuantumCircuit(5, 5)
    circuit.measure(range(5), range(5))
    env = _environment([bell, circuit])

    with pytest.raises(ValueError, match="more qubits"):
        env.reset(options={"circuit_index": 1})

    assert compiler.imports == 0


def test_environment_rejects_more_than_one_hundred_passes(bell: QuantumCircuit) -> None:
    """History normalization has a fixed 100-pass ABI."""
    with pytest.raises(ValueError, match="between 1 and 100"):
        _environment([bell], max_passes=101)


def test_current_core_compiles_bell_for_iqm_garnet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optional current-Core dependency supports the complete environment boundary."""
    monkeypatch.chdir(tmp_path)
    compiler = pytest.importorskip("mqt.core.mlir")
    target = compiler.CompilerTarget.from_device_id("mqt.sc.iqm.garnet")
    env = CorePredictorEnv([qasm3.load(INPUTS / "bell.qasm")], target, max_passes=1)

    observation, _ = env.reset(options={"circuit_index": 0})
    assert env.observation_space.contains(observation)
    np.testing.assert_allclose(
        observation,
        [0.15, 0.1408497, 1.0, 1.0, 0.6, 0.0, 0.6111111, 0.0, 0.0, 0.0, 0.0, 0.0],
        rtol=1e-6,
        atol=1e-6,
    )

    env.step(0)
    observation, reward, terminated, truncated, info = env.step(4)

    assert env.observation_space.contains(observation)
    assert reward == pytest.approx(-0.001)
    assert terminated
    assert not truncated
    assert info["passes"] == 1


def test_current_core_fusion_keeps_bell_observable(
    bell: QuantumCircuit,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared ``u`` fusion basis remains exportable for the next observation."""
    monkeypatch.chdir(tmp_path)
    compiler = pytest.importorskip("mqt.core.mlir")
    target = compiler.CompilerTarget.from_device_id("mqt.sc.iqm.garnet")
    env = CorePredictorEnv([bell], target, max_passes=1)
    env.reset(options={"circuit_index": 0})

    observation, reward, terminated, truncated, _ = env.step(1)

    assert env.observation_space.contains(observation)
    assert reward == pytest.approx(0.0)
    assert not terminated
    assert not truncated


def test_current_core_masks_wide_fusion_until_decomposition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CCX cannot enter the ``u`` fusion pass before wide-gate decomposition."""
    monkeypatch.chdir(tmp_path)
    compiler = pytest.importorskip("mqt.core.mlir")
    target = compiler.CompilerTarget.from_device_id("mqt.sc.iqm.garnet")
    env = CorePredictorEnv([qasm3.load(INPUTS / "wide.qasm")], target, max_passes=2)
    env.reset(options={"circuit_index": 0})

    assert env.action_masks() == [True, False, True, True, True]
    with pytest.raises(ValueError, match="not legal"):
        env.step(1)

    _, _, terminated, truncated, _ = env.step(2)
    assert not terminated
    assert not truncated
    assert env.action_masks() == [True, True, True, True, True]

    observation, reward, terminated, truncated, _ = env.step(1)
    assert env.observation_space.contains(observation)
    assert reward == pytest.approx(0.0)
    assert not terminated
    assert not truncated


def test_current_core_qft_uses_compiled_critical_path_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Python observation matches C++ tie-breaking on QFT critical paths."""
    monkeypatch.chdir(tmp_path)
    compiler = pytest.importorskip("mqt.core.mlir")
    target = compiler.CompilerTarget.from_device_id("mqt.sc.iqm.garnet")
    circuit = get_benchmark("qft", BenchmarkLevel.ALG, 4)
    env = CorePredictorEnv([circuit], target, max_passes=1)

    observation, _ = env.reset(options={"circuit_index": 0})

    assert observation[3] == pytest.approx(0.75)
