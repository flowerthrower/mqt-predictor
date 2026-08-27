# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the Core-only pass-ordering environment."""

from __future__ import annotations

import math
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
        circuit = self.circuit.copy()
        circuit.metadata = {"fake_ir": self.ir}
        return circuit


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

    def fuse_two_qubit_gates(self) -> None:
        self._apply(ACTION_NAMES[2])

    def decompose_multi_controlled(self) -> None:
        self.compiler.decompositions += 1
        self.ir += "|decomposed"

    def _apply_stage(self, name: str) -> None:
        self.compiler.stage_calls.append((name, self.ir))
        if name in self.compiler.failing_stages:
            self.ir += f"|partial-{name}"
            msg = f"failed {name}"
            raise RuntimeError(msg)
        if name != "verify-target-conformance" and name not in self.compiler.noop_stages:
            self.ir += f"|{name}"

    def place_and_route(self, _target: object) -> None:
        self._apply_stage(ACTION_NAMES[3])

    def synthesize_for_target(self, _target: object) -> None:
        self._apply_stage(ACTION_NAMES[4])

    def verify_target_conformance(self, _target: object) -> None:
        self._apply_stage("verify-target-conformance")

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
        self.decompositions = 0
        self.applied: list[tuple[str, str]] = []
        self.stage_calls: list[tuple[str, str]] = []
        self.fuse_bases: list[str] = []
        self.compilation_inputs: list[str] = []
        self.exports: list[object | None] = []
        self.noop_actions: set[str] = set()
        self.noop_stages: set[str] = set()
        self.failing_actions: set[str] = set()
        self.failing_stages: set[str] = set()
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


def _environment(circuits: Sequence[QuantumCircuit], *, max_passes: int = 20) -> CorePredictorEnv:
    return CorePredictorEnv(circuits, _FakeTarget(), max_passes=max_passes)


def test_core_environment_has_compact_stable_abi(bell: QuantumCircuit, compiler: _FakeCompiler) -> None:
    """The Core environment exposes six actions and the v3 flat observation."""
    env = _environment([bell])

    observation, info = env.reset(seed=7)

    assert ACTION_NAMES == (
        "merge-single-qubit-rotation-gates",
        "fuse-single-qubit-unitary-runs",
        "fuse-two-qubit-gates",
        "place-and-route",
        "synthesize-for-target",
        "terminate",
    )
    assert FEATURE_NAMES == (
        "c3sqrtx",
        "c3x",
        "c4x",
        "ccx",
        "ch",
        "cp",
        "critical_depth",
        "crx",
        "cry",
        "crz",
        "cswap",
        "csx",
        "cu",
        "cu1",
        "cu3",
        "cx",
        "cy",
        "cz",
        "depth",
        "entanglement_ratio",
        "h",
        "id",
        "liveness",
        "measure",
        "num_qubits",
        "p",
        "parallelism",
        "program_communication",
        "rc3x",
        "rccx",
        "rx",
        "rxx",
        "ry",
        "rz",
        "rzz",
        "s",
        "sdg",
        "swap",
        "sx",
        "sxdg",
        "t",
        "tdg",
        "u",
        "u0",
        "u1",
        "u2",
        "u3",
        "x",
        "y",
        "z",
        "step_fraction",
        "merge-single-qubit-rotation-gates_count",
        "fuse-single-qubit-unitary-runs_count",
        "fuse-two-qubit-gates_count",
        "place-and-route_count",
        "synthesize-for-target_count",
    )
    assert isinstance(env.action_space, Discrete)
    assert env.action_space.n == len(ACTION_NAMES)
    assert env.observation_space.contains(observation)
    assert observation.dtype == np.float32
    assert observation[FEATURE_NAMES.index("h")] == pytest.approx(0.25)
    assert observation[FEATURE_NAMES.index("cx")] == pytest.approx(0.25)
    assert observation[FEATURE_NAMES.index("measure")] == pytest.approx(0.5)
    assert observation[FEATURE_NAMES.index("num_qubits")] == pytest.approx(0.5)
    assert observation[FEATURE_NAMES.index("depth")] == pytest.approx(math.log1p(3) / math.log1p(999_999))
    assert env.action_masks() == [True, True, True, True, True, False]
    assert info["circuit_index"] == 0
    assert info["potential"] == pytest.approx(0.0)
    assert (info["mapped"], info["routed"], info["synthesized"]) == (False, False, False)
    assert compiler.imports == 1
    assert compiler.cleanups == 1
    assert compiler.decompositions == 1


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
    assert first_info["changed"]
    assert first_info["potential"] == pytest.approx(0.0)
    assert not first_info["mapped"]
    assert first_observation[FEATURE_NAMES.index("step_fraction")] == pytest.approx(0.05)
    assert first_observation[FEATURE_NAMES.index(f"{ACTION_NAMES[0]}_count")] == pytest.approx(0.05)
    assert second_observation[FEATURE_NAMES.index("step_fraction")] == pytest.approx(0.1)
    assert second_observation[FEATURE_NAMES.index(f"{ACTION_NAMES[1]}_count")] == pytest.approx(0.05)


def test_gate_frequencies_keep_unknown_operations_in_denominator(compiler: _FakeCompiler) -> None:
    """The v3 gate-frequency denominator includes operations outside its vocabulary."""
    circuit = QuantumCircuit(1, 1)
    circuit.r(0.5, 0.25, 0)
    circuit.barrier()
    circuit.x(0)
    circuit.measure(0, 0)
    env = _environment([circuit])

    observation, _ = env.reset(options={"circuit_index": 0})

    assert compiler.imports == 1
    assert observation[FEATURE_NAMES.index("x")] == pytest.approx(1 / 3)
    assert observation[FEATURE_NAMES.index("measure")] == pytest.approx(1 / 3)


@pytest.mark.parametrize("action", range(3))
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


@pytest.mark.parametrize(("action", "marker"), [(3, "place-and-route"), (4, "synthesize-for-target")])
def test_each_core_stage_calls_its_bound_method(
    action: int,
    marker: str,
    bell: QuantumCircuit,
    compiler: _FakeCompiler,
) -> None:
    """Placement and synthesis are policy-visible Core stages."""
    env = _environment([bell])
    env.reset()

    env.step(action)

    assert f"|{marker}" in env.program.ir
    assert any(stage == marker for stage, _ in compiler.stage_calls)


def test_noop_suppression_still_permits_repeated_passes_after_change(
    bell: QuantumCircuit,
    compiler: _FakeCompiler,
) -> None:
    """Only an exact action retry on identical IR is masked."""
    compiler.noop_actions.add(ACTION_NAMES[0])
    env = _environment([bell])
    env.reset()
    stage_calls = len(compiler.stage_calls)

    _, _, _, _, info = env.step(0)

    assert not info["changed"]
    assert len(compiler.stage_calls) == stage_calls
    assert not env.action_masks()[0]
    with pytest.raises(ValueError, match="not legal"):
        env.step(0)

    env.step(1)
    assert env.action_masks()[0]
    env.step(0)
    assert env.used_actions == [ACTION_NAMES[0], ACTION_NAMES[1], ACTION_NAMES[0]]


@pytest.mark.usefixtures("compiler")
def test_phase_masks_require_explicit_mapping_synthesis_and_termination(
    bell: QuantumCircuit,
) -> None:
    """The staged Core state machine matches the compiled action phases."""
    env = _environment([bell])
    env.reset()

    env.step(4)
    assert env.action_masks() == [True, True, True, True, False, False]

    env.step(3)
    assert env.action_masks() == [True, True, True, False, True, False]

    env.step(4)
    assert env.action_masks() == [True, True, True, False, False, True]

    env.step(0)
    assert env.action_masks() == [True, True, True, False, True, False]


def test_required_stage_reopens_after_optimizer_retries_are_suppressed(
    bell: QuantumCircuit,
    compiler: _FakeCompiler,
) -> None:
    """Exact-IR optimizer suppression must not leave a nonconformant dead end."""
    compiler.noop_actions.update(ACTION_NAMES[:3])
    compiler.noop_stages.update(ACTION_NAMES[3:5])
    env = _environment([bell])
    env.reset()

    env.step(0)
    env.step(1)
    env.step(2)
    env.step(4)
    env.step(3)

    assert env.action_masks() == [False, False, False, False, True, False]
    env.step(4)
    assert env.action_masks() == [False, False, False, False, False, True]


@pytest.mark.usefixtures("compiler")
def test_shaped_reward_is_potential_delta_and_terminal_reward_is_absolute(
    bell: QuantumCircuit,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Intermediate rewards telescope while terminate reports final quality."""

    def metrics(circuit: QuantumCircuit) -> core_env.CompileMetrics:
        ir = str(circuit.metadata["fake_ir"])
        gates = 8 if ACTION_NAMES[0] in ir else 10
        return core_env.CompileMetrics(two_qubit_depth=10, two_qubit=10, depth=10, gates=gates)

    monkeypatch.setattr(core_env, "_circuit_metrics", metrics)
    env = _environment([bell])
    _, reset_info = env.reset()

    _, reward, _, _, info = env.step(0)

    assert reset_info["potential"] == pytest.approx(0.0)
    assert info["potential"] == pytest.approx(0.01)
    assert reward == pytest.approx(0.009)

    env.step(3)
    env.step(4)
    _, terminal_reward, terminated, truncated, _ = env.step(5)
    assert terminal_reward == pytest.approx(0.01)
    assert terminated
    assert not truncated


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
    assert "failed fuse-two-qubit-gates" in str(info["error"])


def test_potential_completion_failure_does_not_commit_stage(
    bell: QuantumCircuit,
    compiler: _FakeCompiler,
) -> None:
    """Reward evaluation remains in the same transaction as the selected stage."""
    env = _environment([bell])
    env.reset()
    original = env.program.ir
    compiler.failing_stages.add(ACTION_NAMES[4])

    _, reward, terminated, truncated, info = env.step(3)

    assert env.program.ir == original
    assert env.num_passes == 0
    assert reward == pytest.approx(-2.0)
    assert not terminated
    assert truncated
    assert "failed synthesize-for-target" in str(info["error"])


def test_terminal_failure_does_not_commit_partial_program(bell: QuantumCircuit, compiler: _FakeCompiler) -> None:
    """Failed target verification leaves the pre-terminal QCO available."""
    env = _environment([bell])
    env.reset()
    env.step(3)
    env.step(4)
    original = env.program.ir
    compiler.failing_stages.add("verify-target-conformance")

    _, reward, terminated, truncated, _ = env.step(5)

    assert env.program.ir == original
    assert reward == pytest.approx(-2.0)
    assert not terminated
    assert truncated


@pytest.mark.parametrize("action", [0, 5])
def test_export_failure_does_not_commit_candidate(
    bell: QuantumCircuit,
    compiler: _FakeCompiler,
    action: int,
) -> None:
    """Observation export is part of the same transaction as the Core action."""
    env = _environment([bell])
    env.reset()
    if action == 5:
        env.step(3)
        env.step(4)
    original = env.program.ir
    compiler.failing_export_markers.add(ACTION_NAMES[0] if action == 0 else ACTION_NAMES[4])

    _, reward, terminated, truncated, info = env.step(action)

    assert env.program.ir == original
    assert env.num_passes == (0 if action == 0 else 2)
    assert reward == pytest.approx(-2.0)
    assert not terminated
    assert truncated
    assert "failed export" in str(info["error"])


def test_termination_only_verifies_compiled_state(bell: QuantumCircuit, compiler: _FakeCompiler) -> None:
    """Terminate verifies the staged result and returns absolute terminal quality."""
    env = _environment([bell])
    env.reset()
    env.step(3)
    env.step(4)
    before = env.program.ir
    baseline_compilations = compiler.compilations

    observation, reward, terminated, truncated, info = env.step(5)

    assert compiler.compilations == baseline_compilations
    assert env.program.ir == before
    assert env.observation_space.contains(observation)
    assert reward == pytest.approx(0.0)
    assert terminated
    assert not truncated
    assert info["passes"] == 2
    assert compiler.exports[-1] is env.target


@pytest.mark.usefixtures("compiler")
def test_exact_pass_cap_truncates_nonconformant_state(
    bell: QuantumCircuit,
) -> None:
    """Exhausting the pass budget without conformance ends the episode."""
    env = _environment([bell], max_passes=1)
    env.reset()

    _, reward, terminated, truncated, _ = env.step(0)

    assert env.num_passes == 1
    assert reward == pytest.approx(-2.001)
    assert not terminated
    assert truncated
    assert env.action_masks() == [False] * 6


@pytest.mark.usefixtures("compiler")
def test_exact_pass_cap_allows_conformant_termination(
    bell: QuantumCircuit,
) -> None:
    """The final pass slots are reserved for target completion."""
    env = _environment([bell], max_passes=2)
    env.reset()

    assert env.action_masks() == [False, False, False, True, False, False]
    env.step(3)
    assert env.action_masks() == [False, False, False, False, True, False]
    _, reward, terminated, truncated, _ = env.step(4)

    assert reward == pytest.approx(-0.001)
    assert not terminated
    assert not truncated
    assert env.action_masks() == [False, False, False, False, False, True]
    _, _, terminated, truncated, info = env.step(5)
    assert terminated
    assert not truncated
    assert info["passes"] == 2


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


def test_environment_rejects_more_than_twenty_passes(bell: QuantumCircuit) -> None:
    """History normalization has a fixed 20-pass ABI."""
    with pytest.raises(ValueError, match="between 1 and 20"):
        _environment([bell], max_passes=21)


def test_current_core_compiles_bell_for_iqm_garnet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optional current-Core dependency supports the complete environment boundary."""
    monkeypatch.chdir(tmp_path)
    compiler = pytest.importorskip("mqt.core.mlir")
    target = compiler.CompilerTarget.from_device_id("mqt.sc.iqm.garnet")
    env = CorePredictorEnv([qasm3.load(INPUTS / "bell.qasm")], target, max_passes=2)

    observation, _ = env.reset(options={"circuit_index": 0})
    assert env.observation_space.contains(observation)
    expected = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    expected[FEATURE_NAMES.index("critical_depth")] = 1.0
    expected[FEATURE_NAMES.index("cx")] = 3 / 8
    expected[FEATURE_NAMES.index("depth")] = math.log1p(6) / math.log1p(999_999)
    expected[FEATURE_NAMES.index("entanglement_ratio")] = 0.6
    expected[FEATURE_NAMES.index("h")] = 1 / 8
    expected[FEATURE_NAMES.index("liveness")] = 0.6111111
    expected[FEATURE_NAMES.index("measure")] = 3 / 8
    expected[FEATURE_NAMES.index("num_qubits")] = 0.15
    expected[FEATURE_NAMES.index("program_communication")] = 1.0
    expected[FEATURE_NAMES.index("rz")] = 1 / 8
    np.testing.assert_allclose(observation, expected, rtol=1e-6, atol=1e-6)

    mapped_observation, _, _, _, _ = env.step(3)
    mapped_qiskit = env.program.copy().to_qc().to_qiskit(target=target)
    assert mapped_qiskit.num_qubits == target.num_qubits
    assert mapped_observation[FEATURE_NAMES.index("num_qubits")] == pytest.approx(0.15)

    env.step(4)
    observation, _, terminated, truncated, info = env.step(5)

    assert env.observation_space.contains(observation)
    assert terminated
    assert not truncated
    assert info["passes"] == 2


def test_current_core_staged_api_matches_canonical_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The patched staged API reproduces Core's canonical target pipeline."""
    monkeypatch.chdir(tmp_path)
    compiler = pytest.importorskip("mqt.core.mlir")
    target = compiler.CompilerTarget(
        "staged target",
        [
            compiler.CompilerTarget.Site(10),
            compiler.CompilerTarget.Site(20),
            compiler.CompilerTarget.Site(30),
        ],
        couplings=[(10, 20), (20, 30)],
        operations=[
            compiler.CompilerTarget.Operation("u", 1, 3),
            compiler.CompilerTarget.Operation("cz", 2, 0),
            compiler.CompilerTarget.Operation("measure", 1, 0),
        ],
    )
    circuit = qasm3.loads(
        """OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
h q[0];
cx q[0], q[2];
cx q[0], q[2];
cx q[0], q[1];
cx q[1], q[2];
bit[3] c = measure q;
"""
    )
    source = compiler.QCProgram.from_qiskit(circuit).to_qco()

    canonical = source.copy()
    canonical.compile_for_target(target)

    staged = source.copy()
    staged.cleanup()
    staged.decompose_multi_controlled()
    staged.run_pass_pipeline("mqt-qco-default")
    staged.fuse_two_qubit_gates()
    staged.place_and_route(target)
    staged.synthesize_for_target(target)
    staged.verify_target_conformance(target)

    assert staged.ir == canonical.ir


def test_current_core_fusion_keeps_bell_observable(
    bell: QuantumCircuit,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared ``u`` fusion basis remains exportable for the next observation."""
    monkeypatch.chdir(tmp_path)
    compiler = pytest.importorskip("mqt.core.mlir")
    target = compiler.CompilerTarget.from_device_id("mqt.sc.iqm.garnet")
    env = CorePredictorEnv([bell], target, max_passes=3)
    env.reset(options={"circuit_index": 0})

    observation, reward, terminated, truncated, _ = env.step(1)

    assert env.observation_space.contains(observation)
    assert np.isfinite(reward)
    assert not terminated
    assert not truncated


def test_current_core_decomposes_wide_gates_during_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reset makes the optimization actions safe by decomposing wide controls."""
    monkeypatch.chdir(tmp_path)
    compiler = pytest.importorskip("mqt.core.mlir")
    target = compiler.CompilerTarget.from_device_id("mqt.sc.iqm.garnet")
    env = CorePredictorEnv([qasm3.load(INPUTS / "wide.qasm")], target, max_passes=3)
    env.reset(options={"circuit_index": 0})

    assert env.action_masks() == [True, True, True, True, True, False]

    observation, reward, terminated, truncated, _ = env.step(1)
    assert env.observation_space.contains(observation)
    assert np.isfinite(reward)
    assert not terminated
    assert not truncated


def test_current_core_native_input_uses_action_derived_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training and compiled inference share conservative phase transitions."""
    monkeypatch.chdir(tmp_path)
    compiler = pytest.importorskip("mqt.core.mlir")
    target = compiler.CompilerTarget.from_device_id("mqt.sc.iqm.garnet")
    circuit = QuantumCircuit(2, 2)
    circuit.r(0.5, 0.2, 0)
    circuit.cz(0, 1)
    circuit.measure([0, 1], [0, 1])
    env = CorePredictorEnv([circuit], target, max_passes=3)

    env.reset(options={"circuit_index": 0})
    assert env.action_masks() == [True, True, True, True, True, False]

    _, _, _, _, info = env.step(3)
    assert info == {
        "changed": True,
        "potential": pytest.approx(0.0),
        "mapped": True,
        "routed": True,
        "synthesized": False,
    }
    assert env.action_masks() == [True, True, True, False, True, False]


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

    assert observation[FEATURE_NAMES.index("critical_depth")] == pytest.approx(0.5)
