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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pytest
from gymnasium.spaces import Dict as DictSpace
from gymnasium.spaces import Discrete
from mqt.bench import BenchmarkLevel, get_benchmark
from qiskit import QuantumCircuit, qasm3
from qiskit.circuit import Measure
from qiskit.circuit.library import CXGate, CZGate, HGate
from qiskit.transpiler import InstructionProperties, Target

from mqt.predictor.compiled import ACTION_NAMES, FEATURE_NAMES, CorePredictorEnv, core_env

if TYPE_CHECKING:
    from collections.abc import Sequence

INPUTS = Path(__file__).parents[2] / "cpp/test/Inputs"


class _TargetLike(Protocol):
    """Compiler-target surface consumed by the environment."""

    num_qubits: int


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
        if name in self.compiler.action_delays:
            time.sleep(self.compiler.action_delays[name])
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
        self.action_delays: dict[str, float] = {}
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


def _reward_target(*, num_qubits: int = 4) -> Target:
    target = Target(num_qubits=num_qubits)
    target.add_instruction(
        HGate(),
        {(qubit,): InstructionProperties(error=0.01) for qubit in range(num_qubits)},
    )
    target.add_instruction(
        CXGate(),
        {
            (control, target_qubit): InstructionProperties(error=0.02)
            for control in range(num_qubits)
            for target_qubit in range(num_qubits)
            if control != target_qubit
        },
    )
    target.add_instruction(
        Measure(),
        {(qubit,): InstructionProperties(error=0.03) for qubit in range(num_qubits)},
    )
    return target


def _environment(
    circuits: Sequence[QuantumCircuit],
    *,
    max_steps: int = 20,
    pass_timeout: float | None = None,
    reward_target: Target | None = None,
) -> CorePredictorEnv:
    return CorePredictorEnv(
        circuits,
        _FakeTarget(),
        _reward_target() if reward_target is None else reward_target,
        max_steps=max_steps,
        pass_timeout=pass_timeout,
    )


def _feature(observation: dict[str, np.ndarray], name: str) -> float:
    return float(observation[name][0])


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
    )
    assert isinstance(env.action_space, Discrete)
    assert env.action_space.n == len(ACTION_NAMES)
    assert isinstance(env.observation_space, DictSpace)
    assert tuple(env.observation_space.spaces) == FEATURE_NAMES
    assert env.observation_space.contains(observation)
    assert tuple(observation) == FEATURE_NAMES
    assert all(value.dtype == np.float32 and value.shape == (1,) for value in observation.values())
    assert _feature(observation, "h") == pytest.approx(0.25)
    assert _feature(observation, "cx") == pytest.approx(0.25)
    assert _feature(observation, "measure") == pytest.approx(0.5)
    assert _feature(observation, "num_qubits") == pytest.approx(0.5)
    assert _feature(observation, "depth") == pytest.approx(math.log1p(3) / math.log1p(999_999))
    assert env.action_masks() == [True, True, True, True, True, False]
    assert info["circuit_index"] == 0
    assert (info["mapped"], info["routed"], info["synthesized"]) == (False, False, False)
    assert compiler.imports == 1
    assert compiler.cleanups == 1
    assert compiler.decompositions == 1


def test_actions_keep_one_persistent_qco_state(bell: QuantumCircuit, compiler: _FakeCompiler) -> None:
    """Passes consume the prior QCO result without reimporting Qiskit."""
    env = _environment([bell])
    env.reset(options={"circuit_index": 0})

    first_before = env.program.ir
    first_observation, first_reward, _, _, first_info = env.step(0)
    second_before = env.program.ir
    second_observation, second_reward, _, _, _ = env.step(1)

    assert compiler.imports == 1
    assert compiler.applied == [
        (ACTION_NAMES[0], first_before),
        (ACTION_NAMES[1], second_before),
    ]
    assert compiler.fuse_bases == ["u"]
    assert first_info["changed"]
    assert not first_info["mapped"]
    assert first_reward == pytest.approx(0.0)
    assert second_reward == pytest.approx(0.0)
    assert all(np.array_equal(first_observation[name], second_observation[name]) for name in FEATURE_NAMES)


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
    assert _feature(observation, "x") == pytest.approx(1 / 3)
    assert _feature(observation, "measure") == pytest.approx(1 / 3)


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


def test_repeated_noop_passes_remain_legal(
    bell: QuantumCircuit,
    compiler: _FakeCompiler,
) -> None:
    """The v3 policy may select the same ineffective pass repeatedly."""
    compiler.noop_actions.add(ACTION_NAMES[0])
    env = _environment([bell])
    env.reset()

    _, first_reward, _, _, first_info = env.step(0)
    _, second_reward, _, _, second_info = env.step(0)

    assert not first_info["changed"]
    assert not second_info["changed"]
    assert first_reward == pytest.approx(0.0)
    assert second_reward == pytest.approx(0.0)
    assert env.action_masks()[0]
    assert env.used_actions == [ACTION_NAMES[0], ACTION_NAMES[0]]


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


@pytest.mark.usefixtures("compiler")
def test_only_termination_returns_absolute_expected_fidelity(
    bell: QuantumCircuit,
) -> None:
    """Core actions return zero; terminate returns the configured device fidelity."""
    env = _environment([bell])
    env.reset()

    _, optimizer_reward, _, _, _ = env.step(0)
    assert optimizer_reward == pytest.approx(0.0)

    env.step(3)
    env.step(4)
    _, terminal_reward, terminated, truncated, _ = env.step(5)
    assert terminal_reward == pytest.approx(0.99 * 0.98 * 0.97**2)
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
    assert env.num_steps == 0
    assert reward == pytest.approx(0.0)
    assert not terminated
    assert truncated
    assert "failed fuse-two-qubit-gates" in str(info["Truncated because of error"])


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
    assert reward == pytest.approx(0.0)
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
    assert env.num_steps == (0 if action == 0 else 2)
    assert reward == pytest.approx(0.0)
    assert not terminated
    assert truncated
    assert "failed export" in str(info["Truncated because of error"])


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
    assert reward == pytest.approx(0.99 * 0.98 * 0.97**2)
    assert terminated
    assert not truncated
    assert info["steps"] == 3
    assert env.num_steps == 3
    assert compiler.exports[-1] is env.target


@pytest.mark.usefixtures("compiler")
def test_exact_step_cap_truncates_with_zero_reward(
    bell: QuantumCircuit,
) -> None:
    """Every nonterminal action consumes a horizon slot regardless of state."""
    env = _environment([bell], max_steps=1)
    env.reset()

    _, reward, terminated, truncated, info = env.step(0)

    assert env.num_steps == 1
    assert reward == pytest.approx(0.0)
    assert not terminated
    assert truncated
    assert info["truncation_reason"] == "max_steps_exceeded"
    with pytest.raises(RuntimeError, match="episode has ended"):
        env.step(0)


@pytest.mark.usefixtures("compiler")
def test_horizon_counts_termination_and_does_not_reserve_stage_slots(
    bell: QuantumCircuit,
) -> None:
    """Terminate must fit in the horizon; masks remain phase-only."""
    env = _environment([bell], max_steps=2)
    env.reset()

    assert env.action_masks() == [True, True, True, True, True, False]
    env.step(3)
    assert env.action_masks() == [True, True, True, False, True, False]
    _, reward, terminated, truncated, _ = env.step(4)

    assert reward == pytest.approx(0.0)
    assert not terminated
    assert truncated

    env = _environment([bell], max_steps=3)
    env.reset()
    env.step(3)
    env.step(4)
    _, _, terminated, truncated, info = env.step(5)
    assert terminated
    assert not truncated
    assert info["steps"] == 3


def test_reset_rejects_circuit_larger_than_target(bell: QuantumCircuit, compiler: _FakeCompiler) -> None:
    """Oversized training circuits fail before entering Core."""
    circuit = QuantumCircuit(5, 5)
    circuit.measure(range(5), range(5))
    env = _environment([bell, circuit])

    with pytest.raises(ValueError, match="more qubits"):
        env.reset(options={"circuit_index": 1})

    assert compiler.imports == 0


def test_environment_rejects_more_than_twenty_steps(bell: QuantumCircuit) -> None:
    """The deployment contract has an exact 20-action horizon."""
    with pytest.raises(ValueError, match="between 1 and 20"):
        _environment([bell], max_steps=21)


def test_environment_requires_matching_core_and_reward_targets(bell: QuantumCircuit) -> None:
    """Training cannot silently combine calibrations from another device width."""
    with pytest.raises(ValueError, match="same number of qubits"):
        CorePredictorEnv([bell], _FakeTarget(), Target(num_qubits=3))


def test_reverse_cz_uses_the_same_symmetric_calibration(
    compiler: _FakeCompiler,
) -> None:
    """Core operand reversal remains scoreable without general edge mirroring."""
    circuit = QuantumCircuit(2)
    circuit.cz(1, 0)
    reward_target = Target(num_qubits=4)
    reward_target.add_instruction(CZGate(), {(0, 1): InstructionProperties(error=0.2)})
    env = _environment([circuit], reward_target=reward_target, max_steps=3)
    env.reset()
    env.step(3)
    env.step(4)

    _, reward, terminated, truncated, _ = env.step(5)

    assert reward == pytest.approx(0.8)
    assert terminated
    assert not truncated
    assert (1, 0) not in reward_target["cz"]
    assert env.reward_target["cz"][1, 0] is env.reward_target["cz"][0, 1]
    assert compiler.imports == 1


def test_reward_target_does_not_mirror_directional_gates() -> None:
    """Only CZ receives the narrow swap-invariant calibration completion."""
    reward_target = Target(num_qubits=2)
    reward_target.add_instruction(CXGate(), {(0, 1): InstructionProperties(error=0.2)})

    prepared = core_env._prepare_reward_target(reward_target)  # ruff: ignore[private-member-access]

    assert (1, 0) not in prepared["cx"]


def test_pass_timeout_truncates_without_committing_partial_state(
    bell: QuantumCircuit,
    compiler: _FakeCompiler,
) -> None:
    """The Predictor v3 signal timeout is transactional for interruptible passes."""
    required = ("SIGALRM", "ITIMER_REAL", "getitimer", "setitimer")
    if not all(hasattr(core_env.signal, attribute) for attribute in required):
        pytest.skip("POSIX interval timers are unavailable")
    compiler.action_delays[ACTION_NAMES[0]] = 0.1
    env = _environment([bell], pass_timeout=0.01)
    env.reset()
    original = env.program.ir

    _, reward, terminated, truncated, info = env.step(0)

    assert env.program.ir == original
    assert env.num_steps == 0
    assert reward == pytest.approx(0.0)
    assert not terminated
    assert truncated
    assert "TimeoutError" in str(info["Truncated because of error"])


def test_pass_timeout_must_be_positive(bell: QuantumCircuit) -> None:
    """Invalid timeout values fail during environment construction."""
    with pytest.raises(ValueError, match="pass_timeout must be positive"):
        _environment([bell], pass_timeout=0.0)


def _iqm_targets() -> tuple[_TargetLike, Target]:
    compiler = pytest.importorskip("mqt.core.mlir")
    qiskit_plugin = pytest.importorskip("mqt.core.plugins.qiskit")
    backend = qiskit_plugin.QDMIBackend.from_device_id("mqt.sc.iqm.garnet")
    return compiler.CompilerTarget.from_device(backend.device), backend.target


def test_current_core_compiles_bell_for_iqm_garnet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optional current-Core dependency supports the complete environment boundary."""
    monkeypatch.chdir(tmp_path)
    pytest.importorskip("mqt.core.mlir")
    target, reward_target = _iqm_targets()
    env = CorePredictorEnv([qasm3.load(INPUTS / "bell.qasm")], target, reward_target, max_steps=3)

    observation, _ = env.reset(options={"circuit_index": 0})
    assert env.observation_space.contains(observation)
    expected = dict.fromkeys(FEATURE_NAMES, 0.0)
    expected["critical_depth"] = 1.0
    expected["cx"] = 3 / 8
    expected["depth"] = math.log1p(6) / math.log1p(999_999)
    expected["entanglement_ratio"] = 0.6
    expected["h"] = 1 / 8
    expected["liveness"] = 0.6111111
    expected["measure"] = 3 / 8
    expected["num_qubits"] = 0.15
    expected["program_communication"] = 1.0
    expected["rz"] = 1 / 8
    for name, value in expected.items():
        assert _feature(observation, name) == pytest.approx(value, rel=1e-6, abs=1e-6)

    mapped_observation, _, _, _, _ = env.step(3)
    mapped_qiskit = env.program.copy().to_qc().to_qiskit(target=target)
    assert mapped_qiskit.num_qubits == target.num_qubits
    assert _feature(mapped_observation, "num_qubits") == pytest.approx(0.15)

    env.step(4)
    observation, reward, terminated, truncated, info = env.step(5)

    assert env.observation_space.contains(observation)
    assert reward == pytest.approx(0.851076634)
    assert terminated
    assert not truncated
    assert info["steps"] == 3


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
    pytest.importorskip("mqt.core.mlir")
    target, reward_target = _iqm_targets()
    env = CorePredictorEnv([bell], target, reward_target, max_steps=3)
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
    pytest.importorskip("mqt.core.mlir")
    target, reward_target = _iqm_targets()
    env = CorePredictorEnv([qasm3.load(INPUTS / "wide.qasm")], target, reward_target, max_steps=3)
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
    pytest.importorskip("mqt.core.mlir")
    target, reward_target = _iqm_targets()
    circuit = QuantumCircuit(2, 2)
    circuit.r(0.5, 0.2, 0)
    circuit.cz(0, 1)
    circuit.measure([0, 1], [0, 1])
    env = CorePredictorEnv([circuit], target, reward_target, max_steps=3)

    env.reset(options={"circuit_index": 0})
    assert env.action_masks() == [True, True, True, True, True, False]

    _, _, _, _, info = env.step(3)
    assert info == {
        "changed": True,
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
    pytest.importorskip("mqt.core.mlir")
    target, reward_target = _iqm_targets()
    circuit = get_benchmark("qft", BenchmarkLevel.ALG, 4)
    env = CorePredictorEnv([circuit], target, reward_target, max_steps=1)

    observation, _ = env.reset(options={"circuit_index": 0})

    assert _feature(observation, "critical_depth") == pytest.approx(0.5)
