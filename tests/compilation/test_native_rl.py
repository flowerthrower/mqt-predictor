# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for native MLIR policy-gradient training."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.predictor.compiled.native_rl import (
    CompileMetrics,
    Episode,
    Transition,
    parse_episode,
    parse_exhaustive_metrics,
    reinforce_update,
    terminal_reward,
)
from mqt.predictor.compiled.policy import ACTION_NAMES, FEATURE_NAMES, LinearPolicy


def test_parse_sampled_native_episode() -> None:
    """The trainer consumes the C++ feature, mask, action, and metric order."""
    trace = """
[mqt-predictor] step=0 action=merge-single-qubit-rotation-gates qubits=4 depth=8 two_qubit_depth=2 gates=12 two_qubit=3 mapped=0 routed=0 synthesized=0 legal=11111 features={relative_qubits=1,log_depth=0.2,program_communication=0.5,critical_depth=0.4,entanglement_ratio=0.25,parallelism=0.5,liveness=0.75,step_fraction=0,merge-single-qubit-rotation-gates_count=0,fuse-single-qubit-unitary-runs_count=0,decompose-multi-controlled_count=0,hadamard-lifting_count=0}
[mqt-predictor] step=1 action=merge-single-qubit-rotation-gates qubits=4 depth=7 two_qubit_depth=2 gates=11 two_qubit=3 mapped=0 routed=0 synthesized=0 legal=11111 features={relative_qubits=1,log_depth=0.19,program_communication=0.5,critical_depth=0.4,entanglement_ratio=0.27,parallelism=0.5,liveness=0.75,step_fraction=0.01,merge-single-qubit-rotation-gates_count=0.01,fuse-single-qubit-unitary-runs_count=0,decompose-multi-controlled_count=0,hadamard-lifting_count=0}
[mqt-predictor] step=2 action=terminate qubits=4 depth=10 two_qubit_depth=4 gates=16 two_qubit=6 mapped=1 routed=1 synthesized=1 legal=00001 features={relative_qubits=1,log_depth=0.22,program_communication=0.5,critical_depth=0.67,entanglement_ratio=0.38,parallelism=0.4,liveness=0.6,step_fraction=0.02,merge-single-qubit-rotation-gates_count=0.02,fuse-single-qubit-unitary-runs_count=0,decompose-multi-controlled_count=0,hadamard-lifting_count=0}
"""

    episode = parse_episode(trace)

    assert episode.terminated
    assert episode.pass_count == 2
    assert episode.repeated_passes == 1
    assert episode.repeated_optimizations == 1
    assert episode.action_counts == {"merge-single-qubit-rotation-gates": 2, "terminate": 1}
    assert episode.actions == (
        "merge-single-qubit-rotation-gates",
        "merge-single-qubit-rotation-gates",
        "terminate",
    )
    assert episode.final_metrics == CompileMetrics(two_qubit_depth=4, two_qubit=6, depth=10, gates=16)


def test_parse_exhaustive_single_use_result() -> None:
    """The comparison baseline is the best exhaustive one-use ordering."""
    trace = """
[mqt-predictor] search-result winner=7 schedule=merge-single-qubit-rotation-gates>hadamard-lifting valid=17 unique_outputs=14 two_qubit_depth=4 two_qubit=6 depth=10 gates=16 total_compile_us=42
"""

    assert parse_exhaustive_metrics(trace) == CompileMetrics(two_qubit_depth=4, two_qubit=6, depth=10, gates=16)


def test_terminal_reward_uses_core_relative_quality_and_pass_cost() -> None:
    """Matching Core quality is slightly penalized for unnecessary passes."""
    metrics = CompileMetrics(two_qubit_depth=4, two_qubit=6, depth=10, gates=16)
    transition = Transition(
        action=ACTION_NAMES.index("terminate"),
        features=(0.5,) * len(FEATURE_NAMES),
        legal=(False, False, False, False, True),
        metrics=metrics,
    )
    pass_transition = Transition(
        action=ACTION_NAMES.index("merge-single-qubit-rotation-gates"),
        features=(0.5,) * len(FEATURE_NAMES),
        legal=(True, True, True, True, True),
        metrics=metrics,
    )
    episode = Episode((pass_transition, pass_transition, transition), terminated=True, fell_back=False)

    assert terminal_reward(episode, metrics) == pytest.approx(-0.002)


def test_reinforce_increases_positive_action_logit() -> None:
    """A positive centered return raises its sampled action probability."""
    policy = LinearPolicy(
        np.zeros((len(ACTION_NAMES), len(FEATURE_NAMES)), dtype=np.float32),
        np.zeros(len(ACTION_NAMES), dtype=np.float32),
    )
    legal = (True, True, False, False, True)
    metrics = CompileMetrics(two_qubit_depth=1, two_qubit=1, depth=1, gates=1)
    positive = Episode(
        (
            Transition(
                action=0,
                features=(1.0,) + (0.0,) * (len(FEATURE_NAMES) - 1),
                legal=legal,
                metrics=metrics,
            ),
        ),
        terminated=False,
        fell_back=False,
    )
    negative = Episode(
        (
            Transition(
                action=1,
                features=(1.0,) + (0.0,) * (len(FEATURE_NAMES) - 1),
                legal=legal,
                metrics=metrics,
            ),
        ),
        terminated=False,
        fell_back=False,
    )

    updated = reinforce_update(policy, [(positive, 1.0), (negative, -1.0)], learning_rate=0.1, l2=0.0)

    assert updated.weights[0, 0] > updated.weights[1, 0]
    assert updated.bias[0] > updated.bias[1]
