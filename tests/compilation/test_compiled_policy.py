# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the experimental compiled-policy trainer and artifact ABI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

from mqt.predictor.compiled import (
    ACTION_NAMES,
    FEATURE_NAMES,
    TanhMlpPolicy,
    TrainingExample,
    fit_linear_policy,
    load_training_dataset,
    target_fingerprint,
    train_maskable_ppo,
)
from mqt.predictor.compiled.policy import LinearPolicy, export_linear_policy, parameter_checksum

INPUTS = Path(__file__).parents[2] / "cpp" / "test" / "Inputs"


def test_minimal_training_is_deterministic() -> None:
    """The 32-step smoke training run is reproducible."""
    dataset = json.loads((INPUTS / "line-4-training.json").read_text(encoding="utf-8"))
    assert dataset["purpose"] == (
        "manually curated ABI smoke samples; not a reproducible Core trajectory or performance dataset"
    )
    examples = load_training_dataset(INPUTS / "line-4-training.json")
    first = fit_linear_policy(examples)
    second = fit_linear_policy(examples)

    assert first.accuracy >= 0.7
    assert first.loss == pytest.approx(second.loss)
    np.testing.assert_array_equal(first.policy.weights, second.policy.weights)
    np.testing.assert_array_equal(first.policy.bias, second.policy.bias)


def test_checked_in_artifact_matches_python_contract() -> None:
    """The checked-in artifact matches its target and parameter digests."""
    artifact = json.loads((INPUTS / "line-4-policy.json").read_text(encoding="utf-8"))
    weights = np.asarray(artifact["parameters"]["weights"], dtype=np.float32)
    bias = np.asarray(artifact["parameters"]["bias"], dtype=np.float32)

    assert artifact["compatibility"]["target_fingerprint"] == target_fingerprint(INPUTS / "line-4-target.json")
    assert artifact["parameters_sha256"] == parameter_checksum(weights, bias)
    assert artifact["training"]["epochs"] == 32
    assert artifact["training"]["objective"] == (
        "manually curated ABI smoke samples; not a reproducible Core trajectory or performance dataset"
    )


def test_training_fixture_only_enables_terminate_after_core_stages() -> None:
    """Recorded terminal masks follow staged target conformance."""
    dataset = json.loads((INPUTS / "line-4-training.json").read_text(encoding="utf-8"))
    terminal_samples = [sample for sample in dataset["samples"] if sample["legal"][-1]]

    assert terminal_samples
    assert all(sample["action"] == "terminate" for sample in terminal_samples)
    assert all(sample["legal"][3:] == [False, False, True] for sample in terminal_samples)
    assert all(sample["features"][11] > 0 and sample["features"][12] > 0 for sample in terminal_samples)


def test_policy_masks_illegal_actions() -> None:
    """Masked actions cannot win native inference."""
    weights = np.zeros((len(ACTION_NAMES), len(FEATURE_NAMES)), dtype=np.float32)
    bias = np.arange(len(ACTION_NAMES), dtype=np.float32)
    policy = LinearPolicy(weights, bias)

    selected, logits = policy.select([0.5] * len(FEATURE_NAMES), [True, True, False, False, False, False])

    assert selected == 1
    assert np.isneginf(logits[2:]).all()


def test_linear_policy_rejects_runtime_logit_overflow() -> None:
    """Python rejects parameters that the float32 C++ evaluator cannot score."""
    weights = np.full(
        (len(ACTION_NAMES), len(FEATURE_NAMES)),
        np.finfo(np.float32).max,
        dtype=np.float32,
    )
    policy = LinearPolicy(weights, np.zeros(len(ACTION_NAMES), dtype=np.float32))

    with pytest.raises(ValueError, match="float32 runtime range"):
        policy.select([1.0] * len(FEATURE_NAMES), [True] * len(ACTION_NAMES))


def test_training_rejects_illegal_label() -> None:
    """A training label must be legal in its recorded state."""
    with pytest.raises(ValueError, match="enabled"):
        TrainingExample(
            features=(0.0,) * len(FEATURE_NAMES),
            legal=(False, True, False, False, False, False),
            action=0,
        )


def test_training_accepts_preterminal_stage_mask() -> None:
    """Imitation data may record states before target conformance."""
    example = TrainingExample(
        features=(0.0,) * len(FEATURE_NAMES),
        legal=(True, True, True, True, True, False),
        action=0,
    )

    assert example.action == 0


@pytest.mark.parametrize("invalid_operation", ["cnot", "U", " cx "])
def test_target_fingerprint_rejects_noncanonical_operations(tmp_path: Path, invalid_operation: str) -> None:
    """The exporter rejects spellings rejected by the C++ v1 loader."""
    document = json.loads((INPUTS / "line-4-target.json").read_text(encoding="utf-8"))
    document["operations"][1]["name"] = invalid_operation
    target = tmp_path / "target.json"
    target.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical"):
        target_fingerprint(target)


def test_export_accepts_runtime_target_fingerprint(tmp_path: Path) -> None:
    """QDMI training can bind an artifact to a fingerprint reported by C++."""
    policy = LinearPolicy(
        np.zeros((len(ACTION_NAMES), len(FEATURE_NAMES)), dtype=np.float32),
        np.zeros(len(ACTION_NAMES), dtype=np.float32),
    )
    fingerprint = f"sha256:{'a' * 64}"
    output = tmp_path / "policy.json"

    export_linear_policy(
        output,
        policy,
        target_fingerprint_override=fingerprint,
        core_revision="core",
        source_revision="source",
        algorithm="test",
        objective="test",
        samples=1,
        epochs=1,
        learning_rate=0.1,
        l2=0.0,
        seed=0,
    )

    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["compatibility"]["target_fingerprint"] == fingerprint


class _MinimalMaskedEnv(Env):
    """Two-step environment for checking the deployable actor shape."""

    observation_space = Box(0.0, 1.0, shape=(len(FEATURE_NAMES),), dtype=np.float32)
    action_space = Discrete(len(ACTION_NAMES))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed, options=options)
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        observation = np.full(len(FEATURE_NAMES), 0.5, dtype=np.float32)
        return observation, float(action == 0), True, False, {}

    def action_masks(self) -> list[bool]:
        return [True] * len(ACTION_NAMES)


def test_maskable_ppo_trains_deployable_tanh_actor() -> None:
    """Minimal PPO training returns the fixed compact ONNX actor."""
    policy = train_maskable_ppo(_MinimalMaskedEnv(), timesteps=2, seed=7)

    assert isinstance(policy, TanhMlpPolicy)
    assert policy.hidden_weights.shape == (16, len(FEATURE_NAMES))
    assert policy.hidden_bias.shape == (16,)
    assert policy.output_weights.shape == (len(ACTION_NAMES), 16)
    assert policy.output_bias.shape == (len(ACTION_NAMES),)


def test_maskable_ppo_rejects_rollout_larger_than_training_budget() -> None:
    """Requested timesteps remain an upper bound on compiler transitions."""
    with pytest.raises(ValueError, match="parameters are invalid"):
        train_maskable_ppo(_MinimalMaskedEnv(), timesteps=2, rollout_steps=3)


def test_maskable_ppo_retains_linear_actor_ablation() -> None:
    """The direct-linear actor remains available as an experiment baseline."""
    policy = train_maskable_ppo(_MinimalMaskedEnv(), timesteps=2, seed=7, hidden_dim=0)

    assert isinstance(policy, LinearPolicy)
    assert policy.weights.shape == (len(ACTION_NAMES), len(FEATURE_NAMES))
    assert policy.bias.shape == (len(ACTION_NAMES),)
