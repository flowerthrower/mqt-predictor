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
from inspect import signature
from pathlib import Path

import numpy as np
import pytest
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from gymnasium.spaces import Dict as DictSpace

from mqt.predictor.compiled import (
    ACTION_NAMES,
    FEATURE_NAMES,
    TanhMlpPolicy,
    TrainingExample,
    extract_maskable_ppo_policy,
    fit_linear_policy,
    load_training_dataset,
    target_fingerprint,
    train_maskable_ppo,
)
from mqt.predictor.compiled.policy import (
    OBSERVATION_SCHEMA,
    V3_FEATURE_NAMES,
    LinearPolicy,
    export_linear_policy,
    parameter_checksum,
)

INPUTS = Path(__file__).parents[2] / "cpp" / "test" / "Inputs"


def test_observation_matches_predictor_v3_flat_features() -> None:
    """The compiled actor consumes the exact 50 non-GNN Predictor v3 features."""
    assert FEATURE_NAMES == V3_FEATURE_NAMES
    assert tuple(sorted(FEATURE_NAMES)) == FEATURE_NAMES
    assert len(FEATURE_NAMES) == 50
    assert OBSERVATION_SCHEMA == "mqt-predictor-core-stages/3"


def test_minimal_training_is_deterministic() -> None:
    """The smoke dataset produces a reproducible 32-epoch fit."""
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
    assert artifact["training"]["epochs"] == 1
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
    """The exporter rejects spellings rejected by the C++ target loader."""
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

    observation_space = DictSpace({name: Box(0.0, 1.0, shape=(1,), dtype=np.float32) for name in FEATURE_NAMES})
    action_space = Discrete(len(ACTION_NAMES))

    @staticmethod
    def _observation(value: float) -> dict[str, np.ndarray]:
        return {name: np.asarray([value], dtype=np.float32) for name in FEATURE_NAMES}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        super().reset(seed=seed, options=options)
        return self._observation(0.0), {}

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        observation = self._observation(0.5)
        return observation, float(action == 0), True, False, {}

    def action_masks(self) -> list[bool]:
        return [True] * len(ACTION_NAMES)


def test_maskable_ppo_trains_deployable_tanh_actor() -> None:
    """Minimal PPO training keeps the exact Predictor v3 actor shape."""
    policy = train_maskable_ppo(_MinimalMaskedEnv(), timesteps=2, seed=7, n_steps=2, batch_size=2, n_epochs=1)

    assert isinstance(policy, TanhMlpPolicy)
    assert policy.first_hidden_weights.shape == (64, len(FEATURE_NAMES))
    assert policy.first_hidden_bias.shape == (64,)
    assert policy.second_hidden_weights.shape == (64, 64)
    assert policy.second_hidden_bias.shape == (64,)
    assert policy.output_weights.shape == (len(ACTION_NAMES), 64)
    assert policy.output_bias.shape == (len(ACTION_NAMES),)


def test_maskable_ppo_defaults_match_predictor_v3() -> None:
    """Training defaults retain the PPO rollout and optimizer contract from PR 798."""
    parameters = signature(train_maskable_ppo).parameters

    assert parameters["timesteps"].default == 1000
    assert parameters["n_steps"].default == 2048
    assert parameters["batch_size"].default == 64
    assert parameters["n_epochs"].default == 10


def test_maskable_ppo_extractor_preserves_pytorch_logits_and_critic_shape() -> None:
    """Actor export is numerically equivalent and the training critic stays 64x64."""
    torch = pytest.importorskip("torch")
    maskable_ppo_class = pytest.importorskip("sb3_contrib").MaskablePPO
    multi_input_policy = pytest.importorskip("sb3_contrib.common.maskable.policies").MaskableMultiInputActorCriticPolicy
    model = maskable_ppo_class(
        multi_input_policy,
        _MinimalMaskedEnv(),
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        gamma=0.98,
        seed=7,
        device="cpu",
    )
    actor = extract_maskable_ppo_policy(model)
    features = np.linspace(0.0, 1.0, len(FEATURE_NAMES), dtype=np.float32)
    named_features = {name: np.asarray([features[index]], dtype=np.float32) for index, name in enumerate(FEATURE_NAMES)}
    observation, _ = model.policy.obs_to_tensor(named_features)
    with torch.no_grad():
        extracted_features = model.policy.extract_features(observation)
        latent_actor = model.policy.mlp_extractor.forward_actor(extracted_features)
        expected = model.policy.action_net(latent_actor).cpu().numpy()[0]

    np.testing.assert_allclose(actor.logits(features.tolist()), expected, rtol=1e-6, atol=1e-6)
    value_layers = model.policy.mlp_extractor.value_net
    assert (value_layers[0].in_features, value_layers[0].out_features) == (len(FEATURE_NAMES), 64)
    assert (value_layers[2].in_features, value_layers[2].out_features) == (64, 64)
    assert model.policy.ortho_init is True
