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

from mqt.predictor.compiled import ACTION_NAMES, fit_linear_policy, load_training_dataset, target_fingerprint
from mqt.predictor.compiled.policy import LinearPolicy, parameter_checksum

INPUTS = Path(__file__).parents[2] / "cpp" / "test" / "Inputs"


def test_minimal_training_is_deterministic() -> None:
    """The 32-step smoke training run is reproducible."""
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


def test_policy_masks_illegal_actions() -> None:
    """Masked actions cannot win native inference."""
    weights = np.zeros((len(ACTION_NAMES), 7), dtype=np.float32)
    bias = np.arange(len(ACTION_NAMES), dtype=np.float32)
    policy = LinearPolicy(weights, bias)

    selected, logits = policy.select([0.5] * 7, [True, True, False, False, False, False])

    assert selected == 1
    assert np.isneginf(logits[2:]).all()


def test_training_rejects_illegal_label() -> None:
    """A training label must be legal in its recorded state."""
    examples = load_training_dataset(INPUTS / "line-4-training.json")
    example = examples[0]

    with pytest.raises(ValueError, match="enabled"):
        type(example)(features=example.features, legal=(False, True, False, False, False, False), action=0)


@pytest.mark.parametrize("invalid_operation", ["cnot", "U", " cx "])
def test_target_fingerprint_rejects_noncanonical_operations(tmp_path: Path, invalid_operation: str) -> None:
    """The exporter rejects spellings rejected by the C++ v1 loader."""
    document = json.loads((INPUTS / "line-4-target.json").read_text(encoding="utf-8"))
    document["operations"][1]["name"] = invalid_operation
    target = tmp_path / "target.json"
    target.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical"):
        target_fingerprint(target)
