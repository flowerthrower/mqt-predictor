# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the optional ONNX linear-policy boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from mqt.predictor.compiled import (
    ACTION_NAMES,
    FEATURE_NAMES,
    ONNX_POLICY_SCHEMA,
    LinearPolicy,
    export_onnx_policy,
    load_onnx_policy,
    onnx_policy,
)

if TYPE_CHECKING:
    from pathlib import Path


TARGET_FINGERPRINT = f"sha256:{'a' * 64}"


def _policy() -> LinearPolicy:
    weights = np.arange(len(ACTION_NAMES) * len(FEATURE_NAMES), dtype=np.float32).reshape(
        len(ACTION_NAMES), len(FEATURE_NAMES)
    )
    weights = (weights - 20.0) / 32.0
    bias = np.linspace(-0.2, 0.3, len(ACTION_NAMES), dtype=np.float32)
    return LinearPolicy(weights, bias)


def _export(path: Path) -> LinearPolicy:
    policy = _policy()
    export_onnx_policy(
        path,
        policy,
        target_fingerprint_override=TARGET_FINGERPRINT,
        core_revision="core-revision",
        training_algorithm="test-linear",
        objective="raw-logit parity",
    )
    return policy


def test_onnx_dependency_is_loaded_only_for_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing compiled support does not require the optional ONNX package."""

    def missing_module(name: str) -> object:
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(onnx_policy.importlib, "import_module", missing_module)

    with pytest.raises(ImportError, match="optional dependency 'onnx'"):
        _export(tmp_path / "policy.onnx")


def test_onnx_round_trip_preserves_raw_logits_and_metadata(tmp_path: Path) -> None:
    """ONNX Runtime reproduces the native actor before action masking."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    path = tmp_path / "policy.onnx"
    policy = _export(path)
    model = onnx.load_model(str(path))

    assert {entry.key: entry.value for entry in model.metadata_props} == {
        "schema": ONNX_POLICY_SCHEMA,
        "observation_schema": onnx_policy.OBSERVATION_SCHEMA,
        "feature_names": ",".join(FEATURE_NAMES),
        "action_names": ",".join(ACTION_NAMES),
        "target_fingerprint": TARGET_FINGERPRINT,
        "core_revision": "core-revision",
        "training_algorithm": "test-linear",
        "objective": "raw-logit parity",
    }

    features = np.linspace(0.0, 1.0, len(FEATURE_NAMES), dtype=np.float32)
    expected = policy.weights.astype(np.float64) @ features.astype(np.float64) + policy.bias.astype(np.float64)
    actual = load_onnx_policy(
        path,
        expected_target_fingerprint=TARGET_FINGERPRINT,
        expected_core_revision="core-revision",
    ).logits(features.tolist())

    np.testing.assert_allclose(actual, expected.astype(np.float32), rtol=1e-6, atol=1e-6)


def test_onnx_loader_rejects_incompatible_metadata(tmp_path: Path) -> None:
    """Runtime compatibility is checked before actor inference."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    path = tmp_path / "policy.onnx"
    _export(path)

    with pytest.raises(ValueError, match="metadata does not match"):
        load_onnx_policy(
            path,
            expected_target_fingerprint=f"sha256:{'b' * 64}",
            expected_core_revision="core-revision",
        )
