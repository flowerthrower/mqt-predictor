# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Deterministic masked-softmax trainer for the native linear actor."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .policy import ACTION_NAMES, FEATURE_NAMES, LinearPolicy

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


TRAINING_DATASET_SCHEMA = "mqt-predictor-native-dataset/1"


@dataclass(frozen=True)
class TrainingExample:
    """One masked policy-imitation example."""

    features: tuple[float, ...]
    legal: tuple[bool, ...]
    action: int
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate the feature, mask, label, and sample weight."""
        features = np.asarray(self.features, dtype=np.float64)
        if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
            msg = "training features must be a finite seven-float vector"
            raise ValueError(msg)
        if np.any((features < 0) | (features > 1)):
            msg = "training features must lie in [0, 1]"
            raise ValueError(msg)
        if len(self.legal) != len(ACTION_NAMES) or not any(self.legal):
            msg = "training mask must enable at least one of six actions"
            raise ValueError(msg)
        if not 0 <= self.action < len(ACTION_NAMES) or not self.legal[self.action]:
            msg = "training action must be enabled by its mask"
            raise ValueError(msg)
        if not np.isfinite(self.weight) or self.weight <= 0:
            msg = "training weight must be finite and positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class TrainingResult:
    """A fitted native policy and basic fit diagnostics."""

    policy: LinearPolicy
    loss: float
    accuracy: float


def _object(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        msg = f"{context} must be a JSON object"
        raise ValueError(msg)
    return cast("dict[str, Any]", value)


def _array(value: object, context: str) -> list[Any]:
    if not isinstance(value, list):
        msg = f"{context} must be a JSON array"
        raise TypeError(msg)
    return cast("list[Any]", value)


def load_training_dataset(path: Path) -> list[TrainingExample]:
    """Load and validate a native policy-imitation dataset."""
    try:
        root = _object(json.loads(path.read_text(encoding="utf-8")), "dataset")
    except (OSError, json.JSONDecodeError) as error:
        msg = f"failed to read training dataset: {path}"
        raise ValueError(msg) from error
    if set(root) != {"schema", "feature_names", "action_names", "samples"}:
        msg = "training dataset fields do not match the native schema"
        raise ValueError(msg)
    if root["schema"] != TRAINING_DATASET_SCHEMA or tuple(root["feature_names"]) != FEATURE_NAMES:
        msg = "training observation schema does not match the native ABI"
        raise ValueError(msg)
    if tuple(root["action_names"]) != ACTION_NAMES:
        msg = "training action order does not match the native ABI"
        raise ValueError(msg)

    examples: list[TrainingExample] = []
    for index, value in enumerate(_array(root["samples"], "dataset.samples")):
        sample = _object(value, f"dataset.samples[{index}]")
        if set(sample) - {"features", "legal", "action", "weight"} or not {"features", "legal", "action"} <= set(
            sample
        ):
            msg = "training sample fields do not match the native schema"
            raise ValueError(msg)
        action_name = sample["action"]
        if not isinstance(action_name, str) or action_name not in ACTION_NAMES:
            msg = "training sample action is unknown"
            raise ValueError(msg)
        features = tuple(float(value) for value in _array(sample["features"], "sample.features"))
        legal_values = _array(sample["legal"], "sample.legal")
        if not all(type(value) is bool for value in legal_values):
            msg = "training legal mask must contain booleans"
            raise ValueError(msg)
        weight = float(sample.get("weight", 1.0))
        examples.append(
            TrainingExample(
                features=features,
                legal=tuple(legal_values),
                action=ACTION_NAMES.index(action_name),
                weight=weight,
            )
        )
    if not examples:
        msg = "training dataset must contain at least one sample"
        raise ValueError(msg)
    return examples


def fit_linear_policy(
    examples: Sequence[TrainingExample],
    *,
    epochs: int = 32,
    learning_rate: float = 0.2,
    l2: float = 1e-4,
    seed: int = 7,
) -> TrainingResult:
    """Fit an action-masked linear actor by weighted behavioral cloning."""
    if not examples:
        msg = "at least one training example is required"
        raise ValueError(msg)
    if epochs <= 0 or learning_rate <= 0 or l2 < 0 or seed < 0:
        msg = "training hyperparameters are invalid"
        raise ValueError(msg)
    features = np.asarray([example.features for example in examples], dtype=np.float64)
    legal = np.asarray([example.legal for example in examples], dtype=np.bool_)
    actions = np.asarray([example.action for example in examples], dtype=np.intp)
    sample_weights = np.asarray([example.weight for example in examples], dtype=np.float64)

    generator = np.random.default_rng(seed)
    weights = generator.normal(0.0, 0.01, size=(len(ACTION_NAMES), len(FEATURE_NAMES)))
    bias = np.zeros(len(ACTION_NAMES), dtype=np.float64)
    rows = np.arange(len(examples))
    normalization = float(sample_weights.sum())

    loss = float("inf")
    probabilities = np.zeros_like(legal, dtype=np.float64)
    for _ in range(epochs):
        logits = features @ weights.T + bias
        logits[~legal] = -np.inf
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities[~legal] = 0.0
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        chosen = np.maximum(probabilities[rows, actions], np.finfo(np.float64).tiny)
        loss = float(-(sample_weights * np.log(chosen)).sum() / normalization + 0.5 * l2 * np.square(weights).sum())

        error = probabilities.copy()
        error[rows, actions] -= 1.0
        error *= sample_weights[:, None] / normalization
        weights -= learning_rate * (error.T @ features + l2 * weights)
        bias -= learning_rate * error.sum(axis=0)

    predictions = np.argmax(np.where(legal, features @ weights.T + bias, -np.inf), axis=1)
    accuracy = float(np.average(predictions == actions, weights=sample_weights))
    policy = LinearPolicy(weights=weights.astype(np.float32), bias=bias.astype(np.float32))
    return TrainingResult(policy=policy, loss=loss, accuracy=accuracy)
