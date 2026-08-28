# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Native linear-policy artifact contract shared with the C++ runtime."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray

OBSERVATION_SCHEMA = "mqt-predictor-core-stages/5"
NATIVE_POLICY_SCHEMA = "mqt-predictor-native-policy/1"
COMPILER_TARGET_SCHEMA = "mqt-compiler-target/1"
TARGET_FINGERPRINT_SCHEMA = b"mqt-compiler-target-fingerprint/2"
TARGET_FINGERPRINT_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
MAX_STEPS = 20

V3_OPERATION_NAMES = (
    "u3",
    "u2",
    "u1",
    "cx",
    "id",
    "u0",
    "u",
    "p",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "rx",
    "ry",
    "rz",
    "sx",
    "sxdg",
    "cz",
    "cy",
    "swap",
    "ch",
    "ccx",
    "cswap",
    "crx",
    "cry",
    "crz",
    "cu1",
    "cp",
    "cu3",
    "csx",
    "cu",
    "rxx",
    "rzz",
    "rccx",
    "rc3x",
    "c3x",
    "c3sqrtx",
    "c4x",
    "measure",
    "r",
)

V3_FEATURE_NAMES = (
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
    "r",
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

FEATURE_NAMES = V3_FEATURE_NAMES
ACTION_NAMES = (
    "merge-single-qubit-rotation-gates",
    "fuse-single-qubit-unitary-runs",
    "fuse-two-qubit-gates",
    "place-and-route",
    "synthesize-for-target",
    "terminate",
)


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        msg = f"{context} must be a JSON object"
        raise ValueError(msg)
    return cast("dict[str, Any]", value)


def _list(value: object, context: str) -> list[Any]:
    if not isinstance(value, list):
        msg = f"{context} must be a JSON array"
        raise TypeError(msg)
    return cast("list[Any]", value)


def _integer(value: object, context: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive) or value > 2**63 - 1:
        qualifier = "positive" if positive else "nonnegative"
        msg = f"{context} must be a {qualifier} integer"
        raise ValueError(msg)
    return value


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        msg = f"{context} must be a nonempty string"
        raise ValueError(msg)
    return value


def _append_u64(payload: bytearray, value: int) -> None:
    payload.extend(struct.pack("<Q", value))


def _append_string(payload: bytearray, value: str) -> None:
    encoded = value.encode()
    _append_u64(payload, len(encoded))
    payload.extend(encoded)


def _normalized_target(
    document: object,
) -> tuple[str, list[int], list[tuple[int, int]] | None, list[tuple[str, int, int]]]:
    root = _mapping(document, "target")
    if set(root) - {"schema", "name", "sites", "couplings", "operations"}:
        msg = "target contains unexpected fields"
        raise ValueError(msg)
    if root.get("schema") != COMPILER_TARGET_SCHEMA:
        msg = "target schema is not supported"
        raise ValueError(msg)
    name = _string(root.get("name"), "target.name")
    sites = [_integer(value, "target.sites[]") for value in _list(root.get("sites"), "target.sites")]
    if len(sites) < 2 or len(sites) != len(set(sites)):
        msg = "target.sites must contain at least two unique IDs"
        raise ValueError(msg)
    site_set = set(sites)

    couplings: list[tuple[int, int]] | None = None
    if "couplings" in root:
        couplings_value = root["couplings"]
        couplings = []
        for index, value in enumerate(_list(couplings_value, "target.couplings")):
            pair = _list(value, f"target.couplings[{index}]")
            if len(pair) != 2:
                msg = "each target coupling must contain two site IDs"
                raise ValueError(msg)
            source = _integer(pair[0], "coupling source")
            destination = _integer(pair[1], "coupling destination")
            if source not in site_set or destination not in site_set or source == destination:
                msg = "target coupling contains an unknown or repeated site"
                raise ValueError(msg)
            couplings.append((min(source, destination), max(source, destination)))
        couplings.sort()
        if len(couplings) != len(set(couplings)):
            msg = "target couplings must be unique"
            raise ValueError(msg)

    operations: list[tuple[str, int, int]] = []
    for index, value in enumerate(_list(root.get("operations"), "target.operations")):
        operation = _mapping(value, f"target.operations[{index}]")
        if set(operation) != {"name", "num_qubits", "num_parameters"}:
            msg = "target operation fields do not match the v1 schema"
            raise ValueError(msg)
        operation_name = _string(operation.get("name"), "operation.name")
        canonical_name = operation_name.strip().lower()
        canonical_name = {"prx": "r", "u3": "u", "cnot": "cx"}.get(canonical_name, canonical_name)
        if operation_name != canonical_name:
            msg = "operation names must use canonical lowercase spelling"
            raise ValueError(msg)
        operations.append((
            operation_name,
            _integer(operation.get("num_qubits"), "operation.num_qubits", positive=True),
            _integer(operation.get("num_parameters"), "operation.num_parameters"),
        ))
    if not operations or len(operations) != len(set(operations)):
        msg = "target operations must contain unique capabilities"
        raise ValueError(msg)
    operations.sort()
    return name, sites, couplings, operations


def target_fingerprint(path: Path) -> str:
    """Return the calibration-sensitive fingerprint produced by the C++ target loader."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"failed to read target artifact: {path}"
        raise ValueError(msg) from error
    name, sites, couplings, operations = _normalized_target(document)

    payload = bytearray(TARGET_FINGERPRINT_SCHEMA)
    payload.append(0)
    payload.append(1)
    _append_string(payload, name)
    payload.append(0)  # no duration unit in the JSON target schema
    _append_u64(payload, len(sites))
    for site in sites:
        _append_u64(payload, site)
        payload.extend((0, 0, 0))  # no site name, T1, or T2
    payload.append(couplings is not None)
    if couplings is not None:
        _append_u64(payload, len(couplings))
        for source, destination in couplings:
            _append_u64(payload, source)
            _append_u64(payload, destination)
    payload.append(1)
    _append_u64(payload, len(operations))
    for operation_name, num_qubits, num_parameters in operations:
        _append_string(payload, operation_name)
        _append_u64(payload, num_qubits)
        _append_u64(payload, num_parameters)
        payload.extend((0, 0))  # no default duration or fidelity
        _append_u64(payload, 0)  # no calibrated site tuples
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def parameter_checksum(weights: NDArray[np.float32], bias: NDArray[np.float32]) -> str:
    """Hash action-major float32 parameters using the native ABI."""
    checked_weights = np.asarray(weights, dtype=np.float32)
    checked_bias = np.asarray(bias, dtype=np.float32)
    if checked_weights.shape != (len(ACTION_NAMES), len(FEATURE_NAMES)) or checked_bias.shape != (len(ACTION_NAMES),):
        msg = "linear parameter dimensions do not match the native ABI"
        raise ValueError(msg)
    if not np.isfinite(checked_weights).all() or not np.isfinite(checked_bias).all():
        msg = "linear parameters must be finite"
        raise ValueError(msg)
    normalized_weights = checked_weights.copy()
    normalized_bias = checked_bias.copy()
    normalized_weights[normalized_weights == 0] = 0
    normalized_bias[normalized_bias == 0] = 0
    payload = bytearray(NATIVE_POLICY_SCHEMA.encode())
    payload.append(0)
    payload.extend(struct.pack("<II", len(FEATURE_NAMES), len(ACTION_NAMES)))
    payload.extend(normalized_weights.astype("<f4", copy=False).tobytes(order="C"))
    payload.extend(normalized_bias.astype("<f4", copy=False).tobytes(order="C"))
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True)
class LinearPolicy:
    """A validated Python view of the native linear actor."""

    weights: NDArray[np.float32]
    bias: NDArray[np.float32]

    def __post_init__(self) -> None:
        """Validate and normalize the native parameters."""
        weights = np.asarray(self.weights, dtype=np.float32)
        bias = np.asarray(self.bias, dtype=np.float32)
        parameter_checksum(weights, bias)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "bias", bias)

    def select(self, features: Sequence[float], legal: Sequence[bool]) -> tuple[int, NDArray[np.float32]]:
        """Select the highest-logit legal action with C++ tie semantics."""
        feature_array = np.asarray(features, dtype=np.float32)
        legal_array = np.asarray(legal, dtype=np.bool_)
        if feature_array.shape != (len(FEATURE_NAMES),) or not np.isfinite(feature_array).all():
            msg = f"features must be a finite {len(FEATURE_NAMES)}-float vector"
            raise ValueError(msg)
        if np.any((feature_array < 0) | (feature_array > 1)):
            msg = "features must lie in [0, 1]"
            raise ValueError(msg)
        if legal_array.shape != (len(ACTION_NAMES),) or not legal_array.any():
            msg = f"legal must enable at least one of {len(ACTION_NAMES)} actions"
            raise ValueError(msg)
        logits64 = self.weights.astype(np.float64) @ feature_array.astype(np.float64) + self.bias.astype(np.float64)
        if not np.isfinite(logits64).all() or np.any(np.abs(logits64) > np.finfo(np.float32).max):
            msg = "linear policy logits exceed the float32 runtime range"
            raise ValueError(msg)
        logits = logits64.astype(np.float32)
        logits[~legal_array] = -np.inf
        return int(np.argmax(logits)), logits


def export_linear_policy(
    path: Path,
    policy: LinearPolicy,
    *,
    target: Path | None = None,
    target_fingerprint_override: str | None = None,
    core_revision: str,
    source_revision: str,
    algorithm: str,
    objective: str,
    samples: int,
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
) -> None:
    """Write one strict JSON artifact consumable by the C++ runtime."""
    if not core_revision or not source_revision or not algorithm or not objective:
        msg = "artifact provenance strings must be nonempty"
        raise ValueError(msg)
    if samples <= 0 or epochs <= 0 or learning_rate <= 0 or l2 < 0 or seed < 0:
        msg = "artifact training metadata is invalid"
        raise ValueError(msg)
    if (target is None) == (target_fingerprint_override is None):
        msg = "exactly one target or target fingerprint must be provided"
        raise ValueError(msg)
    compiler_target_fingerprint = (
        target_fingerprint(target) if target is not None else cast("str", target_fingerprint_override)
    )
    if TARGET_FINGERPRINT_PATTERN.fullmatch(compiler_target_fingerprint) is None:
        msg = "target fingerprint must be a lowercase SHA-256 digest"
        raise ValueError(msg)
    document = {
        "schema": NATIVE_POLICY_SCHEMA,
        "observation_schema": OBSERVATION_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "action_names": list(ACTION_NAMES),
        "architecture": {
            "type": "linear",
            "input_size": len(FEATURE_NAMES),
            "output_size": len(ACTION_NAMES),
        },
        "parameters": {
            "weights": [[float(value) for value in row] for row in policy.weights],
            "bias": [float(value) for value in policy.bias],
        },
        "parameters_sha256": parameter_checksum(policy.weights, policy.bias),
        "compatibility": {
            "target_fingerprint": compiler_target_fingerprint,
            "core_revision": core_revision,
        },
        "training": {
            "algorithm": algorithm,
            "objective": objective,
            "source_revision": source_revision,
            "samples": samples,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
            "seed": seed,
        },
    }
    if not math.isfinite(learning_rate) or not math.isfinite(l2):
        msg = "artifact training scalars must be finite"
        raise ValueError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
