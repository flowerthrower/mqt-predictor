# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Optional ONNX serialization and inference for the compiled linear actor."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .policy import (
    ACTION_NAMES,
    FEATURE_NAMES,
    OBSERVATION_SCHEMA,
    TARGET_FINGERPRINT_PATTERN,
    LinearPolicy,
    parameter_checksum,
    target_fingerprint,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray

ONNX_POLICY_SCHEMA = "mqt-predictor-onnx-policy/1"
ONNX_OPSET = 17
ONNX_INPUT_NAME = "features"
ONNX_OUTPUT_NAME = "logits"
MAX_MODEL_BYTES = 1024 * 1024

_WEIGHTS_NAME = "weights"
_BIAS_NAME = "bias"
_METADATA_KEYS = frozenset({
    "schema",
    "observation_schema",
    "feature_names",
    "action_names",
    "target_fingerprint",
    "core_revision",
    "training_algorithm",
    "objective",
})


def _optional_module(name: str, purpose: str) -> Any:  # ruff: ignore[any-type]
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        msg = f"optional dependency '{name}' is required to {purpose}"
        raise ImportError(msg) from error


def _metadata(
    *,
    compiler_target_fingerprint: str,
    core_revision: str,
    training_algorithm: str,
    objective: str,
) -> dict[str, str]:
    if TARGET_FINGERPRINT_PATTERN.fullmatch(compiler_target_fingerprint) is None:
        msg = "target fingerprint must be a lowercase SHA-256 digest"
        raise ValueError(msg)
    if not core_revision or not training_algorithm or not objective:
        msg = "ONNX policy provenance strings must be nonempty"
        raise ValueError(msg)
    return {
        "schema": ONNX_POLICY_SCHEMA,
        "observation_schema": OBSERVATION_SCHEMA,
        "feature_names": ",".join(FEATURE_NAMES),
        "action_names": ",".join(ACTION_NAMES),
        "target_fingerprint": compiler_target_fingerprint,
        "core_revision": core_revision,
        "training_algorithm": training_algorithm,
        "objective": objective,
    }


def _tensor_shape(value_info: Any) -> tuple[int, ...] | None:  # ruff: ignore[any-type]
    tensor_type = value_info.type.tensor_type
    dimensions: list[int] = []
    for dimension in tensor_type.shape.dim:
        if not dimension.HasField("dim_value"):
            return None
        dimensions.append(dimension.dim_value)
    return tuple(dimensions)


def _validate_model(
    model: Any,  # ruff: ignore[any-type]
    onnx: Any,  # ruff: ignore[any-type]
    *,
    expected_target_fingerprint: str,
    expected_core_revision: str,
) -> None:
    onnx.checker.check_model(model, full_check=True)

    metadata = {entry.key: entry.value for entry in model.metadata_props}
    if len(metadata) != len(model.metadata_props) or set(metadata) != _METADATA_KEYS:
        msg = "ONNX policy metadata fields do not match the schema"
        raise ValueError(msg)
    expected_metadata = _metadata(
        compiler_target_fingerprint=expected_target_fingerprint,
        core_revision=expected_core_revision,
        training_algorithm=metadata["training_algorithm"],
        objective=metadata["objective"],
    )
    if metadata != expected_metadata:
        msg = "ONNX policy metadata does not match this runtime"
        raise ValueError(msg)

    imports = {(item.domain, item.version) for item in model.opset_import}
    if imports != {("", ONNX_OPSET)}:
        msg = f"ONNX policy must use default-domain opset {ONNX_OPSET}"
        raise ValueError(msg)
    graph = model.graph
    if len(graph.input) != 1 or len(graph.output) != 1:
        msg = "ONNX policy must have one input and one output"
        raise ValueError(msg)
    input_info = graph.input[0]
    output_info = graph.output[0]
    if (
        input_info.name != ONNX_INPUT_NAME
        or input_info.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
        or _tensor_shape(input_info) != (1, len(FEATURE_NAMES))
    ):
        msg = f"ONNX policy input must be float32 features[1,{len(FEATURE_NAMES)}]"
        raise ValueError(msg)
    if (
        output_info.name != ONNX_OUTPUT_NAME
        or output_info.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
        or _tensor_shape(output_info) != (1, len(ACTION_NAMES))
    ):
        msg = f"ONNX policy output must be float32 logits[1,{len(ACTION_NAMES)}]"
        raise ValueError(msg)

    if len(graph.node) != 1:
        msg = "ONNX linear policy must contain exactly one node"
        raise ValueError(msg)
    node = graph.node[0]
    attributes = {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}
    if (
        node.op_type != "Gemm"
        or node.domain
        or tuple(node.input) != (ONNX_INPUT_NAME, _WEIGHTS_NAME, _BIAS_NAME)
        or tuple(node.output) != (ONNX_OUTPUT_NAME,)
        or attributes != {"transB": 1}
    ):
        msg = "ONNX linear policy graph does not match the supported actor"
        raise ValueError(msg)

    if len(graph.initializer) != 2 or any(
        initializer.data_location != onnx.TensorProto.DEFAULT for initializer in graph.initializer
    ):
        msg = "ONNX linear policy must contain two embedded initializers"
        raise ValueError(msg)
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    if set(initializers) != {_WEIGHTS_NAME, _BIAS_NAME}:
        msg = "ONNX linear policy initializer names do not match the schema"
        raise ValueError(msg)
    weights = np.asarray(onnx.numpy_helper.to_array(initializers[_WEIGHTS_NAME]), dtype=np.float32)
    bias = np.asarray(onnx.numpy_helper.to_array(initializers[_BIAS_NAME]), dtype=np.float32)
    parameter_checksum(weights, bias)


def export_onnx_policy(
    path: Path,
    policy: LinearPolicy,
    *,
    target: Path | None = None,
    target_fingerprint_override: str | None = None,
    core_revision: str,
    training_algorithm: str,
    objective: str,
) -> None:
    """Export a strict linear ONNX actor matching the compiled policy ABI.

    The optional :mod:`onnx` package is imported only when this function is
    called.
    """
    if (target is None) == (target_fingerprint_override is None):
        msg = "exactly one target or target fingerprint must be provided"
        raise ValueError(msg)
    compiler_target_fingerprint = (
        target_fingerprint(target) if target is not None else cast("str", target_fingerprint_override)
    )
    metadata = _metadata(
        compiler_target_fingerprint=compiler_target_fingerprint,
        core_revision=core_revision,
        training_algorithm=training_algorithm,
        objective=objective,
    )
    onnx = _optional_module("onnx", "export an ONNX policy")

    input_info = onnx.helper.make_tensor_value_info(ONNX_INPUT_NAME, onnx.TensorProto.FLOAT, [1, len(FEATURE_NAMES)])
    output_info = onnx.helper.make_tensor_value_info(ONNX_OUTPUT_NAME, onnx.TensorProto.FLOAT, [1, len(ACTION_NAMES)])
    weights = onnx.numpy_helper.from_array(policy.weights, name=_WEIGHTS_NAME)
    bias = onnx.numpy_helper.from_array(policy.bias, name=_BIAS_NAME)
    node = onnx.helper.make_node(
        "Gemm",
        [ONNX_INPUT_NAME, _WEIGHTS_NAME, _BIAS_NAME],
        [ONNX_OUTPUT_NAME],
        name="linear_actor",
        transB=1,
    )
    graph = onnx.helper.make_graph(
        [node],
        "mqt_predictor_linear_actor",
        [input_info],
        [output_info],
        initializer=[weights, bias],
    )
    model = onnx.helper.make_model(
        graph,
        producer_name="mqt.predictor",
        model_version=1,
        opset_imports=[onnx.helper.make_opsetid("", ONNX_OPSET)],
    )
    model.ir_version = 8
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    _validate_model(
        model,
        onnx,
        expected_target_fingerprint=compiler_target_fingerprint,
        expected_core_revision=core_revision,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(path))


@dataclass(frozen=True)
class OnnxPolicy:
    """A validated ONNX Runtime session returning unmasked actor logits."""

    _session: Any

    def logits(self, features: Sequence[float]) -> NDArray[np.float32]:
        """Evaluate one normalized feature vector without applying an action mask."""
        feature_array = np.asarray(features, dtype=np.float32)
        if feature_array.shape != (len(FEATURE_NAMES),) or not np.isfinite(feature_array).all():
            msg = f"features must be a finite {len(FEATURE_NAMES)}-float vector"
            raise ValueError(msg)
        if np.any((feature_array < 0) | (feature_array > 1)):
            msg = "features must lie in [0, 1]"
            raise ValueError(msg)
        outputs = self._session.run([ONNX_OUTPUT_NAME], {ONNX_INPUT_NAME: feature_array[np.newaxis, :]})
        if len(outputs) != 1:
            msg = "ONNX Runtime returned an unexpected number of outputs"
            raise RuntimeError(msg)
        logits = np.asarray(outputs[0])
        if logits.dtype != np.float32 or logits.shape != (1, len(ACTION_NAMES)) or not np.isfinite(logits).all():
            msg = "ONNX Runtime returned invalid policy logits"
            raise RuntimeError(msg)
        return logits[0].copy()


def load_onnx_policy(
    path: Path,
    *,
    expected_target_fingerprint: str,
    expected_core_revision: str,
) -> OnnxPolicy:
    """Validate and load an ONNX actor using the CPU execution provider."""
    try:
        size = path.stat().st_size
    except OSError as error:
        msg = f"failed to read ONNX policy: {path}"
        raise ValueError(msg) from error
    if size > MAX_MODEL_BYTES:
        msg = "ONNX policy exceeds the 1 MiB size limit"
        raise ValueError(msg)

    onnx = _optional_module("onnx", "validate an ONNX policy")
    try:
        model = onnx.load_model(str(path), load_external_data=False)
        _validate_model(
            model,
            onnx,
            expected_target_fingerprint=expected_target_fingerprint,
            expected_core_revision=expected_core_revision,
        )
    except (OSError, onnx.checker.ValidationError) as error:
        msg = f"failed to validate ONNX policy: {path}"
        raise ValueError(msg) from error

    runtime = _optional_module("onnxruntime", "run an ONNX policy")
    options = runtime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = runtime.ExecutionMode.ORT_SEQUENTIAL
    try:
        session = runtime.InferenceSession(
            str(path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as error:
        msg = f"ONNX Runtime rejected policy: {path}"
        raise ValueError(msg) from error

    runtime_inputs = session.get_inputs()
    runtime_outputs = session.get_outputs()
    if (
        len(runtime_inputs) != 1
        or runtime_inputs[0].name != ONNX_INPUT_NAME
        or runtime_inputs[0].type != "tensor(float)"
        or runtime_inputs[0].shape != [1, len(FEATURE_NAMES)]
        or len(runtime_outputs) != 1
        or runtime_outputs[0].name != ONNX_OUTPUT_NAME
        or runtime_outputs[0].type != "tensor(float)"
        or runtime_outputs[0].shape != [1, len(ACTION_NAMES)]
    ):
        msg = "ONNX Runtime policy interface does not match the compiled ABI"
        raise ValueError(msg)
    return OnnxPolicy(session)
