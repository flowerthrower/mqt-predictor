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
_HIDDEN_WEIGHTS_NAME = "hidden_weights"
_HIDDEN_BIAS_NAME = "hidden_bias"
_OUTPUT_WEIGHTS_NAME = "output_weights"
_OUTPUT_BIAS_NAME = "output_bias"
_METADATA_KEYS = frozenset({
    "schema",
    "observation_schema",
    "feature_names",
    "action_names",
    "target_fingerprint",
    "core_revision",
    "architecture",
    "source_revision",
    "training_algorithm",
    "objective",
})


@dataclass(frozen=True)
class TanhMlpPolicy:
    """Parameters of a fixed one-hidden-layer actor exported to ONNX."""

    hidden_weights: NDArray[np.float32]
    hidden_bias: NDArray[np.float32]
    output_weights: NDArray[np.float32]
    output_bias: NDArray[np.float32]

    def __post_init__(self) -> None:
        """Validate dimensions and immutable float32 parameter storage."""
        hidden_weights = np.asarray(self.hidden_weights, dtype=np.float32)
        hidden_bias = np.asarray(self.hidden_bias, dtype=np.float32)
        output_weights = np.asarray(self.output_weights, dtype=np.float32)
        output_bias = np.asarray(self.output_bias, dtype=np.float32)
        if hidden_weights.ndim != 2 or hidden_weights.shape[1] != len(FEATURE_NAMES):
            msg = f"hidden weights must have shape [hidden,{len(FEATURE_NAMES)}]"
            raise ValueError(msg)
        hidden_size = hidden_weights.shape[0]
        if not 0 < hidden_size <= 64 or hidden_bias.shape != (hidden_size,):
            msg = "hidden actor width must be between 1 and 64"
            raise ValueError(msg)
        if output_weights.shape != (len(ACTION_NAMES), hidden_size) or output_bias.shape != (len(ACTION_NAMES),):
            msg = f"output parameters must produce {len(ACTION_NAMES)} action logits"
            raise ValueError(msg)
        if not all(np.isfinite(value).all() for value in (hidden_weights, hidden_bias, output_weights, output_bias)):
            msg = "actor parameters must be finite"
            raise ValueError(msg)
        for name, value in (
            ("hidden_weights", hidden_weights),
            ("hidden_bias", hidden_bias),
            ("output_weights", output_weights),
            ("output_bias", output_bias),
        ):
            copied = value.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)

    def logits(self, features: Sequence[float]) -> NDArray[np.float32]:
        """Evaluate raw actor logits using float32 Tanh semantics."""
        feature_array = np.asarray(features, dtype=np.float32)
        if feature_array.shape != (len(FEATURE_NAMES),) or not np.isfinite(feature_array).all():
            msg = f"features must be a finite {len(FEATURE_NAMES)}-float vector"
            raise ValueError(msg)
        if np.any((feature_array < 0) | (feature_array > 1)):
            msg = "features must lie in [0, 1]"
            raise ValueError(msg)
        with np.errstate(over="ignore", invalid="ignore"):
            hidden = np.tanh(self.hidden_weights @ feature_array + self.hidden_bias)
            logits = np.asarray(self.output_weights @ hidden + self.output_bias, dtype=np.float32)
        if not np.isfinite(logits).all():
            msg = "Tanh policy logits exceed the float32 runtime range"
            raise ValueError(msg)
        return logits


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
    architecture: str,
    source_revision: str,
    training_algorithm: str,
    objective: str,
) -> dict[str, str]:
    if TARGET_FINGERPRINT_PATTERN.fullmatch(compiler_target_fingerprint) is None:
        msg = "target fingerprint must be a lowercase SHA-256 digest"
        raise ValueError(msg)
    if not core_revision or not architecture or not source_revision or not training_algorithm or not objective:
        msg = "ONNX policy provenance strings must be nonempty"
        raise ValueError(msg)
    return {
        "schema": ONNX_POLICY_SCHEMA,
        "observation_schema": OBSERVATION_SCHEMA,
        "feature_names": ",".join(FEATURE_NAMES),
        "action_names": ",".join(ACTION_NAMES),
        "target_fingerprint": compiler_target_fingerprint,
        "core_revision": core_revision,
        "architecture": architecture,
        "source_revision": source_revision,
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
        architecture=metadata["architecture"],
        source_revision=metadata["source_revision"],
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

    if any(initializer.data_location != onnx.TensorProto.DEFAULT for initializer in graph.initializer):
        msg = "ONNX policy parameters must be embedded"
        raise ValueError(msg)
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    if len(initializers) != len(graph.initializer):
        msg = "ONNX policy initializer names must be unique"
        raise ValueError(msg)

    def attributes(node: Any) -> dict[str, Any]:  # ruff: ignore[any-type]
        return {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}

    linear = len(graph.node) == 1 and set(initializers) == {_WEIGHTS_NAME, _BIAS_NAME}
    if linear:
        node = graph.node[0]
        if (
            node.op_type != "Gemm"
            or node.domain
            or tuple(node.input) != (ONNX_INPUT_NAME, _WEIGHTS_NAME, _BIAS_NAME)
            or tuple(node.output) != (ONNX_OUTPUT_NAME,)
            or attributes(node) != {"transB": 1}
        ):
            msg = "ONNX linear policy graph does not match the supported actor"
            raise ValueError(msg)
        weights = np.asarray(onnx.numpy_helper.to_array(initializers[_WEIGHTS_NAME]), dtype=np.float32)
        bias = np.asarray(onnx.numpy_helper.to_array(initializers[_BIAS_NAME]), dtype=np.float32)
        parameter_checksum(weights, bias)
        if metadata["architecture"] != "linear":
            msg = "ONNX policy architecture metadata does not match its graph"
            raise ValueError(msg)
        return

    expected_initializers = {
        _HIDDEN_WEIGHTS_NAME,
        _HIDDEN_BIAS_NAME,
        _OUTPUT_WEIGHTS_NAME,
        _OUTPUT_BIAS_NAME,
    }
    if len(graph.node) != 3 or set(initializers) != expected_initializers:
        msg = "ONNX policy must be a linear or one-hidden-layer Tanh actor"
        raise ValueError(msg)
    hidden_gemm, activation, output_gemm = graph.node
    if (
        hidden_gemm.op_type != "Gemm"
        or hidden_gemm.domain
        or tuple(hidden_gemm.input) != (ONNX_INPUT_NAME, _HIDDEN_WEIGHTS_NAME, _HIDDEN_BIAS_NAME)
        or tuple(hidden_gemm.output) != ("hidden_pre_activation",)
        or attributes(hidden_gemm) != {"transB": 1}
        or activation.op_type != "Tanh"
        or activation.domain
        or tuple(activation.input) != ("hidden_pre_activation",)
        or tuple(activation.output) != ("hidden",)
        or attributes(activation)
        or output_gemm.op_type != "Gemm"
        or output_gemm.domain
        or tuple(output_gemm.input) != ("hidden", _OUTPUT_WEIGHTS_NAME, _OUTPUT_BIAS_NAME)
        or tuple(output_gemm.output) != (ONNX_OUTPUT_NAME,)
        or attributes(output_gemm) != {"transB": 1}
    ):
        msg = "ONNX Tanh policy graph does not match the supported actor"
        raise ValueError(msg)
    policy = TanhMlpPolicy(
        hidden_weights=np.asarray(onnx.numpy_helper.to_array(initializers[_HIDDEN_WEIGHTS_NAME]), dtype=np.float32),
        hidden_bias=np.asarray(onnx.numpy_helper.to_array(initializers[_HIDDEN_BIAS_NAME]), dtype=np.float32),
        output_weights=np.asarray(onnx.numpy_helper.to_array(initializers[_OUTPUT_WEIGHTS_NAME]), dtype=np.float32),
        output_bias=np.asarray(onnx.numpy_helper.to_array(initializers[_OUTPUT_BIAS_NAME]), dtype=np.float32),
    )
    if metadata["architecture"] != f"tanh-mlp-{policy.hidden_weights.shape[0]}":
        msg = "ONNX policy architecture metadata does not match its graph"
        raise ValueError(msg)


def export_onnx_policy(
    path: Path,
    policy: LinearPolicy | TanhMlpPolicy,
    *,
    target: Path | None = None,
    target_fingerprint_override: str | None = None,
    core_revision: str,
    source_revision: str,
    training_algorithm: str,
    objective: str,
) -> None:
    """Export a strict linear or one-hidden-layer ONNX actor.

    The optional :mod:`onnx` package is imported only when this function is
    called.
    """
    if (target is None) == (target_fingerprint_override is None):
        msg = "exactly one target or target fingerprint must be provided"
        raise ValueError(msg)
    compiler_target_fingerprint = (
        target_fingerprint(target) if target is not None else cast("str", target_fingerprint_override)
    )
    architecture = "linear" if isinstance(policy, LinearPolicy) else f"tanh-mlp-{policy.hidden_weights.shape[0]}"
    metadata = _metadata(
        compiler_target_fingerprint=compiler_target_fingerprint,
        core_revision=core_revision,
        architecture=architecture,
        source_revision=source_revision,
        training_algorithm=training_algorithm,
        objective=objective,
    )
    onnx = _optional_module("onnx", "export an ONNX policy")

    input_info = onnx.helper.make_tensor_value_info(ONNX_INPUT_NAME, onnx.TensorProto.FLOAT, [1, len(FEATURE_NAMES)])
    output_info = onnx.helper.make_tensor_value_info(ONNX_OUTPUT_NAME, onnx.TensorProto.FLOAT, [1, len(ACTION_NAMES)])
    if isinstance(policy, LinearPolicy):
        initializers = [
            onnx.numpy_helper.from_array(policy.weights, name=_WEIGHTS_NAME),
            onnx.numpy_helper.from_array(policy.bias, name=_BIAS_NAME),
        ]
        nodes = [
            onnx.helper.make_node(
                "Gemm",
                [ONNX_INPUT_NAME, _WEIGHTS_NAME, _BIAS_NAME],
                [ONNX_OUTPUT_NAME],
                name="linear_actor",
                transB=1,
            )
        ]
        graph_name = "mqt_predictor_linear_actor"
    else:
        initializers = [
            onnx.numpy_helper.from_array(policy.hidden_weights, name=_HIDDEN_WEIGHTS_NAME),
            onnx.numpy_helper.from_array(policy.hidden_bias, name=_HIDDEN_BIAS_NAME),
            onnx.numpy_helper.from_array(policy.output_weights, name=_OUTPUT_WEIGHTS_NAME),
            onnx.numpy_helper.from_array(policy.output_bias, name=_OUTPUT_BIAS_NAME),
        ]
        nodes = [
            onnx.helper.make_node(
                "Gemm",
                [ONNX_INPUT_NAME, _HIDDEN_WEIGHTS_NAME, _HIDDEN_BIAS_NAME],
                ["hidden_pre_activation"],
                name="hidden_linear",
                transB=1,
            ),
            onnx.helper.make_node("Tanh", ["hidden_pre_activation"], ["hidden"], name="hidden_tanh"),
            onnx.helper.make_node(
                "Gemm",
                ["hidden", _OUTPUT_WEIGHTS_NAME, _OUTPUT_BIAS_NAME],
                [ONNX_OUTPUT_NAME],
                name="action_logits",
                transB=1,
            ),
        ]
        graph_name = "mqt_predictor_tanh_actor"
    graph = onnx.helper.make_graph(
        nodes,
        graph_name,
        [input_info],
        [output_info],
        initializer=initializers,
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
