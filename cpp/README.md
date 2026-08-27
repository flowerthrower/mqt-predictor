# Compiled Core pass-ordering experiment

This directory contains an experimental MLIR `ModuleOp` pass that orders only
MQT Core QCO passes. Python is used to train and export an actor. Deployment is
C++ only: ONNX Runtime evaluates the actor inside the MLIR pass, and MQT Core
executes every selected transformation. No Qiskit, TKET, BQSKit, or other SDK
compiler pass is a policy action.

The current Python environment and compiled runtime use this compatibility
identity:

```text
3036c91238449452d53cb6aca5d02ce503d8f1ac+patch.e761b935ad001c122eb34044da3468844a7caf2d35edaff41e63492fcd665a01
```

The first component is the exact MQT Core revision. The second is the SHA-256 of
`patches/mqt-core-predictor-stages.patch`, which exposes the staged target
operations and orders mapping-ready operations by MLIR block order.

## Runtime design

```text
Python: CorePredictorEnv -> MaskablePPO -> 13x16x6 Tanh actor -> ONNX
                                                             |
C++: MLIR pass -> features and legal mask -> ONNX Runtime ----+
       |                                      |
       +------ persistent Core QCO program <--+
```

`CorePredictorEnv` converts input circuits to QCO once and keeps one QCO program
alive for the complete episode. C++ does the same inside the pass. Cleanup and
multi-control decomposition are preparation steps outside the policy. Mapping,
routing, and synthesis state is derived from the selected stages rather than
from a Python round trip.

The staged completion path is:

1. `place-and-route` maps and routes the persistent QCO program;
2. `synthesize-for-target` lowers it to the target-native operation set; and
3. `terminate` only runs Core's target-conformance verifier.

Optimization passes may run more than once, including after mapping or
synthesis. A changed optimization invalidates synthesized state. Exact no-op
retries are masked for the current QCO state. The budget mask reserves the last
required mapping and synthesis slots.

Python may use Qiskit circuit objects for input conversion and reward metrics,
but the transition selected by the agent is always one of the Core actions
below. The deployed C++ path does not require Python or Qiskit.

## Policy ABI

Schema `mqt-predictor-core-stages/1` has 13 ordered, clamped float32 features:

1. logical qubits relative to target width;
2. normalized logarithmic depth;
3. two-qubit interaction density;
4. normalized two-qubit critical depth;
5. two-qubit-gate ratio;
6. normalized parallelism;
7. qubit liveness;
8. transformation step divided by 100; and
9. through 13. execution counts divided by 100 for each transformation action in
   the order below.

The six ordered actions are:

1. `merge-single-qubit-rotation-gates`;
2. `fuse-single-qubit-unitary-runs` with basis `u`;
3. `fuse-two-qubit-gates`;
4. `place-and-route`;
5. `synthesize-for-target`; and
6. `terminate`.

The hard limit is 100 transformation actions per compilation. Termination does
not consume a transformation slot. The current deployment recommendation is
`--max-steps=16`; see the measured budget comparison below.

The selected actor has one 16-unit Tanh hidden layer and 326 float32 parameters.
The C++ loader checks the ONNX schema, feature and action order, target
fingerprint, complete Core compatibility identity, required provenance, runtime
tensor types and shapes, and finite logits. It logs a SHA-256 ID of the complete
ONNX file. It does not independently checksum the ONNX initializer tensors, so
that artifact ID must not be described as a separately verified parameter
checksum.

## Build and run

The experiment requires CMake 3.24+, Ninja, LLVM/MLIR 22.1+, C++20, and Python
3.11+ for the optional training dependencies. It has been tested with LLVM/MLIR
22.1.8.

With no `MQT_CORE_SOURCE_DIR`, CMake fetches the exact Core revision and applies
the patch. To reuse a checkout, put it at that revision and apply the patch
before configuring:

```console
export LLVM_DIR=/path/to/llvm/lib/cmake/llvm
export MLIR_DIR=/path/to/llvm/lib/cmake/mlir
export MQT_CORE_SOURCE_DIR=/path/to/mqt-core

git -C "$MQT_CORE_SOURCE_DIR" apply \
  "$PWD/cpp/patches/mqt-core-predictor-stages.patch"
```

Build with an ONNX Runtime C/C++ SDK and run the complete C++ test set:

```console
cmake --preset release -DMQT_PREDICTOR_ENABLE_ONNX=ON \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime-sdk
cmake --build --preset release
ctest --preset release
```

Run deterministic argmax inference on the Core-hosted IQM Garnet target:

```console
build/release/cpp/mqt-predictor-cc --policy=model --trace --max-steps=16 \
  --qdmi-device=mqt.sc.iqm.garnet \
  --model=cpp/models/iqm-garnet-ppo-tanh16.onnx \
  -o build/release/predicted.mlir input.qasm
```

The driver accepts OpenQASM 3 or QCO MLIR and emits QCO MLIR. The model is
target-specific; a metadata or target mismatch is a configuration error.

For Python-side training and ABI tests, install the opt-in group and replace its
clean Core wheel with bindings built from the same patched checkout:

```console
uv sync --group compiled
CMAKE_ARGS="-DENABLE_CACHE=OFF -DMLIR_DIR=$MLIR_DIR -DLLVM_DIR=$LLVM_DIR" \
  uv pip install --reinstall --no-deps "$MQT_CORE_SOURCE_DIR"
uv run pytest tests/compilation/test_core_env.py \
  tests/compilation/test_compiled_policy.py \
  tests/compilation/test_onnx_policy.py
```

These compiled-dependency Python tests and the CMake/CTest build are optional
local validation. The repository's normal Nox/CI test sessions neither install
the `compiled` dependency group nor configure this C++ experiment.

## Selected actor and measured result

The retained actor was trained with MaskablePPO for 2,048 timesteps using a
13-to-16-to-6 Tanh policy, rollout size 64, four epochs, `gamma=0.98`, learning
rate `3e-4`, and seed 19. Its weights were trained against older Core revision
`27980b4ec5b2ef6a8ada3629944238f5f66700c2` with the same patch digest. They were
re-exported for, and revalidated against, the current compatibility identity
above. A minimal retraining run on the current pin scored worse on its
validation circuits, so it did not replace the retained weights.

The broad revalidation used the Core-hosted 20-qubit IQM Garnet target and 48
MQT Bench circuits selected from the prior Garnet-supported screen. `bv-16`
failed for every method during conversion with
`QC measurement destination must follow the measurement in the same block`; the
table therefore reports the 47 paired supported circuits. Scores are weighted
structural improvement relative to Core's canonical target pipeline, so zero is
a tie with Core and larger is better.

| Method                      | Mean score |    Median | Minimum | Positive / tie / negative | Mean passes |  Mean time |
| --------------------------- | ---------: | --------: | ------: | ------------------------: | ----------: | ---------: |
| Core canonical baseline     |          0 |         0 |       0 |                0 / 47 / 0 |           0 | 0.102229 s |
| Fixed five-pass schedule    |  0.0323792 | 0.0304044 |       0 |                40 / 7 / 0 |           5 | 0.412701 s |
| Selected Tanh actor, cap 16 |  0.1249747 | 0.1213732 |       0 |                43 / 4 / 0 |     15.2766 | 1.074931 s |

Against the fixed schedule, the actor was better on 38 circuits, tied on seven,
and worse on two; its mean score delta was `+0.0925955`. Against canonical Core
it had no negative result on the 47 supported circuits.

In the same-actor budget study, cap 100 obtained mean score `0.1249807` with
93.8511 mean passes and 5.27559 s mean time. Cap 16 retained 99.9952% of that
mean score while executing 83.72% fewer passes. Times are local wall-clock
measurements, not portable performance guarantees.

The broad corpus was untouched when first evaluated, but its results later
influenced the decision to retain this actor and deploy it with cap 16 instead
of the separately trained cap-16 actor. It is therefore engineering selection
data, not an untouched final test set. These results show a promising local
experiment, not a generalization claim.

## Input and fallback boundary

Model inference starts only from a fully unmapped, straight-line QCO program.
The current analyzer supports scalar qubits and statically indexed,
one-dimensional QTensor registers. An initially mapped or partially mapped
program, dynamic tensor indexing, quantum control flow, an invalid action
result, or unsupported structure rejects the model attempt, restores the
original module, and attempts Core's canonical target pipeline. Compilation can
still fail if that pipeline also rejects the input.

Every successful model result is checked with Core's target-conformance verifier
and with the experiment's static-site and linear-qubit checks before it is
returned.
