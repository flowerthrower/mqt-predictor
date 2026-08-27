# Compiled Core pass-ordering experiment

This directory contains an experimental MLIR `ModuleOp` pass that orders only
MQT Core QCO passes. Python trains and exports the actor. Deployment is C++
only: ONNX Runtime evaluates the actor inside the MLIR pass, and MQT Core
executes each selected transformation. Qiskit, TKET, BQSKit, and other SDK
compiler passes are not policy actions.

The Python environment and compiled runtime use this compatibility identity:

```text
99fd4d2ef93a8680ed17a9e7bed72bce77aaadce+patch.e761b935ad001c122eb34044da3468844a7caf2d35edaff41e63492fcd665a01
```

The first component is the exact MQT Core revision. The second is the SHA-256 of
`patches/mqt-core-predictor-stages.patch`, which exposes the staged target
operations and orders mapping-ready operations by MLIR block order.

## Runtime design

```text
Python: CorePredictorEnv -> MaskablePPO -> 56x16x6 Tanh actor -> ONNX
                                                             |
C++: MLIR pass -> features and legal mask -> ONNX Runtime ----+
       |                                      |
       +------ persistent Core QCO program <--+
```

`CorePredictorEnv` converts an input circuit to QCO once and keeps that QCO
program alive for the complete episode. C++ does the same inside the pass.
Cleanup and multi-control decomposition are preparation outside the policy.
Python may use Qiskit objects for input conversion and reward metrics, but each
transition is one of these six Core actions:

1. `merge-single-qubit-rotation-gates`;
2. `fuse-single-qubit-unitary-runs` with basis `u`;
3. `fuse-two-qubit-gates`;
4. `place-and-route`;
5. `synthesize-for-target`; and
6. `terminate`.

Placement and routing operate on the persistent program, target synthesis lowers
it to the target-native operation set, and termination only invokes Core's
target-conformance verifier. Optimization actions may repeat, including after
mapping or synthesis. A changed optimization invalidates synthesized state,
exact no-op retries are masked, and the budget mask reserves the mapping and
synthesis slots still required for a legal result.

## Policy ABI

Schema `mqt-predictor-core-stages/2` contains 56 ordered, clamped `float32`
features. Its first 50 slots are the exact flat, non-GNN v3 observation order:

```text
c3sqrtx, c3x, c4x, ccx, ch, cp, critical_depth, crx, cry, crz,
cswap, csx, cu, cu1, cu3, cx, cy, cz, depth, entanglement_ratio,
h, id, liveness, measure, num_qubits, p, parallelism,
program_communication, rc3x, rccx, rx, rxx, ry, rz, rzz, s, sdg,
swap, sx, sxdg, t, tdg, u, u0, u1, u2, u3, x, y, z
```

These are 43 operation frequencies plus target-relative logical qubits,
log-normalized depth, and the five SuperMarQ structural features: critical
depth, entanglement ratio, parallelism, program communication, and liveness.
Barriers are omitted; unknown operations remain in the frequency denominator.
The final six slots are `step_fraction` followed by the execution counts for the
five transformation actions above. Each history value is divided by 20;
termination has no count feature.

The transformation limit and CLI default are both 20. Termination does not
consume a transformation slot. The selected actor has one 16-unit Tanh hidden
layer and 1,014 `float32` parameters. The C++ loader checks the ONNX schema,
feature and action order, target fingerprint, complete Core compatibility
identity, provenance, tensor types and shapes, and finite logits. It logs a
SHA-256 ID of the complete ONNX file.

## Build and run

The experiment requires CMake 3.24+, Ninja, LLVM/MLIR 22.1+, C++20, and Python
3.11+ for optional training dependencies. It has been tested with LLVM/MLIR
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

Run deterministic argmax inference on Core's 20-qubit IQM Garnet target:

```console
build/release/cpp/mqt-predictor-cc --policy=model --trace --max-steps=20 \
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
  tests/compilation/test_onnx_policy.py \
  tests/compilation/test_native_rl.py
```

These compiled-dependency Python tests and the CMake/CTest build are optional
local validation. Normal Nox/CI sessions do not install the `compiled` group or
configure this C++ experiment.

## Training and model selection

The historical Predictor compilation archive contains 500 QASM circuits. All 440
inputs that fit Garnet's 20-qubit width were used: 225 Qiskit-origin and 215
TKET-origin circuits. The other 60 require 21--30 qubits. Thus 440 is the full
target-compatible training corpus, not a sample, and every one of those inputs
was visited during training.

The selected actor used `sb3_contrib.MaskablePPO` for 10,240 environment
timesteps with rollout size 128, batch size 64, four epochs, `gamma=0.98`,
learning rate `3e-4`, and seed 19. A seeded shuffled cycle supplied all 440
circuits. The sweep compared a linear seed-19 actor with 16-unit Tanh actors at
seeds 7, 19, and 43.

Model selection used only a split fixed before policy evaluation: after sorting
the 68 current MQT Bench spot-grid circuits that fit Garnet, every fourth label
was assigned to validation and the other 51 to heldout evaluation. The Tanh
seed-19 actor won on validation. Heldout circuits did not affect training or
model selection.

The weights were trained with Core `3036c91238449452d53cb6aca5d02ce503d8f1ac`
and the same patch. They were re-exported with the current compatibility
metadata above. Repeating the full 68-case current-Bench study on Core
`99fd4d2ef93a8680ed17a9e7bed72bce77aaadce` preserved statuses, selected actions,
and structural scores. Direct compiled C++ inference against that current Core
pin also passed.

## Results

Scores are weighted structural improvement relative to Core's canonical target
pipeline, so zero is a tie and larger is better. The current MQT Bench spot grid
generated 99 circuits, of which 68 fit Garnet. Five failed or timed out before
the three methods could run. Of the remaining 63, 55 produced paired results;
the table reports those successful pairs from the Core `99fd4d2` replay.

| Split          | Pairs | Actor mean | Fixed mean | Actor better / tie / worse | Actor passes | Actor time |
| -------------- | ----: | ---------: | ---------: | -------------------------: | -----------: | ---------: |
| Validation     |    14 |  0.1646336 |  0.0360053 |                 12 / 2 / 0 |      19.8571 |  2.26324 s |
| Heldout        |    41 |  0.1260203 |  0.0338313 |                 37 / 4 / 0 |      19.8780 |  1.50597 s |
| All successful |    55 |  0.1358492 |  0.0343847 |                 49 / 6 / 0 |      19.8727 |  1.69873 s |

The actor had no negative result relative to canonical Core among those 55
pairs. Its heldout mean improvement over the fixed five-pass schedule was
`+0.0921890`. Times are local wall-clock measurements, not portable performance
guarantees.

The selected actor was also replayed on all 440 target-compatible training
circuits. This is a training-set check on Core `3036c912`, not heldout evidence
and not a Core `99fd4d2` result.

| Method          |      Mean |    Median |    Minimum | Positive / tie / negative | Mean passes |  Mean time |
| --------------- | --------: | --------: | ---------: | ------------------------: | ----------: | ---------: |
| Core canonical  |         0 |         0 |          0 |               0 / 440 / 0 |           0 | 0.054194 s |
| Fixed five-pass | 0.0306329 | 0.0227808 | -0.0037383 |             359 / 61 / 20 |           5 | 0.220538 s |
| Selected actor  | 0.0889999 | 0.0674816 | -0.0426113 |              409 / 24 / 7 |     19.9227 | 0.507330 s |

Against the fixed schedule on that training corpus, the actor was better on 363
circuits, tied on 65, and worse on 12; its mean score delta was `+0.0583670`.
All three methods completed all 440 circuits.

The earlier 47-circuit number was not the repository's theoretical support
limit. It came from a separate conservative screen that retained 48 entries
after width, frontend, dynamic-circuit, and timeout exclusions; `bv-16` then
failed during metric export, leaving 47 paired measurements. The full Predictor
archive supplies 440 Garnet-compatible training circuits, while the newer MQT
Bench study deliberately reports its broader selection and failures separately.

## Input and fallback boundary

Model inference starts only from a fully unmapped, straight-line scalar QCO
program. Tensor-backed qubit registers, initially or partially mapped programs,
dynamic indexing, quantum control flow, an invalid action result, or unsupported
structure reject the model attempt. The original module is then restored and
Core's canonical target pipeline is attempted; compilation can still fail if
that pipeline rejects the input too.

Every successful model result is checked with Core's target-conformance verifier
and with the experiment's static-site and linear-qubit checks before it is
returned.
