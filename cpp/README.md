# Compiled MLIR predictor experiment

This directory contains a compiled C++ policy experiment. It runs as an MLIR
`ModuleOp` pass and uses MQT Core's native optimization, mapping, routing,
synthesis, and verification passes. The default build has no ML inference
runtime; an optional ONNX Runtime backend accepts any actor with the fixed
feature/logit tensor interface. The bundled exporter starts with a linear actor
for the smallest end-to-end experiment.

The driver supports both its original hand-written bootstrap actor and a
target-specific linear model exported by the matching Python trainer. The
artifact is validated against its feature/action order, parameter checksum,
target fingerprint, and exact MQT Core revision before inference. It is not
compatible with existing Predictor v3 models.

## Build

The experiment requires CMake 3.24+, Ninja, and LLVM/MLIR 22.1+ (tested with
22.1.8). MQT Core's installed package does not yet export its MLIR targets, so
this build embeds Core main revision `27980b4ec5b2ef6a8ada3629944238f5f66700c2`.
Set `MQT_CORE_SOURCE_DIR` to reuse a local checkout; otherwise CMake downloads
that revision. The build applies `patches/mqt-core-ready-block-order.patch` to
downloaded sources. A local override must be a Git checkout at the exact
revision with that patch already applied.

```console
export LLVM_DIR=/path/to/llvm/lib/cmake/llvm
export MLIR_DIR=/path/to/llvm/lib/cmake/mlir
export MQT_CORE_SOURCE_DIR=/path/to/mqt-core  # optional local override

cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

To enable ONNX inference, point CMake at an ONNX Runtime C/C++ SDK. It remains a
Predictor-only dependency and is not linked into MQT Core.

```console
cmake --preset debug -DMQT_PREDICTOR_ENABLE_ONNX=ON \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime-sdk
cmake --build --preset debug
```

Compiler caching is disabled in the presets for reproducibility; pass
`-DENABLE_CACHE=ON` to `cmake` to opt in with a working cache installation.

## Run

```console
build/debug/cpp/mqt-predictor-cc --trace \
  --target=cpp/test/Inputs/line-4-target.json \
  --model=cpp/test/Inputs/line-4-policy.json \
  -o build/debug/predicted.mlir cpp/test/Inputs/bell.qasm
```

`--model` accepts the native JSON actor or an `.onnx` actor when ONNX support is
enabled. Use `--policy=bootstrap` for the hand-written actor, `--policy=core`
for Core's canonical target pipeline, or `--policy=exhaustive` for a
training-free search. `--model` implies `--policy=model`. The driver accepts
OpenQASM 3 or QCO MLIR and emits QCO MLIR.

The exhaustive mode evaluates Core's canonical pipeline and all 16 ordered
subsets of the three native optimization actions before the same mapping and
target-finalization stages. It selects lexicographically by two-qubit critical
depth, two-qubit gate count, total depth, and total gate count. `--trace`
reports every candidate, its compile time, and the selected schedule.

The experiment sorts each mapping ready set by MLIR block order before
evaluating it. The change is carried as a patch because the pinned upstream Core
revision does not contain it.

The Core-hosted IQM snapshots are available through their stable QDMI IDs:

```console
build/release/cpp/mqt-predictor-cc --policy=exhaustive --trace \
  --qdmi-device=mqt.sc.iqm.garnet input.qasm
build/release/cpp/mqt-predictor-cc --policy=exhaustive --trace \
  --qdmi-device=mqt.sc.iqm.emerald input.qasm
```

## Minimal trainer/exporter

The checked-in dataset contains 20 synthetic imitation examples and only 32
full-batch updates. It exists solely to smoke-test serialization, training,
export, and loading; its samples are not compiler trajectories or a quality
result.

```console
uv run python -m mqt.predictor.compiled \
  --dataset cpp/test/Inputs/line-4-training.json \
  --target cpp/test/Inputs/line-4-target.json \
  --output cpp/test/Inputs/line-4-policy.json \
  --core-revision 27980b4ec5b2ef6a8ada3629944238f5f66700c2 \
  --objective "synthetic serialization and imitation-training smoke" \
  --epochs 32
```

The exporter records the current Git revision (with `+dirty` when applicable)
unless `--source-revision` is provided.

## Native pass-ordering RL experiment

The native trainer uses masked episodic REINFORCE. Action sampling and every
pass transition happen in the compiled MLIR driver; Python only updates and
exports the 65 parameters of the linear actor between complete compilations.
Deterministic deployment remains C++-only.

```console
uv run python -m mqt.predictor.compiled.native_rl bench/*.qasm \
  --binary build/release/cpp/mqt-predictor-cc \
  --qdmi-device mqt.sc.iqm.garnet \
  --max-passes 100 --updates 8 --episodes-per-circuit 2 \
  --output build/release/native-policy.json \
  --report build/release/native-policy-report.json
```

`--max-passes` limits executed transformations; the terminate decision does not
consume that budget. The reward compares two-qubit depth, two-qubit gates, total
depth, and total gates with Core's canonical pipeline and charges a small cost
per pass. The report preserves complete best schedules, repeated-pass counts,
deterministic evaluation, and the best exhaustive ordering in which each of the
three optimization passes is used at most once. Circuits without a valid Core
baseline are listed under `excluded` and are not trained on.

## Python/Core training environment

`CorePredictorEnv` keeps one `QCOProgram` alive for the complete episode. Qiskit
is used only at reset and on copied snapshots for observations and reward; every
chosen pass mutates a transactional QCO copy. This avoids losing MLIR mapping
state in Qiskit round trips. Python and C++ both run Core's QCO cleanup pipeline
once before the first observation so Qiskit and OpenQASM inputs have the same
policy state.

This first profile selects four pre-target actions exposed by Core: merge
rotations, fuse one-qubit runs in the `u` basis, decompose multi-controlled
gates, and lift Hadamards. The fifth action terminates and invokes Core's
canonical `compile_for_target()` pipeline. The qubit-reuse pipeline is excluded
because it can introduce quantum control flow that the straight-line feature
analyzer cannot represent. Core main does not yet expose fuse-two-qubit,
place-and-route, or target-native synthesis as separate Python actions. The
environment therefore has a distinct 12-feature ABI and permits exactly 100
transformation actions before only termination remains legal. It does not
register any Qiskit, TKET, BQSKit, or other SDK compilation action; Qiskit is
only the circuit conversion and metric view.

Install the opt-in experiment dependencies with `uv sync --group compiled`.
Before collecting training rewards, replace the clean Core wheel with bindings
built from the patched checkout so Python training and C++ inference use the
same deterministic routing implementation:

```console
CMAKE_ARGS="-DENABLE_CACHE=OFF -DMLIR_DIR=$MLIR_DIR -DLLVM_DIR=$LLVM_DIR" \
  uv pip install --reinstall --no-deps "$MQT_CORE_SOURCE_DIR"
```

The smallest real RL trial uses the existing MaskablePPO dependency with a
linear actor and one update. In this precise pseudocode, `circuits` is a
non-empty sequence of Qiskit circuits, `target` is the matching Core
`CompilerTarget`, and `target_fingerprint` is the fingerprint reported by the
compiled driver for that target:

```python
from pathlib import Path

from mqt.predictor.compiled import CorePredictorEnv, export_onnx_policy, train_maskable_ppo

env = CorePredictorEnv(circuits, target, max_passes=100)
policy = train_maskable_ppo(env, timesteps=8, seed=7)
export_onnx_policy(
    Path("policy.onnx"),
    policy,
    target_fingerprint_override=target_fingerprint,
    core_revision="27980b4ec5b2ef6a8ada3629944238f5f66700c2",
    training_algorithm="MaskablePPO linear actor smoke",
    objective="Core-relative structural cost",
)
```

Eight timesteps prove training, export, and compiled inference; they are not
enough to claim compilation-quality improvement. Richer ONNX actors can reuse
the same C++ feature/logit interface.

This experiment requires Python 3.11 or newer because the pinned Core-main
bindings do not support Python 3.10.

Fusion uses the shared `u` basis. It is masked until any unitary acting on more
than two qubits has been decomposed because Core's QCO-to-Qiskit exporter cannot
represent the resulting multiply controlled `u` operation. Any failed pass or
snapshot conversion is rolled back transactionally.

## Experimental policy contract

Schema `mqt-predictor-core-passes/1` contains this ordered, clamped 12-float
vector:

1. logical qubits divided by target width;
2. `log1p(depth) / log1p(1,000,000)`;
3. unique two-qubit interaction density;
4. two-qubit critical-path length divided by the number of two-qubit gates;
5. two-qubit gates divided by all unitary gates;
6. normalized gates-per-depth parallelism;
7. active qubit-operation slots divided by qubits times depth;
8. executed transformation steps divided by 100; and
9. through 12. each Core transformation's execution count divided by 100.

The ordered actions are merge rotations, fuse single-qubit runs, decompose
multi-controlled gates, lift Hadamards, and terminate. A no-op transformation is
suppressed for the unchanged circuit, and only termination remains legal after
100 transformations. Termination invokes Core's canonical target pipeline.
`--trace` prints the ordered feature values and every decision. No Qiskit, TKET,
BQSKit, or other SDK compiler action is represented.

The native artifact schema is `mqt-predictor-native-policy/1`; it contains an
action-major float32 linear layer with 12 inputs and five outputs. The optional
`mqt-predictor-onnx-policy/1` schema accepts a fixed float32 `features[1,12]`
input and returns raw `logits[1,5]`. In either case C++ applies the Core
legality mask and deterministic argmax.

## Compiler target JSON

Target schema `mqt-compiler-target/1` contains a name, ordered integer site IDs,
an optional undirected coupling list, and native operation signatures. See
`test/Inputs/line-4-target.json`. The loader constructs Core's validated
`CompilerTarget` and derives a deterministic fingerprint from its normalized
topology and operation set.

This first target schema intentionally excludes calibration, durations,
site-specific gate support, and live QDMI device discovery. Those need a reward
contract before they can contribute meaningfully to training.

## Experiment boundary

The native path currently supports straight-line scalar-QCO entry points and
statically indexed, straight-line one-dimensional QTensor registers. Each
completed result is checked with Core's target-conformance verifier. Direct MLIR
inputs are also checked for exactly-once linear-qubit use; static-site aliases
are rejected conservatively across the whole module.

Dynamic tensor indexing, quantum control flow, failed actions, and exhausted
transformation budgets restore the original module and use Core's canonical
pipeline. The built-in line target remains available through `--target-qubits`
for the bootstrap and Core policies. Model artifacts are deliberately
target-specific; a target, Core, schema, ordering, dimension, or checksum
mismatch is a hard configuration error rather than a silent fallback.
