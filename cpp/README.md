# Compiled MLIR predictor experiment

This directory contains an inference-runtime-free C++ policy experiment. It runs
as an MLIR `ModuleOp` pass and uses MQT Core's native optimization, mapping,
routing, synthesis, and verification passes.

The driver supports both its original hand-written bootstrap actor and a
target-specific linear model exported by the matching Python trainer. The
artifact is validated against its feature/action order, parameter checksum,
target fingerprint, and exact MQT Core revision before inference. It is not
compatible with existing Predictor v3 models.

## Build

The experiment requires CMake 3.24+, Ninja, and LLVM/MLIR 22.1+ (tested with
22.1.8). MQT Core's installed package does not yet export its MLIR targets, so
this build embeds a pinned Core source revision. Set `MQT_CORE_SOURCE_DIR` to
reuse a local checkout; otherwise CMake downloads the pinned revision. A local
override must be a Git checkout at that exact revision.

```console
export LLVM_DIR=/path/to/llvm/lib/cmake/llvm
export MLIR_DIR=/path/to/llvm/lib/cmake/mlir
export MQT_CORE_SOURCE_DIR=/path/to/mqt-core  # optional local override

cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

The same commands are available as VS Code tasks. Compiler caching is disabled
in the presets for reproducibility; pass `-DENABLE_CACHE=ON` to `cmake` to opt
in with a working cache installation.

## Run

```console
build/debug/cpp/mqt-predictor-cc --trace \
  --target=cpp/test/Inputs/line-4-target.json \
  --model=cpp/test/Inputs/line-4-policy.json \
  -o build/debug/predicted.mlir cpp/test/Inputs/bell.qasm
```

Use `--policy=bootstrap` for the hand-written actor, `--policy=core` for Core's
canonical target pipeline, or `--policy=exhaustive` for a training-free search.
`--model` implies `--policy=model`. The driver accepts OpenQASM 3 or QCO MLIR
and emits QCO MLIR.

The exhaustive mode evaluates Core's canonical pipeline and all 16 ordered
subsets of the three native optimization actions before the same mapping and
target-finalization stages. It selects lexicographically by two-qubit critical
depth, two-qubit gate count, total depth, and total gate count. `--trace`
reports every candidate, its compile time, and the selected schedule.

Core's current routing heuristic can produce different layouts across fresh
processes despite its fixed default seed. Repeat constrained-topology
measurements; a one-shot difference is not necessarily caused by pass order.

## Minimal trainer/exporter

The checked-in demo uses 20 tiny imitation examples and only 32 full-batch
updates. This is enough to prove the train/export/load boundary locally; it is
not an RL campaign or a quality result.

```console
uv run python -m mqt.predictor.compiled \
  --dataset cpp/test/Inputs/line-4-training.json \
  --target cpp/test/Inputs/line-4-target.json \
  --output cpp/test/Inputs/line-4-policy.json \
  --core-revision 0c50dd30815638517aa159d20e78290cd449323e \
  --epochs 32
```

The exporter records the current Git revision (with `+dirty` when applicable)
unless `--source-revision` is provided. The same command is available as the
`Train Native Demo Policy` VS Code task.

## Experimental policy contract

Schema `mqt-predictor-bootstrap/1` contains this ordered, clamped seven-float
vector:

1. logical qubits divided by target width;
2. `log1p(depth) / log1p(1,000,000)`;
3. unique two-qubit interaction density;
4. two-qubit critical-path length divided by the number of two-qubit gates;
5. two-qubit gates divided by all unitary gates;
6. normalized gates-per-depth parallelism; and
7. active qubit-operation slots divided by qubits times depth.

The ordered actions are merge rotations, fuse single-qubit runs, fuse two-qubit
gates, place-and-route, native synthesis, and terminate. Core's mapping pass
combines placement and routing. Before placement, optimization, mapping, and any
required synthesis are legal. After placement and routing, optimization remains
legal until the program is native; a native program may terminate. `--trace`
prints the ordered feature values and every decision. The full SDK action space,
separate layout and routing stages, learned objective, and v3 final-optimization
stage are not represented.

The learned artifact schema is `mqt-predictor-native-policy/1`. Its only
supported architecture is an action-major float32 linear layer with seven inputs
and six outputs. C++ performs masked deterministic argmax directly; no Python,
PyTorch, ONNX, or other inference runtime is loaded.

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

The native path currently supports straight-line, scalar-QCO entry points. Each
completed result is checked with Core's target-conformance verifier. Direct MLIR
inputs are also checked for exactly-once linear-qubit use; static-site aliases
are rejected conservatively across the whole module.

Tensor-backed registers, quantum control flow, failed actions, and exhausted
decision budgets restore the original module and use Core's canonical pipeline.
The built-in line target remains available through `--target-qubits` for the
bootstrap and Core policies. Model artifacts are deliberately target-specific; a
target, Core, schema, ordering, dimension, or checksum mismatch is a hard
configuration error rather than a silent fallback.
