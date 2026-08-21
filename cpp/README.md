# Compiled MLIR predictor experiment

This directory contains an inference-runtime-free C++ bootstrap for a
v3-inspired compiled policy experiment. It runs as an MLIR `ModuleOp` pass and
uses MQT Core's native optimization, mapping, routing, synthesis, and
verification passes.

The current actor is a deterministic linear bootstrap policy. Its coefficients
are not trained Predictor weights, and its schema is not compatible with the
Python Predictor v3 model. The feature and action contracts are experimental
boundaries for later matching training and export code.

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
build/debug/cpp/mqt-predictor-cc --trace --target-qubits=4 \
  -o build/debug/predicted.mlir cpp/test/Inputs/bell.qasm
```

Use `--policy=core` to run Core's canonical target pipeline as a baseline. The
driver accepts OpenQASM 3 or QCO MLIR and emits QCO MLIR.

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

## Experiment boundary

The native path currently supports straight-line, scalar-QCO entry points. Each
completed result is checked with Core's target-conformance verifier. Direct MLIR
inputs are also checked for exactly-once linear-qubit use; static-site aliases
are rejected conservatively across the whole module.

Tensor-backed registers, quantum control flow, failed actions, and exhausted
decision budgets restore the original module and use Core's canonical pipeline.
The built-in trial target is a line topology with `u`, `cx`, `measure`, and
`reset`; production device loading and trained model import are deliberately out
of scope for this first experiment.
