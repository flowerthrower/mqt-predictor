# Compiled Core pass-ordering experiment

This directory contains an experimental MLIR `ModuleOp` pass that uses a trained
policy to order only MQT Core QCO passes. Python supplies training,
Qiskit-to-Core input conversion, reward calculation, and ONNX export. Deployment
is C++ only: ONNX Runtime evaluates the actor inside the MLIR pass, while MQT
Core applies every selected transformation. Qiskit, TKET, BQSKit, and other SDK
compiler passes are not policy actions.

The alignment target is Predictor pull request
[#798](https://github.com/munich-quantum-toolkit/predictor/pull/798) at commit
`9446b53fad345f27c04f65193e23bcf4c803f9d1`. The experiment preserves its flat
observation, PPO architecture and hyperparameters, episode horizon, terminal
reward timing, and stochastic inference. It deliberately replaces that
predictor's action set and circuit state with Core-only actions over one
persistent QCO program.

The policy-training and export source revision is
`54e8b6a87900fa11987b81fb13779e6ec5a3fbe5`. The Python environment and compiled
runtime share this exact Core compatibility identity:

```text
99fd4d2ef93a8680ed17a9e7bed72bce77aaadce+patch.e761b935ad001c122eb34044da3468844a7caf2d35edaff41e63492fcd665a01
```

The first component is the MQT Core revision. The second is the SHA-256 of
`patches/mqt-core-predictor-stages.patch`, which exposes the staged target
operations and orders mapping-ready operations by MLIR block order. The IQM
Garnet target fingerprint for the trained policy is:

```text
sha256:d9be5c92985ee59418ff58317a2a7ce2c24a6c08a515fe34d192d3dde8f00599
```

## Python training and C++ deployment

```text
Python training
  Qiskit circuit -> Core QCO -> CorePredictorEnv
    -> MaskablePPO actor  50 -> 64 -> 64 -> 6
    -> MaskablePPO critic 50 -> 64 -> 64 -> 1  (training only)
    -> actor ONNX

C++ deployment
  raw unmapped QCO ModuleOp -> 50 features + legal-action mask
    -> ONNX Runtime logits -> masked categorical sample
    -> selected Core QCO pass -> same ModuleOp
```

`CorePredictorEnv` calls `QCProgram.from_qiskit(circuit).to_qco()` once, runs
the cleanup and multi-control-decomposition prelude, and retains that QCO
program for the complete episode. The MLIR pass likewise mutates one persistent
`ModuleOp`. The policy has six actions:

1. `merge-single-qubit-rotation-gates`;
2. `fuse-single-qubit-unitary-runs` with basis `u`;
3. `fuse-two-qubit-gates`;
4. `place-and-route`;
5. `synthesize-for-target`; and
6. `terminate`.

The three optimization actions are legal in every phase. They may be repeated,
including when they have no effect and after mapping or synthesis. Every
selection still consumes one decision. A changed optimization after synthesis
invalidates synthesized state; a no-op does not. Placement and routing is legal
only before mapping, synthesis is legal while the state is not synthesized, and
termination is legal only after full target conformance. There is no retry
suppression, pass-history feature, or budget reservation for later actions.

The horizon is 20 total policy decisions, including `terminate`. Termination may
succeed as the twentieth decision. A twentieth non-termination action truncates
the episode. Non-terminal actions return zero reward. Successful termination
returns the absolute expected fidelity; pass errors, timeouts, and truncation
return zero. This is the terminal-only reward contract of #798, not the
intermediate-reward change from later work.

Expected fidelity is computed from the Qiskit target produced by the same IQM
Garnet QDMI device snapshot as the Core compiler target. The reward adapter only
mirrors a missing reverse CZ calibration needed for Core's legal orientation.
ESP is not used because the Qiskit reward target does not expose the operation
durations and qubit-coherence properties required by that metric.

## Policy ABI

Schema `mqt-predictor-core-stages/3` contains exactly 50 ordered `float32`
features. Python exposes them as a Gymnasium `Dict` of scalar `Box(0, 1)`
spaces; the effective Stable-Baselines3 concatenation order and the C++/ONNX
order are:

```text
c3sqrtx, c3x, c4x, ccx, ch, cp, critical_depth, crx, cry, crz,
cswap, csx, cu, cu1, cu3, cx, cy, cz, depth, entanglement_ratio,
h, id, liveness, measure, num_qubits, p, parallelism,
program_communication, rc3x, rccx, rx, rxx, ry, rz, rzz, s, sdg,
swap, sx, sxdg, t, tdg, u, u0, u1, u2, u3, x, y, z
```

These are 43 operation frequencies, target-relative logical-qubit count,
log-normalized depth, and the five structural features `critical_depth`,
`entanglement_ratio`, `parallelism`, `program_communication`, and `liveness`.
Values are clamped to `[0, 1]`. Barriers are omitted, while unknown operations
remain in the frequency denominator. The observation contains no step fraction,
action counts, pass history, or other hidden state.

The PPO actor and critic each have two 64-unit Tanh hidden layers and retain
Stable-Baselines3 orthogonal initialization. Only the actor is exported. Its
`50 -> 64 -> 64 -> 6` graph contains 7,814 `float32` parameters and returns six
unmasked logits. The host applies the legal-action mask.

Native inference samples the masked softmax at temperature 1 by default. With no
sampling option, the C++ runtime seeds its generator from system entropy.
`--sampling-seed=<n>` makes a local run reproducible, while
`--deterministic-policy` selects the highest legal logit. Python and C++ use
different random-number engines, so the same integer seed does not promise an
identical cross-language action trace; it defines independent samples from the
same masked categorical policy.

The loader checks the ONNX schema, feature and action order, target fingerprint,
complete Core compatibility identity, provenance, tensor interface, and finite
logits. It logs the SHA-256 of the complete ONNX file as its artifact ID.

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

Run seeded stochastic inference on Core's 20-qubit IQM Garnet target:

```console
build/release/cpp/mqt-predictor-cc --trace --max-steps=20 \
  --qdmi-device=mqt.sc.iqm.garnet --sampling-seed=7 \
  --model=cpp/models/iqm-garnet-ppo-tanh64x64.onnx \
  -o build/release/predicted.mlir input.mlir
```

Omit `--sampling-seed` for entropy-seeded inference, or add
`--deterministic-policy` for argmax. Supplying `--model` implies
`--policy=model`.

The driver accepts OpenQASM 3 or QCO MLIR and emits QCO MLIR. The exact trained
frontend contract, however, is Qiskit circuit to Core QCO to raw unmapped QCO
MLIR. Native experiments therefore serialize the result of
`QCProgram.from_qiskit(circuit).to_qco()` and pass that `.mlir` file to the C++
driver. Direct QASM input remains useful for general compiler use and smoke
tests with the non-model policies. Model policy rejects direct QASM input
because Core's QASM frontend can lower some operations, including controlled
unitaries, differently from the Qiskit-to-Core adapter.

The model is target-specific. A metadata, target-fingerprint, or Core-identity
mismatch is a configuration error.

For Python-side environment and ABI tests, install the opt-in group and replace
its clean Core wheel with bindings built from the same patched checkout:

```console
uv sync --group compiled
CMAKE_ARGS="-DENABLE_CACHE=OFF -DMLIR_DIR=$MLIR_DIR -DLLVM_DIR=$LLVM_DIR" \
  uv pip install --reinstall --no-deps "$MQT_CORE_SOURCE_DIR"
uv run pytest tests/compilation/test_core_env.py \
  tests/compilation/test_compiled_policy.py \
  tests/compilation/test_onnx_policy.py
```

These compiled-dependency Python tests and the CMake/CTest build are optional
local validation. Normal Nox/CI sessions do not install the `compiled` group or
configure this C++ experiment.

## Training and model selection

The Predictor compilation archive contains 500 QASM circuits. Exactly 440 have
at most 20 qubits and form the full Garnet-compatible training corpus. The other
60 use 21--30 qubits and are excluded only because they exceed Garnet's 20-qubit
capacity. They are not frontend or pass failures. A deterministic shuffled cycle
visits every one of the 440 eligible circuits before reshuffling.

The frozen corpus identities are:

```text
archive:       sha256:eac7f551b7a68e5d70274dc26e8831f02f1e8d76e4ad359a5d5889b2f110a604
ordered names: sha256:a75e74282409b5ae88f30ad91dd34659418b8e20f44d2eb0b462d45488016ee9
```

Three independent MaskablePPO candidates use seeds 7, 19, and 43. Each receives
10,240 environment timesteps over the full corpus. The training configuration
matches #798: `MaskableMultiInputActorCriticPolicy`, rollout size 2,048, batch
size 64, ten epochs per update, `gamma=0.98`, learning rate `3e-4`, two 64-unit
Tanh actor layers, two 64-unit Tanh critic layers, orthogonal initialization,
and otherwise the same Stable-Baselines3 defaults. Every episode uses the
20-decision horizon and terminal-only expected-fidelity reward described above.

Model selection is separate from training. The frozen MQT Bench study generated
99 circuits from 35 families; 68 fit Garnet. Sorting those 68 labels assigns
every fourth one-based entry to a 17-circuit validation split and the other 51
to selection-heldout evaluation. Each candidate receives five stochastic trials
per validation circuit, using sampling seeds 7, 19, 43, 71, and 97. Selection
uses only mean validation episode return with every error or truncation counted
as zero. The selection-heldout scores were not read until one candidate had been
selected. This split is in-distribution and overlaps the training archive, so it
is not evidence on unseen circuits.

The fixed comparison schedule is `synthesize-for-target`,
`fuse-two-qubit-gates`, `merge-single-qubit-rotation-gates`, `place-and-route`,
`synthesize-for-target`, and `terminate`. The canonical comparison is Core's
target compilation pipeline. All scores are absolute expected fidelity, not a
structural proxy.

## Results

All means below include every failure, truncation, or frontend exclusion as a
zero return. The complete r5 report has SHA-256
`fd1163ce82c3da74d13b5d09f9e12e050824b253d9e54d4eab2b1c6c778bac8a`.

| PPO seed | Validation mean | Successes | Mean successful decisions |
| -------: | --------------: | --------: | ------------------------: |
|        7 |        0.229484 |     70/85 |                     5.086 |
|       19 |        0.230406 |     70/85 |                     5.357 |
|       43 |        0.230175 |     70/85 |                     4.929 |

Seed 19 won by validation return. Its deployed ONNX file has SHA-256
`384c2a5ba98e9d415f46ad1f2703b25958f10ab93df4c1f9f2b551280e64d4c4`.

| Python split                    |   Policy | Canonical Core | Fixed staged | Policy successes |
| ------------------------------- | -------: | -------------: | -----------: | ---------------: |
| Validation (17 circuits)        | 0.230406 |       0.228347 |     0.250389 |            70/85 |
| Selection-heldout (51 circuits) | 0.208807 |       0.202299 |     0.221173 |          210/255 |
| All 68 circuits                 | 0.214207 |       0.208811 |     0.228477 |          280/340 |

Thus the learned actor improved over canonical Core by `+0.005396` across the
68-circuit study, but trailed the fixed staged schedule by `-0.014271`. On the
56 circuits where the fixed baseline compiled, the policy was better, tied, or
worse on 19, 8, and 29 circuits, respectively.

The compiled MLIR/ONNX runtime scored `0.213356` on the same 340 stochastic
trials. It completed 280 successfully; the 60 zero returns comprise 35 canonical
fallbacks, 20 frontend exclusions, and 5 output-scoring failures. There were no
watchdog or compiler-process failures. Native and Python success rates are both
82.35%. Native minus Python return is `+0.002429` on validation, `-0.001943` on
selection-heldout, and `-0.000850` overall. These are aggregate comparisons:
NumPy and C++ use different random-number engines, so equal sampling-seed labels
do not create paired trajectories.

The selected actor was also replayed five times on every circuit in the complete
440-circuit training corpus. All 2,200 episodes terminated successfully within
the 20-action horizon. Its mean expected fidelity was `0.372315` at 5.079 mean
decisions, versus `0.364881` for canonical Core (`+0.007434`; better/tied/worse
on 399/22/19 circuits) and `0.383805` for the fixed schedule (`-0.011490`;
128/22/290). This is training-corpus evidence, not a generalization estimate.
The full-corpus report has SHA-256
`2d17eaf74d0238fa098e525bdf5b53d2fb18e124e314230569a84a51ca686e55`.

## Input, timeout, and fallback boundary

Model inference starts only from a fully unmapped, straight-line QCO program.
Initially or partially mapped programs, dynamic indexing, quantum control flow,
a failed action, or unsupported structure reject the model attempt. A successful
model episode must pass Core's target-conformance verifier and the experiment's
static-site and linear-qubit checks.

The Python environment applies each pass transactionally to a copy. On POSIX,
its optional per-pass deadline uses `SIGALRM`; this is best effort, works only
on the main thread, and cannot interrupt every native call safely. The training
and Python evaluation runs use a 30-second setting.

The in-process C++ MLIR pass has no safe hard-cancellation mechanism for a Core
pass. The native experiment therefore runs each compiler invocation in a child
process with an external watchdog. Embedders that require a hard wall-clock
limit must provide the same process boundary; the 20-action horizon is not a
wall-clock timeout.

For an ordinary model error or horizon truncation that returns control, the C++
pass restores the original module and attempts Core's canonical target pipeline.
This protects compilation, but evaluation still assigns the failed model episode
zero rather than crediting it with the fallback's result. A pass that never
returns cannot reach the in-process fallback before the external watchdog ends
its process.
