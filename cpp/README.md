# Compiled Core pass-ordering experiment

This directory contains an experimental MLIR `ModuleOp` pass that uses a trained
policy to order only MQT Core QCO passes. Python supplies training, one-time
Qiskit-to-Core input conversion, and ONNX export. MQT Core supplies the shared
QCO feature analysis and target-calibration reward. Deployment is C++ only: ONNX
Runtime evaluates the actor inside the MLIR pass, while MQT Core applies every
selected transformation. Qiskit, TKET, BQSKit, and other SDK compiler passes are
not policy actions.

The original alignment target was Predictor pull request
[#798](https://github.com/munich-quantum-toolkit/predictor/pull/798) at commit
`9446b53fad345f27c04f65193e23bcf4c803f9d1`. The deployed follow-up study actor
keeps its 51 flat circuit features, PPO architecture, 20-decision horizon, and
stochastic inference, and adds 13 episode-context inputs. It uses Core-only
actions over one persistent QCO program, a small transform cost in training, and
best verified circuit retention during deployment.

The follow-up study uses Predictor source revision
`a2ae0470fc589dc88428d47dc33aad12c210f6ff`. The Python environment and compiled
runtime share this exact Core compatibility identity:

```text
99fd4d2ef93a8680ed17a9e7bed72bce77aaadce+patch.904aee31e1dc5f4796bb45c9931246cb72c9bedaa6aa6064a457d0b4de01aa66
```

The first component is the MQT Core revision. The second is the SHA-256 of
`patches/mqt-core-predictor-stages.patch`, which exposes the staged target
operations, orders mapping-ready operations by MLIR block order, and provides
the shared QCO analysis and native expected-fidelity calculation. The IQM Garnet
target fingerprint for the trained policy is:

```text
sha256:d9be5c92985ee59418ff58317a2a7ce2c24a6c08a515fe34d192d3dde8f00599
```

## Python training and C++ deployment

```text
Python training
  Qiskit circuit -> Core QCO -> study wrapper over CorePredictorEnv
    -> Core QCO analysis after each action
    -> Core target-calibration fidelity + episode context
    -> MaskablePPO actor  64 -> 64 -> 64 -> 6
    -> MaskablePPO critic 64 -> 64 -> 64 -> 1  (training only)
    -> actor ONNX

C++ deployment
  raw unmapped QCO ModuleOp -> Core analysis + episode context + controlled mask
    -> ONNX Runtime logits -> masked categorical sample
    -> selected Core QCO pass -> invalidated/recomputed analysis -> same ModuleOp
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

The three optimization actions are legal in every phase. After every action, the
shared Core analysis recomputes mapping, routing, and target-native synthesis
from the resulting QCO program. As in #798, the first decision exposes every
transform and excludes termination; later decisions derive their mask from the
factual analysis. Placement and routing is then legal only while the program is
unmapped, synthesis is legal while non-native operations remain, and termination
is exposed only for a mapped, routed, target-native program. An action that
leaves the IR unchanged is additionally masked until another action changes the
IR; several no-op masks may accumulate. Every selection still consumes one
decision. Core separately verifies target conformance before accepting
termination. No budget is reserved for later actions.

The horizon is 20 total policy decisions, including `terminate`. Termination may
succeed as the twentieth decision. Training retains terminal expected fidelity
and charges 0.0001 per transform, without potential-based intermediate rewards.
At deployment, Core scores each already-conformant state and retains the first
strictly best circuit (improvement tolerance `1e-15`). No canonical compilation
is needed for intermediate scoring. Termination, reactive stopping, or budget
exhaustion returns that incumbent, even if the current circuit is worse or
non-conformant. Context fractions always use the trained horizon of 20,
including when a smaller runtime budget is requested.

The default reactive-stop controller identifies a visible state by the 51
`float32` circuit features and the base legal mask. Revisiting without incumbent
improvement temporarily masks that state's previous transform. A second
non-improving recurrence stops once an incumbent exists. Before one exists, mask
tenure doubles up to four decisions. An improvement resets the recurrence count.
Termination is never controller-masked. `--controller=baseline` disables this
controller but still supplies context and preserves the incumbent.

Expected fidelity is computed directly from the same Core `CompilerTarget`
snapshot used for compilation. The target is populated by the QDMI device and
contains operation defaults and per-site fidelities. Lookup prefers the exact
ordered site tuple, permits a reverse-tuple fallback only for CZ, then uses the
operation default. Measurement contributes to the product. This removes the
intermediate and terminal Qiskit conversions. ESP remains out of scope because
the native reward currently models only the calibrated operation-fidelity
product, not scheduled durations and qubit-coherence decay.

## Policy ABI

Schema `mqt-predictor-markov-stop-features/1` contains exactly 64 ordered
`float32` features. The first 51 retain the baseline order:

```text
c3sqrtx, c3x, c4x, ccx, ch, cp, critical_depth, crx, cry, crz,
cswap, csx, cu, cu1, cu3, cx, cy, cz, depth, entanglement_ratio,
h, id, liveness, measure, num_qubits, p, parallelism,
program_communication, r, rc3x, rccx, rx, rxx, ry, rz, rzz, s, sdg,
swap, sx, sxdg, t, tdg, u, u0, u1, u2, u3, x, y, z
```

These are 44 operation frequencies, target-relative logical-qubit count,
log-normalized depth, and the five structural features `critical_depth`,
`entanglement_ratio`, `parallelism`, `program_communication`, and `liveness`.
Values are clamped to `[0, 1]`. Barriers are omitted, while unknown operations
remain in the frequency denominator. The remaining inputs, in order, are:

```text
zz_decision_fraction, zz_incumbent_available, zz_current_fidelity,
zz_incumbent_fidelity, zz_incumbent_age_fraction, zz_visible_visit_fraction,
zz_last_action_changed_ir, zz_previous_action_0, zz_previous_action_1,
zz_previous_action_2, zz_previous_action_3, zz_previous_action_4,
zz_previous_action_5
```

Unavailable scores are zero, visits count prior observations of the same visible
state, and previous action is one-hot. The driver supplies this context; circuit
features and expected fidelity still come directly from Core's QCO analysis. The
frozen study wrapper provides the matching Python observation. The original
`CorePredictorEnv` and its legacy export helpers still describe the 51-feature
baseline, not the deployed actor.

The PPO actor and critic each have two 64-unit Tanh hidden layers and retain
Stable-Baselines3 orthogonal initialization. Only the actor is exported. Its
`64 -> 64 -> 64 -> 6` graph contains 8,710 `float32` parameters and returns six
unmasked logits. The host applies the legal-action mask.

Native inference samples the masked softmax at temperature 1 by default. With no
sampling option, the C++ runtime seeds its generator from system entropy.
`--sampling-seed=<n>` makes a local run reproducible, while
`--deterministic-policy` selects the highest legal logit. Python and C++ use
different random-number engines, so the same integer seed does not promise an
identical cross-language action trace; it defines independent samples from the
same masked categorical policy. The native sampler uses Gumbel-max and includes
its noise draws in `--trace`, allowing Python to replay identical randomness.

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

## Deployed study checkpoint

The checked-in actor is H1-D0-C1 at 100,352 timesteps, using the first
preregistered screening seed, 1266875488. Selection did not use confirmation
performance. Training uses context, no potential shaping, transform cost 0.0001,
and the existing two-64-unit Tanh PPO architecture. Deployment repackages only
metadata; neither the graph nor any learned parameter was changed.

```text
source ONNX: sha256:86eb159ffc5208236650c501a6c75156b55615de9179f7621327810471e719d6
deployed:    sha256:4607d50ec5347decca550a79e2a222d4526648954edd4a676d34afc403ad9f8c
parameters: sha256:5550dbe8caa94123dea2d60f625fd14a2a3434325cee284dbbb6a2853c49e5ab
```

The follow-up's 64 paired confirmation training seeds improved mean stochastic
expected fidelity from 0.211460 to 0.217501 under reactive stopping. This is
evidence for the training configuration, not a guaranteed score for this single
checkpoint or a comparison against Core's canonical pipeline.

Matched Release validation covered 1,010 episodes and 5,808 decisions across the
440 training entries, 47 broad Bench entries, and 24 confirmation entries.
Features, masks, actions, intermediate/returned IR, and expected fidelity agreed
exactly with the frozen Python evaluator. Native ONNX logits matched Python ONNX
Runtime exactly and the original SB3 actor within `2.861e-6`. All 866
full-budget episodes produced a valid incumbent; shortened-budget diagnostics
also exercised incumbent retention and canonical fallback without retraining.

## Earlier 51-feature training and evaluation

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

The factual-state actor was retrained once with seed 19. The request was 10,000
environment timesteps; Stable-Baselines3 completes whole 2,048-step rollouts, so
five updates produced 10,240 actual timesteps. The deterministic shuffled corpus
cycle visited all 440 circuits and completed two full cycles in 53.25 seconds.

The training configuration matches #798: `MaskableMultiInputActorCriticPolicy`,
rollout size 2,048, batch size 64, ten epochs per update, `gamma=0.98`, learning
rate `3e-4`, two 64-unit Tanh actor layers, two 64-unit Tanh critic layers,
orthogonal initialization, and otherwise the same Stable-Baselines3 defaults.
Every episode used the 20-decision horizon and terminal-only expected-fidelity
reward. That earlier actor predates the horizon correction: max-step exhaustion
was still reported as a truncation during that training run. Training used
source revision `87ebb46aed758ede7c06f576eb78a873e9f63256` and Core identity
`99fd4d2ef93a8680ed17a9e7bed72bce77aaadce+patch.904aee31e1dc5f4796bb45c9931246cb72c9bedaa6aa6064a457d0b4de01aa66`.

The fixed comparison schedule is `synthesize-for-target`,
`fuse-two-qubit-gates`, `merge-single-qubit-rotation-gates`, `place-and-route`,
`synthesize-for-target`, and `terminate`. The canonical comparison is Core's
target compilation pipeline. All scores are absolute expected fidelity, not a
structural proxy.

### Earlier results

The earlier factual-state ONNX actor had SHA-256
`ac2069f022cdda15d33ef0e2502b85ba04765c8b5f441ee76dc25a5f9ceb197b`. Its new `r`
input column is zero-initialized, so its logits remain unchanged until
retraining. Its logits agree with the exported PyTorch actor within `3.58e-7`
over 100 random feature vectors.

The actor was evaluated on every training circuit once deterministically and
five times stochastically with seeds 7, 19, 43, 71, and 97. All means include
errors and truncations as zero.

| Inference              | Episodes | Successes | Mean expected fidelity | Mean decisions |
| ---------------------- | -------: | --------: | ---------------------: | -------------: |
| Deterministic          |      440 |       440 |               0.360133 |          3.130 |
| Stochastic, five seeds |    2,200 |     2,200 |               0.365817 |          8.075 |

There were no errors or horizon exhaustions. Stochastic inference improved over
deterministic inference by `+0.005685` (`+1.58%`) at the cost of 4.95 additional
decisions on average. Against fresh current-code baselines on the same 440
circuits, it improved over canonical Core by `+0.000936` (`+0.257%`) and trailed
the fixed staged schedule by `-0.017988` (`-4.687%`). Both baselines completed
440/440 circuits without errors or timeouts; their mean expected fidelities were
`0.364881` and `0.383805`, respectively.

This is training-corpus evidence, not a generalization estimate. The complete
episode-level policy report has SHA-256
`551db39d44559501932731a9543a41f133ed4d367d33aea86944f571d0be74b5`; the baseline
report has SHA-256
`e3480f9589f5d2939d7ea27483761de2526269786a4d17346867c5b8d3e32085`.

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

For Python/native trajectory comparisons, use matching Release builds of Core
with the same compiler options. Debug versus Release synthesis can differ in
floating-point angles and even gate counts, changing observations and scores
despite an identical source revision. Native traces expose input/outputs via IR
hashes, 64 features, legal masks, logits, and sampled noise for this check.

For an ordinary model error or horizon exhaustion that returns control, the C++
pass returns the best verified incumbent if available. Otherwise it restores the
original module and attempts Core's canonical target pipeline. Evaluation does
not credit that canonical fallback to the policy. Uncalibrated smoke targets can
still accept a verified explicit termination without an incumbent score. A pass
that never returns cannot reach the in-process fallback before the external
watchdog ends its process.
