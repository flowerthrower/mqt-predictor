# SCASIA rerun

Experiment-only branch based on #830 at `b391dca3`, including #831 and the paper
stack. This branch is not intended for merging.

Use Linux or macOS and the checked-in lock: `uv sync --frozen --extra gnn`. Edit
`scasia.toml`, especially `output`, before submitting four independent cluster
jobs:

```sh
uv run --frozen --extra gnn python -m mqt.predictor.rl.experiments.scasia --compiler qiskit --config experiments/scasia.toml
uv run --frozen --extra gnn python -m mqt.predictor.rl.experiments.scasia --compiler tket --config experiments/scasia.toml
uv run --frozen --extra gnn python -m mqt.predictor.rl.experiments.scasia --compiler original --config experiments/scasia.toml
uv run --frozen --extra gnn python -m mqt.predictor.rl.experiments.scasia --compiler paper --config experiments/scasia.toml
```

`--stage train|evaluate|all` defaults to `all`; baselines always evaluate. Each
row writes to its own directory below `output` and locks that directory against
concurrent writers. Paths are relative to the TOML file. Do not set
`GITHUB_ACTIONS`: the shared BQSKit actions use reduced settings under that
flag.

## Configuration and recovery

Defaults are ESP, 100,000 requested training steps, 32 actions per episode, seed
0, and 10 independent evaluations of each test circuit. Both RL policies sample
stochastically with a recorded seed for each repetition. Every repetition is
retained; there is no best-of-N policy selection. The native TKET pipeline
retains its fixed LightSABRE seed 0; shared BQSKit actions retain seed 10.
Qiskit uses the repetition seed. The Qiskit actions used by the teacher use O3's
SDK parameters, including `QiskitSabreMapping`'s search budget and VF2 settings,
with one compilation seed per episode. The separate `SabreSwap` action retains
its shared implementation. This aligns the teacher's primitives with the native
reference; it does not add an outer best-of-N compilation selection.

Both RL rows exclude `QiskitO3`. They expose
`Optimize1qGatesDecomposition_preserve` and `Opt2qBlocks_preserve` as separate
actions for optimization in Boston's native basis. AI routing and MGD remain
excluded. This changes the action schema: train fresh models in a new output
directory; existing checkpoints cannot be resumed with this action set. The
paper graph input also keeps normalized qubit count and depth instead of
overwriting them with raw values. This input change requires fresh training too.

Both RL rows additionally expose canonical `ConsolidateBlocks` and
`TwoQubitPeepholeOptimization`. `ElidePermutations` removes virtual swaps
without committing to a physical layout. Its output permutation survives
subsequent mapping. `VF2PostLayout_2q` is the SDK's standard configuration
before translation; `VF2PostLayout` is its configuration for native gates. The
paper mask allows the former on routed physical circuits before synthesis.
Qiskit's native graph is kept inside the worker between actions: rebuilding it
can change later synthesis choices even when the rebuilt circuit is equivalent.
Intermediate consolidated unitary blocks have unavailable proxy rewards; final
scores still use the shared ESP calculation.

The paper mask follows v3's compilation stages: broad optimization before
layout, then optimizations that preserve layout, routing, and the native gate
set. Termination requires all three properties. This deliberately excludes some
optimization-and-repair sequences. SDK preconditions further restrict available
passes. Barriers do not count as multi-qubit gates or routing interactions.
Fully preserving TKET optimizations, currently `RemoveRedundancies`, remain
available after layout in paper mode; original mode retains its legacy filter.

The RL `VF2PostLayout` action uses the standard Qiskit pass on the unchanged
physical target, with the SDK's seed. It has no custom placement cost or ESP
acceptance check. The native Qiskit and TKET baseline pipelines are unchanged.

`original.ppo` lists the legacy PPO settings, including gamma 0.98 and
2,048-step rollouts. `paper.gnn` overrides `GNNConfig.paper()`; an empty table
uses that configuration unchanged. Full rollouts make the default effective
budget 100,352 steps. The manifest records requested, effective, and actual
steps and the optimizer defaults. The original discrete million-value depth
encoding is preserved, including SB3's large one-hot input layer.

### O3 demonstrations and warm start

`paper.warmstart` defaults to five imitation epochs with batches of eight
graphs. Before fitting, the runner compiles all 321 training circuits with
native O3 and replays its accepted individual actions through the actual
environment. It omits no-ops and suffixes that O3 itself rolled back. Layout
bookkeeping stays attached to the relevant canonical action. The replay must fit
the configured episode limit, obey the action masks, and reproduce the final
circuit, logical-output mapping, and ESP. Any mismatch stops training; no
circuit is silently excluded. The evaluation split supplies no demonstrations.

The GNN learns masked action labels and discounted demonstration returns, then
continues with ordinary PPO rollouts. The teacher is never called during RL
evaluation. `teacher.json` records each verified sequence, reference score,
native pass count, and runtime. The manifest records imitation losses, training
action accuracy, and separate demonstration/fit costs. Imitation is additional
training work; its transitions are not counted as PPO timesteps. The five-epoch
default is an initial setting, not a claim that the learned policy matches O3.

Configured dropout applies during imitation. PPO disables dropout during both
rollout collection and updates so unchanged weights produce the same action
likelihoods. The KL guard remains active for actual policy changes. Runs
predating this correction need a fresh output directory because resume checks
source identity.

Use separate TOML files and output roots for these ablations:

```bash
uv run python -m mqt.predictor.rl.experiments.scasia --compiler paper --config PATH
```

| Variant                  | `paper.warmstart.epochs` | `experiment.training_timesteps` |
| ------------------------ | -----------------------: | ------------------------------: |
| RL from scratch          |                        0 |                          100000 |
| Imitation only           |                        5 |                               0 |
| Imitation followed by RL |                        5 |                          100000 |

Keep the other settings identical and retain the native Qiskit/TKET references.
This separates gains from imitation and subsequent RL. The original PPO row has
no imitation stage. Discounting and reward shaping are unchanged; faster or
higher-quality inference must be demonstrated by evaluation, including GNN time.

One rolling `checkpoint.zip` is replaced after each complete imitation epoch and
every 2,048-step PPO update. `warmstart.zip` retains the imitation model, and
`final.zip` is saved at completion. `checkpoint_steps` must be a multiple of
both configured rollout sizes. Use the same command with `--resume` after
interruption. Resume checks configuration, code, dependency versions, frozen
inputs, and the identity embedded in the checkpoint. It records the restart and
trains only the remaining budget. An interrupted imitation phase regenerates and
verifies the demonstrations, then resumes after its last complete epoch.
Environment state, partial rollouts, and random-generator state are not
restored; continuation is not bit-identical. Completed evaluation rows are
skipped on resume, and an incomplete final JSON line is discarded. Use a new
output directory when changing settings or code.

BQSKit block synthesis now rejects results above its synthesis tolerance,
including inaccurate results returned when QSearch or LEAP exhausts the
three-layer search. Both RL rows treat these as pass failures. Restart RL
training in a new output directory after updating from `93244f66`; its
checkpoints may have learned from inequivalent circuits and must not be resumed.

## Comparison plots

During or after evaluation, point the report script at the parent of the
compiler output folders:

```sh
uv run experiments/compare.py /path/to/scasia-runs
```

Open `scasia-runs/comparison/comparison.html` in a browser. It is a standalone,
offline report with the paper's grouped ESP plot, completion/failure rates, ESP
distributions, runtime distributions and per-circuit paper-minus-original ESP
differences. SVG/PNG plots and the plotted summary/per-circuit CSV values are
saved beside it. Use `--output /path/to/report` to choose another report
directory.

Rows without a manifest or completed evaluation records are excluded and listed
in the report. An unfinished final JSON line is ignored without changing the
results. If no rows are available yet, the command prints a message and exits.
The paper-minus-original plot requires both RL rows.

Quality comparisons use repetitions valid in **all included** rows, then average
within each circuit and give circuits equal weight. Failed and missing runs stay
visible in the coverage counts; they are never assigned zero ESP. Runtime
includes all completed attempts, including failures, startup and scoring.
Incompatible inputs, targets, lockfiles or settings are rejected;
code/dependency differences are flagged. Training steps and run commits are
shown for provenance.

The script declares Matplotlib as its only plotting dependency; `uv` runs it in
an isolated script environment. It does not load models, change training outputs
or modify the project's dependency lock. TensorBoard training curves remain
available in each RL row's `logs/` directory.

### Jupyter notebook

Open `experiments/compare.ipynb`, set `RESULTS`, and run all cells. Rerun the
last cell to refresh during evaluation. The notebook uses `compare.py` and
displays the same report inline. Keep the two files together.

Start Jupyter with Matplotlib in an isolated reporting environment:

```sh
cd experiments
uvx --from jupyterlab --with 'matplotlib>=3.10,<4' jupyter lab compare.ipynb
```

Use the default Python 3 kernel. This follows uv's
[standalone Jupyter setup](https://docs.astral.sh/uv/guides/integration/jupyter/#using-jupyter-as-a-standalone-tool)
and leaves the training environment and lockfile unchanged. Over SSH, add
`--no-browser` to the Jupyter command and forward its port from your computer,
for example `ssh -L 8888:localhost:8888 hopf@cda-server-4`. Open the
token-bearing localhost URL printed by Jupyter. While training is running, copy
only `compare.py` and `compare.ipynb` from the updated branch into a separate
directory; start Jupyter there to preserve the run's Git revision.

The `reward_comp_esp_grouped.svg` and `.png` plots reproduce the manuscript's
ESP figure layout using the rerun results. Algorithm names come from the frozen
`<algorithm>_<qubits>_indep.qasm` filenames. Each point averages matched circuit
means within an algorithm; the displayed overall means give each circuit equal
weight. Algorithm order follows the paper row, or the mean across included rows
when paper is unavailable. Missing algorithms are omitted. The report does not
reuse historical paper values or claim identical results.

## Frozen inputs and method differences

- `assets/circuits.zip` preserves all 321 training and 41 evaluation QASM files
  byte-for-byte from prototype `80366c6ac4ec49f65bfc659c38d492029898ccda`;
  filenames and split are unchanged. The archive and every circuit are hashed in
  `assets/manifest.json` and verified at startup.
- Boston's 156-qubit calibration is the
  **selected 17 April 2026 rerun snapshot**, not a verified original-paper
  calibration. The gzip files preserve the original `conf_boston.json` and
  `props_boston.json` bytes from Qiskit IBM Runtime 0.49.0, with uncompressed
  hashes in the manifest. These IBM files are Copyright IBM 2026, Apache-2.0;
  see `assets/LICENSE`. The circuit archive comes from MQT Predictor's
  MIT-licensed prototype.
- All rows use the same snapshot, physical basis gates, scoring code, and
  current dependency lock. The target excludes control flow and paired
  reset/measurement aliases; it retains calibrated physical gate indices,
  durations, errors, coherence times, and timing constraints. Qiskit uses its
  native level-3 preset; TKET uses its native offline IBM level-2 pipeline. TKET
  conversion preserves sparse physical indices and composes implicit output
  permutations.
- `original` ports v2.0.0's transition priorities, seven observations and flat
  masked PPO. Qubit count and depth remain discrete; only the qubit range grows
  to support Boston. It has no gate-frequency or remaining-budget features and
  no reward shaping. Explicit termination earns the objective; pass failures
  terminate with zero reward. The new 32-action cap gives zero reward and
  external truncation with SB3 bootstrapping.
- `paper` uses the current v3 transitions, normalized observations, remaining
  budget, GNN, intermediate rewards and #830 endings: valid termination/horizon
  earns final ESP; invalid horizons and failures receive the terminal penalty
  without bootstrapping. Both RL rows share the updated action registry,
  correctness fixes and SDK availability checks. ESP and the updated actions are
  deliberate additions to the original method. The paper row uses current SB3
  training; it does not restore the prototype's custom PPO trainer.

## Timeouts and recorded results

The default timeout is 60 seconds per registered RL action, or per internal
baseline SDK pass. A parent process enforces deadlines independently of the
compiler's GIL and kills the worker's process group on failure or timeout. The
next request creates a fresh worker. Conversion/scoring and request setup also
have bounded deadlines; worker startup has its own limit. BQSKit uses distinct
server/worker ports for the two rows and bounded process/BLAS counts. The
experiment-local launcher supplies the server port omitted by BQSKit 1.2.1's
attached launcher. For multiple replicas of the same row on one node, assign
different ports and output directories.

`manifest.json` records the commit, source/lock hashes, installed versions,
input hashes, action order, resolved settings, seeds, step counts and checkpoint
provenance. `evaluation.jsonl` records every circuit/repetition, final ESP and
expected fidelity when valid, last observed depth/gate counts, status, actions,
pass traces, and physical output mapping. Unavailable scores are null and
labeled; proxy-to-exact transitions are not comparable deltas. Wall time
includes startup and observation overhead; worker startup and leaf-pass runtimes
are recorded separately. `summary.json` reports pooled efficiency, coverage and
failure counts. Training logs are below `logs/`; native compiler diagnostics go
to each row's `worker.log`.

For comparable ESP deltas, optimization efficiency is
`sum(max(delta, 0)) / sum(abs(delta))` (zero if there is no movement).
Structural success is the fraction of structural invocations that newly reach
their requested target state; mapping requires synthesis, layout and routing.
Overall efficiency weights those two values by their respective invocation
counts. PPE is the mean **signed** ESP delta over comparable
structural/optimization pass observations. Analysis and administrative passes
are counted separately; composite containers are excluded. Repeated executions
count separately. Scores include exact and proxy coverage, excluded transitions,
failed passes and timed-out passes. An opaque SDK pass's internal search
iterations are not observable passes. RL traces use registered actions; baseline
traces use SDK passes, so their invocation granularity differs.

For a local smoke run, use a separate output directory, small matching
`n_steps`/`batch_size`/ `checkpoint_steps`, a small training budget, one
repetition, and `evaluation_circuits = 1`. Full cluster experiments are a
separate execution step.
