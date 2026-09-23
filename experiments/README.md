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
Qiskit uses the repetition seed. Native SDK search trials are unchanged.

`original.ppo` lists the legacy PPO settings, including gamma 0.98 and
2,048-step rollouts. `paper.gnn` overrides `GNNConfig.paper()`; an empty table
uses that configuration unchanged. Full rollouts make the default effective
budget 100,352 steps. The manifest records requested, effective, and actual
steps and the optimizer defaults. The original discrete million-value depth
encoding is preserved, including SB3's large one-hot input layer.

One rolling `checkpoint.zip` is replaced after every 2,048-step PPO update;
`final.zip` is saved at completion. `checkpoint_steps` must be a multiple of
both configured rollout sizes. Use the same command with `--resume` after
interruption. Resume checks configuration, code, dependency versions, frozen
inputs, and the identity embedded in the checkpoint. It records the restart and
trains only the remaining budget. Environment state, partial rollouts, and
random-generator state are not restored; continuation is not bit-identical.
Completed evaluation rows are skipped on resume, and an incomplete final JSON
line is discarded. Use a new output directory when changing settings or code.

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
