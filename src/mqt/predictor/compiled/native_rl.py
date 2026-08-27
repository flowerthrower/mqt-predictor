# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Train the compiled linear actor with episodic policy gradients."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .policy import ACTION_NAMES, FEATURE_NAMES, MAX_PASSES, LinearPolicy, export_linear_policy

if TYPE_CHECKING:
    from collections.abc import Sequence


HEADER_PATTERN = re.compile(
    r"schema=\S+ core_pin=(?P<core>[0-9a-f]+) target=.* "
    r"target_fingerprint=(?P<fingerprint>sha256:[0-9a-f]{64}) policy="
)
STEP_PATTERN = re.compile(
    r"step=(?P<step>\d+) action=(?P<action>[a-z-]+) qubits=(?P<qubits>\d+) "
    r"depth=(?P<depth>\d+) two_qubit_depth=(?P<two_qubit_depth>\d+) "
    r"gates=(?P<gates>\d+) two_qubit=(?P<two_qubit>\d+) mapped=(?P<mapped>[01]) "
    r"routed=(?P<routed>[01]) synthesized=(?P<synthesized>[01]) "
    r"legal=(?P<legal>[01]+) features=\{(?P<features>[^}]*)\}"
)
CORE_PATTERN = re.compile(
    r"candidate=0 schedule=core valid=1 two_qubit_depth=(?P<two_qubit_depth>\d+) "
    r"two_qubit=(?P<two_qubit>\d+) depth=(?P<depth>\d+) gates=(?P<gates>\d+)"
)
EXHAUSTIVE_PATTERN = re.compile(
    r"search-result winner=\d+ schedule=\S+ valid=\d+ unique_outputs=\d+ "
    r"two_qubit_depth=(?P<two_qubit_depth>\d+) two_qubit=(?P<two_qubit>\d+) "
    r"depth=(?P<depth>\d+) gates=(?P<gates>\d+)"
)
PASS_PENALTY = 1e-3


@dataclass(frozen=True, order=True)
class CompileMetrics:
    """The ordered native compilation objective."""

    two_qubit_depth: int
    two_qubit: int
    depth: int
    gates: int


@dataclass(frozen=True)
class Transition:
    """One sampled native policy decision."""

    action: int
    features: tuple[float, ...]
    legal: tuple[bool, ...]
    metrics: CompileMetrics


@dataclass(frozen=True)
class Episode:
    """One complete compiled-policy trajectory."""

    transitions: tuple[Transition, ...]
    terminated: bool
    fell_back: bool

    @property
    def pass_count(self) -> int:
        """Count transformation decisions, excluding terminate."""
        return sum(ACTION_NAMES[item.action] != "terminate" for item in self.transitions)

    @property
    def repeated_passes(self) -> int:
        """Count uses beyond the first use of each transformation."""
        counts = Counter(
            ACTION_NAMES[item.action] for item in self.transitions if ACTION_NAMES[item.action] != "terminate"
        )
        return sum(max(count - 1, 0) for count in counts.values())

    @property
    def repeated_optimizations(self) -> int:
        """Count repeated uses of reorderable optimization passes."""
        counts = Counter(ACTION_NAMES[item.action] for item in self.transitions if item.action < 3)
        return sum(max(count - 1, 0) for count in counts.values())

    @property
    def action_counts(self) -> dict[str, int]:
        """Count each action in the trajectory."""
        counts = Counter(self.actions)
        return {name: counts[name] for name in ACTION_NAMES if counts[name]}

    @property
    def final_metrics(self) -> CompileMetrics | None:
        """Metrics at the terminal decision."""
        if not self.terminated or not self.transitions:
            return None
        return self.transitions[-1].metrics

    @property
    def actions(self) -> tuple[str, ...]:
        """Ordered action names."""
        return tuple(ACTION_NAMES[item.action] for item in self.transitions)


def _features(text: str) -> tuple[float, ...]:
    values: dict[str, float] = {}
    for item in text.split(","):
        name, value = item.split("=", maxsplit=1)
        values[name] = float(value)
    if tuple(values) != FEATURE_NAMES:
        msg = "native trace feature order does not match the policy ABI"
        raise ValueError(msg)
    return tuple(values[name] for name in FEATURE_NAMES)


def parse_episode(trace: str) -> Episode:
    """Parse one C++ sampled-policy trace."""
    transitions: list[Transition] = []
    for match in STEP_PATTERN.finditer(trace):
        action_name = match["action"]
        if action_name not in ACTION_NAMES:
            msg = f"native trace contains unknown action: {action_name}"
            raise ValueError(msg)
        legal_text = match["legal"]
        if len(legal_text) != len(ACTION_NAMES):
            msg = "native trace action mask does not match the policy ABI"
            raise ValueError(msg)
        transitions.append(
            Transition(
                action=ACTION_NAMES.index(action_name),
                features=_features(match["features"]),
                legal=tuple(value == "1" for value in legal_text),
                metrics=CompileMetrics(
                    two_qubit_depth=int(match["two_qubit_depth"]),
                    two_qubit=int(match["two_qubit"]),
                    depth=int(match["depth"]),
                    gates=int(match["gates"]),
                ),
            )
        )
    terminated = bool(transitions and ACTION_NAMES[transitions[-1].action] == "terminate")
    return Episode(
        transitions=tuple(transitions),
        terminated=terminated,
        fell_back="falling back to Core's canonical target pipeline" in trace,
    )


def parse_runtime_identity(trace: str) -> tuple[str, str]:
    """Read the exact Core revision and target fingerprint from C++."""
    match = HEADER_PATTERN.search(trace)
    if match is None:
        msg = "native trace does not contain runtime compatibility metadata"
        raise ValueError(msg)
    return match["core"], match["fingerprint"]


def parse_core_metrics(trace: str) -> CompileMetrics:
    """Read Core's candidate metrics from exhaustive-mode output."""
    match = CORE_PATTERN.search(trace)
    if match is None:
        msg = "native trace does not contain a valid Core baseline"
        raise ValueError(msg)
    return CompileMetrics(
        two_qubit_depth=int(match["two_qubit_depth"]),
        two_qubit=int(match["two_qubit"]),
        depth=int(match["depth"]),
        gates=int(match["gates"]),
    )


def parse_exhaustive_metrics(trace: str) -> CompileMetrics:
    """Read the best zero-or-one-use ordering from exhaustive-mode output."""
    match = EXHAUSTIVE_PATTERN.search(trace)
    if match is None:
        msg = "native trace does not contain a valid exhaustive result"
        raise ValueError(msg)
    return CompileMetrics(
        two_qubit_depth=int(match["two_qubit_depth"]),
        two_qubit=int(match["two_qubit"]),
        depth=int(match["depth"]),
        gates=int(match["gates"]),
    )


def terminal_reward(episode: Episode, baseline: CompileMetrics) -> float:
    """Score terminal quality relative to Core and penalize long schedules."""
    metrics = episode.final_metrics
    if metrics is None or episode.fell_back:
        return -2.0 - PASS_PENALTY * episode.pass_count
    weighted_improvement = 0.0
    for weight, candidate, reference in zip(
        (1.0, 0.25, 0.1, 0.05),
        (metrics.two_qubit_depth, metrics.two_qubit, metrics.depth, metrics.gates),
        (baseline.two_qubit_depth, baseline.two_qubit, baseline.depth, baseline.gates),
        strict=True,
    ):
        weighted_improvement += weight * (reference - candidate) / max(reference, 1)
    return weighted_improvement - PASS_PENALTY * episode.pass_count


def _probabilities(policy: LinearPolicy, transition: Transition) -> np.ndarray:
    features = np.asarray(transition.features, dtype=np.float64)
    legal = np.asarray(transition.legal, dtype=np.bool_)
    logits = (policy.weights.astype(np.float64) @ features + policy.bias.astype(np.float64)).astype(np.float32)
    logits = logits.astype(np.float64)
    logits[~legal] = -np.inf
    logits -= np.max(logits)
    probabilities = np.exp(logits)
    probabilities[~legal] = 0.0
    probabilities /= probabilities.sum()
    return probabilities


def reinforce_update(
    policy: LinearPolicy,
    episodes: Sequence[tuple[Episode, float]],
    *,
    learning_rate: float,
    l2: float,
) -> LinearPolicy:
    """Apply one centered-return REINFORCE update to the linear actor."""
    if not episodes or learning_rate <= 0 or l2 < 0:
        msg = "policy-gradient batch or hyperparameters are invalid"
        raise ValueError(msg)
    returns = np.asarray([reward for _, reward in episodes], dtype=np.float64)
    advantages = returns - returns.mean()
    deviation = advantages.std()
    if deviation > 1e-12:
        advantages /= deviation

    weight_gradient = np.zeros_like(policy.weights, dtype=np.float64)
    bias_gradient = np.zeros_like(policy.bias, dtype=np.float64)
    decisions = 0
    for (episode, _), advantage in zip(episodes, advantages, strict=True):
        for transition in episode.transitions:
            probabilities = _probabilities(policy, transition)
            score = -probabilities
            score[transition.action] += 1.0
            weight_gradient += advantage * np.outer(score, transition.features)
            bias_gradient += advantage * score
            decisions += 1
    if decisions == 0:
        return policy

    weight_gradient /= decisions
    bias_gradient /= decisions
    weight_gradient -= l2 * policy.weights
    gradient_norm = np.sqrt(np.square(weight_gradient).sum() + np.square(bias_gradient).sum())
    if gradient_norm > 1.0:
        weight_gradient /= gradient_norm
        bias_gradient /= gradient_norm
    return LinearPolicy(
        weights=(policy.weights + learning_rate * weight_gradient).astype(np.float32),
        bias=(policy.bias + learning_rate * bias_gradient).astype(np.float32),
    )


def _source_revision() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
    revision = result.stdout.strip() or "unknown-working-tree"
    status = subprocess.run(["git", "status", "--porcelain"], check=False, capture_output=True, text=True)
    return f"{revision}+dirty" if status.stdout else revision


def _target_arguments(args: argparse.Namespace) -> list[str]:
    if args.target is not None:
        return [f"--target={args.target}"]
    if args.qdmi_device is not None:
        return [f"--qdmi-device={args.qdmi_device}"]
    return [f"--target-qubits={args.target_qubits}"]


def _run(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)


def _probe_runtime(args: argparse.Namespace, output: Path) -> tuple[str, str]:
    command = [
        str(args.binary),
        "--trace",
        f"--max-steps={args.max_passes}",
        *_target_arguments(args),
        "-o",
        str(output),
        str(args.circuits[0]),
    ]
    process = _run(command, args.timeout)
    if process.returncode != 0:
        msg = f"failed to probe native runtime: {process.stderr[-1000:]}"
        raise RuntimeError(msg)
    return parse_runtime_identity(process.stderr)


def _baseline(args: argparse.Namespace, circuit: Path, output: Path) -> tuple[CompileMetrics, CompileMetrics]:
    command = [
        str(args.binary),
        "--trace",
        "--policy=exhaustive",
        *_target_arguments(args),
        "-o",
        str(output),
        str(circuit),
    ]
    process = _run(command, args.timeout)
    if process.returncode != 0:
        msg = f"failed to compile Core baseline for {circuit}: {process.stderr[-1000:]}"
        raise RuntimeError(msg)
    return parse_core_metrics(process.stderr), parse_exhaustive_metrics(process.stderr)


def _eligible_baseline(
    args: argparse.Namespace, circuit: Path, output: Path
) -> tuple[tuple[CompileMetrics, CompileMetrics] | None, str | None]:
    try:
        return _baseline(args, circuit, output), None
    except RuntimeError as error:
        return None, str(error)


def _episode(
    args: argparse.Namespace,
    circuit: Path,
    model: Path,
    output: Path,
    sampling_seed: int | None,
) -> Episode:
    sampling = [] if sampling_seed is None else ["--sample-policy", f"--sampling-seed={sampling_seed}"]
    command = [
        str(args.binary),
        "--trace",
        f"--max-steps={args.max_passes}",
        *sampling,
        f"--model={model}",
        *_target_arguments(args),
        "-o",
        str(output),
        str(circuit),
    ]
    process = _run(command, args.timeout)
    if process.returncode != 0:
        return Episode(transitions=(), terminated=False, fell_back=True)
    return parse_episode(process.stderr)


def _export(
    args: argparse.Namespace,
    path: Path,
    policy: LinearPolicy,
    *,
    core_revision: str,
    target_fingerprint: str,
    samples: int,
) -> None:
    export_linear_policy(
        path,
        policy,
        target_fingerprint_override=target_fingerprint,
        core_revision=core_revision,
        source_revision=_source_revision(),
        algorithm="masked REINFORCE",
        objective="Core-relative two-qubit depth, gates, depth, and pass count",
        samples=max(samples, 1),
        epochs=args.updates,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
    )


def _episode_report(
    circuit: Path,
    episode: Episode,
    baseline: CompileMetrics,
    single_use_best: CompileMetrics,
) -> dict[str, object]:
    metrics = episode.final_metrics
    return {
        "circuit": str(circuit),
        "reward": terminal_reward(episode, baseline),
        "passes": episode.pass_count,
        "repeated_passes": episode.repeated_passes,
        "repeated_optimizations": episode.repeated_optimizations,
        "action_counts": episode.action_counts,
        "actions": list(episode.actions),
        "metrics": None if metrics is None else metrics.__dict__,
        "core": baseline.__dict__,
        "single_use_best": single_use_best.__dict__,
        "beats_single_use_best": metrics is not None and metrics < single_use_best,
        "fell_back": episode.fell_back,
    }


def train(args: argparse.Namespace) -> dict[str, object]:
    """Train, export, and evaluate one native linear policy."""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mqt-native-rl-") as directory:
        temporary = Path(directory)
        core_revision, target_fingerprint = _probe_runtime(args, temporary / "probe.mlir")
        baselines: dict[Path, CompileMetrics] = {}
        single_use_best: dict[Path, CompileMetrics] = {}
        excluded: list[dict[str, str]] = []
        for index, circuit in enumerate(args.circuits):
            baseline_pair, error = _eligible_baseline(args, circuit, temporary / f"baseline-{index}.mlir")
            if baseline_pair is None:
                excluded.append({"circuit": str(circuit), "reason": error or "unknown baseline failure"})
            else:
                baselines[circuit], single_use_best[circuit] = baseline_pair
        circuits = list(baselines)
        if not circuits:
            msg = "none of the requested circuits has a valid Core baseline"
            raise RuntimeError(msg)
        policy = LinearPolicy(
            weights=np.zeros((len(ACTION_NAMES), len(FEATURE_NAMES)), dtype=np.float32),
            bias=np.zeros(len(ACTION_NAMES), dtype=np.float32),
        )
        history: list[dict[str, object]] = []
        best_sampled: dict[Path, tuple[Episode, float]] = {}
        best_repeated: dict[Path, tuple[Episode, float]] = {}
        best_without_repeats: dict[Path, tuple[Episode, float]] = {}
        total_episodes = 0
        repeated_episodes = 0
        repeated_reward = 0.0
        without_repeats_reward = 0.0
        for update in range(args.updates):
            model = temporary / "policy.json"
            _export(
                args,
                model,
                policy,
                core_revision=core_revision,
                target_fingerprint=target_fingerprint,
                samples=max(total_episodes, 1),
            )
            batch: list[tuple[Episode, float]] = []
            for circuit_index, circuit in enumerate(circuits):
                for episode_index in range(args.episodes_per_circuit):
                    sampling_seed = args.seed + update * 100_000 + circuit_index * 1_000 + episode_index
                    episode = _episode(
                        args,
                        circuit,
                        model,
                        temporary / f"episode-{update}-{circuit_index}-{episode_index}.mlir",
                        sampling_seed,
                    )
                    reward = terminal_reward(episode, baselines[circuit])
                    batch.append((episode, reward))
                    if circuit not in best_sampled or reward > best_sampled[circuit][1]:
                        best_sampled[circuit] = (episode, reward)
                    group = best_repeated if episode.repeated_optimizations else best_without_repeats
                    if circuit not in group or reward > group[circuit][1]:
                        group[circuit] = (episode, reward)
                    if episode.repeated_optimizations:
                        repeated_episodes += 1
                        repeated_reward += reward
                    else:
                        without_repeats_reward += reward
            policy = reinforce_update(
                policy,
                batch,
                learning_rate=args.learning_rate,
                l2=args.l2,
            )
            total_episodes += len(batch)
            rewards = [reward for _, reward in batch]
            repeated_rewards = [reward for episode, reward in batch if episode.repeated_optimizations]
            without_repeats_rewards = [reward for episode, reward in batch if not episode.repeated_optimizations]
            history.append({
                "update": update + 1,
                "episodes": len(batch),
                "mean_reward": float(np.mean(rewards)),
                "best_reward": float(np.max(rewards)),
                "terminated": sum(episode.terminated and not episode.fell_back for episode, _ in batch),
                "mean_passes": float(np.mean([episode.pass_count for episode, _ in batch])),
                "episodes_with_repeats": sum(episode.repeated_passes > 0 for episode, _ in batch),
                "episodes_with_repeated_optimizations": sum(episode.repeated_optimizations > 0 for episode, _ in batch),
                "mean_reward_with_repeated_optimizations": (
                    float(np.mean(repeated_rewards)) if repeated_rewards else None
                ),
                "mean_reward_without_repeated_optimizations": (
                    float(np.mean(without_repeats_rewards)) if without_repeats_rewards else None
                ),
            })
            print(json.dumps(history[-1], sort_keys=True), flush=True)

        _export(
            args,
            args.output,
            policy,
            core_revision=core_revision,
            target_fingerprint=target_fingerprint,
            samples=total_episodes,
        )
        evaluation = []
        for index, circuit in enumerate(circuits):
            episode = _episode(args, circuit, args.output, temporary / f"evaluation-{index}.mlir", None)
            evaluation.append(_episode_report(circuit, episode, baselines[circuit], single_use_best[circuit]))
        report: dict[str, object] = {
            "core_revision": core_revision,
            "target_fingerprint": target_fingerprint,
            "max_passes": args.max_passes,
            "updates": args.updates,
            "episodes": total_episodes,
            "episodes_with_repeated_optimizations": repeated_episodes,
            "mean_reward_with_repeated_optimizations": (
                repeated_reward / repeated_episodes if repeated_episodes else None
            ),
            "mean_reward_without_repeated_optimizations": (
                without_repeats_reward / (total_episodes - repeated_episodes)
                if total_episodes > repeated_episodes
                else None
            ),
            "requested_circuits": len(args.circuits),
            "eligible_circuits": len(circuits),
            "excluded": excluded,
            "history": history,
            "evaluation": evaluation,
            "best_sampled": [
                _episode_report(
                    circuit,
                    best_sampled[circuit][0],
                    baselines[circuit],
                    single_use_best[circuit],
                )
                for circuit in circuits
            ],
            "best_sampled_with_repeated_optimizations": [
                _episode_report(
                    circuit,
                    best_repeated[circuit][0],
                    baselines[circuit],
                    single_use_best[circuit],
                )
                for circuit in circuits
                if circuit in best_repeated
            ],
            "best_sampled_without_repeated_optimizations": [
                _episode_report(
                    circuit,
                    best_without_repeats[circuit][0],
                    baselines[circuit],
                    single_use_best[circuit],
                )
                for circuit in circuits
                if circuit in best_without_repeats
            ],
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("circuits", type=Path, nargs="+")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--target", type=Path)
    target.add_argument("--qdmi-device")
    target.add_argument("--target-qubits", type=int, default=5)
    parser.add_argument("--max-passes", type=int, default=MAX_PASSES)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--episodes-per-circuit", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=int, default=120)
    return parser


def main() -> None:
    """Run the native policy-gradient experiment."""
    args = _parser().parse_args()
    if (
        args.max_passes <= 0
        or args.updates <= 0
        or args.episodes_per_circuit <= 0
        or args.learning_rate <= 0
        or args.l2 < 0
        or args.seed < 0
        or args.timeout <= 0
    ):
        msg = "training arguments must be positive (with nonnegative l2 and seed)"
        raise ValueError(msg)
    train(args)


if __name__ == "__main__":
    main()
