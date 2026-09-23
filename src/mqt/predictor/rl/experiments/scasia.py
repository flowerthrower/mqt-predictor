# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Four independent compiler rows with frozen inputs and recoverable SB3 runs."""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
import platform
import subprocess
import time
import tomllib
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

from mqt.predictor.rl.gnn import GNNConfig, GNNMaskablePPO, GNNObservationWrapper, create_gnn_model

from .environment import ExperimentEnv, OriginalEnv
from .inputs import Inputs, digest, load_target
from .metrics import efficiency, observe
from .worker import CompilerWorker

if TYPE_CHECKING:
    from collections.abc import Sequence


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Replace small run metadata atomically."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def resolve_config(path: Path) -> dict[str, Any]:
    """Resolve readable settings and fail before starting an invalid run."""
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    config["assets"] = str((path.parent / config["assets"]).resolve())
    config["output"] = str((path.parent / config["output"]).resolve())
    config["paper"]["gnn"] = asdict(replace(GNNConfig.paper(), **config["paper"]["gnn"]))
    config["original"]["ppo"] = {"clip_range_vf": None, "target_kl": None, **config["original"]["ppo"]}
    for key in ("episode_actions", "training_timesteps", "evaluation_repetitions", "checkpoint_steps"):
        if config["experiment"][key] <= 0:
            raise ValueError(key)
    for key in ("pass_timeout_seconds", "startup_timeout_seconds", "threads", "bqskit_workers"):
        if config["worker"][key] <= 0:
            raise ValueError(key)
    ports = [config[mode][key] for mode in ("original", "paper") for key in ("bqskit_port", "bqskit_worker_port")]
    if len(set(ports)) != 4 or any(not 1024 <= port <= 65535 for port in ports):
        msg = "The two RL rows require four distinct BQSKit ports in [1024, 65535]."
        raise ValueError(msg)
    if config["experiment"]["objective"] not in {"estimated_success_probability", "expected_fidelity"}:
        msg = "Use ESP or expected_fidelity as the shared training objective."
        raise ValueError(msg)
    for rollout in (config["original"]["ppo"]["n_steps"], config["paper"]["gnn"]["n_steps"]):
        if config["experiment"]["checkpoint_steps"] % rollout:
            msg = "checkpoint_steps must be a multiple of each RL row's n_steps (save after PPO updates)."
            raise ValueError(msg)
    return config


def worker_settings(config: dict[str, Any], compiler: str) -> dict[str, Any]:
    """Resolve bounded worker settings and separate runtime ports."""
    row = config[compiler] if compiler in {"original", "paper"} else config["original"]
    return {
        **config["worker"],
        **{key: row[key] for key in ("bqskit_port", "bqskit_worker_port")},
        "log_file": str(Path(config["output"]) / compiler / "worker.log"),
    }


def make_env(compiler: str, config: dict[str, Any], inputs: Inputs) -> ExperimentEnv:
    """Use one action registry and target for the two RL methods."""
    _, target = load_target(inputs.path)
    worker = CompilerWorker(inputs.path, worker_settings(config, compiler))
    cls = OriginalEnv if compiler == "original" else ExperimentEnv
    return cls(target, inputs, worker, config["experiment"], compiler)


def run_identity(config: dict[str, Any], inputs: Inputs, env: ExperimentEnv, compiler: str) -> dict[str, Any]:
    """Record code, lock, installed dependencies, actions and frozen input hashes."""
    root = Path(__file__).resolve().parents[5]
    sources = {str(path.relative_to(root)): digest(path.read_bytes()) for path in sorted((root / "src").rglob("*.py"))}
    versions = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    settings = {key: value for key, value in config.items() if key not in {"assets", "output"}}
    return {
        "compiler": compiler,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_sha256": digest(json.dumps(sources, sort_keys=True).encode()),
        "lock_sha256": digest((root / "uv.lock").read_bytes()),
        "python": platform.python_version(),
        "dependencies": versions,
        "inputs": inputs.manifest,
        "settings": settings,
        "target": {"num_qubits": env.device.num_qubits, "operations": sorted(env.device.operation_names)},
        "actions": [
            {"index": index, "name": action.name, "origin": action.origin, "category": action.pass_type}
            for index, action in env.action_set.items()
        ],
        "legacy_policy": {"net_arch": {"pi": [64, 64], "vf": [64, 64]}, "activation": "Tanh", "optimizer": "Adam"},
        "evaluation_sampling": "seeded stochastic; every repetition retained; no best-of-N",
        "native_sdk_seeds": {"qiskit": "per-compilation seed", "tket_lightsabre": 0, "bqskit_actions": 10},
    }


class RollingCheckpoint(BaseCallback):
    """Retain one rolling checkpoint after completed PPO updates."""

    def __init__(self, output: Path, manifest: dict[str, Any], interval: int) -> None:
        """Save at update boundaries and retain provenance in the run manifest."""
        super().__init__()
        self.output = output
        self.manifest = manifest
        self.interval = interval
        self.last_saved = manifest.get("actual_training_timesteps", 0)

    def save(self, name: str) -> None:
        """Atomically publish the SB3 model and record its completed step count."""
        temporary = self.output / f".{name}.zip"
        self.model.save(temporary)
        path = self.output / f"{name}.zip"
        temporary.replace(path)
        self.last_saved = self.model.num_timesteps
        self.manifest["actual_training_timesteps"] = self.model.num_timesteps
        self.manifest["checkpoints"][name] = {
            "path": path.name,
            "timesteps": self.model.num_timesteps,
            "saved_at": time.time(),
            "phase": "after PPO update; environment and random-generator state are not restored",
        }
        write_json(self.output / "manifest.json", self.manifest)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        if self.model.num_timesteps - self.last_saved >= self.interval:
            self.save("checkpoint")

    def _on_training_end(self) -> None:
        self.save("checkpoint")
        self.save("final")


def train(
    compiler: str, config: dict[str, Any], env: ExperimentEnv, output: Path, manifest: dict[str, Any], *, resume: bool
) -> None:
    """Train to the requested total budget, completing whole SB3 rollouts."""
    settings = config["experiment"]
    wrapped = GNNObservationWrapper(env) if compiler == "paper" else env
    model_class = GNNMaskablePPO if compiler == "paper" else MaskablePPO
    identity = digest(json.dumps(manifest["identity"], sort_keys=True).encode())
    if resume:
        model = model_class.load(output / "checkpoint.zip", env=wrapped)
        if model.__dict__["scasia_identity"] != identity:
            msg = "Checkpoint identity differs from the current configuration, code or dataset."
            raise ValueError(msg)
        manifest["restarts"].append({
            "stage": "train",
            "checkpoint": "checkpoint.zip",
            "timesteps": model.num_timesteps,
            "time": time.time(),
        })
    elif compiler == "paper":
        assert isinstance(wrapped, GNNObservationWrapper)
        model = create_gnn_model(
            wrapped,
            GNNConfig(**config["paper"]["gnn"]),
            verbose=1,
            tensorboard_log=str(output / "logs"),
            seed=settings["training_seed"],
        )
    else:
        model = MaskablePPO(
            "MultiInputPolicy",
            wrapped,
            **config["original"]["ppo"],
            verbose=1,
            tensorboard_log=str(output / "logs"),
            seed=settings["training_seed"],
        )
    model.__dict__["scasia_identity"] = identity
    manifest["model_device"] = str(model.device)
    manifest["optimizer_defaults"] = model.policy.optimizer.defaults
    manifest["effective_training_timesteps"] = (
        (settings["training_timesteps"] + model.n_steps - 1) // model.n_steps * model.n_steps
    )
    manifest["actual_training_timesteps"] = model.num_timesteps
    write_json(output / "manifest.json", manifest)
    remaining = max(0, settings["training_timesteps"] - model.num_timesteps)
    callback = RollingCheckpoint(output, manifest, settings["checkpoint_steps"])
    if remaining:
        model.learn(total_timesteps=remaining, reset_num_timesteps=not resume, callback=callback)
    elif not (output / "final.zip").exists():
        callback.init_callback(model)
        callback.save("final")


def evaluate(
    compiler: str, config: dict[str, Any], env: ExperimentEnv, output: Path, manifest: dict[str, Any], *, resume: bool
) -> None:
    """Retain every independent compilation, including failed repetitions."""
    settings = config["experiment"]
    model = None
    wrapped = GNNObservationWrapper(env) if compiler == "paper" else env
    if compiler in {"original", "paper"}:
        cls = GNNMaskablePPO if compiler == "paper" else MaskablePPO
        model = cls.load(output / "final.zip")
        identity = digest(json.dumps(manifest["identity"], sort_keys=True).encode())
        if model.__dict__["scasia_identity"] != identity:
            msg = "Final model identity differs from this run."
            raise ValueError(msg)
    path = output / "evaluation.jsonl"
    records: list[dict[str, Any]] = []
    if path.exists():
        if not resume:
            msg = "Evaluation exists; use --resume to continue it."
            raise FileExistsError(msg)
        # A killed job can leave only its final JSON line incomplete.
        with path.open("rb+") as stream:
            end = 0
            for line in stream:
                if not line.endswith(b"\n"):
                    break
                records.append(json.loads(line))
                end = stream.tell()
            stream.truncate(end)
    completed = {(row["circuit"], row["repetition"]) for row in records}
    if resume:
        manifest["restarts"].append({"stage": "evaluate", "completed_compilations": len(records), "time": time.time()})
        write_json(output / "manifest.json", manifest)
    names = env.inputs.names("test")
    if settings["evaluation_circuits"]:
        names = names[: settings["evaluation_circuits"]]
    with path.open("a") as stream:
        for index, name in enumerate(names):
            for repetition in range(settings["evaluation_repetitions"]):
                if (name, repetition) in completed:
                    continue
                seed = settings["evaluation_seed"] + index * settings["evaluation_repetitions"] + repetition
                set_random_seed(seed)
                circuit = env.inputs.circuit(name)
                started = time.monotonic()
                result: dict[str, Any]
                if model is None:
                    result = env.worker.run({"compiler": compiler, "circuit": circuit, "seed": seed})
                    actions = []
                else:
                    observation, _ = wrapped.reset(circuit, seed=seed)
                    terminated = truncated = False
                    info: dict[str, Any] = {}
                    while not (terminated or truncated):
                        policy_input = (
                            wrapped.graph_observation if isinstance(wrapped, GNNObservationWrapper) else observation
                        )
                        action, _ = model.predict(
                            cast("Any", policy_input), deterministic=False, action_masks=np.array(env.action_masks())
                        )
                        observation, _, terminated, truncated, info = wrapped.step(int(action))
                    result = {
                        "status": env.last_result.get("status", "error"),
                        "score": observe(env.state, env.device, physical=env.layout is not None),
                        "traces": env.trace,
                        "terminated": terminated,
                        "truncated": truncated,
                        "ending": info,
                        "worker_startup_seconds": env.worker_startup_seconds,
                        "final_layout": env.last_result.get("final_layout"),
                    }
                    if env.error_occurred and result["status"] == "ok":
                        result["status"] = "invalid"
                    actions = env.used_actions
                result.pop("circuit", None)
                score = result.get("score", {})
                if result["status"] == "ok" and score.get("esp_kind") != "exact":
                    result["status"] = "invalid"
                record = {
                    "circuit": name,
                    "repetition": repetition,
                    "seed": seed,
                    "compiler": compiler,
                    **result,
                    "runtime_seconds": time.monotonic() - started,
                    "actions": actions,
                    "final_esp": score.get("esp")
                    if result["status"] == "ok" and score.get("esp_kind") == "exact"
                    else None,
                    "final_expected_fidelity": score.get("expected_fidelity") if result["status"] == "ok" else None,
                    "final_depth": score.get("depth") if result["status"] == "ok" else None,
                    "final_gate_counts": score.get("gate_counts") if result["status"] == "ok" else None,
                    "pass_runtime_seconds": sum(
                        entry.get("runtime_seconds", 0.0) for entry in result["traces"] if not entry.get("container")
                    ),
                    "efficiency": efficiency(result["traces"]),
                }
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                records.append(record)
    traces = [trace for row in records for trace in row["traces"]]
    write_json(
        output / "summary.json",
        {
            "compilations": len(records),
            "successful_compilations": sum(row["status"] == "ok" for row in records),
            "failure_counts": {
                status: sum(row["status"] == status for row in records) for status in ("error", "timeout", "invalid")
            },
            "efficiency": efficiency(traces),
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run one compiler row from a shared TOML configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True, choices=("qiskit", "tket", "original", "paper"))
    parser.add_argument("--config", type=Path, default=Path("experiments/scasia.toml"))
    parser.add_argument("--stage", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = resolve_config(args.config.resolve())
    if os.environ.get("GITHUB_ACTIONS") == "true":
        msg = "Unset GITHUB_ACTIONS: it changes the shared BQSKit action implementations."
        raise ValueError(msg)
    threads = config["worker"]["threads"]
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[name] = str(threads)
    os.environ["QISKIT_PARALLEL"] = "FALSE"
    torch.set_num_threads(threads)
    inputs = Inputs(Path(config["assets"]))
    output = Path(config["output"]) / args.compiler
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = make_env(args.compiler, config, inputs)
        try:
            identity = run_identity(config, inputs, env, args.compiler)
            manifest_path = output / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                if manifest["identity"] != identity:
                    msg = "Existing run differs in code, dependencies, configuration or frozen inputs; choose a new output."
                    raise ValueError(msg)
                if args.stage != "evaluate" and not args.resume:
                    msg = "Existing training run; use --resume or choose a new output directory."
                    raise FileExistsError(msg)
            else:
                if args.resume:
                    msg = "No run manifest to resume."
                    raise FileNotFoundError(msg)
                manifest = {"identity": identity, "restarts": [], "checkpoints": {}, "actual_training_timesteps": 0}
                manifest["requested_training_timesteps"] = (
                    config["experiment"]["training_timesteps"] if args.compiler in {"original", "paper"} else 0
                )
                write_json(manifest_path, manifest)
            if args.compiler in {"original", "paper"} and args.stage in {"train", "all"}:
                train(args.compiler, config, env, output, manifest, resume=args.resume)
            if args.stage in {"evaluate", "all"} or args.compiler in {"qiskit", "tket"}:
                evaluate(args.compiler, config, env, output, manifest, resume=args.resume)
        finally:
            env.close()


if __name__ == "__main__":
    main()
