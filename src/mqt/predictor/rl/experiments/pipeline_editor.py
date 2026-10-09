# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Run small edits of native presets against the frozen experiment inputs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import time
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bqskit import MachineModel
from bqskit.ext import qiskit_to_bqskit

from mqt.predictor.rl.actions.bqskit_actions import (
    bqskit_to_qiskit,
    final_layout_bqskit_to_qiskit,
    get_bqskit_native_gates,
)

from .inputs import Inputs, digest
from .pipeline_actions import MAX_EDITS
from .pipeline_bqskit import compile_bqskit
from .pipeline_qiskit import run_native_qiskit
from .pipeline_tket import compile_tket
from .worker import CompilerWorker, PassObserver, RuntimeCompiler, observe_qiskit_passes

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from qiskit import QuantumCircuit
    from qiskit.transpiler import Target
    from qiskit_ibm_runtime.fake_provider import FakeBoston


class EditPolicy:
    """Apply explicit stage edits; unmentioned stages retain native behavior."""

    def __init__(self, edits: dict[int, list[str]]) -> None:
        """Keep stage indices local to one selected native preset."""
        self.edits = edits
        self.seen: set[int] = set()
        self.consumed: dict[int, int] = {}

    def __call__(self, state: dict[str, Any]) -> str:
        """Use the same state interface that a learned policy can consume."""
        index = state["stage_index"]
        self.seen.add(index)
        position = self.consumed.get(index, 0)
        actions = self.edits.get(index, [])
        action = actions[position] if position < len(actions) else "KEEP"
        if action not in state["allowed_actions"]:
            msg = f"Invalid {action!r} for stage {index}:{state['stage_name']}."
            raise ValueError(msg)
        self.consumed[index] = position + 1
        return action

    def validate(self) -> None:
        """Reject typos instead of silently executing an unedited baseline."""
        if missing := self.edits.keys() - self.seen:
            msg = f"Unknown editable stage indices: {sorted(missing)}"
            raise ValueError(msg)
        if unused := [index for index, actions in self.edits.items() if self.consumed.get(index, 0) < len(actions)]:
            msg = f"Unconsumed edits at stage indices: {unused}"
            raise ValueError(msg)


class _UnscoredObserver(PassObserver):
    def score(self, circuit: QuantumCircuit) -> dict[str, Any]:
        del circuit
        return {}


def compile_pipeline(
    request: dict[str, Any],
    backend: FakeBoston,
    target: Target,
    settings: dict[str, Any],
    connection: Connection,
) -> tuple[QuantumCircuit, dict[str, Any]]:
    """Run in CompilerWorker so a stalled native stage cannot hang the caller."""
    name = request["pipeline"]
    policy = EditPolicy(request.get("edits", {}))
    max_edits = request.get("max_edits", MAX_EDITS)
    stages: list[dict[str, Any]] = []
    pending: dict[str, Any] = {}

    def event(entry: dict[str, Any]) -> None:
        nonlocal pending
        if entry["event"] == "before":
            pending = {
                **entry,
                "index": len(stages),
                "name": entry["stage_name"],
                "category": "native_stage",
                "status": "running",
                "started": time.monotonic(),
            }
            connection.send(("phase", entry["stage_name"]))
            if name != "qiskit":
                connection.send(("begin", pending))
        else:
            pending.update(entry)
            pending["runtime_seconds"] = time.monotonic() - pending.pop("started")
            pending["status"] = "skipped" if entry["occurrence"] is None else "ok"
            stages.append(pending)
            if name != "qiskit":
                connection.send(("end", pending))

    circuit = request["circuit"]
    started = time.monotonic()
    if name == "qiskit":
        # Keep internal pass deadlines but omit expensive intermediate ESP scores.
        with observe_qiskit_passes(_UnscoredObserver(connection, target)):
            result, _ = run_native_qiskit(circuit, target, request["seed"], policy, max_edits=max_edits, on_event=event)
    elif name == "tket":
        result, _, _ = compile_tket(
            circuit,
            backend,
            policy,
            optimisation_level=request.get("tket_level", 2),
            max_edits=max_edits,
            on_event=event,
        )
    elif name == "bqskit":
        model = MachineModel(
            target.num_qubits,
            coupling_graph=target.build_coupling_map().get_edges(),
            gate_set=get_bqskit_native_gates(target),
        )
        with RuntimeCompiler(
            settings["bqskit_workers"],
            port=settings["bqskit_port"],
            worker_port=settings["bqskit_worker_port"],
            num_blas_threads=settings["threads"],
        ) as compiler:
            compiled, data, _ = compile_bqskit(
                qiskit_to_bqskit(circuit),
                model,
                compiler,
                policy,
                optimization_level=request.get("bqskit_level", 1),
                max_edits=max_edits,
                seed=request["seed"],
                on_event=event,
            )
        result = bqskit_to_qiskit(compiled)
        result._layout = final_layout_bqskit_to_qiskit(  # ruff: ignore[private-member-access]
            tuple(data.initial_mapping), tuple(data.final_mapping), result, circuit
        )
    else:
        raise ValueError(name)
    policy.validate()
    return result, {"stage_trace": stages, "compilation_seconds": time.monotonic() - started}


def main() -> None:
    """Write every requested compilation, including failures, to a fresh directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", choices=("qiskit", "tket", "bqskit"), required=True)
    parser.add_argument("--config", type=Path, default=Path("experiments/scasia.toml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--circuit", nargs="+", required=True, help="Exact archive names, e.g. train/qft_3_indep.qasm")
    parser.add_argument("--seed", nargs="+", type=int, default=[0])
    parser.add_argument("--max-edits", type=int, default=MAX_EDITS, help="Maximum edits per compilation (default: 2)")
    parser.add_argument(
        "--edit", action="append", default=[], help="INDEX:SKIP, INDEX:END, or INDEX:INSERT:NATIVE_STAGE_INDEX"
    )
    parser.add_argument("--bqskit-level", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--tket-level", type=int, choices=(2, 3), default=2)
    parser.add_argument("--bqskit-port", type=int, default=37472)
    parser.add_argument("--bqskit-worker-port", type=int, default=37474)
    args = parser.parse_args()
    if args.max_edits < 0:
        parser.error("--max-edits must be nonnegative.")
    edits: dict[int, list[str]] = {}
    for edit in args.edit:
        index, separator, action = edit.partition(":")
        if (
            not separator
            or not index.isdecimal()
            or (
                action not in {"KEEP", "SKIP", "END"}
                and not (action.startswith("INSERT:") and action.removeprefix("INSERT:").isdecimal())
            )
        ):
            parser.error(f"Invalid stage edit: {edit}")
        actions = edits.setdefault(int(index), [])
        if actions and not actions[-1].startswith("INSERT:"):
            parser.error(f"Only INSERT allows another decision at the same stage: {edit}")
        actions.append(action)
    if sum(action != "KEEP" for actions in edits.values() for action in actions) > args.max_edits:
        parser.error(f"At most {args.max_edits} edits are allowed per compilation.")
    config = tomllib.loads(args.config.read_text())
    inputs = Inputs((args.config.resolve().parent / config["assets"]).resolve())
    for name in args.circuit:
        if name not in inputs.circuits:
            parser.error(f"Unknown circuit: {name}")
    settings = {
        **config["worker"],
        "bqskit_port": args.bqskit_port,
        "bqskit_worker_port": args.bqskit_worker_port,
        "log_file": str(args.output.resolve() / "worker.log"),
    }
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[name] = str(settings["threads"])
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[5]
    manifest = {
        "compiler": args.compiler,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "sources": {str(p.relative_to(root)): digest(p.read_bytes()) for p in sorted((root / "src").rglob("*.py"))},
        "lock_sha256": digest((root / "uv.lock").read_bytes()),
        "dependencies": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()},
        "inputs": inputs.manifest,
        "circuits": args.circuit,
        "seeds": args.seed,
        "edits": edits,
        "max_edits": args.max_edits,
        "bqskit_level": args.bqskit_level,
        "tket_level": args.tket_level,
        "tket_greedy_thread_timeout_seconds": 300 if args.compiler == "tket" and args.tket_level == 3 else None,
        "worker": settings,
        "tket_lightsabre": {"seed": 0, "attempts": 50},
        "timing": "compilation_seconds includes editor, conversion and BQSKit runtime startup; excludes final ESP",
        "timeout": "Qiskit internal pass; TKET/BQSKit native top-level stage",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    worker = CompilerWorker(inputs.path, settings)
    try:
        with (args.output / "results.jsonl").open("w") as output:
            for name in args.circuit:
                for seed in args.seed:
                    result = worker.run({
                        "compiler": "pipeline_editor",
                        "pipeline": args.compiler,
                        "circuit": inputs.circuit(name),
                        "seed": seed,
                        "edits": edits,
                        "max_edits": args.max_edits,
                        "bqskit_level": args.bqskit_level,
                        "tket_level": args.tket_level,
                    })
                    result.pop("circuit", None)
                    result.update(circuit_name=name, seed=seed)
                    output.write(json.dumps(result, allow_nan=False) + "\n")
                    output.flush()
                    print(
                        f"{name} seed={seed}: {result['status']} ESP={result.get('score', {}).get('esp')}", flush=True
                    )
    finally:
        worker.close()


if __name__ == "__main__":
    main()
