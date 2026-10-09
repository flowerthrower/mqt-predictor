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

    def __init__(self, edits: dict[int, str]) -> None:
        """Keep stage indices local to one selected native preset."""
        self.edits = edits
        self.seen: set[int] = set()

    def __call__(self, state: dict[str, Any]) -> str:
        """Use the same state interface that a learned policy can consume."""
        index = state["stage_index"]
        self.seen.add(index)
        action = self.edits.get(index, "KEEP")
        if action not in state["allowed_actions"]:
            msg = f"Invalid {action!r} for stage {index}:{state['stage_name']}."
            raise ValueError(msg)
        return action

    def validate(self) -> None:
        """Reject typos instead of silently executing an unedited baseline."""
        if missing := self.edits.keys() - self.seen:
            msg = f"Unknown editable stage indices: {sorted(missing)}"
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
            result, _ = run_native_qiskit(circuit, target, request["seed"], policy, on_event=event)
    elif name == "tket":
        result, _, _ = compile_tket(circuit, backend, policy, on_event=event)
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
    parser.add_argument("--edit", action="append", default=[], help="STAGE_INDEX:SKIP or STAGE_INDEX:REPEAT")
    parser.add_argument("--bqskit-level", type=int, choices=(1, 2), default=1)
    parser.add_argument("--bqskit-port", type=int, default=37472)
    parser.add_argument("--bqskit-worker-port", type=int, default=37474)
    args = parser.parse_args()
    edits: dict[int, str] = {}
    for edit in args.edit:
        index, action = edit.split(":")
        if action not in {"KEEP", "SKIP", "REPEAT"} or int(index) in edits:
            parser.error(f"Invalid or repeated stage edit: {edit}")
        edits[int(index)] = action
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
        "bqskit_level": args.bqskit_level,
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
                        "bqskit_level": args.bqskit_level,
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
