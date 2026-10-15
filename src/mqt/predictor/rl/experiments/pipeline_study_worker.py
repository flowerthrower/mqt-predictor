# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Run cleanly timed native pipeline plans in bounded, isolated workers."""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Any

import joblib
import psutil  # ty: ignore[unresolved-import] -- supplied by the study's existing gnn extra.
from bqskit import MachineModel
from bqskit import compile as native_bqskit
from bqskit.ext import qiskit_to_bqskit
from pytket.extensions.qiskit import IBMQBackend, qiskit_to_tk
from pytket.predicates import CompilationUnit

from mqt.predictor.rl.actions.bqskit_actions import (
    bqskit_to_qiskit,
    final_layout_bqskit_to_qiskit,
    get_bqskit_native_gates,
)

from .inputs import Inputs, load_target
from .metrics import observe
from .pipeline_bqskit import compile_bqskit
from .pipeline_editor import EditPolicy
from .pipeline_learning import choose_plan, input_features
from .pipeline_qiskit import run_native_qiskit, run_qiskit_variant
from .pipeline_tket import compile_tket
from .worker import CompilerWorker, RuntimeCompiler, tket_physical_circuit

if TYPE_CHECKING:
    from collections.abc import Callable
    from multiprocessing.connection import Connection

    from qiskit import QuantumCircuit
    from qiskit.transpiler import Target
    from qiskit_ibm_runtime.fake_provider import FakeBoston


class StudyWorker(CompilerWorker):
    """Bound total request time and process-tree memory without scoring failures."""

    def __init__(self, assets: Path, settings: dict[str, Any], slot: int) -> None:
        """Reserve separate runtime ports and one log for each concurrent slot."""
        port = settings.get("bqskit_port_base", 43472) + 4 * slot
        resolved = {
            "startup_timeout_seconds": 120.0,
            "compilation_timeout_seconds": 600.0,
            "memory_limit_bytes": 4 * 1024**3,
            "threads": 1,
            "max_edits": 6,
            **settings,
            "bqskit_port": port,
            "bqskit_worker_port": port + 2,
            "log_file": str(Path(settings["output"]) / f"worker-{slot}.log"),
        }
        super().__init__(assets, resolved)
        self.peak_rss_bytes = 0
        self.cancel_requested = Event()
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
            os.environ[variable] = str(resolved["threads"])

    def cancel(self) -> None:
        """Ask the active request to clean up its own process before pool shutdown."""
        self.cancel_requested.set()

    def close(self) -> None:
        """Use owned process identities if the host denies process-group signals."""
        owned = []
        if self.process is not None and self.process.pid is not None:
            with contextlib.suppress(psutil.NoSuchProcess):
                process = psutil.Process(self.process.pid)
                owned = [process, *process.children(recursive=True)]
        try:
            super().close()
        except PermissionError:
            for process in owned:
                with contextlib.suppress(psutil.NoSuchProcess):
                    process.kill()
            assert self.process is not None
            self.process.join(timeout=1)
            _, remaining = psutil.wait_procs(owned[1:], timeout=1)
            if self.process.is_alive() or any(
                process.is_running() and process.status() != psutil.STATUS_ZOMBIE for process in remaining
            ):
                msg = "Owned compiler processes survived cleanup."
                raise RuntimeError(msg) from None
            self.process.close()
            self.process = None
            if self.connection is not None:
                self.connection.close()
                self.connection = None

    def _receive(self, deadline: float) -> tuple[str, Any]:
        assert self.process is not None
        assert self.process.pid is not None
        assert self.connection is not None
        while True:
            if self.cancel_requested.is_set():
                msg = "Study interrupted."
                raise InterruptedError(msg)
            rss = 0
            with contextlib.suppress(psutil.NoSuchProcess):
                process = psutil.Process(self.process.pid)
                for child in [process, *process.children(recursive=True)]:
                    with contextlib.suppress(psutil.NoSuchProcess):
                        rss += child.memory_info().rss
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            if rss > self.settings["memory_limit_bytes"]:
                msg = f"Process-tree RSS {rss} exceeds {self.settings['memory_limit_bytes']} bytes."
                raise MemoryError(msg)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                msg = "Aggregate deadline exceeded."
                raise TimeoutError(msg)
            if self.connection.poll(min(0.25, remaining)):
                return self.connection.recv()

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        """Restart failed workers; stage messages never extend the deadline."""
        started = time.monotonic()
        startup_seconds = 0.0
        phase = "worker startup"
        self.peak_rss_bytes = 0
        try:  # ruff: ignore[too-many-statements-in-try-clause] -- every protocol failure must close the worker tree.
            if self.process is None:
                context = mp.get_context("spawn")
                self.connection, child = context.Pipe()
                self.process = context.Process(target=_worker_main, args=(child, self.assets, self.settings))
                self.process.start()
                child.close()
                kind, payload = self._receive(started + self.settings["startup_timeout_seconds"])
                if kind != "ready":
                    raise RuntimeError(payload)  # ruff: ignore[raise-within-try] -- dispose of failed initialization.
                startup_seconds = time.monotonic() - started
            assert self.connection is not None
            self.connection.send(request)
            phase = "request setup"
            deadline = time.monotonic() + self.settings["compilation_timeout_seconds"]
            while True:
                kind, payload = self._receive(deadline)
                if kind == "phase":
                    phase = payload
                elif kind == "result":
                    return {
                        **payload,
                        "wall_seconds": time.monotonic() - started,
                        "worker_startup_seconds": startup_seconds,
                        "peak_rss_bytes": self.peak_rss_bytes,
                    }
                elif kind == "error":
                    raise RuntimeError(payload)  # ruff: ignore[raise-within-try] -- dispose of failed compilation.
        except (TimeoutError, MemoryError, EOFError, BrokenPipeError, OSError, RuntimeError) as error:
            self.close()
            status = (
                "cancelled"
                if isinstance(error, InterruptedError)
                else "timeout"
                if isinstance(error, TimeoutError)
                else "memory_limit"
                if isinstance(error, MemoryError)
                else "error"
            )
            return {
                "status": status,
                "error": f"{type(error).__name__}: {error}",
                "offending_phase": phase,
                "seconds": None,
                "score": None,
                "plan_id": request.get("plan", {}).get("id"),
                "wall_seconds": time.monotonic() - started,
                "worker_startup_seconds": startup_seconds,
                "peak_rss_bytes": self.peak_rss_bytes,
            }
        except BaseException:
            self.close()
            raise


def _compile(
    circuit: QuantumCircuit,
    plan: dict[str, Any],
    seed: int,
    backend: FakeBoston,
    target: Target,
    settings: dict[str, Any],
    event: Callable[[dict[str, Any]], None],
) -> tuple[QuantumCircuit, list[dict[str, Any]]]:
    policy = EditPolicy({int(index): actions for index, actions in plan.get("edits", {}).items()})
    trace: list[dict[str, Any]] = []
    native = plan.get("native", False)
    level = plan["level"]
    budget = settings["max_edits"]
    if plan["compiler"] == "qiskit":
        if native or plan.get("optimization", "native") != "native":
            compiled = run_qiskit_variant(
                circuit, target, seed, optimization_level=level, optimization=plan.get("optimization", "native")
            )
        else:
            compiled, trace = run_native_qiskit(circuit, target, seed, policy, max_edits=budget, on_event=event)
    elif plan["compiler"] == "tket":
        if native:
            pipeline = IBMQBackend.default_compilation_pass_offline(
                backend.configuration(), backend.properties(), optimisation_level=level
            )
            unit = CompilationUnit(qiskit_to_tk(circuit))
            pipeline.apply(unit)
            compiled = tket_physical_circuit(unit, circuit)
        else:
            compiled, _, trace = compile_tket(
                circuit, backend, policy, optimisation_level=level, max_edits=budget, on_event=event
            )
    elif plan["compiler"] == "bqskit":
        model = MachineModel(
            target.num_qubits,
            coupling_graph=target.build_coupling_map().get_edges(),
            gate_set=get_bqskit_native_gates(target),
        )
        with RuntimeCompiler(
            1,
            port=settings["bqskit_port"],
            worker_port=settings["bqskit_worker_port"],
            num_blas_threads=settings["threads"],
        ) as compiler:
            source = qiskit_to_bqskit(circuit)
            if native:
                output, initial, final = native_bqskit(
                    source, model, optimization_level=level, compiler=compiler, seed=seed, with_mapping=True
                )
            else:
                output, data, trace = compile_bqskit(
                    source,
                    model,
                    compiler,
                    policy,
                    optimization_level=level,
                    seed=seed,
                    max_edits=budget,
                    on_event=event,
                )
                initial, final = tuple(data.initial_mapping), tuple(data.final_mapping)
            compiled = bqskit_to_qiskit(output)
            compiled._layout = final_layout_bqskit_to_qiskit(  # ruff: ignore[private-member-access]
                tuple(initial), tuple(final), compiled, circuit
            )
    else:
        raise ValueError(plan["compiler"])
    policy.validate()
    return compiled, trace


def _worker_main(connection: Connection, assets: Path, settings: dict[str, Any]) -> None:
    os.setsid()
    log_path = Path(settings["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
    try:  # ruff: ignore[too-many-statements-in-try-clause] -- SDK exceptions cross the worker boundary as data.
        inputs = Inputs(assets)
        backend, target = load_target(assets)
        fitted: dict[str, Any] = {}
        connection.send(("ready", None))
        while (request := connection.recv()) is not None:
            circuit = inputs.circuit(request["circuit"])
            policy_file = request.get("policy_file")
            if policy_file and policy_file not in fitted:
                fitted[policy_file] = joblib.load(policy_file)
            features = {} if policy_file else input_features(circuit, target.num_qubits)
            connection.send(("phase", "compilation"))
            started = time.perf_counter()
            selection_seconds = 0.0
            if policy_file:
                features = input_features(circuit, target.num_qubits)
                plan = choose_plan(fitted[policy_file], features, request["weight"])
                selection_seconds = time.perf_counter() - started
            else:
                plan = request["plan"]

            def event(entry: dict[str, Any]) -> None:
                connection.send(("phase", entry["stage_name"] if entry["event"] == "before" else "between stages"))

            compiled, trace = _compile(circuit, plan, request["seed"], backend, target, settings, event)
            seconds = time.perf_counter() - started
            connection.send(("phase", "final ESP scoring"))
            score = observe(compiled, target, physical=True)
            if score["esp_kind"] != "exact" or score["esp"] is None or not all(score["state"].values()):
                msg = f"Unavailable physical score: {score}"
                raise ValueError(msg)  # ruff: ignore[raise-within-try] -- reject invalid scores through worker cleanup.
            connection.send((
                "result",
                {
                    "status": "ok",
                    "seconds": seconds,
                    "score": score,
                    "features": features,
                    "plan_id": plan["id"],
                    "stage_trace": trace,
                    "selection_seconds": selection_seconds,
                    "final_layout": compiled.layout.final_index_layout() if compiled.layout is not None else None,
                },
            ))
    except Exception:  # ruff: ignore[blind-except] -- native SDK failures must reach the parent watchdog.
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()
