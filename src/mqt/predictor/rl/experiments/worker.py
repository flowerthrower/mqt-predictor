# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Disposable compiler processes with parent-enforced pass deadlines."""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from bqskit.compiler import Compiler
from pytket.circuit import Qubit
from pytket.extensions.qiskit import IBMQBackend, qiskit_to_tk, tk_to_qiskit
from pytket.passes import BasePass as TketBasePass
from qiskit.converters import dag_to_circuit
from qiskit.transpiler import Layout, TranspileLayout
from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from mqt.predictor.rl.actions import bqskit_actions
from mqt.predictor.rl.predictorenv import PredictorEnv

from .inputs import load_target
from .metrics import observe

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from multiprocessing.connection import Connection

    from pytket import Circuit
    from pytket.predicates import CompilationUnit
    from qiskit import QuantumCircuit
    from qiskit.dagcircuit import DAGCircuit
    from qiskit.passmanager import PassManagerState
    from qiskit.transpiler import Target
    from qiskit_ibm_runtime.fake_provider import FakeBoston


class CompilerWorker:
    """Keep compilation failures and native hangs outside the learning process."""

    def __init__(self, assets: Path, settings: dict[str, Any]) -> None:
        """Configure a persistent worker; start it only on the first request."""
        self.assets = assets
        self.settings = settings
        self.process = None
        self.connection = None

    def close(self) -> None:
        """Kill the worker's process group, including BQSKit children."""
        if self.process is not None:
            assert self.process.pid is not None
            for sig in (signal.SIGTERM, signal.SIGKILL):
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.process.pid, sig)
                self.process.join(timeout=0.5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run a request, restarting the deadline on each pass boundary."""
        traces: dict[int, dict[str, Any]] = {}
        phase = "worker startup"
        started = time.monotonic()
        startup_seconds = 0.0
        try:  # ruff: ignore[too-many-statements-in-try-clause] -- all protocol failures must dispose of the worker.
            if self.process is None:
                context = mp.get_context("spawn")
                self.connection, child = context.Pipe()
                self.process = context.Process(target=_worker_main, args=(child, self.assets, self.settings))
                self.process.start()
                child.close()
                if not self.connection.poll(self.settings["startup_timeout_seconds"]):
                    raise TimeoutError(phase)  # ruff: ignore[raise-within-try] -- use the common worker cleanup path.
                assert self.connection.recv() == ("ready", None)
                startup_seconds = time.monotonic() - started
            assert self.connection is not None
            self.connection.send(request)
            phase = "request setup"
            while True:
                if not self.connection.poll(self.settings["pass_timeout_seconds"]):
                    raise TimeoutError(phase)  # ruff: ignore[raise-within-try] -- use the common worker cleanup path.
                kind, payload = self.connection.recv()
                if kind == "phase":
                    phase = payload
                elif kind in {"begin", "end", "update"}:
                    traces[payload["index"]] = payload
                    if kind != "update":
                        phase = payload["name"] if kind == "begin" else "between passes"
                elif kind == "result":
                    return {
                        **payload,
                        "traces": list(traces.values()),
                        "runtime_seconds": time.monotonic() - started,
                        "worker_startup_seconds": startup_seconds,
                    }
                elif kind == "error":
                    raise RuntimeError(payload)  # ruff: ignore[raise-within-try] -- use the common worker cleanup path.
        except (TimeoutError, EOFError, BrokenPipeError, OSError, RuntimeError) as error:
            status = "timeout" if isinstance(error, TimeoutError) else "error"
            for trace in traces.values():
                if trace["status"] == "running":
                    trace["status"] = status
                    trace["error"] = f"{phase}: {error}"
                    trace["runtime_seconds"] = time.monotonic() - trace.pop("started", started)
            self.close()
            return {
                "status": status,
                "error": f"{phase}: {type(error).__name__}: {error}",
                "offending_pass": phase,
                "traces": list(traces.values()),
                "runtime_seconds": time.monotonic() - started,
                "worker_startup_seconds": startup_seconds,
            }


class RuntimeCompiler(Compiler):
    """Supply the server port omitted by BQSKit 1.2.1's attached launcher."""

    def __init__(self, num_workers: int, *, port: int, worker_port: int, num_blas_threads: int) -> None:
        """Use the SDK compiler and shutdown behavior with an isolated runtime."""
        self.server_port = port
        super().__init__(port=port, num_workers=num_workers, worker_port=worker_port, num_blas_threads=num_blas_threads)

    def _start_server(self, num_workers: int, runtime_log_level: int, worker_port: int, num_blas_threads: int) -> None:
        launch = (
            "from bqskit.runtime.attached import start_attached_server; "
            f"start_attached_server({num_workers}, port={self.server_port}, worker_port={worker_port}, "
            f"log_level={runtime_log_level}, num_blas_threads={num_blas_threads})"
        )
        self.p = subprocess.Popen([sys.executable, "-c", launch])


class PassObserver:
    """Record actual invocations, excluding nested composite containers."""

    def __init__(self, connection: Connection, target: Target) -> None:
        """Send pass boundaries to the independent watchdog."""
        self.connection = connection
        self.target = target
        self.index = 0
        self.stack: list[dict[str, Any]] = []
        self.measuring = False
        self.physical = False

    def score(self, circuit: QuantumCircuit) -> dict[str, Any]:
        """Score outside compilation timers and suppress instrumentation recursion."""
        self.connection.send(("phase", "ESP observation"))
        self.measuring = True
        try:
            return observe(circuit, self.target, physical=self.physical)
        finally:
            self.measuring = False

    def begin(self, name: str, category: str, circuit: QuantumCircuit) -> dict[str, Any]:
        """Start one invocation after observing its input circuit."""
        if self.stack:
            self.stack[-1]["container"] = True
            self.connection.send(("update", self.stack[-1]))
        entry = {
            "index": self.index,
            "name": name,
            "category": category,
            "before": self.score(circuit),
            "status": "running",
            "container": False,
        }
        self.index += 1
        self.stack.append(entry)
        entry["started"] = time.monotonic()
        self.connection.send(("begin", entry))
        return entry

    def end(self, entry: dict[str, Any], circuit: QuantumCircuit) -> None:
        """Stop the pass timer and record its output state."""
        entry["runtime_seconds"] = time.monotonic() - entry.pop("started")
        entry["status"] = "ok"
        # Stop the compilation timer before scoring, which runs under its own deadline.
        self.connection.send(("phase", f"score after {entry['name']}"))
        entry["after"] = self.score(circuit)
        self.stack.pop()
        self.connection.send(("end", entry))
        if self.stack:
            self.connection.send(("phase", self.stack[-1]["name"]))


@contextlib.contextmanager
def observe_qiskit_passes(observer: PassObserver) -> Iterator[None]:
    """Observe native Qiskit passes, including those inside TKET's LightSABRE map."""
    structural = {
        "BasisTranslator": "synthesis",
        "HighLevelSynthesis": "synthesis",
        "UnitarySynthesis": "synthesis",
        "ApplyLayout": "layout",
        "SabreLayout": "mapping",
        "SabreSwap": "routing",
    }
    administrative = {
        "FullAncillaAllocation",
        "EnlargeWithAncilla",
        "SetLayout",
        "BarrierBeforeFinalMeasurements",
        "FilterOpNodes",
        "MinimumPoint",
    }
    execute = BasePass.execute

    def observed_execute(
        self: BasePass,
        passmanager_ir: DAGCircuit,
        state: PassManagerState,
        callback: Callable | None = None,
    ) -> tuple[DAGCircuit, PassManagerState]:
        run = self.run

        def observed_run(dag: DAGCircuit) -> DAGCircuit | None:
            if observer.measuring:
                return run(dag)
            name = self.name()
            category = structural.get(name, "analysis" if self.is_analysis_pass else "optimization")
            layout_selection_only = name == "SabreLayout" and getattr(self, "skip_routing", False)
            if layout_selection_only:
                category = "analysis"
            if name in administrative:
                category = "administrative"
            entry = observer.begin(name, category, dag_to_circuit(dag))
            result = run(dag)
            if name == "ApplyLayout" or (
                name == "SabreLayout" and not layout_selection_only and self.property_set.get("layout") is not None
            ):
                observer.physical = True
            observer.end(entry, dag_to_circuit(result if result is not None else dag))
            return result

        self.__dict__["run"] = observed_run
        try:
            # Keep Qiskit's dependency execution, skip policy and status bookkeeping.
            return execute(self, passmanager_ir, state, callback)
        finally:
            self.__dict__["run"] = run

    BasePass.execute = observed_execute
    try:
        yield
    finally:
        BasePass.execute = execute


def tket_physical_circuit(unit: CompilationUnit, original: QuantumCircuit) -> QuantumCircuit:
    """Keep node indices and output permutations when converting a TKET result."""
    permutation = {destination: source for source, destination in unit.circuit.implicit_qubit_permutation().items()}
    # Keep implicit swaps in metadata; materializing them can introduce non-native SWAP gates.
    result = _tket_physical_wires(unit.circuit)
    logical = {
        Qubit(register.name, index): qubit for register in original.qregs for index, qubit in enumerate(register)
    }
    initial = {logical[q]: p.index[0] for q, p in unit.initial_map.items() if isinstance(q, Qubit) and q in logical}
    final = {
        result.qubits[unit.initial_map[q].index[0]]: permutation[p].index[0]
        for q, p in unit.final_map.items()
        if q in logical and isinstance(p, Qubit)
    }
    result._layout = TranspileLayout(  # ruff: ignore[private-member-access]
        initial_layout=Layout(initial),
        input_qubit_mapping={q: i for i, q in enumerate(original.qubits)},
        final_layout=Layout(final),
        _input_qubit_count=original.num_qubits,
        _output_qubit_list=result.qubits,
    )
    return result


def _tket_physical_wires(circuit: Circuit) -> QuantumCircuit:
    physical = circuit.copy()
    physical.rename_units({qubit: Qubit("q", qubit.index[0]) for qubit in physical.qubits})
    return tk_to_qiskit(physical, replace_implicit_swaps=False, perm_warning=False)


def _tket_pipeline(circuit: QuantumCircuit, backend: FakeBoston, observer: PassObserver) -> QuantumCircuit:
    pending: list[dict[str, Any]] = []
    final_unit = None

    def trace_circuit(unit: CompilationUnit) -> QuantumCircuit:
        observer.measuring = True
        try:
            return (
                _tket_physical_wires(unit.circuit)
                if observer.physical
                else tk_to_qiskit(unit.circuit, perm_warning=False)
            )
        finally:
            observer.measuring = False

    def before(unit: CompilationUnit, config: dict[str, Any]) -> None:
        if config["pass_class"] != "StandardPass":
            return
        name = config["StandardPass"]["name"]
        label = config["StandardPass"].get("label", "")
        category = "mapping" if label == "lightsabrepass" else "optimization"
        if name in {"AutoRebase", "RebaseCustom", "RebaseTket", "DecomposeBoxes"}:
            category = "synthesis"
        qc = trace_circuit(unit)
        pending.append(observer.begin(f"{name}:{label}" if label else name, category, qc))

    def after(unit: CompilationUnit, config: dict[str, Any]) -> None:
        nonlocal final_unit
        final_unit = unit
        if config["pass_class"] != "StandardPass":
            return
        if config["StandardPass"].get("label") == "lightsabrepass":
            observer.physical = True
        qc = trace_circuit(unit)
        observer.end(pending.pop(), qc)

    pipeline = IBMQBackend.default_compilation_pass_offline(
        backend.configuration(), backend.properties(), optimisation_level=2
    )
    apply = TketBasePass.apply
    tk_circuit = qiskit_to_tk(circuit)

    def observed_apply(self: TketBasePass, tk_circuit: Circuit, *args: object, **kwargs: object) -> bool:
        if args or kwargs or observer.measuring:
            return apply(self, tk_circuit, *args, **kwargs)
        return apply(self, tk_circuit, before, after)

    cast("Any", TketBasePass).apply = observed_apply
    try:
        pipeline.apply(tk_circuit, before, after)
    finally:
        TketBasePass.apply = apply
    assert final_unit is not None
    return tket_physical_circuit(final_unit, circuit)


def _worker_main(connection: Connection, assets: Path, settings: dict[str, Any]) -> None:
    os.setsid()
    log_path = Path(settings["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
    cast("Any", bqskit_actions).Compiler = partial(
        RuntimeCompiler,
        port=settings["bqskit_port"],
        worker_port=settings["bqskit_worker_port"],
        num_blas_threads=settings["threads"],
    )
    bqskit_actions._BQSKIT_NUM_WORKERS = settings["bqskit_workers"]  # ruff: ignore[private-member-access]
    backend, target = load_target(assets)
    env = PredictorEnv(
        target, reward_function="estimated_success_probability", pass_timeout=settings["pass_timeout_seconds"]
    )
    connection.send(("ready", None))
    while True:
        request = connection.recv()
        observer = PassObserver(connection, target)
        try:  # ruff: ignore[too-many-statements-in-try-clause] -- SDK exceptions are returned to the watchdog.
            mode = request["compiler"]
            circuit = request["circuit"]
            if mode in {"original", "paper"}:
                env.reset(circuit, seed=request["seed"])
                env.layout = request["layout"]
                env.num_qubits_uncompiled_circuit = request["input_qubits"]
                observer.physical = env.layout is not None
                action = env.action_set[request["action"]]
                entry = observer.begin(action.name, action.pass_type.value, circuit)
                result = env.apply_action(request["action"])
                result._layout = env.layout  # ruff: ignore[private-member-access]
                observer.physical = env.layout is not None
                observer.end(entry, result)
            else:
                with observe_qiskit_passes(observer):
                    if mode == "qiskit":
                        result = generate_preset_pass_manager(
                            optimization_level=3, target=target, seed_transpiler=request["seed"]
                        ).run(circuit)
                    else:
                        result = _tket_pipeline(circuit, backend, observer)
            observer.physical = result.layout is not None
            connection.send((
                "result",
                {
                    "status": "ok",
                    "circuit": result,
                    "score": observer.score(result),
                    "final_layout": result.layout.final_index_layout() if result.layout is not None else None,
                },
            ))
        except Exception as error:  # ruff: ignore[blind-except]
            connection.send(("error", f"{type(error).__name__}: {error}"))
