# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Collect, learn, and time bounded edits of native compiler pipelines."""

from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import os
import random
import re
import subprocess
import time
import tomllib
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import joblib
import psutil  # ty: ignore[unresolved-import] -- supplied by the existing gnn experiment extra.

from .inputs import Inputs, digest, load_target
from .pipeline_learning import paired_rows, train_models
from .pipeline_study_worker import StudyWorker

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from pytket.passes import SequencePass

ROOT = Path(__file__).resolve().parents[5]
COMPILERS = ("qiskit", "tket", "bqskit")


def balanced(names: Iterable[str]) -> list[str]:
    """Visit every circuit family before advancing to larger instances."""
    groups: dict[str, list[str]] = defaultdict(list)
    for name in names:
        groups[re.sub(r"_\d+_indep\.qasm$", "", Path(name).name)].append(name)
    ordered = [sorted(group, key=lambda n: (int(n.rsplit("_", 2)[1]), n)) for _, group in sorted(groups.items())]
    return [name for row in itertools.zip_longest(*ordered) for name in row if name is not None]


def candidate_plans(inputs: Inputs, limit: int, max_edits: int, seed: int) -> list[dict[str, Any]]:
    """Bound a deterministic native-only search, including repeats and reordering."""
    from bqskit import MachineModel  # ruff: ignore[import-outside-top-level]
    from bqskit.compiler.compile import build_workflow  # ruff: ignore[import-outside-top-level]
    from bqskit.ext import qiskit_to_bqskit  # ruff: ignore[import-outside-top-level]
    from pytket.extensions.qiskit import IBMQBackend  # ruff: ignore[import-outside-top-level]

    from mqt.predictor.rl.actions.bqskit_actions import get_bqskit_native_gates  # ruff: ignore[import-outside-top-level]

    from .pipeline_bqskit import OPTIONAL_STAGES as BQ_OPTIONAL  # ruff: ignore[import-outside-top-level]
    from .pipeline_tket import OPTIONAL_STAGES as TK_OPTIONAL  # ruff: ignore[import-outside-top-level]

    backend, target = load_target(inputs.path)
    result: list[dict[str, Any]] = []
    for compiler, levels in (("qiskit", (3, 2, 1)), ("tket", (3, 2)), ("bqskit", (3, 2, 1))):
        for level in levels:
            prefix = f"{compiler}{level}"
            base = {"compiler": compiler, "level": level, "optimization": "native", "edits": {}}
            result.append({**base, "id": f"{prefix}-native", "native": True, "priority": 0})
            if level == 1 or (compiler == "qiskit" and level != 3):
                continue
            options: list[dict[str, Any]] = [{**base, "id": f"{prefix}-keep", "native": False, "priority": 1}]
            if compiler == "qiskit":
                options.extend(
                    {**base, "id": f"{prefix}-{mode}", "native": False, "optimization": mode, "priority": 2}
                    for mode in ("post_layout", "one_sweep", "peephole_once", "skip_optimization")
                )
                options.extend(
                    {
                        **base,
                        "id": f"{prefix}-repeat-{count}",
                        "native": False,
                        "edits": {"4": ["INSERT:4"] * count},
                        "priority": 3,
                    }
                    for count in range(1, max_edits + 1)
                )
                result.extend(options[:limit])
                continue
            if compiler == "tket":
                stages = cast(
                    "SequencePass",
                    IBMQBackend.default_compilation_pass_offline(
                        backend.configuration(), backend.properties(), optimisation_level=level
                    ),
                ).get_sequence()
                optional = [
                    i for i, stage in enumerate(stages) if stage.to_dict()["StandardPass"]["name"] in TK_OPTIONAL
                ]
                swaps = [(left, right) for left, right in itertools.pairwise(optional) if right == left + 1]
            else:
                model = MachineModel(
                    target.num_qubits,
                    coupling_graph=target.build_coupling_map().get_edges(),
                    gate_set=get_bqskit_native_gates(target),
                )
                stages = build_workflow(
                    qiskit_to_bqskit(inputs.circuit("train/ghz_2_indep.qasm")),
                    model,
                    optimization_level=level,
                    seed=seed,
                )
                optional = [i for i, stage in enumerate(stages) if stage.name in BQ_OPTIONAL]
                swaps = []
            edits: list[tuple[int, dict[str, list[str]]]] = []
            for index in optional:
                edits.extend([(2, {str(index): ["SKIP"]}), (2, {str(index): [f"INSERT:{index}"]})])
            if len(optional) <= max_edits:
                edits.append((2, {str(i): ["SKIP"] for i in optional}))
            if max_edits >= 2:
                edits.extend((3, {str(left): [f"INSERT:{right}"], str(right): ["SKIP"]}) for left, right in swaps)
                edits.extend(
                    (3, {str(i): [f"INSERT:{i}"] * count}) for count in range(2, max_edits + 1) for i in optional
                )
            rng = random.Random(seed + level)
            for _ in range(256):
                proposal = {}
                budget = max_edits
                for index in rng.sample(optional, len(optional)):
                    action = rng.choice(("KEEP", "SKIP", "REPEAT"))
                    if action != "KEEP" and budget:
                        count = 1 if action == "SKIP" else rng.randint(1, budget)
                        proposal[str(index)] = ["SKIP" if action == "SKIP" else f"INSERT:{index}"] * count
                        budget -= count
                edits.append((4, proposal))
            seen = {"{}"}
            for priority, edit in edits:
                if sum(len(actions) for actions in edit.values()) > max_edits:
                    continue
                identity = json.dumps(edit, sort_keys=True)
                if identity in seen:
                    continue
                seen.add(identity)
                options.append({
                    **base,
                    "id": f"{prefix}-edit-{digest(identity.encode())[:10]}",
                    "native": False,
                    "edits": edit,
                    "priority": priority,
                })
            required = [option for option in options if option["priority"] <= 2]
            exploration = [
                option
                for pair in itertools.zip_longest(
                    (option for option in options if option["priority"] == 3),
                    (option for option in options if option["priority"] == 4),
                )
                for option in pair
                if option is not None
            ]
            result.extend((required + exploration)[:limit])
    return sorted(result, key=lambda plan: (plan["priority"], plan["compiler"], -plan["level"], plan["id"]))


def read_rows(path: Path, *, repair_tail: bool = False) -> list[dict[str, Any]]:
    """Read flushed records; preserve and discard only an interrupted final write."""
    if not path.exists():
        return []
    rows = []
    with path.open("rb") as stream:
        offset = 0
        while line := stream.readline():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if not repair_tail or stream.read(1):
                    raise
                path.with_suffix(f".partial-{time.time_ns()}").write_bytes(line)
                with path.open("r+b") as repair:
                    repair.truncate(offset)
                break
            offset = stream.tell()
            if repair_tail and not line.endswith(b"\n"):
                with path.open("ab") as repair:
                    repair.write(b"\n")
                break
    return rows


def prepare(config_path: Path, output: Path | None, *, resume: bool, smoke: bool) -> tuple[dict[str, Any], Inputs]:
    """Freeze inputs, candidates, settings and source identity before compilation."""
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    config["assets"] = str((config_path.parent / config["assets"]).resolve())
    config["output"] = str(output.resolve() if output else (config_path.parent / config["output"]).resolve())
    if smoke:
        config["study"].update(
            workers=1,
            plans_per_preset=7,
            train_circuits=8,
            test_circuits=2,
            timing_circuits=2,
            compiler_seeds=[0],
            model_seeds=[0],
            timing_repetitions=1,
            collect_hours=0.15,
            evaluate_hours=0.1,
            timing_hours=0.1,
        )
        config["worker"]["compilation_timeout_seconds"] = 20
    study = config["study"]
    if study["workers"] < 1 or study["max_edits"] < 1 or study["plans_per_preset"] < 1:
        msg = "Workers, edit budget and plan limit must be positive."
        raise ValueError(msg)
    if config["worker"]["threads"] != 1:
        msg = "Use one native thread per study worker."
        raise ValueError(msg)
    if 0 not in study["compiler_seeds"] or not study["model_seeds"]:
        msg = "Compiler seeds must include 0, and model_seeds must not be empty."
        raise ValueError(msg)
    if any(study[key] <= 0 for key in ("collect_hours", "evaluate_hours", "timing_hours", "timing_repetitions")):
        msg = "Phase budgets and timing repetitions must be positive."
        raise ValueError(msg)
    inputs = Inputs(Path(config["assets"]))
    plans = candidate_plans(inputs, study["plans_per_preset"], study["max_edits"], study["plan_seed"])
    splits = {
        split: balanced(name for name in inputs.names(split) if not smoke or int(name.rsplit("_", 2)[1]) <= 3)
        for split in ("train", "test")
    }
    evaluation = splits["test"][: study["test_circuits"] or None]
    timing = random.Random(study["plan_seed"]).sample(
        evaluation, min(study["timing_circuits"] or len(evaluation), len(evaluation))
    )
    manifest = {
        "config": config,
        "smoke": smoke,
        "plans": plans,
        "training": splits["train"][: study["train_circuits"] or None],
        "evaluation": evaluation,
        "timing": timing,
        "inputs": inputs.manifest,
        "sources": {
            str(path.relative_to(ROOT)): digest(path.read_bytes())
            for path in sorted((ROOT / "src").rglob("*.py"))
            if path != ROOT / "src/mqt/predictor/_version.py"
        },
        "lock_sha256": digest((ROOT / "uv.lock").read_bytes()),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("qiskit", "pytket", "pytket-qiskit", "bqskit", "numpy", "scikit-learn", "psutil")
        },
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "protocol": {
            "learning": "Predict signed ESP change and full compilation time relative to native level3. Choose one plan; no best-of-N compilation. Family/identical-content grouped CV uses training only.",
            "historical_test": "The original 41 circuits have been inspected in earlier research; this is confirmation, not a pristine final holdout.",
            "timing": "Parallel collect/evaluate times are exploratory. The timing phase uses one active compiler worker, rotated variants, one excluded warmup and measured repeats. No per-pass circuit conversion observer. Feature/inference time is included for learned policies; input parsing, model loading and final scoring are excluded.",
            "failures": "Timeout/memory/error rows stay in coverage counts. Never impute zero ESP or include censored time as a learning target.",
            "seeds": "Qiskit/BQSKit use compiler seeds; native TKET keeps its fixed LightSABRE seed0 and50attempts. Timing repetitions use compiler seed0.",
            "limits": "A finite candidate study, not exhaustive pipeline search. Sampled worker-tree RSS limits may briefly overshoot. Phase time budgets stop new work and cap pending requests.",
        },
    }
    directory = Path(config["output"])
    path = directory / "manifest.json"
    if directory.exists() and not resume:
        msg = f"Use a new output directory or --resume: {directory}"
        raise FileExistsError(msg)
    if resume:
        previous = json.loads(path.read_text())
        if {k: v for k, v in previous.items() if k != "commit"} != {k: v for k, v in manifest.items() if k != "commit"}:
            msg = "Run configuration, sources, dependencies or dataset changed; use a new output directory."
            raise ValueError(msg)
        manifest = previous
    else:
        directory.mkdir(parents=True)
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    with (directory / "restarts.jsonl").open("a") as stream:
        stream.write(json.dumps({"time_unix": time.time(), "resume": resume}) + "\n")
    return manifest, inputs


def collection_jobs(manifest: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Collect native references and all edit tiers in small circuit batches."""
    seeds = sorted(manifest["config"]["study"]["compiler_seeds"], key=lambda seed: seed != 0)
    for seed in seeds:
        for start in range(0, len(manifest["training"]), 8):
            names = manifest["training"][start : start + 8]
            for priority in sorted({plan["priority"] for plan in manifest["plans"]}):
                plans = [plan for plan in manifest["plans"] if plan["priority"] == priority]
                for name in names:
                    for plan in plans:
                        if plan["compiler"] != "tket" or seed == 0:
                            yield {
                                "circuit": name,
                                "seed": seed,
                                "variant": plan["id"],
                                "compiler": plan["compiler"],
                                "plan": plan,
                            }


def run_jobs(manifest: dict[str, Any], phase: str, jobs: Iterable[dict[str, Any]], workers: int, hours: float) -> None:
    """Keep one isolated compiler per slot and flush each completed result."""
    config = manifest["config"]
    output = Path(config["output"])
    path = output / f"{phase}.jsonl"
    completed = {
        row["job_id"]
        for row in read_rows(path, repair_tail=True)
        if row["status"] not in {"cancelled", "budget_exhausted"}
    }
    settings = {**config["worker"], "output": str(output / phase), "max_edits": config["study"]["max_edits"]}
    settings["memory_limit_bytes"] = int(settings.pop("memory_gib") * 1024**3)
    if workers * settings["memory_limit_bytes"] > psutil.virtual_memory().total * 0.8:
        msg = "Worker memory limits exceed 80% of system RAM; reduce workers or memory_gib."
        raise ValueError(msg)
    Path(settings["output"]).mkdir(exist_ok=True)
    clients = [StudyWorker(Path(config["assets"]), settings.copy(), slot) for slot in range(workers)]
    deadline = time.monotonic() + hours * 3600
    iterator = iter(jobs)
    idle = deque(range(workers))
    pending = {}
    exhausted = False
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        with path.open("a") as stream:
            while pending or (not exhausted and time.monotonic() < deadline):
                while idle and not exhausted and time.monotonic() < deadline:
                    request = next(iterator, None)
                    if request is None:
                        exhausted = True
                        break
                    job_id = digest(json.dumps(request, sort_keys=True).encode())
                    if job_id in completed:
                        continue
                    slot = idle.popleft()
                    clients[slot].settings["compilation_timeout_seconds"] = min(
                        settings["compilation_timeout_seconds"], max(0.1, deadline - time.monotonic())
                    )
                    pending[executor.submit(clients[slot].run, request)] = (slot, request, job_id)
                if not pending:
                    break
                ready, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in ready:
                    slot, request, job_id = pending.pop(future)
                    result = future.result()
                    if (
                        result["status"] == "timeout"
                        and clients[slot].settings["compilation_timeout_seconds"]
                        < settings["compilation_timeout_seconds"]
                    ):
                        result["status"] = "budget_exhausted"
                    trace = result.get("stage_trace", [])
                    if phase == "collect":
                        result["stage_trace"] = [
                            {
                                key: entry[key]
                                for key in (
                                    "stage_index",
                                    "stage_name",
                                    "action",
                                    "occurrence",
                                    "runtime_seconds",
                                    "inserted_stage_index",
                                )
                                if key in entry
                            }
                            for entry in trace
                        ]
                    row = {**request, **result, "job_id": job_id, "phase": phase}
                    row.setdefault("plan_id", request.get("plan", {}).get("id", request["variant"]))
                    row.pop("plan", None)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    completed.add(job_id)
                    idle.append(slot)
                    print(
                        f"{phase} {len(completed)}: {request['compiler']} {request['circuit']} {request['variant']} {result['status']}",
                        flush=True,
                    )
    finally:
        for client in clients:
            client.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        for client in clients:
            client.close()


def static_choices(rows: list[dict[str, Any]], plans: list[dict[str, Any]], weights: list[float]) -> list[str]:
    """Select fixed-plan competitors on training data, requiring broad coverage."""
    selected = set()
    for compiler in COMPILERS:
        candidates = [p for p in plans if p["compiler"] == compiler]
        paired, _ = paired_rows(rows, candidates, f"{compiler}3-native")
        baseline_count = len({r["circuit"] for r in paired})
        by_plan: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in paired:
            by_plan[row["plan_id"]].append(row)
        eligible = {
            name: measurements
            for name, measurements in by_plan.items()
            if len({r["circuit"] for r in measurements}) >= max(4, baseline_count * 0.8)
        }
        for weight in weights:
            if eligible:
                selected.add(
                    min(
                        eligible,
                        key=lambda name: (
                            sum(r["time_ratio"] - 1 - weight * r["delta_esp"] for r in eligible[name])
                            / len(eligible[name]),
                            name,
                        ),
                    )
                )
    return sorted(selected)


def fit_models(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Freeze all models before reading historical benchmark outcomes."""
    output = Path(manifest["config"]["output"])
    directory = output / "models"
    index_path = directory / "index.json"
    training_hash = digest((output / "collect.jsonl").read_bytes())
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if index["training_sha256"] != training_hash:
            msg = "Collected data changed after models were frozen; use a new output directory."
            raise ValueError(msg)
        for entry in index["policies"]:
            if digest((output / entry["file"]).read_bytes()) != entry["sha256"]:
                msg = "Frozen model identity changed."
                raise ValueError(msg)
        return index["policies"]
    rows = read_rows(output / "collect.jsonl")
    directory.mkdir(exist_ok=True)
    study = manifest["config"]["study"]
    policies = []
    for compiler in COMPILERS:
        try:
            bundles, report = train_models(
                rows,
                manifest["plans"],
                compiler,
                weights=study["tradeoff_weights"],
                model_seeds=study["model_seeds"],
                circuit_hashes=manifest["inputs"]["circuits"],
            )
        except ValueError as error:
            (directory / f"{compiler}-training.json").write_text(
                json.dumps({"unavailable": str(error)}, indent=2) + "\n"
            )
            continue
        (directory / f"{compiler}-training.json").write_text(json.dumps(report, indent=2) + "\n")
        for bundle in bundles:
            name = f"{compiler}-w{bundle['weight']}-seed{bundle['model_seed']}"
            path = directory / f"{name}.joblib"
            joblib.dump(bundle, path)
            policies.append({
                "id": name,
                "compiler": compiler,
                "weight": bundle["weight"],
                "model_seed": bundle["model_seed"],
                "file": str(path.relative_to(output)),
                "sha256": digest(path.read_bytes()),
            })
    pending_index = index_path.with_suffix(".tmp")
    pending_index.write_text(
        json.dumps(
            {
                "training_sha256": training_hash,
                "policies": policies,
                "static_plan_ids": static_choices(rows, manifest["plans"], study["tradeoff_weights"]),
            },
            indent=2,
        )
        + "\n"
    )
    pending_index.replace(index_path)
    return policies


def evaluation_jobs(
    manifest: dict[str, Any], policies: list[dict[str, Any]], *, timing: bool
) -> Iterator[dict[str, Any]]:
    """Keep static controls and independently deploy every frozen preference."""
    config = manifest["config"]
    index_path = Path(config["output"]) / "models/index.json"
    fixed = json.loads(index_path.read_text())["static_plan_ids"] if index_path.exists() else []
    variants = [
        {"variant": p["id"], "compiler": p["compiler"], "plan": p}
        for p in manifest["plans"]
        if p["native"] or p["id"].endswith("-keep") or p["compiler"] == "qiskit" or p["id"] in fixed
    ]
    variants.extend(
        {
            "variant": p["id"],
            "compiler": p["compiler"],
            "policy_file": str(Path(config["output"]) / p["file"]),
            "policy_sha256": p["sha256"],
            "weight": p["weight"],
        }
        for p in policies
        if not timing or p["model_seed"] == config["study"]["model_seeds"][0]
    )
    names = manifest["timing"] if timing else manifest["evaluation"]
    for index, name in enumerate(names):
        seeds = [0] if timing else config["study"]["compiler_seeds"]
        for seed in seeds:
            repeats = range(-1, config["study"]["timing_repetitions"]) if timing else [0]
            for repeat in repeats:
                order = variants.copy()
                random.Random(index * 100 + seed * 10 + repeat).shuffle(order)
                for variant in order:
                    if variant["compiler"] == "tket" and seed != 0:
                        continue
                    yield {**variant, "circuit": name, "seed": seed, "repeat": repeat, "warmup": repeat < 0}


def write_report(manifest: dict[str, Any]) -> dict[str, Any]:
    """Report paired quality and serial latency, with exclusions and coverage."""
    output = Path(manifest["config"]["output"])
    report: dict[str, Any] = {"protocol": manifest["protocol"], "phases": {}}
    for phase in ("collect", "evaluate", "timing"):
        rows = [r for r in read_rows(output / f"{phase}.jsonl") if not r.get("warmup", False)]
        bases = {
            (r["compiler"], r["circuit"], r["seed"], r.get("repeat", 0)): r
            for r in rows
            if r["status"] == "ok" and r["variant"] == f"{r['compiler']}3-native"
        }
        summary = {}
        for variant in sorted({r["variant"] for r in rows}):
            selected = [r for r in rows if r["variant"] == variant]
            pairs = [
                (r, bases[key])
                for r in selected
                if r["status"] == "ok"
                and (key := (r["compiler"], r["circuit"], r["seed"], r.get("repeat", 0))) in bases
            ]
            deltas = [100 * (r["score"]["esp"] - b["score"]["esp"]) for r, b in pairs]
            summary[variant] = {
                "compiler": selected[0]["compiler"],
                "statuses": dict(Counter(r["status"] for r in selected)),
                "matched_compilations": len(pairs),
                "matched_circuits": len({r["circuit"] for r, _ in pairs}),
                "mean_delta_esp_pp": sum(deltas) / len(deltas) if deltas else None,
                "worst_delta_esp_pp": min(deltas) if deltas else None,
                "runtime_ratio": sum(r["seconds"] for r, _ in pairs) / sum(b["seconds"] for _, b in pairs)
                if pairs
                else None,
                "selected_plans": dict(Counter(r.get("plan_id", variant) for r in selected if r["status"] == "ok")),
            }
        report["phases"][phase] = {
            "records": len(rows),
            "variants": summary,
            "latency_claims_supported": phase == "timing",
            "coverage": "Paired successful compilations only. Different coverage is not a common-set Pareto comparison; timeouts/errors are not zero quality.",
        }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Run the finite study, or resume only its missing compilations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/pipeline-weekend.toml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=("all", "collect", "train", "evaluate", "timing", "report"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[name] = "1"
    manifest, _ = prepare(args.config.resolve(), args.output, resume=args.resume, smoke=args.smoke)
    config = manifest["config"]
    study = config["study"]
    output = Path(config["output"])
    try:
        if args.stage in {"all", "collect"} and not (output / "models/index.json").exists():
            run_jobs(manifest, "collect", collection_jobs(manifest), study["workers"], study["collect_hours"])
        policies = fit_models(manifest) if args.stage in {"all", "train", "evaluate", "timing"} else []
        if args.stage in {"all", "evaluate"}:
            run_jobs(
                manifest,
                "evaluate",
                evaluation_jobs(manifest, policies, timing=False),
                study["workers"],
                study["evaluate_hours"],
            )
        if args.stage in {"all", "timing"}:
            run_jobs(manifest, "timing", evaluation_jobs(manifest, policies, timing=True), 1, study["timing_hours"])
    finally:
        write_report(manifest)


if __name__ == "__main__":
    main()
