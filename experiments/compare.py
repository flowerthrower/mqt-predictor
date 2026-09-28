# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.10,<4"]
# ///

"""Compare available SCASIA evaluations without loading models or compilers."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

COMPILERS = ("qiskit", "tket", "original", "paper")
COLORS = ("#527b9f", "#b4773c", "#79836a", "#795aaa")


def compare(results: Path) -> dict[str, Any]:
    """Match valid repetitions across available rows and retain every failure count."""
    manifests = {}
    runs = {}
    warnings = []
    for name in COMPILERS:
        manifest_path = results / name / "manifest.json"
        evaluation_path = results / name / "evaluation.jsonl"
        if not manifest_path.is_file() or not evaluation_path.is_file():
            warnings.append(f"{name}: excluded; manifest or evaluation file is missing.")
            continue
        records = {}
        with evaluation_path.open() as stream:
            for line in stream:
                if not line.endswith("\n"):
                    warnings.append(f"{name}: ignored unfinished final evaluation record.")
                    break
                row = json.loads(line)
                key = (row["circuit"], row["repetition"])
                if row["compiler"] != name or key in records:
                    msg = f"{name}: wrong compiler or duplicate repetition {key}."
                    raise ValueError(msg)
                records[key] = row
        if not records:
            warnings.append(f"{name}: excluded; no completed evaluation records yet.")
            continue
        runs[name] = records
        manifests[name] = json.loads(manifest_path.read_text())
    if not runs:
        return {"summary": [], "paired": [], "runtimes": {}, "warnings": warnings, "matched": 0}
    reference_name = next(iter(manifests))
    reference = manifests[reference_name]["identity"]
    for name, manifest in manifests.items():
        identity = manifest["identity"]
        if identity["compiler"] != name:
            msg = f"{name}: incorrect compiler in manifest."
            raise ValueError(msg)
        for field in ("inputs", "target", "lock_sha256", "settings"):
            if identity[field] != reference[field]:
                msg = f"{name}: incompatible {field}; use runs from the same experiment configuration."
                raise ValueError(msg)
        warnings.extend(
            f"{name}: {field} differs from {reference_name}; check provenance before interpreting differences."
            for field in ("source_sha256", "dependencies", "python")
            if identity[field] != reference[field]
        )
    if {"original", "paper"} <= manifests.keys() and (
        manifests["original"]["identity"]["actions"] != manifests["paper"]["identity"]["actions"]
    ):
        msg = "The two RL rows have different action registries."
        raise ValueError(msg)
    settings = reference["settings"]["experiment"]
    circuits = sorted(name for name in reference["inputs"]["circuits"] if name.startswith("test/"))
    if settings["evaluation_circuits"]:
        circuits = circuits[: settings["evaluation_circuits"]]
    expected = {(name, repetition) for name in circuits for repetition in range(settings["evaluation_repetitions"])}
    valid = {}
    for name, records in runs.items():
        for key in records:
            if key not in expected:
                msg = f"{name}: unexpected repetition {key}."
                raise ValueError(msg)
        valid[name] = {
            key
            for key, row in records.items()
            if row["status"] == "ok" and row["final_esp"] is not None and math.isfinite(row["final_esp"])
        }
    common = set.intersection(*valid.values())
    for key in set.intersection(*(set(records) for records in runs.values())):
        if len({records[key]["seed"] for records in runs.values()}) != 1:
            msg = f"Evaluation seeds differ for {key}."
            raise ValueError(msg)
    paired = [
        {
            "circuit": circuit,
            "matched_repetitions": len(keys),
            **{name: mean(records[key]["final_esp"] for key in keys) for name, records in runs.items()},
        }
        for circuit in circuits
        if (keys := sorted(key for key in common if key[0] == circuit))
    ]
    summary = []
    runtimes = {}
    for name, records in runs.items():
        counts = Counter(row["status"] for row in records.values())
        runtimes[name] = [row["runtime_seconds"] for row in records.values()]
        summary.append({
            "compiler": name,
            "completed": len(records),
            "expected": len(expected),
            "valid": len(valid[name]),
            "error": counts["error"],
            "timeout": counts["timeout"],
            "invalid": counts["invalid"],
            "unavailable": counts["ok"] - len(valid[name]),
            "missing": len(expected) - len(records),
            "paired_mean_esp": mean(row[name] for row in paired) if paired else None,
            "median_runtime_seconds": median(runtimes[name]) if runtimes[name] else None,
            "actual_training_timesteps": manifests[name]["actual_training_timesteps"],
            "commit": manifests[name]["identity"]["commit"],
        })
    return {"summary": summary, "paired": paired, "runtimes": runtimes, "warnings": warnings, "matched": len(common)}


def grouped_esp(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Average matched circuit means within each frozen benchmark algorithm."""
    compilers = [row["compiler"] for row in report["summary"]]
    algorithms = defaultdict(list)
    for row in report["paired"]:
        algorithm = Path(row["circuit"]).stem.rsplit("_", 2)[0]
        algorithms[algorithm].append(row)
    grouped = [
        {
            "algorithm": algorithm,
            **{name: mean(row[name] for row in rows) for name in compilers},
        }
        for algorithm, rows in algorithms.items()
    ]
    return sorted(
        grouped, key=lambda row: row["paper"] if "paper" in compilers else mean(row[name] for name in compilers)
    )


def write_report(report: dict[str, Any], output: Path) -> None:
    """Save an offline HTML view, SVG/PNG plots and the plotted CSV values."""
    # Keep aggregation usable without the plotting dependency supplied by uv.
    from matplotlib import pyplot as plt  # ty: ignore[unresolved-import]  # ruff: ignore[import-outside-top-level]

    plt.switch_backend("Agg")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    output.mkdir(parents=True, exist_ok=True)
    summary, paired = report["summary"], report["paired"]
    compilers = [row["compiler"] for row in summary]
    for filename, rows in (("summary.csv", summary), ("per_circuit.csv", paired)):
        with (output / filename).open("w", newline="") as stream:
            fields = list(rows[0]) if rows else ["circuit", "matched_repetitions", *compilers]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    figures = []
    grouped = grouped_esp(report)
    if grouped:
        with plt.rc_context({"font.family": "serif"}):
            fig, axis = plt.subplots(figsize=(15, 7), layout="constrained")
            styles = {
                "qiskit": ("Qiskit", "red", "o"),
                "tket": ("TKET", "green", "s"),
                "original": ("MQT Predictor", "#703b9c", "D"),
                "paper": ("Proposed Method", "blue", "^"),
            }
            for index, row in enumerate(grouped):
                axis.axvspan(index - 0.5, index + 0.5, color="0.90" if index % 2 else "0.95", zorder=0)
                values = [row[name] for name in compilers]
                axis.vlines(index, min(values), max(values), color="0.75", linewidth=1, zorder=1)
            for name in compilers:
                label, color, marker = styles[name]
                axis.scatter(
                    range(len(grouped)),
                    [row[name] for row in grouped],
                    label=label,
                    color=color,
                    marker=marker,
                    s=65,
                    zorder=2,
                )
            axis.set_xticks(range(len(grouped)), [row["algorithm"] for row in grouped], rotation=45, ha="right")
            axis.tick_params(axis="x", labelsize=9)
            axis.set(
                xlabel="Benchmarks (MQT Bench)",
                ylabel="Estimated Success Probability",
                xlim=(-0.6, len(grouped) - 0.4),
                ylim=(-0.05, 1.05),
            )
            axis.grid(axis="y", linestyle="--", color="0.75", linewidth=0.6)
            axis.legend(loc="upper left")
            means = "Means (per circuit)\n" + "\n".join(
                f"{styles[row['compiler']][0]:<16}: {row['paired_mean_esp']:.3f}" for row in summary
            )
            axis.text(
                0.99,
                0.02,
                means,
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontfamily="monospace",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
            figures.append(("reward_comp_esp_grouped", fig))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    bottom = [0.0] * len(compilers)
    for status, color in (
        ("valid", "#527b9f"),
        ("error", "#c05c51"),
        ("timeout", "#d99f42"),
        ("invalid", "#9179a9"),
        ("unavailable", "#777777"),
        ("missing", "#dddddd"),
    ):
        heights = [100 * row[status] / row["expected"] if row["expected"] else 0 for row in summary]
        axes[0].bar(compilers, heights, bottom=bottom, label=status, color=color)
        bottom = [a + b for a, b in zip(bottom, heights, strict=True)]
    axes[0].set(title="Completion and failures", ylabel="% of expected repetitions", ylim=(0, 105))
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
    for index, name in enumerate(compilers, start=1):
        color = COLORS[COMPILERS.index(name)]
        if paired:
            values = [row[name] for row in paired]
            axes[1].boxplot(values, positions=[index], widths=0.45, showfliers=False)
            axes[1].scatter([index + (i % 7 - 3) * 0.025 for i in range(len(values))], values, s=13, color=color)
        times = [value for value in report["runtimes"][name] if value > 0]
        if times:
            axes[2].boxplot(times, positions=[index], widths=0.45)
    axes[1].set(title="ESP · matched valid repetitions", ylabel="Mean ESP per circuit", ylim=(0, 1.03))
    axes[2].set(title="Runtime · all completed attempts", ylabel="Wall time (seconds, log scale)", yscale="log")
    for axis in axes[1:]:
        axis.set_xticks(range(1, len(compilers) + 1), compilers)
        axis.set_xlim(0.5, len(compilers) + 0.5)
        axis.grid(axis="y", alpha=0.15)
    if not paired:
        axes[1].text(0.5, 0.5, "No common valid repetitions", ha="center", transform=axes[1].transAxes)
    figures.append(("overview", fig))
    if paired and {"original", "paper"} <= set(compilers):
        ordered = sorted(paired, key=lambda row: row["paper"] - row["original"])
        fig, axis = plt.subplots(figsize=(10, max(3, 0.25 * len(ordered) + 1.4)), layout="constrained")
        deltas = [row["paper"] - row["original"] for row in ordered]
        axis.barh(range(len(ordered)), deltas, color=[COLORS[3] if value >= 0 else COLORS[2] for value in deltas])
        axis.set_yticks(range(len(ordered)), [Path(row["circuit"]).stem for row in ordered])
        axis.axvline(0, color="#444444", linewidth=0.7)
        axis.set(title="Paper - original · same circuit and repetitions", xlabel="ESP difference (paper - original)")
        axis.grid(axis="x", alpha=0.15)
        figures.append(("paper_vs_original", fig))
    plots = []
    for name, fig in figures:
        fig.savefig(output / f"{name}.svg")
        fig.savefig(output / f"{name}.png", dpi=150)
        plots.append("<svg" + (output / f"{name}.svg").read_text().split("<svg", 1)[1])
        plt.close(fig)
    columns = (
        "compiler",
        "completed",
        "expected",
        "valid",
        "error",
        "timeout",
        "invalid",
        "missing",
        "paired_mean_esp",
        "median_runtime_seconds",
        "actual_training_timesteps",
    )
    header = "".join(f"<th>{key.replace('_', ' ')}</th>" for key in columns)
    rows = ""
    for row in summary:
        cells = [
            f"{value:.5g}" if isinstance(value, float) else str(value) if value is not None else "—"
            for key in columns
            for value in [row[key]]
        ]
        rows += "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in cells) + "</tr>"
    warnings = "".join(f"<p class='warning'>{html.escape(message)}</p>" for message in report["warnings"])
    commits = "; ".join(f"{row['compiler']}: {row['commit']}" for row in summary)
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>SCASIA comparison</title>
<style>body{{font:15px system-ui,sans-serif;color:#253344;background:#fafaf8;margin:32px auto;padding:0 24px;max-width:1400px}}
svg{{width:100%;height:auto;background:white;margin:16px 0}} table{{border-collapse:collapse;font-size:13px}}
th,td{{padding:10px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}
.scroll{{overflow:auto}}.warning{{background:#fff0d9;padding:12px}}.muted{{color:#606870;overflow-wrap:anywhere}}</style>
<h1>SCASIA comparison</h1>{warnings}
<p>Included rows: {", ".join(compilers)}.</p>
<p>Quality uses {report["matched"]} repetitions valid in all included rows, covering {len(paired)} circuits.
Each circuit contributes one mean, with the same repetitions for every compiler; no best-of-N selection.
Failures and missing runs are shown separately, never converted to zero ESP.</p>
<p>The paper-style ESP plot averages circuit means within each algorithm. Its overall means give every circuit equal
weight. Algorithms are sorted by the paper row, or by the mean across included rows when paper is unavailable.
This recreates the figure layout with rerun data, not the manuscript's historical values.</p>
<p>Runtime includes all completed attempts, including failures, worker startup and scoring.
Partial results are visible in the completed/expected counts. These plots are descriptive, without significance claims.</p>
<div class="scroll"><table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>
{"".join(plots)}<p class="muted">Run commits: {html.escape(commits)}</p>
<p class="muted">SVG/PNG plots and summary.csv / per_circuit.csv are saved beside this report.</p></html>"""
    (output / "comparison.html").write_text(page, encoding="utf-8")


def main() -> None:
    """Read available evaluations from the compiler output directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Directory containing qiskit/, tket/, original/ and paper/.")
    parser.add_argument("--output", type=Path, help="Report directory; defaults to RESULTS/comparison/.")
    args = parser.parse_args()
    output = args.output or args.results / "comparison"
    report = compare(args.results)
    for message in report["warnings"]:
        print(message)
    if not report["summary"]:
        print("No completed evaluation records yet; no report written.")
        return
    write_report(report, output)
    print(output / "comparison.html")


if __name__ == "__main__":
    main()
