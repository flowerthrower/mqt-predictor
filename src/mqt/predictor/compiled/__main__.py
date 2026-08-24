# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Train and export the experimental native linear predictor."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .policy import export_linear_policy
from .trainer import fit_linear_policy, load_training_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--core-revision", required=True)
    parser.add_argument("--source-revision")
    parser.add_argument("--objective", default="bootstrap action imitation")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _source_revision(override: str | None) -> str:
    if override:
        return override
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown-working-tree"
    revision = result.stdout.strip()
    if not revision:
        return "unknown-working-tree"
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    return f"{revision}+dirty" if status.stdout else revision


def main() -> None:
    """Train a deterministic actor and export its native artifact."""
    args = _parser().parse_args()
    examples = load_training_dataset(args.dataset)
    result = fit_linear_policy(
        examples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
    )
    export_linear_policy(
        args.output,
        result.policy,
        target=args.target,
        core_revision=args.core_revision,
        source_revision=_source_revision(args.source_revision),
        algorithm="masked-softmax behavioral cloning",
        objective=args.objective,
        samples=len(examples),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
    )
    print(f"wrote {args.output} (loss={result.loss:.6f}, accuracy={result.accuracy:.3f})")


if __name__ == "__main__":
    main()
