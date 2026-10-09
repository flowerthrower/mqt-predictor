# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Bound edits while retaining each native preset's required finalization."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

Policy = Callable[[dict[str, Any]], str]
MAX_EDITS = 2


class PipelineActions:
    """Share edit budgets and stopping rules across the three native adapters."""

    def __init__(self, policy: Policy | None) -> None:
        """Start one compilation with the full edit budget."""
        self.policy = policy
        self.remaining = MAX_EDITS
        self.ended = False

    def decide(
        self,
        state: dict[str, Any],
        *,
        optional: bool,
        insertions: dict[int, dict[str, Any]],
        can_end: bool,
    ) -> str:
        """Expose legal actions and charge each explicit edit once.

        Adapters supply only insertions that preserve their native prerequisites.
        END spends one edit, then required stages run and optional stages skip.
        INSERT does not advance the native cursor; adapters request another
        decision after its execution with updated circuit features.
        """
        allowed = ["SKIP" if self.ended and optional else "KEEP"]
        if not self.ended and self.remaining:
            if optional:
                allowed.append("SKIP")
            allowed.extend(f"INSERT:{index}" for index in insertions)
            if can_end:
                allowed.append("END")
        state.update(
            allowed_actions=allowed,
            insertion_candidates=insertions,
            edits_remaining=self.remaining,
            automatic=self.ended,
        )
        action = allowed[0] if self.ended or self.policy is None else self.policy(state)
        if action not in allowed:
            msg = f"Invalid {action!r} for stage {state['stage_index']}:{state['stage_name']}; allowed: {allowed}."
            raise ValueError(msg)
        if not self.ended and action != "KEEP":
            self.remaining -= 1
            self.ended = action == "END"
        return action
