/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/predictor/mlir/Policy.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>

namespace mqt::predictor::compiler {
namespace {

constexpr auto WEIGHTS =
    std::array<std::array<float, NUM_FEATURES>, NUM_ACTIONS>{
        std::array{0.00F, 1.60F, 0.00F, 0.00F, -0.20F, 0.00F, 0.10F, 0.00F,
                   0.00F, 0.00F, 0.00F, 0.00F},
        std::array{0.00F, 1.30F, 0.00F, 0.00F, -0.15F, 0.00F, 0.00F, 0.00F,
                   0.00F, 0.00F, 0.00F, 0.00F},
        std::array{0.00F, 0.10F, 0.35F, 0.25F, 0.35F, 0.00F, 0.00F, 0.00F,
                   0.00F, 0.00F, 0.00F, 0.00F},
        std::array{0.00F, 0.00F, 0.20F, 0.00F, 0.00F, 0.00F, 0.00F, 0.00F,
                   0.00F, 0.00F, 0.00F, 0.00F},
        std::array{0.00F, 0.00F, 0.00F, 0.00F, 0.00F, 0.00F, 0.00F, 0.00F,
                   0.00F, 0.00F, 0.00F, 0.00F},
    };

constexpr auto BIASES =
    std::array<float, NUM_ACTIONS>{0.05F, 0.04F, 0.03F, 0.02F, 0.25F};

[[nodiscard]] constexpr std::size_t index(const Action action) {
  return static_cast<std::size_t>(action);
}

} // namespace

std::string_view actionName(const Action action) {
  const auto actionIndex = index(action);
  if (actionIndex >= ACTION_NAMES.size()) {
    return "unknown";
  }
  return ACTION_NAMES[actionIndex];
}

ActionMask legalActions(const CompilerState& state,
                        const ActionMask& suppressed) {
  ActionMask legal{};
  std::transform(suppressed.begin(), suppressed.end(), legal.begin(),
                 [](const bool isSuppressed) { return !isSuppressed; });
  legal[index(Action::FuseSingleQubitUnitaryRuns)] &= !state.hasWideUnitary;
  return legal;
}

std::optional<Decision>
BootstrapLinearPolicy::select(const FeatureVector& features,
                              const ActionMask& legal) const {
  Decision decision{.action = Action::Terminate};
  decision.logits.fill(-std::numeric_limits<float>::infinity());

  std::optional<std::size_t> selected;
  for (std::size_t action = 0; action < NUM_ACTIONS; ++action) {
    if (!legal[action]) {
      continue;
    }
    auto logit = BIASES[action];
    for (std::size_t feature = 0; feature < NUM_FEATURES; ++feature) {
      logit += WEIGHTS[action][feature] * features[feature];
    }
    decision.logits[action] = logit;
    if (!selected || logit > decision.logits[*selected]) {
      selected = action;
    }
  }

  if (!selected) {
    return std::nullopt;
  }
  decision.action = static_cast<Action>(*selected);
  return decision;
}

} // namespace mqt::predictor::compiler
