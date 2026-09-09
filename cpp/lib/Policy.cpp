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

constexpr auto WEIGHTS = [] {
  std::array<std::array<float, NUM_FEATURES>, NUM_ACTIONS> weights{};
  weights[0][18] = 1.60F;  // depth
  weights[0][19] = -0.20F; // entanglement_ratio
  weights[0][22] = 0.10F;  // liveness
  weights[1][18] = 1.30F;
  weights[1][19] = -0.15F;
  weights[2][6] = 0.05F; // critical_depth
  weights[2][19] = 0.12F;
  return weights;
}();

constexpr auto BIASES =
    std::array<float, NUM_ACTIONS>{0.05F, 0.04F, 0.02F, 3.00F, 2.00F, 4.00F};

[[nodiscard]] constexpr std::size_t index(const Action action) {
  return static_cast<std::size_t>(action);
}

} // namespace

std::optional<ActionMask> EpisodeContext::observe(FeatureVector& features,
                                                  ActionMask legal) {
  std::string key(reinterpret_cast<const char*>(features.data()),
                  NUM_CIRCUIT_FEATURES * sizeof(float));
  for (const auto enabled : legal) {
    key.push_back(static_cast<char>(enabled));
  }
  visit_ = &visits_[key];
  const auto fraction = [](const std::size_t count) {
    return static_cast<float>(std::min(count, MAX_STEPS)) /
           static_cast<float>(MAX_STEPS);
  };
  std::fill(features.begin() + NUM_CIRCUIT_FEATURES, features.end(), 0.0F);
  features[51] = fraction(step_);
  features[52] = incumbent_.has_value() ? 1.0F : 0.0F;
  features[53] = static_cast<float>(current_.value_or(0.0));
  features[54] = static_cast<float>(incumbent_.value_or(0.0));
  features[55] = incumbent_ ? fraction(step_ - incumbentStep_) : 0.0F;
  features[56] = fraction(visit_->count);
  features[57] = changed_ ? 1.0F : 0.0F;
  if (previousAction_) {
    features[58 + index(*previousAction_)] = 1.0F;
  }
  ++visit_->count;

  if (reactiveStop_ && visit_->action) {
    constexpr auto absent = -std::numeric_limits<double>::infinity();
    if (incumbent_.value_or(absent) > visit_->best.value_or(absent) + 1e-15) {
      visit_->recurrences = 0;
    } else {
      ++visit_->recurrences;
      if (visit_->recurrences >= 2 && incumbent_) {
        return std::nullopt;
      }
      const auto tenure = 1U
                          << std::min<std::size_t>(2, visit_->recurrences - 1);
      auto& before = visit_->tabuBefore[index(*visit_->action)];
      before = std::max(before, step_ + tenure);
    }
  }
  for (std::size_t action = 0; action < index(Action::Terminate); ++action) {
    legal[action] &= step_ >= visit_->tabuBefore[action];
  }
  return legal;
}

bool EpisodeContext::recordResult(const Action action, const bool changed,
                                  const std::optional<double> fidelity) {
  visit_->action = action;
  visit_->best = incumbent_;
  previousAction_ = action;
  changed_ = changed;
  ++step_;
  current_ = fidelity;
  if (fidelity && (!incumbent_ || *fidelity > *incumbent_ + 1e-15)) {
    incumbent_ = fidelity;
    incumbentStep_ = step_;
    return true;
  }
  return false;
}

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
  legal[index(Action::PlaceAndRoute)] &= !state.mapped;
  legal[index(Action::SynthesizeForTarget)] &= !state.synthesized;
  legal[index(Action::Terminate)] &=
      state.mapped && state.routed && state.synthesized;
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
