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

#include <cstdlib>
#include <iostream>
#include <optional>

namespace {

using namespace mqt::predictor::compiler;

[[nodiscard]] bool expectAction(const BootstrapLinearPolicy& policy,
                                const FeatureVector& features,
                                const ActionMask& mask, const Action expected) {
  const auto decision = policy.select(features, mask);
  if (!decision || decision->action != expected) {
    std::cerr << "expected " << actionName(expected) << '\n';
    return false;
  }
  return true;
}

} // namespace

int main() {
  using namespace mqt::predictor::compiler;

  const BootstrapLinearPolicy policy;
  const FeatureVector features{0.6F, 0.2F, 0.5F, 0.4F, 0.3F, 0.7F, 0.6F};
  ActionMask suppressed{};

  auto mask = legalActions({}, suppressed);
  if (!expectAction(policy, features, mask, Action::FuseTwoQubit)) {
    return EXIT_FAILURE;
  }
  suppressed[static_cast<std::size_t>(Action::FuseTwoQubit)] = true;
  mask = legalActions({}, suppressed);
  if (!expectAction(policy, features, mask, Action::PlaceAndRoute)) {
    return EXIT_FAILURE;
  }

  const CompilerState mapped{
      .mapped = true, .routed = true, .synthesized = false};
  ActionMask optimizationsSuppressed{};
  optimizationsSuppressed[static_cast<std::size_t>(Action::MergeRotations)] =
      true;
  optimizationsSuppressed[static_cast<std::size_t>(Action::FuseSingleQubit)] =
      true;
  optimizationsSuppressed[static_cast<std::size_t>(Action::FuseTwoQubit)] =
      true;
  mask = legalActions(mapped, optimizationsSuppressed);
  if (!expectAction(policy, features, mask, Action::NativeSynthesis)) {
    return EXIT_FAILURE;
  }

  const CompilerState conformant{
      .mapped = true, .routed = true, .synthesized = true};
  mask = legalActions(conformant, {});
  if (!expectAction(policy, features, mask, Action::Terminate)) {
    return EXIT_FAILURE;
  }

  const CompilerState invalid{
      .mapped = true, .routed = false, .synthesized = false};
  if (policy.select(features, legalActions(invalid, {}))) {
    std::cerr << "invalid mapped state unexpectedly selected an action\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
