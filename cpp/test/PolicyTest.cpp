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
  const FeatureVector features{0.6F, 0.2F, 0.0F, 0.0F, 0.0F, 0.7F,
                               0.6F, 0.1F, 0.0F, 0.0F, 0.0F, 0.0F};
  ActionMask suppressed{};

  auto mask = legalActions({}, suppressed);
  if (!expectAction(policy, features, mask,
                    Action::MergeSingleQubitRotationGates)) {
    return EXIT_FAILURE;
  }
  suppressed[static_cast<std::size_t>(Action::MergeSingleQubitRotationGates)] =
      true;
  mask = legalActions({}, suppressed);
  if (!expectAction(policy, features, mask,
                    Action::FuseSingleQubitUnitaryRuns)) {
    return EXIT_FAILURE;
  }

  const CompilerState wideCircuit{.hasWideUnitary = true};
  mask = legalActions(wideCircuit, suppressed);
  if (mask != ActionMask{false, false, true, true, true}) {
    std::cerr << "wide circuit did not mask single-qubit fusion\n";
    return EXIT_FAILURE;
  }
  mask = legalActions({}, suppressed);
  if (mask != ActionMask{false, true, true, true, true}) {
    std::cerr << "decomposed circuit did not restore single-qubit fusion\n";
    return EXIT_FAILURE;
  }

  ActionMask transformsSuppressed{};
  for (std::size_t action = 0;
       action < static_cast<std::size_t>(Action::Terminate); ++action) {
    transformsSuppressed[action] = true;
  }
  mask = legalActions({}, transformsSuppressed);
  if (!expectAction(policy, features, mask, Action::Terminate)) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
