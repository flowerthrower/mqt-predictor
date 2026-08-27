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

#include "mqt/predictor/mlir/PredictorPass.h"

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

  if (EXPERIMENT_SCHEMA != "mqt-predictor-core-stages/2" ||
      FEATURE_NAMES.size() != 56 || FEATURE_NAMES[0] != "c3sqrtx" ||
      FEATURE_NAMES[49] != "z" || FEATURE_NAMES[50] != "step_fraction" ||
      MAX_TRANSFORM_PASSES != 20 || PredictorOptions{}.maxSteps != 20) {
    std::cerr << "v3 observation contract is inconsistent\n";
    return EXIT_FAILURE;
  }

  const BootstrapLinearPolicy policy;
  FeatureVector features{};
  features[18] = 0.2F; // depth
  features[22] = 0.6F; // liveness
  ActionMask suppressed{};

  auto mask = legalActions({}, suppressed);
  if (mask != ActionMask{true, true, true, true, true, false}) {
    std::cerr << "pre-mapping phase exposed the wrong actions\n";
    return EXIT_FAILURE;
  }
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

  const CompilerState synthesizedBeforeMapping{.synthesized = true};
  mask = legalActions(synthesizedBeforeMapping, {});
  if (mask != ActionMask{true, true, true, true, false, false}) {
    std::cerr << "pre-mapping native phase exposed the wrong actions\n";
    return EXIT_FAILURE;
  }

  const CompilerState routed{.mapped = true, .routed = true};
  mask = legalActions(routed, {});
  if (mask != ActionMask{true, true, true, false, true, false}) {
    std::cerr << "routed non-native phase exposed the wrong actions\n";
    return EXIT_FAILURE;
  }

  ActionMask attemptedOptimizations{true, true, true, false, false, false};
  mask = legalActions(routed, attemptedOptimizations);
  if (mask != ActionMask{false, false, false, false, true, false} ||
      !expectAction(policy, features, mask, Action::SynthesizeForTarget)) {
    std::cerr << "optimization retries prevented required re-synthesis\n";
    return EXIT_FAILURE;
  }

  if (!isOptimizationAction(Action::MergeSingleQubitRotationGates) ||
      !isOptimizationAction(Action::FuseSingleQubitUnitaryRuns) ||
      !isOptimizationAction(Action::FuseTwoQubitGates) ||
      isOptimizationAction(Action::PlaceAndRoute) ||
      isOptimizationAction(Action::SynthesizeForTarget) ||
      isOptimizationAction(Action::Terminate)) {
    std::cerr << "exact-state suppression covered a Core stage action\n";
    return EXIT_FAILURE;
  }

  const CompilerState compiled{
      .mapped = true, .routed = true, .synthesized = true};
  mask = legalActions(compiled, {});
  if (mask != ActionMask{true, true, true, false, false, true}) {
    std::cerr << "compiled phase exposed the wrong actions\n";
    return EXIT_FAILURE;
  }

  ActionMask transformsSuppressed{};
  for (std::size_t action = 0;
       action < static_cast<std::size_t>(Action::Terminate); ++action) {
    transformsSuppressed[action] = true;
  }
  mask = legalActions(compiled, transformsSuppressed);
  if (!expectAction(policy, features, mask, Action::Terminate)) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
