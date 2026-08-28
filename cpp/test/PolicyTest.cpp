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
#include "mqt/predictor/mlir/Target.h"

#include <llvm/Support/Error.h>
#include <mlir/Compiler/Target.h>

#include <cstdlib>
#include <iostream>
#include <optional>
#include <vector>

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

  if (EXPERIMENT_SCHEMA != "mqt-predictor-core-stages/4" ||
      FEATURE_NAMES.size() != 50 || FEATURE_NAMES[0] != "c3sqrtx" ||
      FEATURE_NAMES[49] != "z" || MAX_STEPS != 20 ||
      PredictorOptions{}.maxSteps != 20 ||
      PredictorOptions{}.deterministicPolicy ||
      PredictorOptions{}.samplingSeed.has_value()) {
    std::cerr << "v3 observation contract is inconsistent\n";
    return EXIT_FAILURE;
  }

  const BootstrapLinearPolicy policy;
  FeatureVector features{};
  features[18] = 0.2F; // depth
  features[22] = 0.6F; // liveness

  auto mask = legalActions({});
  if (mask != ActionMask{true, true, true, true, true, false}) {
    std::cerr << "pre-mapping phase exposed the wrong actions\n";
    return EXIT_FAILURE;
  }
  if (!expectAction(policy, features, mask, Action::PlaceAndRoute)) {
    return EXIT_FAILURE;
  }

  const CompilerState synthesizedBeforeMapping{.synthesized = true};
  mask = legalActions(synthesizedBeforeMapping);
  if (mask != ActionMask{true, true, true, true, false, false}) {
    std::cerr << "pre-mapping native phase exposed the wrong actions\n";
    return EXIT_FAILURE;
  }

  const CompilerState routed{.mapped = true, .routed = true};
  mask = legalActions(routed);
  if (mask != ActionMask{true, true, true, false, true, false}) {
    std::cerr << "routed non-native phase exposed the wrong actions\n";
    return EXIT_FAILURE;
  }
  if (legalActions(routed) != mask) {
    std::cerr << "repeated no-op optimizations did not remain legal\n";
    return EXIT_FAILURE;
  }

  const CompilerState compiled{
      .mapped = true, .routed = true, .synthesized = true};
  mask = legalActions(compiled);
  if (mask != ActionMask{true, true, true, false, false, true}) {
    std::cerr << "compiled phase exposed the wrong actions\n";
    return EXIT_FAILURE;
  }
  if (!expectAction(policy, features, mask, Action::Terminate)) {
    return EXIT_FAILURE;
  }

  using Target = ::mlir::CompilerTarget;
  const auto makeTarget = [](const double fidelity,
                             const bool reverseTuples = false) {
    std::vector<Target::Site> sites;
    sites.emplace_back(llvm::cantFail(Target::Site::create(0, "q0", 100, 80)));
    sites.emplace_back(llvm::cantFail(Target::Site::create(1, "q1", 110, 90)));
    std::vector<Target::SiteTuple> tuples;
    auto firstTuple =
        llvm::cantFail(Target::SiteTuple::create({0}, 10, fidelity));
    auto secondTuple = llvm::cantFail(Target::SiteTuple::create({1}, 12, 0.97));
    if (reverseTuples) {
      tuples.emplace_back(std::move(secondTuple));
      tuples.emplace_back(std::move(firstTuple));
    } else {
      tuples.emplace_back(std::move(firstTuple));
      tuples.emplace_back(std::move(secondTuple));
    }
    std::vector<Target::Operation> operations;
    operations.emplace_back(llvm::cantFail(
        Target::Operation::create("r", 1, 2, tuples, 11, 0.995)));
    auto unit = llvm::cantFail(Target::DurationUnit::create("ns", 1.0));
    return llvm::cantFail(Target::create("calibrated", std::move(sites),
                                         std::nullopt, std::move(operations),
                                         std::move(unit)));
  };
  const auto firstFingerprint = compilerTargetFingerprint(makeTarget(0.99));
  if (firstFingerprint != compilerTargetFingerprint(makeTarget(0.99, true)) ||
      firstFingerprint == compilerTargetFingerprint(makeTarget(0.98))) {
    std::cerr << "target fingerprint is not calibration-sensitive\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
