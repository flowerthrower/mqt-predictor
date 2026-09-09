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

#include <algorithm>
#include <array>
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

  if (EXPERIMENT_SCHEMA != "mqt-predictor-markov-stop-features/1" ||
      FEATURE_NAMES.size() != 64 || NUM_CIRCUIT_FEATURES != 51 ||
      FEATURE_NAMES[0] != "c3sqrtx" || FEATURE_NAMES[28] != "r" ||
      FEATURE_NAMES[50] != "z" || FEATURE_NAMES[51] != "zz_decision_fraction" ||
      FEATURE_NAMES[63] != "zz_previous_action_5" || MAX_STEPS != 20 ||
      PredictorOptions{}.maxSteps != 20 ||
      PredictorOptions{}.deterministicPolicy ||
      !PredictorOptions{}.reactiveStop ||
      PredictorOptions{}.samplingSeed.has_value()) {
    std::cerr << "native observation contract is inconsistent\n";
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
  if (!expectAction(policy, features, mask, Action::PlaceAndRoute)) {
    return EXIT_FAILURE;
  }

  const CompilerState synthesizedBeforeMapping{.synthesized = true};
  mask = legalActions(synthesizedBeforeMapping, suppressed);
  if (mask != ActionMask{true, true, true, true, false, false}) {
    std::cerr << "pre-mapping native phase exposed the wrong actions\n";
    return EXIT_FAILURE;
  }

  const CompilerState routed{.mapped = true, .routed = true};
  mask = legalActions(routed, suppressed);
  if (mask != ActionMask{true, true, true, false, true, false}) {
    std::cerr << "routed non-native phase exposed the wrong actions\n";
    return EXIT_FAILURE;
  }
  suppressed[0] = true;
  suppressed[1] = true;
  if (legalActions(routed, suppressed) !=
      ActionMask{false, false, true, false, true, false}) {
    std::cerr << "no-op suppression did not intersect the factual mask\n";
    return EXIT_FAILURE;
  }
  suppressed.fill(false);
  if (legalActions(routed, suppressed) != mask) {
    std::cerr << "clearing no-op suppression did not restore the factual mask\n";
    return EXIT_FAILURE;
  }

  const CompilerState compiled{
      .mapped = true, .routed = true, .synthesized = true};
  mask = legalActions(compiled, suppressed);
  if (mask != ActionMask{true, true, true, false, false, true}) {
    std::cerr << "compiled phase exposed the wrong actions\n";
    return EXIT_FAILURE;
  }
  if (!expectAction(policy, features, mask, Action::Terminate)) {
    return EXIT_FAILURE;
  }

  EpisodeContext episode;
  const auto circuitFeatures = features;
  if (episode.observe(features, mask) != mask || features != circuitFeatures ||
      !episode.recordResult(Action::FuseSingleQubitUnitaryRuns, true, 0.8)) {
    std::cerr << "initial episode context is inconsistent\n";
    return EXIT_FAILURE;
  }
  const auto expectContext = [&](const std::array<float, 13>& expected) {
    return std::equal(features.begin(), features.begin() + NUM_CIRCUIT_FEATURES,
                      circuitFeatures.begin()) &&
           std::equal(expected.begin(), expected.end(),
                      features.begin() + NUM_CIRCUIT_FEATURES);
  };
  if (episode.observe(features, mask) != mask ||
      !expectContext({0.05F, 1, 0.8F, 0.8F, 0, 0.05F, 1, 0, 1, 0, 0, 0, 0}) ||
      episode.recordResult(Action::FuseSingleQubitUnitaryRuns, false, 0.7)) {
    std::cerr << "incumbent improvement did not release the repeated state\n";
    return EXIT_FAILURE;
  }
  auto tabuMask = mask;
  tabuMask[1] = false;
  if (episode.observe(features, mask) != tabuMask ||
      !expectContext({0.1F, 1, 0.7F, 0.8F, 0.05F, 0.1F, 0, 0, 1, 0, 0, 0, 0}) ||
      episode.recordResult(Action::FuseTwoQubitGates, true, std::nullopt)) {
    std::cerr
        << "first non-improving recurrence did not mask the previous action\n";
    return EXIT_FAILURE;
  }
  if (episode.observe(features, mask) ||
      !expectContext({0.15F, 1, 0, 0.8F, 0.1F, 0.15F, 1, 0, 0, 1, 0, 0, 0}) ||
      episode.incumbentFidelity() != 0.8) {
    std::cerr
        << "second non-improving recurrence did not request incumbent stop\n";
    return EXIT_FAILURE;
  }

  EpisodeContext baseline(false);
  features = circuitFeatures;
  for (std::size_t step = 0; step < MAX_STEPS; ++step) {
    if (baseline.observe(features, mask) != mask ||
        features[51] != static_cast<float>(step) / 20.0F ||
        features[56] != static_cast<float>(step) / 20.0F) {
      std::cerr << "baseline controller changed the mask or visit context\n";
      return EXIT_FAILURE;
    }
    static_cast<void>(
        baseline.recordResult(Action::FuseTwoQubitGates, false, 0.8));
  }

  EpisodeContext unscored;
  features = circuitFeatures;
  const ActionMask allTransforms{true, true, true, true, true, false};
  const std::array expectedMasks{
      allTransforms, ActionMask{false, true, true, true, true, false},
      ActionMask{true, false, true, true, true, false},
      ActionMask{true, false, false, true, true, false},
      ActionMask{false, true, false, true, true, false}};
  const std::array actions{
      Action::MergeSingleQubitRotationGates, Action::FuseSingleQubitUnitaryRuns,
      Action::FuseTwoQubitGates, Action::MergeSingleQubitRotationGates,
      Action::FuseSingleQubitUnitaryRuns};
  for (std::size_t step = 0; step < actions.size(); ++step) {
    if (unscored.observe(features, allTransforms) != expectedMasks[step] ||
        unscored.recordResult(actions[step], true, std::nullopt)) {
      std::cerr
          << "unscored recurrence stopped or used the wrong mask tenure\n";
      return EXIT_FAILURE;
    }
  }
  if (unscored.observe(features, mask) != mask || features[56] != 0.0F) {
    std::cerr << "visible-state identity ignored the base legal mask\n";
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
