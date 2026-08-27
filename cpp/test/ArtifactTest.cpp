/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/predictor/mlir/Model.h"
#include "mqt/predictor/mlir/Target.h"

#include <llvm/Support/Error.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>

int main(const int argc, char** argv) {
  using namespace mqt::predictor::compiler;
  if (argc != 3) {
    std::cerr << "expected target and model paths\n";
    return EXIT_FAILURE;
  }

  const auto targetPath = std::filesystem::path(argv[1]);
  const auto modelPath = std::filesystem::path(argv[2]);
  auto target = loadCompilerTarget(targetPath);
  if (!target || target->target.numQubits() != 4) {
    if (!target) {
      std::cerr << llvm::toString(target.takeError()) << '\n';
    }
    return EXIT_FAILURE;
  }
  auto model = LinearPolicyModel::load(modelPath, target->fingerprint,
                                       MQT_PREDICTOR_CORE_REVISION);
  if (!model) {
    std::cerr << llvm::toString(model.takeError()) << '\n';
    return EXIT_FAILURE;
  }

  const FeatureVector features{};
  const ActionMask legal{true, true, true, true, true, false};
  const auto decision = model->select(features, legal);
  if (!decision || decision->action != Action::PlaceAndRoute) {
    std::cerr << "native model did not reproduce the exported decision\n";
    return EXIT_FAILURE;
  }

  std::mt19937_64 generator(7);
  const ActionMask onlyMapping{false, false, false, true, false, false};
  const auto sampled = model->sample(features, onlyMapping, generator);
  if (!sampled || sampled->action != Action::PlaceAndRoute) {
    std::cerr << "native model did not respect the sampled action mask\n";
    return EXIT_FAILURE;
  }

  auto builtIn = createLineTarget(4);
  if (!builtIn) {
    std::cerr << llvm::toString(builtIn.takeError()) << '\n';
    return EXIT_FAILURE;
  }
  auto mismatch = LinearPolicyModel::load(modelPath, builtIn->fingerprint,
                                          MQT_PREDICTOR_CORE_REVISION);
  if (mismatch) {
    std::cerr << "target-specific model accepted a different target\n";
    return EXIT_FAILURE;
  }
  llvm::consumeError(mismatch.takeError());
  return EXIT_SUCCESS;
}
