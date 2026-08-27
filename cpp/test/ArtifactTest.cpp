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
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <string_view>

int main(const int argc, char** argv) {
  using namespace mqt::predictor::compiler;
  if (argc != 3 && argc != 4) {
    std::cerr << "expected target and model paths, with an optional invalid "
                 "ONNX output path\n";
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

  if (argc == 4) {
    std::ifstream input(modelPath, std::ios::binary);
    if (!input) {
      std::cerr << "failed to read ONNX model\n";
      return EXIT_FAILURE;
    }
    std::string contents(std::istreambuf_iterator<char>{input}, {});
    constexpr std::string_view supportedArchitecture = "tanh-mlp-64x64";
    constexpr std::string_view unsupportedArchitecture = "unsupported-v1";
    static_assert(supportedArchitecture.size() ==
                  unsupportedArchitecture.size());
    const auto architecture = contents.find(supportedArchitecture);
    if (architecture == std::string::npos) {
      std::cerr << "ONNX model does not contain the supported architecture\n";
      return EXIT_FAILURE;
    }
    contents.replace(architecture, supportedArchitecture.size(),
                     unsupportedArchitecture);

    const auto invalidModelPath = std::filesystem::path(argv[3]);
    std::ofstream output(invalidModelPath, std::ios::binary | std::ios::trunc);
    output.write(contents.data(),
                 static_cast<std::streamsize>(contents.size()));
    output.close();
    if (!output) {
      std::cerr << "failed to write invalid ONNX model\n";
      return EXIT_FAILURE;
    }

    auto invalid = loadPolicyModel(invalidModelPath, target->fingerprint,
                                   MQT_PREDICTOR_CORE_REVISION);
    std::filesystem::remove(invalidModelPath);
    if (invalid) {
      std::cerr << "ONNX model accepted an unsupported architecture\n";
      return EXIT_FAILURE;
    }
    const auto diagnostic = llvm::toString(invalid.takeError());
    if (diagnostic.find(
            "architecture metadata does not match the supported actor") ==
        std::string::npos) {
      std::cerr << diagnostic << '\n';
      return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
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

  std::mt19937_64 firstGenerator(11);
  std::mt19937_64 secondGenerator(11);
  const auto firstSample = model->sample(features, legal, firstGenerator);
  const auto secondSample = model->sample(features, legal, secondGenerator);
  if (!firstSample || !secondSample ||
      firstSample->action != secondSample->action ||
      firstSample->logits != secondSample->logits) {
    std::cerr << "seeded stochastic inference is not reproducible\n";
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
