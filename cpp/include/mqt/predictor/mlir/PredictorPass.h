/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <mlir/Pass/Pass.h>

#include <cstddef>
#include <memory>
#include <string>

namespace mqt::predictor::compiler {

enum class PolicyMode { Core, Bootstrap, Model };

struct PredictorOptions {
  PolicyMode policy = PolicyMode::Bootstrap;
  std::size_t targetQubits = 5;
  std::size_t maxSteps = 16;
  std::string targetPath;
  std::string modelPath;
  bool trace = false;
};

[[nodiscard]] std::unique_ptr<::mlir::Pass>
createPredictorPass(PredictorOptions options = {});

void registerPredictorPass();

} // namespace mqt::predictor::compiler
