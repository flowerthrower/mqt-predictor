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

#include "mqt/predictor/mlir/Policy.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>

namespace mlir {
class CompilerTarget;
namespace qco {
class QCOCircuitAnalysis;
} // namespace qco
} // namespace mlir

namespace mqt::predictor::compiler {

struct CircuitFeatures {
  std::size_t numQubits = 0;
  std::size_t depth = 0;
  std::size_t twoQubitDepth = 0;
  std::size_t numGates = 0;
  std::size_t numTwoQubitGates = 0;
  FeatureVector values{};
};

struct CircuitAnalysis {
  CircuitFeatures features;
  CompilerState state;
  bool fullyUnmapped = false;
};

/**
 * Analyze a straight-line scalar QCO entry point.
 *
 * Static one-dimensional tensor registers are supported. Quantum control flow
 * and dynamic tensor indexing return failure in schema v3; the predictor pass
 * then uses the canonical Core pipeline as its safe fallback.
 */
[[nodiscard]] ::mlir::FailureOr<CircuitAnalysis>
analyzeCircuit(::mlir::ModuleOp module, const ::mlir::CompilerTarget& target);

/** Materialize Predictor features from Core's cached QCO analysis. */
[[nodiscard]] ::mlir::FailureOr<CircuitAnalysis>
analyzeCircuit(const ::mlir::qco::QCOCircuitAnalysis& analysis,
               const ::mlir::CompilerTarget& target);

} // namespace mqt::predictor::compiler
