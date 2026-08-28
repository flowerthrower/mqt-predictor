/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/predictor/mlir/CircuitFeatures.h"

#include "mlir/Compiler/QCOAnalysis.h"
#include "mlir/Compiler/Target.h"

#include <llvm/ADT/StringRef.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace mqt::predictor::compiler {
namespace {

constexpr std::size_t CRITICAL_DEPTH_INDEX = 6;
constexpr std::size_t DEPTH_INDEX = 18;
constexpr std::size_t ENTANGLEMENT_RATIO_INDEX = 19;
constexpr std::size_t LIVENESS_INDEX = 22;
constexpr std::size_t NUM_QUBITS_INDEX = 24;
constexpr std::size_t PARALLELISM_INDEX = 26;
constexpr std::size_t PROGRAM_COMMUNICATION_INDEX = 27;

[[nodiscard]] float clampUnit(const double value) {
  return static_cast<float>(std::clamp(value, 0., 1.));
}

} // namespace

::mlir::FailureOr<CircuitAnalysis>
analyzeCircuit(const ::mlir::qco::QCOCircuitAnalysis& analysis,
               const ::mlir::CompilerTarget& target) {
  const auto metrics = analysis.forTarget(target);
  if (failed(metrics)) {
    return ::mlir::failure();
  }

  FeatureVector values{};
  if (metrics->totalOperations > 0) {
    for (const auto& [operationName, count] : metrics->operationCounts) {
      for (std::size_t index = 0; index < NUM_FEATURES; ++index) {
        const auto featureName = FEATURE_NAMES[index];
        if (llvm::StringRef(operationName) ==
            llvm::StringRef(featureName.data(), featureName.size())) {
          values[index] =
              clampUnit(static_cast<double>(count) /
                        static_cast<double>(metrics->totalOperations));
          break;
        }
      }
    }
  }
  values[CRITICAL_DEPTH_INDEX] = clampUnit(metrics->criticalDepth);
  values[DEPTH_INDEX] =
      clampUnit(std::log1p(std::min(static_cast<double>(metrics->depth),
                                    DEPTH_NORMALIZATION_MAX)) /
                std::log1p(DEPTH_NORMALIZATION_MAX));
  values[ENTANGLEMENT_RATIO_INDEX] = clampUnit(metrics->entanglementRatio);
  values[LIVENESS_INDEX] = clampUnit(metrics->liveness);
  values[NUM_QUBITS_INDEX] =
      target.numQubits() > 0
          ? clampUnit(static_cast<double>(metrics->numQubits) /
                      static_cast<double>(target.numQubits()))
          : 1.F;
  values[PARALLELISM_INDEX] = clampUnit(metrics->parallelism);
  values[PROGRAM_COMMUNICATION_INDEX] =
      clampUnit(metrics->programCommunication);

  CircuitAnalysis result;
  result.features = CircuitFeatures{
      .numQubits = metrics->numQubits,
      .depth = metrics->depth,
      .twoQubitDepth = metrics->twoQubitDepth,
      .numGates = metrics->numGates,
      .numTwoQubitGates = metrics->numTwoQubitGates,
      .values = values,
  };
  result.state = CompilerState{
      .mapped = metrics->mapped,
      .routed = metrics->routed,
      .synthesized = metrics->synthesized,
      .hasWideUnitary = metrics->hasWideUnitary,
  };
  result.fullyUnmapped = metrics->fullyUnmapped;
  return result;
}

::mlir::FailureOr<CircuitAnalysis>
analyzeCircuit(const ::mlir::ModuleOp module,
               const ::mlir::CompilerTarget& target) {
  return analyzeCircuit(::mlir::qco::QCOCircuitAnalysis(module), target);
}

} // namespace mqt::predictor::compiler
