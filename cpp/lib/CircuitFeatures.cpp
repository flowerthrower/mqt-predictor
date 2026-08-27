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

#include "mlir/Compiler/Target.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/QCOUtils.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <utility>

namespace mqt::predictor::compiler {
namespace {

using SiteId = ::mlir::CompilerTarget::SiteId;

struct WireState {
  std::size_t logical = 0;
  std::optional<SiteId> site;
  std::size_t depth = 0;
  std::size_t twoQubitCriticalDepth = 0;
};

[[nodiscard]] float clampUnit(const double value) {
  return static_cast<float>(std::clamp(value, 0.0, 1.0));
}

[[nodiscard]] bool usesLinearQubit(::mlir::Operation& operation) {
  const auto isQubit = [](const ::mlir::Value value) {
    return ::mlir::qco::isLinearQubitType(value.getType());
  };
  return llvm::any_of(operation.getOperands(), isQubit) ||
         llvm::any_of(operation.getResults(), isQubit);
}

} // namespace

::mlir::FailureOr<CircuitAnalysis>
analyzeCircuit(::mlir::ModuleOp module, const ::mlir::CompilerTarget& target) {
  using namespace ::mlir;
  using namespace ::mlir::qco;

  auto entry = ::mlir::mqt::getEntryPoint(module);
  if (!entry || !entry.getBody().hasOneBlock() ||
      llvm::any_of(entry.getArgumentTypes(), isLinearQubitType)) {
    return failure();
  }

  llvm::DenseMap<Value, WireState> wires;
  using TensorState = SmallVector<std::optional<WireState>>;
  llvm::DenseMap<Value, TensorState> tensors;
  std::set<std::pair<std::size_t, std::size_t>> interactions;
  std::set<SiteId> staticSites;
  std::size_t nextLogical = 0;
  std::size_t dynamicRoots = 0;
  std::size_t staticRoots = 0;
  std::size_t numGates = 0;
  std::size_t numTwoQubitGates = 0;
  std::size_t maxDepth = 0;
  std::size_t twoQubitCriticalDepth = 0;
  std::size_t activity = 0;
  bool routed = true;
  bool synthesized = true;
  bool hasWideUnitary = false;

  const auto recordDepth = [&](const WireState& wire) {
    if (wire.depth > maxDepth ||
        (wire.depth == maxDepth &&
         wire.twoQubitCriticalDepth > twoQubitCriticalDepth)) {
      maxDepth = wire.depth;
      twoQubitCriticalDepth = wire.twoQubitCriticalDepth;
    }
  };

  const auto advanceWire = [&](const Value input, const Value output,
                               const bool isTwoQubit,
                               const std::size_t operationDepth,
                               const std::size_t operationCriticalDepth) {
    const auto found = wires.find(input);
    if (found == wires.end()) {
      return false;
    }
    auto next = found->second;
    wires.erase(found);
    next.depth = operationDepth;
    next.twoQubitCriticalDepth =
        operationCriticalDepth + static_cast<std::size_t>(isTwoQubit);
    wires[output] = next;
    recordDepth(next);
    return true;
  };

  for (Operation& operation : entry.getBody().front()) {
    if (auto alloc = dyn_cast<AllocOp>(operation)) {
      wires[alloc.getResult()] = WireState{.logical = nextLogical++};
      ++dynamicRoots;
      continue;
    }
    if (auto staticQubit = dyn_cast<StaticOp>(operation)) {
      const auto site = static_cast<SiteId>(staticQubit.getIndex());
      if (!target.vertexForSite(site) || !staticSites.insert(site).second) {
        return failure();
      }
      wires[staticQubit.getQubit()] =
          WireState{.logical = nextLogical++, .site = site};
      ++staticRoots;
      continue;
    }
    if (auto alloc = dyn_cast<qtensor::AllocOp>(operation)) {
      const auto type = dyn_cast<RankedTensorType>(alloc.getResult().getType());
      if (!type || !type.hasStaticShape() || type.getRank() != 1) {
        return failure();
      }
      const auto size = static_cast<std::size_t>(type.getDimSize(0));
      TensorState tensor(size);
      for (auto& wire : tensor) {
        wire = WireState{.logical = nextLogical++};
      }
      tensors[alloc.getResult()] = std::move(tensor);
      dynamicRoots += size;
      continue;
    }
    if (auto fromElements = dyn_cast<qtensor::FromElementsOp>(operation)) {
      TensorState tensor;
      tensor.reserve(fromElements.getElements().size());
      for (const auto element : fromElements.getElements()) {
        const auto found = wires.find(element);
        if (found == wires.end()) {
          return failure();
        }
        tensor.emplace_back(found->second);
        wires.erase(found);
      }
      tensors[fromElements.getResult()] = std::move(tensor);
      continue;
    }
    if (auto extract = dyn_cast<qtensor::ExtractOp>(operation)) {
      const auto index = getConstantIntValue(extract.getIndex());
      const auto found = tensors.find(extract.getTensor());
      if (!index || *index < 0 || found == tensors.end() ||
          std::cmp_greater_equal(*index, found->second.size())) {
        return failure();
      }
      auto tensor = std::move(found->second);
      tensors.erase(found);
      auto& wire = tensor[static_cast<std::size_t>(*index)];
      if (!wire) {
        return failure();
      }
      wires[extract.getResult()] = *wire;
      wire.reset();
      tensors[extract.getOutTensor()] = std::move(tensor);
      continue;
    }
    if (auto insert = dyn_cast<qtensor::InsertOp>(operation)) {
      const auto index = getConstantIntValue(insert.getIndex());
      const auto tensorFound = tensors.find(insert.getDest());
      const auto wireFound = wires.find(insert.getScalar());
      if (!index || *index < 0 || tensorFound == tensors.end() ||
          wireFound == wires.end() ||
          std::cmp_greater_equal(*index, tensorFound->second.size())) {
        return failure();
      }
      auto tensor = std::move(tensorFound->second);
      tensors.erase(tensorFound);
      auto& slot = tensor[static_cast<std::size_t>(*index)];
      if (slot) {
        return failure();
      }
      slot = wireFound->second;
      wires.erase(wireFound);
      tensors[insert.getResult()] = std::move(tensor);
      continue;
    }
    if (auto dealloc = dyn_cast<qtensor::DeallocOp>(operation)) {
      const auto found = tensors.find(dealloc.getTensor());
      if (found == tensors.end()) {
        return failure();
      }
      tensors.erase(found);
      continue;
    }
    if (auto sink = dyn_cast<SinkOp>(operation)) {
      const auto found = wires.find(sink.getQubit());
      if (found == wires.end()) {
        return failure();
      }
      wires.erase(found);
      continue;
    }

    if (auto unitary = dyn_cast<UnitaryOpInterface>(operation)) {
      if (isa<BarrierOp, GPhaseOp>(operation)) {
        for (const auto [input, output] : llvm::zip_equal(
                 unitary.getInputQubits(), unitary.getOutputQubits())) {
          const auto found = wires.find(input);
          if (found == wires.end()) {
            return failure();
          }
          const auto next = found->second;
          wires.erase(found);
          wires[output] = next;
        }
        continue;
      }

      const auto arity = unitary.getNumQubits();
      if (arity == 0 || operation.getNumRegions() > 1) {
        return failure();
      }

      std::size_t operationDepth = 0;
      std::size_t operationCriticalDepth = 0;
      for (const auto input : unitary.getInputQubits()) {
        const auto found = wires.find(input);
        if (found == wires.end()) {
          return failure();
        }
        const auto candidate =
            std::pair{found->second.depth, found->second.twoQubitCriticalDepth};
        const auto current = std::pair{operationDepth, operationCriticalDepth};
        if (candidate > current) {
          operationDepth = candidate.first;
          operationCriticalDepth = candidate.second;
        }
      }
      ++operationDepth;

      const auto isTwoQubit = arity == 2;
      hasWideUnitary |= arity > 2;
      ++numGates;
      numTwoQubitGates += static_cast<std::size_t>(isTwoQubit);
      activity += arity;
      synthesized &= target.supports(&operation);

      if (isTwoQubit) {
        const auto first = wires.find(unitary.getInputQubit(0));
        const auto second = wires.find(unitary.getInputQubit(1));
        if (first == wires.end() || second == wires.end()) {
          return failure();
        }
        const auto edge =
            std::minmax(first->second.logical, second->second.logical);
        interactions.emplace(edge.first, edge.second);

        if (first->second.site && second->second.site) {
          const auto firstVertex = target.vertexForSite(*first->second.site);
          const auto secondVertex = target.vertexForSite(*second->second.site);
          routed &= firstVertex && secondVertex &&
                    target.areAdjacent(*firstVertex, *secondVertex);
        }
      }

      for (const auto [input, output] : llvm::zip_equal(
               unitary.getInputQubits(), unitary.getOutputQubits())) {
        if (!advanceWire(input, output, isTwoQubit, operationDepth,
                         operationCriticalDepth)) {
          return failure();
        }
      }
      continue;
    }

    if (auto measure = dyn_cast<MeasureOp>(operation)) {
      const auto found = wires.find(measure.getQubitIn());
      if (found == wires.end()) {
        return failure();
      }
      synthesized &= target.supports(&operation);
      ++activity;
      if (!advanceWire(measure.getQubitIn(), measure.getQubitOut(), false,
                       found->second.depth + 1,
                       found->second.twoQubitCriticalDepth)) {
        return failure();
      }
      continue;
    }

    if (auto reset = dyn_cast<ResetOp>(operation)) {
      const auto found = wires.find(reset.getQubitIn());
      if (found == wires.end()) {
        return failure();
      }
      synthesized &= target.supports(&operation);
      ++activity;
      if (!advanceWire(reset.getQubitIn(), reset.getQubitOut(), false,
                       found->second.depth + 1,
                       found->second.twoQubitCriticalDepth)) {
        return failure();
      }
      continue;
    }

    if (operation.getNumRegions() != 0 || usesLinearQubit(operation)) {
      return failure();
    }
  }

  if (!wires.empty() || !tensors.empty()) {
    return failure();
  }

  const auto numQubits = nextLogical;
  if (numQubits == 0) {
    return failure();
  }

  const auto mapped = dynamicRoots == 0 && staticRoots == numQubits;
  routed &= mapped;

  const auto communication =
      numQubits > 1 ? (2.0 * static_cast<double>(interactions.size())) /
                          (static_cast<double>(numQubits) *
                           static_cast<double>(numQubits - 1))
                    : 0.0;
  const auto criticalDepth = numTwoQubitGates > 0
                                 ? static_cast<double>(twoQubitCriticalDepth) /
                                       static_cast<double>(numTwoQubitGates)
                                 : 0.0;
  const auto entanglement = numGates > 0
                                ? static_cast<double>(numTwoQubitGates) /
                                      static_cast<double>(numGates)
                                : 0.0;
  const auto parallelism =
      numQubits > 1 && maxDepth > 0
          ? std::max(((static_cast<double>(numGates) /
                       static_cast<double>(maxDepth)) -
                      1.0) /
                         static_cast<double>(numQubits - 1),
                     0.0)
          : 0.0;
  const auto liveness =
      maxDepth > 0
          ? static_cast<double>(activity) /
                (static_cast<double>(numQubits) * static_cast<double>(maxDepth))
          : 0.0;
  const auto relativeQubits = target.numQubits() > 0
                                  ? static_cast<double>(numQubits) /
                                        static_cast<double>(target.numQubits())
                                  : 1.0;
  const auto normalizedDepth = std::log1p(static_cast<double>(maxDepth)) /
                               std::log1p(DEPTH_NORMALIZATION_MAX);

  CircuitAnalysis analysis;
  analysis.features = CircuitFeatures{
      .numQubits = numQubits,
      .depth = maxDepth,
      .twoQubitDepth = twoQubitCriticalDepth,
      .numGates = numGates,
      .numTwoQubitGates = numTwoQubitGates,
      .values = {clampUnit(relativeQubits), clampUnit(normalizedDepth),
                 clampUnit(communication), clampUnit(criticalDepth),
                 clampUnit(entanglement), clampUnit(parallelism),
                 clampUnit(liveness)}};
  analysis.state = CompilerState{
      .mapped = mapped,
      .routed = routed,
      .synthesized = synthesized,
      .hasWideUnitary = hasWideUnitary};
  return analysis;
}

} // namespace mqt::predictor::compiler
