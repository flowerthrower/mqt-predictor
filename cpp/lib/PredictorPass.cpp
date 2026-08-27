/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/predictor/mlir/PredictorPass.h"

#include "mlir/Compiler/Target.h"
#include "mlir/Compiler/TargetCompilation.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/QCOUtils.h"
#include "mlir/Dialect/QCO/Transforms/Mapping/Mapping.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Support/Passes.h"
#include "mqt/predictor/mlir/CircuitFeatures.h"
#include "mqt/predictor/mlir/Model.h"
#include "mqt/predictor/mlir/Policy.h"
#include "mqt/predictor/mlir/Target.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/Passes.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mqt::predictor::compiler {
namespace {

using Target = ::mlir::CompilerTarget;
using CandidateScore =
    std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>;

enum class ExhaustiveAction : std::uint8_t {
  MergeRotations,
  FuseSingleQubit,
  FuseTwoQubit,
};

struct ExhaustiveCandidate {
  ::mlir::OwningOpRef<::mlir::ModuleOp> module;
  CandidateScore score;
  std::string schedule;
  std::size_t index = 0;
};

[[nodiscard]] std::string moduleFingerprint(::mlir::ModuleOp module) {
  std::string fingerprint;
  llvm::raw_string_ostream stream(fingerprint);
  module.print(stream);
  return fingerprint;
}

[[nodiscard]] std::string_view
exhaustiveActionName(const ExhaustiveAction action) {
  switch (action) {
  case ExhaustiveAction::MergeRotations:
    return "merge-rotations";
  case ExhaustiveAction::FuseSingleQubit:
    return "fuse-single-qubit";
  case ExhaustiveAction::FuseTwoQubit:
    return "fuse-two-qubit";
  }
  return "unknown";
}

[[nodiscard]] std::vector<std::vector<ExhaustiveAction>>
enumerateOptimizationSchedules() {
  constexpr std::array optimizations{ExhaustiveAction::MergeRotations,
                                     ExhaustiveAction::FuseSingleQubit,
                                     ExhaustiveAction::FuseTwoQubit};
  std::vector<std::vector<ExhaustiveAction>> schedules;
  for (std::size_t mask = 0; mask < (1U << optimizations.size()); ++mask) {
    std::vector<ExhaustiveAction> selected;
    for (std::size_t index = 0; index < optimizations.size(); ++index) {
      if ((mask & (1U << index)) != 0) {
        selected.emplace_back(optimizations[index]);
      }
    }
    do {
      schedules.emplace_back(selected);
    } while (std::next_permutation(selected.begin(), selected.end()));
  }
  return schedules;
}

[[nodiscard]] std::string
scheduleName(const std::vector<ExhaustiveAction>& schedule) {
  if (schedule.empty()) {
    return "none";
  }
  std::string result;
  for (const auto action : schedule) {
    if (!result.empty()) {
      result += '>';
    }
    result += exhaustiveActionName(action);
  }
  return result;
}

[[nodiscard]] CandidateScore scoreFor(const CircuitAnalysis& analysis) {
  return {analysis.features.twoQubitDepth, analysis.features.numTwoQubitGates,
          analysis.features.depth, analysis.features.numGates};
}

[[nodiscard]] ::mlir::LogicalResult verifyStaticSites(::mlir::ModuleOp module,
                                                      const Target& target) {
  std::set<Target::SiteId> sites;
  const auto walkResult = module.walk([&](::mlir::qco::StaticOp staticOp) {
    const auto site = static_cast<Target::SiteId>(staticOp.getIndex());
    if (!target.vertexForSite(site)) {
      staticOp.emitError() << "target does not contain static site " << site;
      return ::mlir::WalkResult::interrupt();
    }
    if (!sites.insert(site).second) {
      staticOp.emitError() << "static site " << site
                           << " is assigned to more than one qubit";
      return ::mlir::WalkResult::interrupt();
    }
    return ::mlir::WalkResult::advance();
  });
  return ::mlir::failure(walkResult.wasInterrupted());
}

[[nodiscard]] ::mlir::LogicalResult
verifyLinearQubitStructure(::mlir::ModuleOp module) {
  const auto walkResult = module.walk([&](::mlir::Operation* operation) {
    const auto dialect = operation->getName().getDialectNamespace();
    const auto supportedCarrier = dialect == "qco" || dialect == "qtensor" ||
                                  dialect == "scf" || dialect == "func";
    const auto rejectUnsupportedCarrier = [&](const ::mlir::Value value) {
      if (!::mlir::qco::isLinearQubitType(value.getType()) ||
          supportedCarrier) {
        return false;
      }
      operation->emitError()
          << "unsupported operation carries a linear qubit value: "
          << operation->getName();
      return true;
    };

    for (const auto operand : operation->getOperands()) {
      if (rejectUnsupportedCarrier(operand)) {
        return ::mlir::WalkResult::interrupt();
      }
    }
    for (const auto result : operation->getResults()) {
      if (rejectUnsupportedCarrier(result)) {
        return ::mlir::WalkResult::interrupt();
      }
      if (::mlir::qco::isLinearQubitType(result.getType()) &&
          !result.hasOneUse()) {
        operation->emitError() << "linear qubit result must have exactly one "
                                  "use";
        return ::mlir::WalkResult::interrupt();
      }
    }
    for (auto& region : operation->getRegions()) {
      for (auto& block : region) {
        for (const auto argument : block.getArguments()) {
          if (rejectUnsupportedCarrier(argument)) {
            return ::mlir::WalkResult::interrupt();
          }
          if (::mlir::qco::isLinearQubitType(argument.getType()) &&
              !argument.hasOneUse()) {
            operation->emitError()
                << "linear qubit block argument must have exactly one use";
            return ::mlir::WalkResult::interrupt();
          }
        }
      }
    }
    return ::mlir::WalkResult::advance();
  });
  return ::mlir::failure(walkResult.wasInterrupted());
}

void populateTargetFinalizationPipeline(::mlir::OpPassManager& pipeline,
                                        const Target& target) {
  ::populateQCOCleanupPipeline(pipeline);
  pipeline.addPass(::mlir::qco::createTargetNativeSynthesis(target));
  pipeline.addPass(::mlir::createCSEPass());
  pipeline.addPass(::mlir::createRemoveDeadValuesPass());
}

class PredictorPass final
    : public ::mlir::PassWrapper<PredictorPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PredictorPass)

  PredictorPass() = default;
  explicit PredictorPass(PredictorOptions options) : options_(options) {}

  [[nodiscard]] llvm::StringRef getArgument() const final {
    return "mqt-predictor-bootstrap";
  }

  [[nodiscard]] llvm::StringRef getDescription() const final {
    return "Run the experimental compiled MQT Predictor bootstrap heuristic";
  }

  void runOnOperation() final {
    if (options_.maxSteps == 0 || options_.maxSteps > MAX_STEPS) {
      getOperation().emitError()
          << "maximum policy decisions must be between 1 and " << MAX_STEPS;
      signalPassFailure();
      return;
    }
    auto loadedTarget = !options_.deviceId.empty()
                            ? loadCompilerTargetFromDevice(options_.deviceId)
                        : !options_.targetPath.empty()
                            ? loadCompilerTarget(options_.targetPath)
                            : createLineTarget(options_.targetQubits);
    if (!loadedTarget) {
      getOperation().emitError() << llvm::toString(loadedTarget.takeError());
      signalPassFailure();
      return;
    }
    const auto& target = loadedTarget->target;
    if (failed(verifyStaticSites(getOperation(), target)) ||
        failed(verifyLinearQubitStructure(getOperation()))) {
      signalPassFailure();
      return;
    }

    if (options_.policy == PolicyMode::Core) {
      ::mlir::OpPassManager pipeline(::mlir::ModuleOp::getOperationName());
      ::mlir::populateTargetCompilationPipeline(pipeline, target);
      if (failed(runPipeline(pipeline, getOperation())) ||
          failed(verifyStaticSites(getOperation(), target)) ||
          failed(verifyLinearQubitStructure(getOperation()))) {
        signalPassFailure();
      }
      return;
    }

    std::unique_ptr<PolicyModel> model;
    if (options_.policy == PolicyMode::Model) {
      auto loadedModel =
          loadPolicyModel(options_.modelPath, loadedTarget->fingerprint,
                          MQT_PREDICTOR_CORE_REVISION);
      if (!loadedModel) {
        getOperation().emitError() << llvm::toString(loadedModel.takeError());
        signalPassFailure();
        return;
      }
      model = std::move(*loadedModel);
    }

    auto backup = ::mlir::OwningOpRef<::mlir::ModuleOp>(
        ::mlir::cast<::mlir::ModuleOp>(getOperation()->clone()));
    const auto result =
        options_.policy == PolicyMode::Exhaustive
            ? runExhaustive(getOperation(), target, loadedTarget->fingerprint)
            : runBootstrap(getOperation(), target, loadedTarget->fingerprint,
                           model.get());
    if (succeeded(result)) {
      if (failed(verifyStaticSites(getOperation(), target)) ||
          failed(verifyLinearQubitStructure(getOperation()))) {
        signalPassFailure();
      }
      return;
    }

    getOperation()->setAttrs(backup.get()->getAttrs());
    getOperation().getBodyRegion().takeBody(backup->getBodyRegion());

    if (options_.trace) {
      llvm::errs() << "[mqt-predictor] falling back to Core's canonical target "
                      "pipeline\n";
    }
    ::mlir::OpPassManager fallback(::mlir::ModuleOp::getOperationName());
    ::mlir::populateTargetCompilationPipeline(fallback, target);
    if (failed(runPipeline(fallback, getOperation())) ||
        failed(verifyStaticSites(getOperation(), target)) ||
        failed(verifyLinearQubitStructure(getOperation()))) {
      signalPassFailure();
    }
  }

private:
  [[nodiscard]] ::mlir::LogicalResult
  runExhaustive(::mlir::ModuleOp module, const Target& target,
                const std::string_view targetFingerprint) {
    const auto schedules = enumerateOptimizationSchedules();
    const auto candidateCount = schedules.size() + 1;
    std::optional<ExhaustiveCandidate> best;
    std::set<std::string> uniqueOutputs;
    std::size_t validCandidates = 0;
    std::chrono::microseconds totalTime{};

    if (options_.trace) {
      llvm::errs()
          << "[mqt-predictor] search=exhaustive core_pin="
          << MQT_PREDICTOR_CORE_REVISION
          << " target=" << target.name().value_or("unnamed")
          << " target_fingerprint=" << targetFingerprint
          << " candidates=" << candidateCount
          << " objective=two-qubit-depth,two-qubit-gates,depth,gates\n";
    }

    const auto evaluate = [&](const std::size_t index, const std::string& name,
                              const std::vector<ExhaustiveAction>* schedule) {
      auto candidate = ::mlir::cast<::mlir::ModuleOp>(module->clone());
      module.getBody()->push_back(candidate);
      const auto started = std::chrono::steady_clock::now();
      const auto compiled =
          schedule == nullptr
              ? runCorePipeline(candidate, target)
              : runOptimizationSchedule(candidate, target, *schedule);
      const auto elapsed =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - started);
      totalTime += elapsed;

      auto analysis = analyzeCircuit(candidate, target);
      const auto valid = succeeded(compiled) && succeeded(analysis) &&
                         analysis->state.mapped && analysis->state.routed &&
                         analysis->state.synthesized &&
                         succeeded(verifyStaticSites(candidate, target)) &&
                         succeeded(verifyLinearQubitStructure(candidate));
      if (!valid) {
        if (options_.trace) {
          llvm::errs() << "[mqt-predictor] candidate=" << index
                       << " schedule=" << name
                       << " valid=0 compile_us=" << elapsed.count() << '\n';
        }
        candidate->erase();
        return;
      }

      ++validCandidates;
      uniqueOutputs.emplace(moduleFingerprint(candidate));
      const auto score = scoreFor(*analysis);
      if (options_.trace) {
        llvm::errs() << "[mqt-predictor] candidate=" << index
                     << " schedule=" << name << " valid=1 two_qubit_depth="
                     << analysis->features.twoQubitDepth
                     << " two_qubit=" << analysis->features.numTwoQubitGates
                     << " depth=" << analysis->features.depth
                     << " gates=" << analysis->features.numGates
                     << " compile_us=" << elapsed.count() << '\n';
      }
      candidate->remove();
      auto ownedCandidate = ::mlir::OwningOpRef<::mlir::ModuleOp>(candidate);
      if (!best || score < best->score) {
        best.emplace(ExhaustiveCandidate{.module = std::move(ownedCandidate),
                                         .score = score,
                                         .schedule = name,
                                         .index = index});
      }
    };

    evaluate(0, "core", nullptr);
    for (std::size_t index = 0; index < schedules.size(); ++index) {
      evaluate(index + 1, scheduleName(schedules[index]), &schedules[index]);
    }
    if (!best) {
      return ::mlir::failure();
    }

    if (options_.trace) {
      const auto [twoQubitDepth, twoQubitGates, depth, gates] = best->score;
      llvm::errs() << "[mqt-predictor] search-result winner=" << best->index
                   << " schedule=" << best->schedule
                   << " valid=" << validCandidates
                   << " unique_outputs=" << uniqueOutputs.size()
                   << " two_qubit_depth=" << twoQubitDepth
                   << " two_qubit=" << twoQubitGates << " depth=" << depth
                   << " gates=" << gates
                   << " total_compile_us=" << totalTime.count() << '\n';
    }
    module->setAttrs(best->module.get()->getAttrs());
    module.getBodyRegion().takeBody(best->module->getBodyRegion());
    return ::mlir::success();
  }

  [[nodiscard]] ::mlir::LogicalResult
  runCorePipeline(const ::mlir::ModuleOp module, const Target& target) {
    ::mlir::OpPassManager pipeline(::mlir::ModuleOp::getOperationName());
    ::mlir::populateTargetCompilationPipeline(pipeline, target);
    return runPipeline(pipeline, module);
  }

  [[nodiscard]] ::mlir::LogicalResult
  runOptimizationSchedule(const ::mlir::ModuleOp module, const Target& target,
                          const std::vector<ExhaustiveAction>& schedule) {
    ::mlir::OpPassManager pipeline(::mlir::ModuleOp::getOperationName());
    ::populateQCOCleanupPipeline(pipeline);
    ::populateDecomposeMultiControlledPipeline(pipeline, 3);
    for (const auto action : schedule) {
      switch (action) {
      case ExhaustiveAction::MergeRotations:
        pipeline.addPass(::mlir::qco::createMergeSingleQubitRotationGates());
        break;
      case ExhaustiveAction::FuseSingleQubit: {
        ::mlir::qco::FuseSingleQubitUnitaryRunsOptions options;
        options.basis = "u";
        pipeline.addPass(
            ::mlir::qco::createFuseSingleQubitUnitaryRuns(options));
        break;
      }
      case ExhaustiveAction::FuseTwoQubit:
        pipeline.addPass(::mlir::qco::createFuseTwoQubitGates());
        break;
      }
    }
    ::populateQCOCleanupPipeline(pipeline);
    pipeline.addPass(::mlir::qco::createMappingPass(target, {}));
    populateTargetFinalizationPipeline(pipeline, target);
    pipeline.addPass(::mlir::qco::createVerifyTargetConformance(target));
    return runPipeline(pipeline, module);
  }

  [[nodiscard]] ::mlir::LogicalResult
  runBootstrap(const ::mlir::ModuleOp module, const Target& target,
               const std::string_view targetFingerprint,
               const PolicyModel* model) {
    ::mlir::OpPassManager preparation(::mlir::ModuleOp::getOperationName());
    ::populateQCOCleanupPipeline(preparation);
    ::populateDecomposeMultiControlledPipeline(preparation, 3);
    if (failed(runPipeline(preparation, module))) {
      return ::mlir::failure();
    }
    const auto initialAnalysis = analyzeCircuit(module, target);
    if (failed(initialAnalysis)) {
      return ::mlir::failure();
    }
    if (model != nullptr && !initialAnalysis->fullyUnmapped) {
      if (options_.trace) {
        llvm::errs() << "[mqt-predictor] model policy requires an initially "
                        "fully unmapped program\n";
      }
      return ::mlir::failure();
    }
    const auto logicalQubits = initialAnalysis->features.numQubits;
    CompilerState policyState{};

    BootstrapLinearPolicy policy;
    std::mt19937_64 generator;
    if (model != nullptr && !options_.deterministicPolicy &&
        options_.samplingSeed) {
      generator.seed(*options_.samplingSeed);
    } else if (model != nullptr && !options_.deterministicPolicy) {
      std::random_device entropy;
      std::seed_seq seed{entropy(), entropy(), entropy(), entropy()};
      generator.seed(seed);
    }
    for (std::size_t step = 0; step < options_.maxSteps; ++step) {
      auto analysis = analyzeCircuit(module, target, logicalQubits);
      if (failed(analysis)) {
        if (options_.trace) {
          llvm::errs() << "[mqt-predictor] unsupported QCO structure at step "
                       << step << '\n';
        }
        return ::mlir::failure();
      }
      if (model != nullptr) {
        analysis->state = policyState;
      }

      const auto moduleBefore = moduleFingerprint(module);
      const auto legal = legalActions(analysis->state);
      const auto decision =
          model != nullptr
              ? options_.deterministicPolicy
                    ? model->select(analysis->features.values, legal)
                    : model->sample(analysis->features.values, legal, generator)
              : policy.select(analysis->features.values, legal);
      if (!decision) {
        return ::mlir::failure();
      }
      if (decision->action == Action::Terminate) {
        traceDecision(step, *analysis, *decision, target, targetFingerprint,
                      legal, model);
        ::mlir::OpPassManager verification(
            ::mlir::ModuleOp::getOperationName());
        verification.addPass(
            ::mlir::qco::createVerifyTargetConformance(target));
        return runPipeline(verification, module);
      }
      traceDecision(step, *analysis, *decision, target, targetFingerprint,
                    legal, model);

      if (failed(runAction(module, decision->action, target))) {
        return ::mlir::failure();
      }
      const auto changed = moduleBefore != moduleFingerprint(module);
      if (model != nullptr) {
        if (isOptimizationAction(decision->action) && changed) {
          policyState.synthesized = false;
        } else if (decision->action == Action::PlaceAndRoute) {
          policyState.mapped = true;
          policyState.routed = true;
          policyState.synthesized = false;
        } else if (decision->action == Action::SynthesizeForTarget) {
          policyState.synthesized = true;
        }
      }
      if (!changed) {
        if (options_.trace) {
          llvm::errs() << "[mqt-predictor] no-effect action="
                       << actionName(decision->action) << '\n';
        }
      }
      if (step + 1 == options_.maxSteps) {
        if (options_.trace) {
          llvm::errs() << "[mqt-predictor] "
                       << (model == nullptr ? "bootstrap" : "model")
                       << " episode truncated after " << options_.maxSteps
                       << " policy decisions\n";
        }
        return ::mlir::failure();
      }
    }
    return ::mlir::failure();
  }

  [[nodiscard]] ::mlir::LogicalResult runAction(const ::mlir::ModuleOp module,
                                                const Action action,
                                                const Target& target) {
    ::mlir::OpPassManager pipeline(::mlir::ModuleOp::getOperationName());
    if (failed(populateActionPipeline(pipeline, action, target))) {
      return ::mlir::failure();
    }
    return runPipeline(pipeline, module);
  }

  [[nodiscard]] static ::mlir::LogicalResult
  populateActionPipeline(::mlir::OpPassManager& pipeline, const Action action,
                         const Target& target) {
    switch (action) {
    case Action::MergeSingleQubitRotationGates:
      pipeline.addPass(::mlir::qco::createMergeSingleQubitRotationGates());
      break;
    case Action::FuseSingleQubitUnitaryRuns: {
      ::mlir::qco::FuseSingleQubitUnitaryRunsOptions options;
      options.basis = "u";
      pipeline.addPass(::mlir::qco::createFuseSingleQubitUnitaryRuns(options));
      break;
    }
    case Action::FuseTwoQubitGates:
      pipeline.addPass(::mlir::qco::createFuseTwoQubitGates());
      break;
    case Action::PlaceAndRoute:
      ::populateQCOCleanupPipeline(pipeline);
      pipeline.addPass(::mlir::qco::createMappingPass(target, {}));
      break;
    case Action::SynthesizeForTarget:
      populateTargetFinalizationPipeline(pipeline, target);
      break;
    case Action::Terminate:
    case Action::Count:
      return ::mlir::failure();
    }
    return ::mlir::success();
  }

  void traceDecision(const std::size_t step, const CircuitAnalysis& analysis,
                     const Decision& decision, const Target& target,
                     const std::string_view targetFingerprint,
                     const ActionMask& legal, const PolicyModel* model) const {
    if (!options_.trace) {
      return;
    }
    if (step == 0) {
      llvm::errs() << "[mqt-predictor] schema=" << EXPERIMENT_SCHEMA
                   << " core_pin=" << MQT_PREDICTOR_CORE_REVISION
                   << " target=" << target.name().value_or("unnamed")
                   << " target_fingerprint=" << targetFingerprint
                   << " policy=" << (model == nullptr ? "bootstrap" : "model");
      if (model == nullptr) {
        llvm::errs() << " objective=none\n";
      } else {
        llvm::errs() << " model_schema=" << model->schema()
                     << " artifact_id=" << model->artifactId();
        if (!model->parametersChecksum().empty()) {
          llvm::errs() << " parameters_sha256=" << model->parametersChecksum();
        }
        llvm::errs() << " training=" << model->trainingAlgorithm()
                     << " objective=" << model->objective() << " sampling="
                     << (options_.deterministicPolicy ? "argmax"
                                                      : "stochastic");
        if (!options_.deterministicPolicy) {
          if (options_.samplingSeed) {
            llvm::errs() << " sampling_seed=" << *options_.samplingSeed;
          } else {
            llvm::errs() << " sampling_seed=entropy";
          }
        }
        llvm::errs() << '\n';
      }
    }
    llvm::errs() << "[mqt-predictor] step=" << step
                 << " action=" << actionName(decision.action)
                 << " qubits=" << analysis.features.numQubits
                 << " depth=" << analysis.features.depth
                 << " two_qubit_depth=" << analysis.features.twoQubitDepth
                 << " gates=" << analysis.features.numGates
                 << " two_qubit=" << analysis.features.numTwoQubitGates
                 << " mapped=" << analysis.state.mapped
                 << " routed=" << analysis.state.routed
                 << " synthesized=" << analysis.state.synthesized << " legal=";
    for (const auto enabled : legal) {
      llvm::errs() << static_cast<unsigned>(enabled);
    }
    llvm::errs() << " features={";
    for (std::size_t index = 0; index < FEATURE_NAMES.size(); ++index) {
      if (index != 0) {
        llvm::errs() << ',';
      }
      llvm::errs() << FEATURE_NAMES[index] << '='
                   << analysis.features.values[index];
    }
    llvm::errs() << "}\n";
  }

  PredictorOptions options_;
};

} // namespace

std::unique_ptr<::mlir::Pass> createPredictorPass(PredictorOptions options) {
  return std::make_unique<PredictorPass>(options);
}

void registerPredictorPass() {
  static const ::mlir::PassRegistration<PredictorPass> registration;
  static_cast<void>(registration);
}

} // namespace mqt::predictor::compiler
