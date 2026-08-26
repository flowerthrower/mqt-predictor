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

struct ExhaustiveCandidate {
  ::mlir::OwningOpRef<::mlir::ModuleOp> module;
  CandidateScore score;
  std::string schedule;
  std::size_t index = 0;
};

[[nodiscard]] constexpr std::size_t actionIndex(const Action action) {
  return static_cast<std::size_t>(action);
}

[[nodiscard]] std::string moduleFingerprint(::mlir::ModuleOp module) {
  std::string fingerprint;
  llvm::raw_string_ostream stream(fingerprint);
  module.print(stream);
  return fingerprint;
}

[[nodiscard]] std::vector<std::vector<Action>>
enumerateOptimizationSchedules() {
  constexpr std::array optimizations{
      Action::MergeRotations, Action::FuseSingleQubit, Action::FuseTwoQubit};
  std::vector<std::vector<Action>> schedules;
  for (std::size_t mask = 0; mask < (1U << optimizations.size()); ++mask) {
    std::vector<Action> selected;
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

[[nodiscard]] std::string scheduleName(const std::vector<Action>& schedule) {
  if (schedule.empty()) {
    return "none";
  }
  std::string result;
  for (const auto action : schedule) {
    if (!result.empty()) {
      result += '>';
    }
    result += actionName(action);
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
    auto loadedTarget = options_.targetPath.empty()
                            ? createLineTarget(options_.targetQubits)
                            : loadCompilerTarget(options_.targetPath);
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

    std::optional<LinearPolicyModel> model;
    if (options_.policy == PolicyMode::Model) {
      auto loadedModel =
          LinearPolicyModel::load(options_.modelPath, loadedTarget->fingerprint,
                                  MQT_PREDICTOR_CORE_REVISION);
      if (!loadedModel) {
        getOperation().emitError() << llvm::toString(loadedModel.takeError());
        signalPassFailure();
        return;
      }
      model.emplace(std::move(*loadedModel));
    }

    auto backup = ::mlir::OwningOpRef<::mlir::ModuleOp>(
        ::mlir::cast<::mlir::ModuleOp>(getOperation()->clone()));
    const auto result =
        options_.policy == PolicyMode::Exhaustive
            ? runExhaustive(getOperation(), target, loadedTarget->fingerprint)
            : runBootstrap(getOperation(), target, loadedTarget->fingerprint,
                           model ? &*model : nullptr);
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
                              const std::vector<Action>* schedule) {
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
                          const std::vector<Action>& schedule) {
    ::mlir::OpPassManager pipeline(::mlir::ModuleOp::getOperationName());
    ::populateQCOCleanupPipeline(pipeline);
    ::populateDecomposeMultiControlledPipeline(pipeline, 3);
    for (const auto action : schedule) {
      if (failed(populateActionPipeline(pipeline, target, action))) {
        return ::mlir::failure();
      }
    }
    if (failed(
            populateActionPipeline(pipeline, target, Action::PlaceAndRoute)) ||
        failed(populateActionPipeline(pipeline, target,
                                      Action::NativeSynthesis))) {
      return ::mlir::failure();
    }
    pipeline.addPass(::mlir::qco::createVerifyTargetConformance(target));
    return runPipeline(pipeline, module);
  }

  [[nodiscard]] ::mlir::LogicalResult
  runBootstrap(const ::mlir::ModuleOp module, const Target& target,
               const std::string_view targetFingerprint,
               const LinearPolicyModel* model) {
    BootstrapLinearPolicy policy;
    std::set<std::pair<std::string, std::size_t>> attemptedTransitions;
    bool needsFinalization = false;
    for (std::size_t step = 0; step < options_.maxSteps; ++step) {
      const auto analysis = analyzeCircuit(module, target);
      if (failed(analysis)) {
        if (options_.trace) {
          llvm::errs() << "[mqt-predictor] unsupported QCO structure at step "
                       << step << '\n';
        }
        return ::mlir::failure();
      }

      const auto fingerprint = moduleFingerprint(module);
      ActionMask suppressed{};
      for (std::size_t action = 0; action < suppressed.size(); ++action) {
        suppressed[action] =
            attemptedTransitions.contains({fingerprint, action});
      }
      const auto legal = legalActions(analysis->state, suppressed);
      const auto decision =
          model != nullptr ? model->select(analysis->features.values, legal)
                           : policy.select(analysis->features.values, legal);
      if (!decision) {
        return ::mlir::failure();
      }
      traceDecision(step, *analysis, *decision, target, targetFingerprint,
                    model);

      if (decision->action == Action::Terminate) {
        if (needsFinalization) {
          return finalizeForTarget(module, target);
        }
        ::mlir::OpPassManager finish(::mlir::ModuleOp::getOperationName());
        finish.addPass(::mlir::qco::createVerifyTargetConformance(target));
        return runPipeline(finish, module);
      }

      if (failed(runAction(module, target, decision->action))) {
        return ::mlir::failure();
      }
      attemptedTransitions.emplace(fingerprint, actionIndex(decision->action));
      const auto changed = fingerprint != moduleFingerprint(module);
      if (decision->action == Action::NativeSynthesis) {
        needsFinalization = false;
      } else if (changed) {
        needsFinalization = true;
      }
      if (!changed) {
        if (options_.trace) {
          llvm::errs() << "[mqt-predictor] no-effect action="
                       << actionName(decision->action) << '\n';
        }
      }
    }
    return ::mlir::failure();
  }

  [[nodiscard]] ::mlir::LogicalResult
  finalizeForTarget(const ::mlir::ModuleOp module, const Target& target) {
    ::mlir::OpPassManager cleanup(::mlir::ModuleOp::getOperationName());
    ::populateQCOCleanupPipeline(cleanup);
    if (failed(runPipeline(cleanup, module))) {
      return ::mlir::failure();
    }

    const auto analysis = analyzeCircuit(module, target);
    if (failed(analysis)) {
      return ::mlir::failure();
    }

    ::mlir::OpPassManager finish(::mlir::ModuleOp::getOperationName());
    if (!analysis->state.synthesized) {
      finish.addPass(::mlir::qco::createTargetNativeSynthesis(target));
    }
    finish.addPass(::mlir::createCSEPass());
    finish.addPass(::mlir::createRemoveDeadValuesPass());
    finish.addPass(::mlir::qco::createVerifyTargetConformance(target));
    return runPipeline(finish, module);
  }

  [[nodiscard]] ::mlir::LogicalResult runAction(const ::mlir::ModuleOp module,
                                                const Target& target,
                                                const Action action) {
    ::mlir::OpPassManager pipeline(::mlir::ModuleOp::getOperationName());
    if (failed(populateActionPipeline(pipeline, target, action))) {
      return ::mlir::failure();
    }
    return runPipeline(pipeline, module);
  }

  [[nodiscard]] static ::mlir::LogicalResult
  populateActionPipeline(::mlir::OpPassManager& pipeline, const Target& target,
                         const Action action) {
    switch (action) {
    case Action::MergeRotations:
      pipeline.addPass(::mlir::qco::createMergeSingleQubitRotationGates());
      break;
    case Action::FuseSingleQubit: {
      ::mlir::qco::FuseSingleQubitUnitaryRunsOptions options;
      options.basis = "u";
      pipeline.addPass(::mlir::qco::createFuseSingleQubitUnitaryRuns(options));
      break;
    }
    case Action::FuseTwoQubit:
      pipeline.addPass(::mlir::qco::createFuseTwoQubitGates());
      break;
    case Action::PlaceAndRoute: {
      pipeline.addPass(::mlir::qco::createMappingPass(target, {}));
      break;
    }
    case Action::NativeSynthesis:
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
                     const LinearPolicyModel* model) const {
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
        llvm::errs() << " model_schema=" << NATIVE_POLICY_SCHEMA
                     << " artifact_id=" << model->artifactId()
                     << " parameters_sha256=" << model->parametersChecksum()
                     << " training=" << model->trainingAlgorithm()
                     << " objective=" << model->objective() << '\n';
      }
    }
    llvm::errs() << "[mqt-predictor] step=" << step
                 << " action=" << actionName(decision.action)
                 << " qubits=" << analysis.features.numQubits
                 << " depth=" << analysis.features.depth
                 << " gates=" << analysis.features.numGates
                 << " two_qubit=" << analysis.features.numTwoQubitGates
                 << " mapped=" << analysis.state.mapped
                 << " routed=" << analysis.state.routed
                 << " synthesized=" << analysis.state.synthesized
                 << " features={";
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
