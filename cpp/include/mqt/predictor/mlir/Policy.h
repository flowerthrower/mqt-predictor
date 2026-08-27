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

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace mqt::predictor::compiler {

inline constexpr std::string_view EXPERIMENT_SCHEMA =
    "mqt-predictor-core-passes/1";
inline constexpr std::size_t NUM_FEATURES = 12;
inline constexpr std::array<std::string_view, NUM_FEATURES> FEATURE_NAMES{
    "relative_qubits",
    "log_depth",
    "program_communication",
    "critical_depth",
    "entanglement_ratio",
    "parallelism",
    "liveness",
    "step_fraction",
    "merge-single-qubit-rotation-gates_count",
    "fuse-single-qubit-unitary-runs_count",
    "decompose-multi-controlled_count",
    "hadamard-lifting_count"};
inline constexpr double DEPTH_NORMALIZATION_MAX = 1'000'000.0;
inline constexpr std::size_t MAX_TRANSFORM_PASSES = 100;

using FeatureVector = std::array<float, NUM_FEATURES>;

enum class Action : std::uint8_t {
  MergeSingleQubitRotationGates,
  FuseSingleQubitUnitaryRuns,
  DecomposeMultiControlled,
  HadamardLifting,
  Terminate,
  Count,
};

inline constexpr std::size_t NUM_ACTIONS =
    static_cast<std::size_t>(Action::Count);
inline constexpr std::array<std::string_view, NUM_ACTIONS> ACTION_NAMES{
    "merge-single-qubit-rotation-gates", "fuse-single-qubit-unitary-runs",
    "decompose-multi-controlled", "hadamard-lifting", "terminate"};
using ActionMask = std::array<bool, NUM_ACTIONS>;

struct CompilerState {
  bool mapped = false;
  bool routed = false;
  bool synthesized = false;
  bool hasWideUnitary = false;
};

struct Decision {
  Action action;
  std::array<float, NUM_ACTIONS> logits{};
};

[[nodiscard]] std::string_view actionName(Action action);

/**
 * Return the legal action mask for the Core-only pass-ordering experiment.
 *
 * Fusion is ineligible while the circuit contains a unitary on more than two
 * qubits. Other actions are eligible unless suppressed by the caller.
 */
[[nodiscard]] ActionMask legalActions(const CompilerState& state,
                                      const ActionMask& suppressed);

/**
 * A dependency-free actor used to exercise the compiled policy boundary.
 *
 * The coefficients are deliberately simple bootstrap values, not trained model
 * weights. The matching artifact path uses LinearPolicyModel instead.
 */
class BootstrapLinearPolicy final {
public:
  [[nodiscard]] std::optional<Decision> select(const FeatureVector& features,
                                               const ActionMask& legal) const;
};

} // namespace mqt::predictor::compiler
