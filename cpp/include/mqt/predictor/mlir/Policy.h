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
    "mqt-predictor-core-stages/1";
inline constexpr std::size_t NUM_FEATURES = 13;
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
    "fuse-two-qubit-gates_count",
    "place-and-route_count",
    "synthesize-for-target_count"};
inline constexpr double DEPTH_NORMALIZATION_MAX = 1'000'000.0;
inline constexpr std::size_t MAX_TRANSFORM_PASSES = 100;

using FeatureVector = std::array<float, NUM_FEATURES>;

enum class Action : std::uint8_t {
  MergeSingleQubitRotationGates,
  FuseSingleQubitUnitaryRuns,
  FuseTwoQubitGates,
  PlaceAndRoute,
  SynthesizeForTarget,
  Terminate,
  Count,
};

inline constexpr std::size_t NUM_ACTIONS =
    static_cast<std::size_t>(Action::Count);
inline constexpr std::array<std::string_view, NUM_ACTIONS> ACTION_NAMES{
    "merge-single-qubit-rotation-gates",
    "fuse-single-qubit-unitary-runs",
    "fuse-two-qubit-gates",
    "place-and-route",
    "synthesize-for-target",
    "terminate"};
using ActionMask = std::array<bool, NUM_ACTIONS>;

[[nodiscard]] constexpr bool isOptimizationAction(const Action action) {
  return action < Action::PlaceAndRoute;
}

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
 * Optimization actions are available in every phase. Placement and routing can
 * only run before mapping, target synthesis can only run while non-native
 * operations remain, and termination requires full target conformance.
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
