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
    "mqt-predictor-core-stages/3";
inline constexpr std::size_t NUM_FEATURES = 50;
inline constexpr std::array<std::string_view, NUM_FEATURES> FEATURE_NAMES{
    "c3sqrtx",
    "c3x",
    "c4x",
    "ccx",
    "ch",
    "cp",
    "critical_depth",
    "crx",
    "cry",
    "crz",
    "cswap",
    "csx",
    "cu",
    "cu1",
    "cu3",
    "cx",
    "cy",
    "cz",
    "depth",
    "entanglement_ratio",
    "h",
    "id",
    "liveness",
    "measure",
    "num_qubits",
    "p",
    "parallelism",
    "program_communication",
    "rc3x",
    "rccx",
    "rx",
    "rxx",
    "ry",
    "rz",
    "rzz",
    "s",
    "sdg",
    "swap",
    "sx",
    "sxdg",
    "t",
    "tdg",
    "u",
    "u0",
    "u1",
    "u2",
    "u3",
    "x",
    "y",
    "z"};
inline constexpr double DEPTH_NORMALIZATION_MAX = 999'999.0;
inline constexpr std::size_t MAX_STEPS = 20;

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
 * Optimization actions are available in every phase. The state supplied by the
 * caller controls whether placement, target synthesis, and termination are
 * available. Termination still requires a separate target-conformance check.
 */
[[nodiscard]] ActionMask legalActions(const CompilerState& state);

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
