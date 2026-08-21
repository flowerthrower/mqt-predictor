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
    "mqt-predictor-bootstrap/1";
inline constexpr std::size_t NUM_FEATURES = 7;
inline constexpr std::array<std::string_view, NUM_FEATURES> FEATURE_NAMES{
    "relative_qubits", "log_depth",          "program_communication",
    "critical_depth",  "entanglement_ratio", "parallelism",
    "liveness"};
inline constexpr double DEPTH_NORMALIZATION_MAX = 1'000'000.0;

using FeatureVector = std::array<float, NUM_FEATURES>;

enum class Action : std::uint8_t {
  MergeRotations,
  FuseSingleQubit,
  FuseTwoQubit,
  PlaceAndRoute,
  NativeSynthesis,
  Terminate,
  Count,
};

inline constexpr std::size_t NUM_ACTIONS =
    static_cast<std::size_t>(Action::Count);
using ActionMask = std::array<bool, NUM_ACTIONS>;

struct CompilerState {
  bool mapped = false;
  bool routed = false;
  bool synthesized = false;
};

struct Decision {
  Action action;
  std::array<float, NUM_ACTIONS> logits{};
};

[[nodiscard]] std::string_view actionName(Action action);

/**
 * Return the legal action mask for the bootstrap experiment.
 *
 * The MLIR mapping pass performs placement and routing atomically.
 * Consequently, this experiment collapses Predictor v3's separate layout and
 * routing stages into one placed-and-routed state.
 */
[[nodiscard]] ActionMask legalActions(const CompilerState& state,
                                      const ActionMask& suppressed);

/**
 * A dependency-free actor used to exercise the compiled policy boundary.
 *
 * The coefficients are deliberately simple bootstrap values, not trained model
 * weights. The interface and schema are experimental and require matching
 * training/export code before a learned actor can replace this bootstrap.
 */
class BootstrapLinearPolicy final {
public:
  [[nodiscard]] std::optional<Decision> select(const FeatureVector& features,
                                               const ActionMask& legal) const;
};

} // namespace mqt::predictor::compiler
