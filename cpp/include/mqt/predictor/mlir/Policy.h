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
#include <map>
#include <optional>
#include <string>
#include <string_view>

namespace mqt::predictor::compiler {

inline constexpr std::string_view EXPERIMENT_SCHEMA =
    "mqt-predictor-markov-stop-features/1";
inline constexpr std::size_t NUM_CIRCUIT_FEATURES = 51;
inline constexpr std::size_t NUM_FEATURES = 64;
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
    "r",
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
    "z",
    "zz_decision_fraction",
    "zz_incumbent_available",
    "zz_current_fidelity",
    "zz_incumbent_fidelity",
    "zz_incumbent_age_fraction",
    "zz_visible_visit_fraction",
    "zz_last_action_changed_ir",
    "zz_previous_action_0",
    "zz_previous_action_1",
    "zz_previous_action_2",
    "zz_previous_action_3",
    "zz_previous_action_4",
    "zz_previous_action_5"};
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

struct CompilerState {
  bool mapped = false;
  bool routed = false;
  bool synthesized = false;
  bool hasWideUnitary = false;
};

struct Decision {
  Action action;
  std::array<float, NUM_ACTIONS> logits{};
  std::array<double, NUM_ACTIONS> samplingNoise{};
};

/** The study's episode context and optional reactive-stop controller. */
class EpisodeContext final {
public:
  explicit EpisodeContext(bool reactiveStop = true)
      : reactiveStop_(reactiveStop) {}

  // Fill context before counting this visit. nullopt requests incumbent stop.
  [[nodiscard]] std::optional<ActionMask> observe(FeatureVector& features,
                                                  ActionMask legal);
  // Record the selected action and result; return whether the incumbent
  // improved.
  [[nodiscard]] bool recordResult(Action action, bool changed,
                                  std::optional<double> fidelity);
  [[nodiscard]] std::optional<double> incumbentFidelity() const {
    return incumbent_;
  }

private:
  struct Visit {
    std::size_t count = 0;
    std::size_t recurrences = 0;
    std::optional<Action> action;
    std::optional<double> best;
    std::array<std::size_t, NUM_ACTIONS> tabuBefore{};
  };
  bool reactiveStop_;
  std::size_t step_ = 0;
  std::size_t incumbentStep_ = 0;
  std::optional<double> current_;
  std::optional<double> incumbent_;
  std::optional<Action> previousAction_;
  bool changed_ = false;
  std::map<std::string, Visit> visits_;
  Visit* visit_ = nullptr;
};

[[nodiscard]] std::string_view actionName(Action action);

/**
 * Return the legal action mask for the Core-only pass-ordering experiment.
 *
 * The state controls the compilation-phase constraints. The suppression mask
 * excludes actions that were no-ops for the current IR. Termination still
 * requires a separate target-conformance check.
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
