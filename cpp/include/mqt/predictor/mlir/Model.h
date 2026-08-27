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

#include <llvm/Support/Error.h>

#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>

namespace mqt::predictor::compiler {

inline constexpr std::string_view NATIVE_POLICY_SCHEMA =
    "mqt-predictor-native-policy/1";
inline constexpr std::string_view ONNX_POLICY_SCHEMA =
    "mqt-predictor-onnx-policy/1";

/** A validated actor used by the compiled predictor. */
class PolicyModel {
public:
  virtual ~PolicyModel() = default;

  [[nodiscard]] std::optional<Decision> select(const FeatureVector& features,
                                               const ActionMask& legal) const;

  [[nodiscard]] std::optional<Decision> sample(const FeatureVector& features,
                                               const ActionMask& legal,
                                               std::mt19937_64& generator,
                                               float temperature = 1.0F) const;

  [[nodiscard]] virtual std::string_view schema() const noexcept = 0;
  [[nodiscard]] virtual std::string_view
  parametersChecksum() const noexcept = 0;
  [[nodiscard]] virtual std::string_view artifactId() const noexcept = 0;
  [[nodiscard]] virtual std::string_view objective() const noexcept = 0;
  [[nodiscard]] virtual std::string_view trainingAlgorithm() const noexcept = 0;

protected:
  [[nodiscard]] virtual std::optional<Decision>
  evaluate(const FeatureVector& features, const ActionMask& legal) const = 0;
};

/** A validated, target-specific linear actor loaded from a JSON artifact. */
class LinearPolicyModel final : public PolicyModel {
public:
  [[nodiscard]] static llvm::Expected<LinearPolicyModel>
  load(const std::filesystem::path& path,
       std::string_view expectedTargetFingerprint,
       std::string_view expectedCoreRevision);

  [[nodiscard]] std::string_view schema() const noexcept final;
  [[nodiscard]] std::string_view parametersChecksum() const noexcept final;
  [[nodiscard]] std::string_view artifactId() const noexcept final;
  [[nodiscard]] std::string_view objective() const noexcept final;
  [[nodiscard]] std::string_view trainingAlgorithm() const noexcept final;

private:
  using WeightMatrix = std::array<std::array<float, NUM_FEATURES>, NUM_ACTIONS>;

  LinearPolicyModel(WeightMatrix weights, std::array<float, NUM_ACTIONS> biases,
                    std::string parametersChecksum, std::string artifactId,
                    std::string objective, std::string trainingAlgorithm);

  [[nodiscard]] std::optional<Decision>
  evaluate(const FeatureVector& features, const ActionMask& legal) const final;

  WeightMatrix weights_{};
  std::array<float, NUM_ACTIONS> biases_{};
  std::string parametersChecksum_;
  std::string artifactId_;
  std::string objective_;
  std::string trainingAlgorithm_;
};

/** Load a JSON linear actor or, when enabled, an ONNX actor. */
[[nodiscard]] llvm::Expected<std::unique_ptr<PolicyModel>>
loadPolicyModel(const std::filesystem::path& path,
                std::string_view expectedTargetFingerprint,
                std::string_view expectedCoreRevision);

} // namespace mqt::predictor::compiler
