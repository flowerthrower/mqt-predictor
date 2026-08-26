/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/predictor/mlir/Model.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SHA256.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace mqt::predictor::compiler {
namespace {

constexpr std::size_t MAX_MODEL_BYTES = 1024U * 1024U;

[[nodiscard]] llvm::Error modelError(const std::filesystem::path& path,
                                     const llvm::StringRef message) {
  return llvm::createStringError("invalid native policy model '%s': %s",
                                 path.string().c_str(), message.str().c_str());
}

[[nodiscard]] bool
hasOnlyFields(const llvm::json::Object& object,
              const std::initializer_list<llvm::StringRef> allowed) {
  return std::all_of(object.begin(), object.end(), [&](const auto& item) {
    const llvm::StringRef key(item.first);
    return std::find(allowed.begin(), allowed.end(), key) != allowed.end();
  });
}

template <std::size_t Size>
[[nodiscard]] bool
validateStringArray(const llvm::json::Array* values,
                    const std::array<std::string_view, Size>& expected) {
  if (values == nullptr || values->size() != expected.size()) {
    return false;
  }
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const auto value = (*values)[index].getAsString();
    if (!value || *value != llvm::StringRef(expected[index])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::optional<float>
readFiniteFloat(const llvm::json::Value& value) {
  const auto number = value.getAsNumber();
  if (!number || !std::isfinite(*number) ||
      *number < -std::numeric_limits<float>::max() ||
      *number > std::numeric_limits<float>::max()) {
    return std::nullopt;
  }
  return static_cast<float>(*number);
}

void appendFloatBytes(std::vector<std::uint8_t>& bytes, const float value) {
  const auto normalized = value == 0.0F ? 0.0F : value;
  const auto bits = std::bit_cast<std::uint32_t>(normalized);
  for (std::size_t byte = 0; byte < sizeof(bits); ++byte) {
    bytes.push_back(static_cast<std::uint8_t>(bits >> (byte * 8U)));
  }
}

template <typename WeightMatrix>
[[nodiscard]] std::string
parameterChecksum(const WeightMatrix& weights,
                  const std::array<float, NUM_ACTIONS>& biases) {
  std::vector<std::uint8_t> bytes;
  constexpr llvm::StringLiteral checksumDomain =
      "mqt-predictor-native-policy/1";
  bytes.insert(bytes.end(), checksumDomain.bytes_begin(),
               checksumDomain.bytes_end());
  bytes.push_back(0);
  const auto appendDimension = [&](const std::uint32_t value) {
    for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
      bytes.push_back(static_cast<std::uint8_t>(value >> (byte * 8U)));
    }
  };
  appendDimension(NUM_FEATURES);
  appendDimension(NUM_ACTIONS);
  for (const auto& row : weights) {
    for (const auto value : row) {
      appendFloatBytes(bytes, value);
    }
  }
  for (const auto value : biases) {
    appendFloatBytes(bytes, value);
  }
  return "sha256:" + llvm::toHex(llvm::SHA256::hash(bytes), true);
}

[[nodiscard]] std::string computeArtifactId(const llvm::StringRef contents) {
  const auto bytes = llvm::ArrayRef(
      reinterpret_cast<const std::uint8_t*>(contents.data()), contents.size());
  return "sha256:" + llvm::toHex(llvm::SHA256::hash(bytes), true);
}

} // namespace

LinearPolicyModel::LinearPolicyModel(WeightMatrix weights,
                                     std::array<float, NUM_ACTIONS> biases,
                                     std::string parametersChecksum,
                                     std::string artifactId,
                                     std::string objective,
                                     std::string trainingAlgorithm)
    : weights_(std::move(weights)), biases_(biases),
      parametersChecksum_(std::move(parametersChecksum)),
      artifactId_(std::move(artifactId)), objective_(std::move(objective)),
      trainingAlgorithm_(std::move(trainingAlgorithm)) {}

llvm::Expected<LinearPolicyModel>
LinearPolicyModel::load(const std::filesystem::path& path,
                        const std::string_view expectedTargetFingerprint,
                        const std::string_view expectedCoreRevision) {
  const auto buffer = llvm::MemoryBuffer::getFile(path.string());
  if (!buffer) {
    return llvm::createStringError(buffer.getError(),
                                   "failed to read native policy model '%s'",
                                   path.string().c_str());
  }
  if ((*buffer)->getBufferSize() > MAX_MODEL_BYTES) {
    return modelError(path, "artifact exceeds the 1 MiB size limit");
  }
  const auto loadedArtifactId = computeArtifactId((*buffer)->getBuffer());
  auto parsed = llvm::json::parse((*buffer)->getBuffer());
  if (!parsed) {
    const auto detail = llvm::toString(parsed.takeError());
    return modelError(path, detail);
  }
  const auto* root = parsed->getAsObject();
  if (root == nullptr ||
      !hasOnlyFields(*root,
                     {"schema", "observation_schema", "feature_names",
                      "action_names", "architecture", "parameters",
                      "parameters_sha256", "compatibility", "training"})) {
    return modelError(path, "unexpected or missing top-level object");
  }

  const auto schema = root->getString("schema");
  if (!schema || *schema != llvm::StringRef(NATIVE_POLICY_SCHEMA)) {
    return modelError(path, "unsupported schema");
  }
  const auto observationSchema = root->getString("observation_schema");
  if (!observationSchema ||
      *observationSchema != llvm::StringRef(EXPERIMENT_SCHEMA)) {
    return modelError(path, "observation schema does not match this binary");
  }
  if (!validateStringArray(root->getArray("feature_names"), FEATURE_NAMES)) {
    return modelError(path, "feature names or order do not match this binary");
  }
  if (!validateStringArray(root->getArray("action_names"), ACTION_NAMES)) {
    return modelError(path, "action names or order do not match this binary");
  }

  const auto* architecture = root->getObject("architecture");
  if (architecture == nullptr ||
      !hasOnlyFields(*architecture, {"type", "input_size", "output_size"}) ||
      architecture->getString("type") != "linear" ||
      architecture->getInteger("input_size") !=
          static_cast<std::int64_t>(NUM_FEATURES) ||
      architecture->getInteger("output_size") !=
          static_cast<std::int64_t>(NUM_ACTIONS)) {
    return modelError(path,
                      "only a 7-input, 6-output linear actor is supported");
  }

  const auto* compatibility = root->getObject("compatibility");
  if (compatibility == nullptr ||
      !hasOnlyFields(*compatibility, {"target_fingerprint", "core_revision"})) {
    return modelError(path, "missing compatibility metadata");
  }
  const auto targetFingerprint = compatibility->getString("target_fingerprint");
  if (!targetFingerprint ||
      *targetFingerprint != llvm::StringRef(expectedTargetFingerprint)) {
    return modelError(path, "compiler target fingerprint mismatch");
  }
  const auto coreRevision = compatibility->getString("core_revision");
  if (!coreRevision || *coreRevision != llvm::StringRef(expectedCoreRevision)) {
    return modelError(path, "MQT Core revision mismatch");
  }

  const auto* training = root->getObject("training");
  if (training == nullptr ||
      !hasOnlyFields(*training,
                     {"algorithm", "objective", "source_revision", "samples",
                      "epochs", "learning_rate", "l2", "seed"})) {
    return modelError(path, "missing or unexpected training metadata");
  }
  const auto algorithm = training->getString("algorithm");
  const auto objective = training->getString("objective");
  const auto sourceRevision = training->getString("source_revision");
  const auto samples = training->getInteger("samples");
  const auto epochs = training->getInteger("epochs");
  const auto learningRate = training->getNumber("learning_rate");
  const auto l2 = training->getNumber("l2");
  const auto seed = training->getInteger("seed");
  if (!algorithm || algorithm->empty() || !objective || objective->empty() ||
      !sourceRevision || sourceRevision->empty() || !samples || *samples <= 0 ||
      !epochs || *epochs <= 0 || !learningRate ||
      !std::isfinite(*learningRate) || *learningRate <= 0.0 || !l2 ||
      !std::isfinite(*l2) || *l2 < 0.0 || !seed || *seed < 0) {
    return modelError(path, "invalid training provenance");
  }

  const auto* parameters = root->getObject("parameters");
  if (parameters == nullptr ||
      !hasOnlyFields(*parameters, {"weights", "bias"})) {
    return modelError(path, "missing linear parameters");
  }
  const auto* weightsJson = parameters->getArray("weights");
  const auto* biasesJson = parameters->getArray("bias");
  if (weightsJson == nullptr || weightsJson->size() != NUM_ACTIONS ||
      biasesJson == nullptr || biasesJson->size() != NUM_ACTIONS) {
    return modelError(path, "linear parameter dimensions do not match");
  }

  WeightMatrix weights{};
  std::array<float, NUM_ACTIONS> biases{};
  for (std::size_t action = 0; action < NUM_ACTIONS; ++action) {
    const auto* row = (*weightsJson)[action].getAsArray();
    if (row == nullptr || row->size() != NUM_FEATURES) {
      return modelError(path, "linear weight matrix dimensions do not match");
    }
    for (std::size_t feature = 0; feature < NUM_FEATURES; ++feature) {
      const auto value = readFiniteFloat((*row)[feature]);
      if (!value) {
        return modelError(path, "linear weights must be finite float values");
      }
      weights[action][feature] = *value;
    }
    const auto bias = readFiniteFloat((*biasesJson)[action]);
    if (!bias) {
      return modelError(path, "linear biases must be finite float values");
    }
    biases[action] = *bias;
  }

  const auto declaredChecksum = root->getString("parameters_sha256");
  const auto actualChecksum = parameterChecksum(weights, biases);
  if (!declaredChecksum || *declaredChecksum != actualChecksum) {
    return modelError(path, "linear parameter checksum mismatch");
  }

  return LinearPolicyModel(std::move(weights), biases, actualChecksum,
                           loadedArtifactId, objective->str(),
                           algorithm->str());
}

std::optional<Decision>
LinearPolicyModel::evaluate(const FeatureVector& features,
                            const ActionMask& legal) const {
  Decision decision{.action = Action::Terminate};
  decision.logits.fill(-std::numeric_limits<float>::infinity());

  bool hasLegalAction = false;
  for (std::size_t action = 0; action < NUM_ACTIONS; ++action) {
    if (!legal[action]) {
      continue;
    }
    hasLegalAction = true;
    double logit = biases_[action];
    for (std::size_t feature = 0; feature < NUM_FEATURES; ++feature) {
      if (!std::isfinite(features[feature]) || features[feature] < 0.0F ||
          features[feature] > 1.0F) {
        return std::nullopt;
      }
      logit += static_cast<double>(weights_[action][feature]) *
               static_cast<double>(features[feature]);
    }
    if (!std::isfinite(logit) || logit < -std::numeric_limits<float>::max() ||
        logit > std::numeric_limits<float>::max()) {
      return std::nullopt;
    }
    const auto storedLogit = static_cast<float>(logit);
    decision.logits[action] = storedLogit;
  }
  if (!hasLegalAction) {
    return std::nullopt;
  }
  return decision;
}

std::optional<Decision>
LinearPolicyModel::select(const FeatureVector& features,
                          const ActionMask& legal) const {
  auto decision = evaluate(features, legal);
  if (!decision) {
    return std::nullopt;
  }

  std::optional<std::size_t> selected;
  for (std::size_t action = 0; action < NUM_ACTIONS; ++action) {
    if (!legal[action]) {
      continue;
    }
    if (!selected || decision->logits[action] > decision->logits[*selected]) {
      selected = action;
    }
  }
  decision->action = static_cast<Action>(*selected);
  return decision;
}

std::optional<Decision>
LinearPolicyModel::sample(const FeatureVector& features,
                          const ActionMask& legal, std::mt19937_64& generator,
                          const float temperature) const {
  if (!std::isfinite(temperature) || temperature <= 0.0F) {
    return std::nullopt;
  }
  auto decision = evaluate(features, legal);
  if (!decision) {
    return std::nullopt;
  }

  const auto maximum =
      *std::max_element(decision->logits.begin(), decision->logits.end());
  std::array<double, NUM_ACTIONS> probabilities{};
  for (std::size_t action = 0; action < NUM_ACTIONS; ++action) {
    if (legal[action]) {
      probabilities[action] =
          std::exp((static_cast<double>(decision->logits[action]) - maximum) /
                   static_cast<double>(temperature));
    }
  }
  std::discrete_distribution<std::size_t> distribution(probabilities.begin(),
                                                       probabilities.end());
  decision->action = static_cast<Action>(distribution(generator));
  return decision;
}

std::string_view LinearPolicyModel::parametersChecksum() const noexcept {
  return parametersChecksum_;
}

std::string_view LinearPolicyModel::artifactId() const noexcept {
  return artifactId_;
}

std::string_view LinearPolicyModel::objective() const noexcept {
  return objective_;
}

std::string_view LinearPolicyModel::trainingAlgorithm() const noexcept {
  return trainingAlgorithm_;
}

} // namespace mqt::predictor::compiler
