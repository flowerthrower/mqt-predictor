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

#ifdef MQT_PREDICTOR_ENABLE_ONNX
#include <onnxruntime_c_api.h>
#endif

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
#include <memory>
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

std::optional<Decision> PolicyModel::select(const FeatureVector& features,
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
  if (!selected) {
    return std::nullopt;
  }
  decision->action = static_cast<Action>(*selected);
  return decision;
}

std::optional<Decision> PolicyModel::sample(const FeatureVector& features,
                                            const ActionMask& legal,
                                            std::mt19937_64& generator,
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
                      "linear actor dimensions do not match this binary");
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

std::string_view LinearPolicyModel::schema() const noexcept {
  return NATIVE_POLICY_SCHEMA;
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

#ifdef MQT_PREDICTOR_ENABLE_ONNX
namespace {

[[nodiscard]] llvm::Error onnxModelError(const std::filesystem::path& path,
                                         const llvm::StringRef message) {
  return llvm::createStringError("invalid ONNX policy model '%s': %s",
                                 path.string().c_str(), message.str().c_str());
}

[[nodiscard]] llvm::Error onnxStatusError(const OrtApi& api,
                                          const std::filesystem::path& path,
                                          const llvm::StringRef operation,
                                          OrtStatus* status) {
  const std::string detail(api.GetErrorMessage(status));
  api.ReleaseStatus(status);
  return onnxModelError(path, operation.str() + ": " + detail);
}

template <typename Value, typename Deleter>
using OrtOwner = std::unique_ptr<Value, Deleter>;

struct TypeInfoDeleter {
  const OrtApi* api;
  void operator()(OrtTypeInfo* value) const noexcept {
    api->ReleaseTypeInfo(value);
  }
};

struct ModelMetadataDeleter {
  const OrtApi* api;
  void operator()(OrtModelMetadata* value) const noexcept {
    api->ReleaseModelMetadata(value);
  }
};

struct ValueDeleter {
  const OrtApi* api;
  void operator()(OrtValue* value) const noexcept { api->ReleaseValue(value); }
};

struct TensorInfoDeleter {
  const OrtApi* api;
  void operator()(OrtTensorTypeAndShapeInfo* value) const noexcept {
    api->ReleaseTensorTypeAndShapeInfo(value);
  }
};

struct OrtState {
  explicit OrtState(const OrtApi* loadedApi) : api(loadedApi) {}
  ~OrtState() {
    if (session != nullptr) {
      api->ReleaseSession(session);
    }
    if (sessionOptions != nullptr) {
      api->ReleaseSessionOptions(sessionOptions);
    }
    if (memoryInfo != nullptr) {
      api->ReleaseMemoryInfo(memoryInfo);
    }
    if (environment != nullptr) {
      api->ReleaseEnv(environment);
    }
  }

  OrtState(const OrtState&) = delete;
  OrtState& operator=(const OrtState&) = delete;

  const OrtApi* api;
  OrtEnv* environment = nullptr;
  OrtSessionOptions* sessionOptions = nullptr;
  OrtSession* session = nullptr;
  OrtMemoryInfo* memoryInfo = nullptr;
};

template <std::size_t Size>
[[nodiscard]] std::string
joinNames(const std::array<std::string_view, Size>& names) {
  std::string joined;
  for (std::size_t index = 0; index < names.size(); ++index) {
    if (index != 0) {
      joined.push_back(',');
    }
    joined.append(names[index]);
  }
  return joined;
}

[[nodiscard]] llvm::Expected<std::optional<std::string>>
lookupMetadata(const OrtApi& api, const std::filesystem::path& path,
               const OrtModelMetadata& metadata, OrtAllocator& allocator,
               const char* key) {
  char* loadedValue = nullptr;
  if (auto* status = api.ModelMetadataLookupCustomMetadataMap(
          &metadata, &allocator, key, &loadedValue)) {
    return onnxStatusError(api, path, "failed to read metadata", status);
  }
  if (loadedValue == nullptr) {
    return std::nullopt;
  }
  std::string value(loadedValue);
  allocator.Free(&allocator, loadedValue);
  return std::optional<std::string>(std::move(value));
}

[[nodiscard]] llvm::Expected<std::string>
requiredMetadata(const OrtApi& api, const std::filesystem::path& path,
                 const OrtModelMetadata& metadata, OrtAllocator& allocator,
                 const char* key) {
  auto value = lookupMetadata(api, path, metadata, allocator, key);
  if (!value) {
    return value.takeError();
  }
  if (!*value || (*value)->empty()) {
    return onnxModelError(
        path,
        ("missing required metadata '" + llvm::StringRef(key) + "'").str());
  }
  return std::move(**value);
}

[[nodiscard]] bool isSha256(const std::string_view value) {
  constexpr std::string_view prefix = "sha256:";
  return value.starts_with(prefix) && value.size() == prefix.size() + 64U &&
         std::all_of(value.begin() + static_cast<std::ptrdiff_t>(prefix.size()),
                     value.end(), [](const char character) {
                       return llvm::isHexDigit(character);
                     });
}

[[nodiscard]] llvm::Error
validateTensorInterface(const OrtApi& api, const std::filesystem::path& path,
                        const OrtSession& session, const bool input,
                        OrtAllocator& allocator, const char* expectedName,
                        const std::int64_t expectedWidth) {
  char* loadedName = nullptr;
  auto* nameStatus =
      input ? api.SessionGetInputName(&session, 0, &allocator, &loadedName)
            : api.SessionGetOutputName(&session, 0, &allocator, &loadedName);
  if (nameStatus != nullptr) {
    return onnxStatusError(api, path, "failed to read tensor name", nameStatus);
  }
  const std::string name(loadedName == nullptr ? "" : loadedName);
  if (loadedName != nullptr) {
    allocator.Free(&allocator, loadedName);
  }
  if (name != expectedName) {
    return onnxModelError(path, input ? "input must be named 'features'"
                                      : "output must be named 'logits'");
  }

  OrtTypeInfo* loadedTypeInfo = nullptr;
  auto* typeStatus =
      input ? api.SessionGetInputTypeInfo(&session, 0, &loadedTypeInfo)
            : api.SessionGetOutputTypeInfo(&session, 0, &loadedTypeInfo);
  if (typeStatus != nullptr) {
    return onnxStatusError(api, path, "failed to read tensor type", typeStatus);
  }
  OrtOwner<OrtTypeInfo, TypeInfoDeleter> typeInfo(loadedTypeInfo,
                                                  TypeInfoDeleter{.api = &api});
  ONNXType onnxType = ONNX_TYPE_UNKNOWN;
  if (auto* status = api.GetOnnxTypeFromTypeInfo(typeInfo.get(), &onnxType)) {
    return onnxStatusError(api, path, "failed to read ONNX value type", status);
  }
  if (onnxType != ONNX_TYPE_TENSOR) {
    return onnxModelError(path, "input and output must be tensors");
  }

  const OrtTensorTypeAndShapeInfo* tensorInfo = nullptr;
  if (auto* status =
          api.CastTypeInfoToTensorInfo(typeInfo.get(), &tensorInfo)) {
    return onnxStatusError(api, path, "failed to read tensor shape", status);
  }
  if (tensorInfo == nullptr) {
    return onnxModelError(path, "input and output must be tensors");
  }
  ONNXTensorElementDataType elementType =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  if (auto* status = api.GetTensorElementType(tensorInfo, &elementType)) {
    return onnxStatusError(api, path, "failed to read tensor element type",
                           status);
  }
  if (elementType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return onnxModelError(path, "input and output must use float32");
  }
  std::size_t dimensionCount = 0;
  if (auto* status = api.GetDimensionsCount(tensorInfo, &dimensionCount)) {
    return onnxStatusError(api, path, "failed to read tensor rank", status);
  }
  if (dimensionCount != 2) {
    return onnxModelError(path, "input and output must have rank two");
  }
  std::array<std::int64_t, 2> dimensions{};
  if (auto* status =
          api.GetDimensions(tensorInfo, dimensions.data(), dimensions.size())) {
    return onnxStatusError(api, path, "failed to read tensor dimensions",
                           status);
  }
  if (dimensions != std::array<std::int64_t, 2>{1, expectedWidth}) {
    return onnxModelError(
        path, input ? "input shape does not match this binary"
                    : "output shape does not match this binary");
  }
  return llvm::Error::success();
}

class OnnxPolicyModel final : public PolicyModel {
public:
  [[nodiscard]] static llvm::Expected<std::unique_ptr<OnnxPolicyModel>>
  load(const std::filesystem::path& path,
       const std::string_view expectedTargetFingerprint,
       const std::string_view expectedCoreRevision) {
    const auto buffer = llvm::MemoryBuffer::getFile(path.string());
    if (!buffer) {
      return llvm::createStringError(buffer.getError(),
                                     "failed to read ONNX policy model '%s'",
                                     path.string().c_str());
    }
    if ((*buffer)->getBufferSize() > MAX_MODEL_BYTES) {
      return onnxModelError(path, "artifact exceeds the 1 MiB size limit");
    }
    const auto loadedArtifactId = computeArtifactId((*buffer)->getBuffer());

    const auto* apiBase = OrtGetApiBase();
    if (apiBase == nullptr) {
      return onnxModelError(path, "ONNX Runtime API is unavailable");
    }
    const auto* api = apiBase->GetApi(ORT_API_VERSION);
    if (api == nullptr) {
      return onnxModelError(path, "ONNX Runtime API version is incompatible");
    }
    auto state = std::make_unique<OrtState>(api);
    if (auto* status = api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "mqt-predictor",
                                      &state->environment)) {
      return onnxStatusError(*api, path, "failed to create environment",
                             status);
    }
    if (auto* status = api->DisableTelemetryEvents(state->environment)) {
      return onnxStatusError(*api, path, "failed to disable telemetry", status);
    }
    if (auto* status = api->CreateSessionOptions(&state->sessionOptions)) {
      return onnxStatusError(*api, path, "failed to create session options",
                             status);
    }
    if (auto* status = api->SetSessionExecutionMode(state->sessionOptions,
                                                    ORT_SEQUENTIAL)) {
      return onnxStatusError(*api, path, "failed to set sequential execution",
                             status);
    }
    if (auto* status = api->SetIntraOpNumThreads(state->sessionOptions, 1)) {
      return onnxStatusError(*api, path, "failed to set intra-op threads",
                             status);
    }
    if (auto* status = api->SetInterOpNumThreads(state->sessionOptions, 1)) {
      return onnxStatusError(*api, path, "failed to set inter-op threads",
                             status);
    }
    if (auto* status = api->SetSessionGraphOptimizationLevel(
            state->sessionOptions, ORT_ENABLE_ALL)) {
      return onnxStatusError(*api, path, "failed to enable graph optimizations",
                             status);
    }
    if (auto* status = api->CreateSessionFromArray(
            state->environment, (*buffer)->getBufferStart(),
            (*buffer)->getBufferSize(), state->sessionOptions,
            &state->session)) {
      return onnxStatusError(*api, path, "failed to load model", status);
    }
    api->ReleaseSessionOptions(state->sessionOptions);
    state->sessionOptions = nullptr;

    std::size_t inputCount = 0;
    std::size_t outputCount = 0;
    if (auto* status = api->SessionGetInputCount(state->session, &inputCount)) {
      return onnxStatusError(*api, path, "failed to read input count", status);
    }
    if (auto* status =
            api->SessionGetOutputCount(state->session, &outputCount)) {
      return onnxStatusError(*api, path, "failed to read output count", status);
    }
    if (inputCount != 1 || outputCount != 1) {
      return onnxModelError(path,
                            "actor must have exactly one input and one output");
    }

    OrtAllocator* allocator = nullptr;
    if (auto* status = api->GetAllocatorWithDefaultOptions(&allocator)) {
      return onnxStatusError(*api, path, "failed to get allocator", status);
    }
    if (auto error =
            validateTensorInterface(*api, path, *state->session, true,
                                    *allocator, "features", NUM_FEATURES)) {
      return std::move(error);
    }
    if (auto error =
            validateTensorInterface(*api, path, *state->session, false,
                                    *allocator, "logits", NUM_ACTIONS)) {
      return std::move(error);
    }

    OrtModelMetadata* loadedMetadata = nullptr;
    if (auto* status =
            api->SessionGetModelMetadata(state->session, &loadedMetadata)) {
      return onnxStatusError(*api, path, "failed to read model metadata",
                             status);
    }
    OrtOwner<OrtModelMetadata, ModelMetadataDeleter> metadata(
        loadedMetadata, ModelMetadataDeleter{.api = api});

    const auto joinedFeatureNames = joinNames(FEATURE_NAMES);
    const auto joinedActionNames = joinNames(ACTION_NAMES);
    struct MetadataRequirement {
      const char* key;
      std::string_view expected;
      const char* mismatch;
    };
    const std::array requirements{
        MetadataRequirement{"schema", ONNX_POLICY_SCHEMA, "unsupported schema"},
        MetadataRequirement{"observation_schema", EXPERIMENT_SCHEMA,
                            "observation schema does not match this binary"},
        MetadataRequirement{"feature_names", joinedFeatureNames,
                            "feature names or order do not match this binary"},
        MetadataRequirement{"action_names", joinedActionNames,
                            "action names or order do not match this binary"},
        MetadataRequirement{"target_fingerprint", expectedTargetFingerprint,
                            "compiler target fingerprint mismatch"},
        MetadataRequirement{"core_revision", expectedCoreRevision,
                            "MQT Core revision mismatch"},
    };
    for (const auto& requirement : requirements) {
      auto value =
          requiredMetadata(*api, path, *metadata, *allocator, requirement.key);
      if (!value) {
        return value.takeError();
      }
      if (*value != requirement.expected) {
        return onnxModelError(path, requirement.mismatch);
      }
    }
    auto trainingAlgorithm = requiredMetadata(*api, path, *metadata, *allocator,
                                              "training_algorithm");
    if (!trainingAlgorithm) {
      return trainingAlgorithm.takeError();
    }
    auto objective =
        requiredMetadata(*api, path, *metadata, *allocator, "objective");
    if (!objective) {
      return objective.takeError();
    }

    auto parametersChecksum =
        lookupMetadata(*api, path, *metadata, *allocator, "parameters_sha256");
    if (!parametersChecksum) {
      return parametersChecksum.takeError();
    }
    if (*parametersChecksum && !isSha256(**parametersChecksum)) {
      return onnxModelError(path,
                            "parameters_sha256 metadata is not a SHA-256 ID");
    }
    for (const auto* optionalKey : {"architecture", "source_revision"}) {
      auto value =
          lookupMetadata(*api, path, *metadata, *allocator, optionalKey);
      if (!value) {
        return value.takeError();
      }
      if (*value && (*value)->empty()) {
        return onnxModelError(
            path, (llvm::StringRef(optionalKey) + " metadata must not be empty")
                      .str());
      }
    }

    if (auto* status = api->CreateCpuMemoryInfo(
            OrtArenaAllocator, OrtMemTypeDefault, &state->memoryInfo)) {
      return onnxStatusError(*api, path, "failed to create CPU memory info",
                             status);
    }
    return std::unique_ptr<OnnxPolicyModel>(new OnnxPolicyModel(
        std::move(state), loadedArtifactId,
        *parametersChecksum ? std::move(**parametersChecksum) : std::string{},
        std::move(*objective), std::move(*trainingAlgorithm)));
  }

  [[nodiscard]] std::string_view schema() const noexcept final {
    return ONNX_POLICY_SCHEMA;
  }
  [[nodiscard]] std::string_view parametersChecksum() const noexcept final {
    return parametersChecksum_;
  }
  [[nodiscard]] std::string_view artifactId() const noexcept final {
    return artifactId_;
  }
  [[nodiscard]] std::string_view objective() const noexcept final {
    return objective_;
  }
  [[nodiscard]] std::string_view trainingAlgorithm() const noexcept final {
    return trainingAlgorithm_;
  }

private:
  OnnxPolicyModel(std::unique_ptr<OrtState> state, std::string artifactId,
                  std::string parametersChecksum, std::string objective,
                  std::string trainingAlgorithm)
      : state_(std::move(state)), artifactId_(std::move(artifactId)),
        parametersChecksum_(std::move(parametersChecksum)),
        objective_(std::move(objective)),
        trainingAlgorithm_(std::move(trainingAlgorithm)) {}

  [[nodiscard]] std::optional<Decision>
  evaluate(const FeatureVector& features, const ActionMask& legal) const final {
    if (std::any_of(features.begin(), features.end(),
                    [](const float feature) {
                      return !std::isfinite(feature) || feature < 0.0F ||
                             feature > 1.0F;
                    }) ||
        std::none_of(legal.begin(), legal.end(),
                     [](const bool value) { return value; })) {
      return std::nullopt;
    }

    auto inputData = features;
    constexpr std::array<std::int64_t, 2> inputShape{1, NUM_FEATURES};
    OrtValue* loadedInput = nullptr;
    if (auto* status = state_->api->CreateTensorWithDataAsOrtValue(
            state_->memoryInfo, inputData.data(), sizeof(inputData),
            inputShape.data(), inputShape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &loadedInput)) {
      state_->api->ReleaseStatus(status);
      return std::nullopt;
    }
    OrtOwner<OrtValue, ValueDeleter> input(loadedInput,
                                           ValueDeleter{.api = state_->api});

    constexpr const char* inputNames[]{"features"};
    constexpr const char* outputNames[]{"logits"};
    const OrtValue* inputValues[]{input.get()};
    OrtValue* loadedOutput = nullptr;
    if (auto* status =
            state_->api->Run(state_->session, nullptr, inputNames, inputValues,
                             std::size(inputNames), outputNames,
                             std::size(outputNames), &loadedOutput)) {
      state_->api->ReleaseStatus(status);
      return std::nullopt;
    }
    OrtOwner<OrtValue, ValueDeleter> output(loadedOutput,
                                            ValueDeleter{.api = state_->api});

    OrtTensorTypeAndShapeInfo* loadedTensorInfo = nullptr;
    if (auto* status = state_->api->GetTensorTypeAndShape(output.get(),
                                                          &loadedTensorInfo)) {
      state_->api->ReleaseStatus(status);
      return std::nullopt;
    }
    OrtOwner<OrtTensorTypeAndShapeInfo, TensorInfoDeleter> tensorInfo(
        loadedTensorInfo, TensorInfoDeleter{.api = state_->api});
    ONNXTensorElementDataType elementType =
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    std::size_t dimensionCount = 0;
    std::array<std::int64_t, 2> dimensions{};
    if (auto* status =
            state_->api->GetTensorElementType(tensorInfo.get(), &elementType)) {
      state_->api->ReleaseStatus(status);
      return std::nullopt;
    }
    if (auto* status = state_->api->GetDimensionsCount(tensorInfo.get(),
                                                       &dimensionCount)) {
      state_->api->ReleaseStatus(status);
      return std::nullopt;
    }
    if (dimensionCount != dimensions.size()) {
      return std::nullopt;
    }
    if (auto* status = state_->api->GetDimensions(
            tensorInfo.get(), dimensions.data(), dimensions.size())) {
      state_->api->ReleaseStatus(status);
      return std::nullopt;
    }
    if (elementType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        dimensions != std::array<std::int64_t, 2>{1, NUM_ACTIONS}) {
      return std::nullopt;
    }

    void* rawLogits = nullptr;
    if (auto* status =
            state_->api->GetTensorMutableData(output.get(), &rawLogits)) {
      state_->api->ReleaseStatus(status);
      return std::nullopt;
    }
    if (rawLogits == nullptr) {
      return std::nullopt;
    }
    const auto* logits = static_cast<const float*>(rawLogits);
    if (std::any_of(logits, logits + NUM_ACTIONS,
                    [](const float logit) { return !std::isfinite(logit); })) {
      return std::nullopt;
    }

    Decision decision{.action = Action::Terminate};
    decision.logits.fill(-std::numeric_limits<float>::infinity());
    for (std::size_t action = 0; action < NUM_ACTIONS; ++action) {
      if (legal[action]) {
        decision.logits[action] = logits[action];
      }
    }
    return decision;
  }

  std::unique_ptr<OrtState> state_;
  std::string artifactId_;
  std::string parametersChecksum_;
  std::string objective_;
  std::string trainingAlgorithm_;
};

} // namespace
#endif

llvm::Expected<std::unique_ptr<PolicyModel>>
loadPolicyModel(const std::filesystem::path& path,
                const std::string_view expectedTargetFingerprint,
                const std::string_view expectedCoreRevision) {
  if (path.extension() == ".json") {
    auto model = LinearPolicyModel::load(path, expectedTargetFingerprint,
                                         expectedCoreRevision);
    if (!model) {
      return model.takeError();
    }
    return std::make_unique<LinearPolicyModel>(std::move(*model));
  }
  if (path.extension() == ".onnx") {
#ifdef MQT_PREDICTOR_ENABLE_ONNX
    auto model = OnnxPolicyModel::load(path, expectedTargetFingerprint,
                                       expectedCoreRevision);
    if (!model) {
      return model.takeError();
    }
    return std::unique_ptr<PolicyModel>(std::move(*model));
#else
    return llvm::createStringError("ONNX policy support is disabled; configure "
                                   "MQT_PREDICTOR_ENABLE_ONNX=ON");
#endif
  }
  return llvm::createStringError("unsupported policy model extension '%s'",
                                 path.extension().string().c_str());
}

} // namespace mqt::predictor::compiler
