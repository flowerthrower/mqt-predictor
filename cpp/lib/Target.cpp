/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/predictor/mlir/Target.h"

#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SHA256.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mqt::predictor::compiler {
namespace {

using Target = ::mlir::CompilerTarget;
constexpr std::size_t MAX_TARGET_BYTES = 1024U * 1024U;

[[nodiscard]] llvm::Error targetError(const std::filesystem::path& path,
                                      const llvm::StringRef message) {
  return llvm::createStringError("invalid compiler target '%s': %s",
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

[[nodiscard]] std::optional<std::size_t>
readSize(const llvm::json::Object& object, const llvm::StringRef key) {
  const auto value = object.getInteger(key);
  if (!value || *value < 0 ||
      static_cast<std::uint64_t>(*value) >
          std::numeric_limits<std::size_t>::max()) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(*value);
}

void appendUnsigned(std::vector<std::uint8_t>& bytes,
                    const std::uint64_t value) {
  for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
    bytes.push_back(static_cast<std::uint8_t>(value >> (byte * 8U)));
  }
}

void appendString(std::vector<std::uint8_t>& bytes,
                  const llvm::StringRef value) {
  appendUnsigned(bytes, value.size());
  bytes.insert(bytes.end(), value.bytes_begin(), value.bytes_end());
}

} // namespace

std::string compilerTargetFingerprint(const Target& target) {
  std::vector<std::uint8_t> bytes;
  constexpr llvm::StringLiteral fingerprintSchema =
      "mqt-compiler-target-fingerprint/1";
  bytes.insert(bytes.end(), fingerprintSchema.bytes_begin(),
               fingerprintSchema.bytes_end());
  bytes.push_back(0);

  if (const auto name = target.name()) {
    bytes.push_back(1);
    appendString(bytes, *name);
  } else {
    bytes.push_back(0);
  }
  appendUnsigned(bytes, target.sites().size());
  for (const auto& site : target.sites()) {
    appendUnsigned(bytes, static_cast<std::uint64_t>(site.id()));
  }

  bytes.push_back(target.hasExplicitTopology() ? 1 : 0);
  if (target.hasExplicitTopology()) {
    appendUnsigned(bytes, target.couplings().size());
    for (const auto& [source, destination] : target.couplings()) {
      appendUnsigned(bytes, static_cast<std::uint64_t>(source));
      appendUnsigned(bytes, static_cast<std::uint64_t>(destination));
    }
  }

  bytes.push_back(target.hasExplicitOperations() ? 1 : 0);
  if (target.hasExplicitOperations()) {
    struct OperationDescription {
      std::string name;
      std::size_t numQubits;
      std::size_t numParameters;
    };
    std::vector<OperationDescription> operations;
    operations.reserve(target.operations().size());
    for (const auto& operation : target.operations()) {
      operations.emplace_back(operation.canonicalName().str(),
                              operation.numQubits(), operation.numParameters());
    }
    std::ranges::sort(operations, {}, [](const auto& operation) {
      return std::tie(operation.name, operation.numQubits,
                      operation.numParameters);
    });
    appendUnsigned(bytes, operations.size());
    for (const auto& operation : operations) {
      appendString(bytes, operation.name);
      appendUnsigned(bytes, operation.numQubits);
      appendUnsigned(bytes, operation.numParameters);
    }
  }

  return "sha256:" + llvm::toHex(llvm::SHA256::hash(bytes), true);
}

llvm::Expected<LoadedCompilerTarget>
loadCompilerTarget(const std::filesystem::path& path) {
  const auto buffer = llvm::MemoryBuffer::getFile(path.string());
  if (!buffer) {
    return llvm::createStringError(buffer.getError(),
                                   "failed to read compiler target '%s'",
                                   path.string().c_str());
  }
  if ((*buffer)->getBufferSize() > MAX_TARGET_BYTES) {
    return targetError(path, "artifact exceeds the 1 MiB size limit");
  }
  auto parsed = llvm::json::parse((*buffer)->getBuffer());
  if (!parsed) {
    const auto detail = llvm::toString(parsed.takeError());
    return targetError(path, detail);
  }
  const auto* root = parsed->getAsObject();
  if (root == nullptr || !hasOnlyFields(*root, {"schema", "name", "sites",
                                                "couplings", "operations"})) {
    return targetError(path, "unexpected or missing top-level object");
  }
  const auto schema = root->getString("schema");
  const auto name = root->getString("name");
  const auto* sitesJson = root->getArray("sites");
  const auto* operationsJson = root->getArray("operations");
  if (!schema || *schema != llvm::StringRef(COMPILER_TARGET_SCHEMA) || !name ||
      name->empty() || sitesJson == nullptr || sitesJson->size() < 2 ||
      operationsJson == nullptr || operationsJson->empty()) {
    return targetError(path, "schema, name, sites, or operations are invalid");
  }

  std::vector<Target::Site> sites;
  sites.reserve(sitesJson->size());
  std::set<Target::SiteId> siteIds;
  for (const auto& siteJson : *sitesJson) {
    const auto siteId = siteJson.getAsInteger();
    if (!siteId || *siteId < 0 || !siteIds.insert(*siteId).second) {
      return targetError(path, "site IDs must be unique nonnegative integers");
    }
    auto site = Target::Site::create(*siteId);
    if (!site) {
      const auto detail = llvm::toString(site.takeError());
      return targetError(path, detail);
    }
    sites.emplace_back(std::move(*site));
  }

  std::optional<std::vector<Target::Coupling>> couplings;
  if (const auto* couplingsJson = root->getArray("couplings")) {
    couplings.emplace();
    couplings->reserve(couplingsJson->size());
    for (const auto& couplingJson : *couplingsJson) {
      const auto* pair = couplingJson.getAsArray();
      if (pair == nullptr || pair->size() != 2) {
        return targetError(path, "each coupling must contain two site IDs");
      }
      const auto source = (*pair)[0].getAsInteger();
      const auto destination = (*pair)[1].getAsInteger();
      if (!source || !destination || *source < 0 || *destination < 0) {
        return targetError(path,
                           "coupling site IDs must be nonnegative integers");
      }
      couplings->emplace_back(*source, *destination);
    }
  } else if (root->get("couplings") != nullptr) {
    return targetError(path, "couplings must be an array when present");
  }

  std::vector<Target::Operation> operations;
  operations.reserve(operationsJson->size());
  for (const auto& operationJson : *operationsJson) {
    const auto* operationObject = operationJson.getAsObject();
    if (operationObject == nullptr ||
        !hasOnlyFields(*operationObject,
                       {"name", "num_qubits", "num_parameters"})) {
      return targetError(path, "each operation must be a capability object");
    }
    const auto operationName = operationObject->getString("name");
    const auto numQubits = readSize(*operationObject, "num_qubits");
    const auto numParameters = readSize(*operationObject, "num_parameters");
    if (!operationName || operationName->empty() || !numQubits ||
        *numQubits == 0 || !numParameters) {
      return targetError(path, "operation capability fields are invalid");
    }
    auto operation = Target::Operation::create(operationName->str(), *numQubits,
                                               *numParameters);
    if (!operation) {
      const auto detail = llvm::toString(operation.takeError());
      return targetError(path, detail);
    }
    if (operation->canonicalName() != *operationName) {
      return targetError(
          path, "operation names must use canonical lowercase spelling");
    }
    operations.emplace_back(std::move(*operation));
  }

  auto target = Target::create(name->str(), std::move(sites),
                               std::move(couplings), std::move(operations));
  if (!target) {
    const auto detail = llvm::toString(target.takeError());
    return targetError(path, detail);
  }
  auto fingerprint = compilerTargetFingerprint(*target);
  return LoadedCompilerTarget{.target = std::move(*target),
                              .fingerprint = std::move(fingerprint)};
}

llvm::Expected<LoadedCompilerTarget> createLineTarget(const std::size_t size) {
  if (size < 2) {
    return llvm::createStringError(
        "the experimental line target needs at least two qubits");
  }

  using Operation = Target::Operation;
  std::vector<Operation> operations;
  operations.emplace_back(llvm::cantFail(Operation::create("u", 1, 3)));
  operations.emplace_back(llvm::cantFail(Operation::create("cx", 2, 0)));
  operations.emplace_back(llvm::cantFail(Operation::create("measure", 1, 0)));
  operations.emplace_back(llvm::cantFail(Operation::create("reset", 1, 0)));

  std::vector<Target::Coupling> couplings;
  couplings.reserve(size - 1);
  for (std::size_t qubit = 0; qubit + 1 < size; ++qubit) {
    couplings.emplace_back(static_cast<Target::SiteId>(qubit),
                           static_cast<Target::SiteId>(qubit + 1));
  }
  auto target = Target::create("predictor-line", size, std::move(couplings),
                               std::move(operations));
  if (!target) {
    return target.takeError();
  }
  auto fingerprint = compilerTargetFingerprint(*target);
  return LoadedCompilerTarget{.target = std::move(*target),
                              .fingerprint = std::move(fingerprint)};
}

} // namespace mqt::predictor::compiler
