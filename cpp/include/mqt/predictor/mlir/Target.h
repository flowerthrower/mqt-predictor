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

#include <llvm/Support/Error.h>
#include <mlir/Compiler/Target.h>

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>

namespace mqt::predictor::compiler {

inline constexpr std::string_view COMPILER_TARGET_SCHEMA =
    "mqt-compiler-target/1";

struct LoadedCompilerTarget {
  ::mlir::CompilerTarget target;
  std::string fingerprint;
};

/** Load a topology and native operation set from a versioned JSON document. */
[[nodiscard]] llvm::Expected<LoadedCompilerTarget>
loadCompilerTarget(const std::filesystem::path& path);

/** Construct the built-in line target retained for bootstrap experiments. */
[[nodiscard]] llvm::Expected<LoadedCompilerTarget>
createLineTarget(std::size_t size);

/** Return the deterministic semantic fingerprint used by policy manifests. */
[[nodiscard]] std::string
compilerTargetFingerprint(const ::mlir::CompilerTarget& target);

} // namespace mqt::predictor::compiler
