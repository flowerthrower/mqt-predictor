/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Compiler/Programs.h"
#include "mqt/predictor/mlir/PredictorPass.h"

#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Pass/PassManager.h>

#include <charconv>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace {

using mqt::predictor::compiler::PolicyMode;
using mqt::predictor::compiler::PredictorOptions;

struct DriverOptions {
  std::filesystem::path input;
  std::filesystem::path output = "-";
  PredictorOptions predictor;
  bool policySpecified = false;
  bool targetQubitsSpecified = false;
};

void printHelp(llvm::raw_ostream& output) {
  output
      << "Usage: mqt-predictor-cc [options] <input.qasm|input.mlir>\n"
         "\n"
         "Options:\n"
         "  -o <path>             Write QCO MLIR to path (default: stdout)\n"
         "  --policy=<name>       exhaustive, model, bootstrap, or core "
         "(default: bootstrap)\n"
         "  --model=<path>        Load a native JSON policy (implies model)\n"
         "  --target=<path>       Load a JSON compiler target\n"
         "  --qdmi-device=<id>    Snapshot a registered QDMI compiler target\n"
         "  --target-qubits=<n>   Built-in line target size (default: 5)\n"
         "  --max-steps=<n>       Maximum policy decisions (default: 16)\n"
         "  --trace               Print features, states, and actions\n"
         "  --help                Show this help\n";
}

[[nodiscard]] std::optional<std::size_t>
parseSize(const std::string_view value) {
  std::size_t result = 0;
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc{} || end != value.data() + value.size()) {
    return std::nullopt;
  }
  return result;
}

[[nodiscard]] std::optional<DriverOptions> parseArguments(const int argc,
                                                          char** argv) {
  DriverOptions options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--help") {
      printHelp(llvm::outs());
      return std::nullopt;
    }
    if (argument == "--trace") {
      options.predictor.trace = true;
      continue;
    }
    if (argument == "-o") {
      if (++index >= argc) {
        llvm::errs() << "-o requires a path\n";
        return std::nullopt;
      }
      options.output = argv[index];
      continue;
    }
    if (argument.starts_with("--policy=")) {
      const auto policy = argument.substr(std::string_view("--policy=").size());
      if (policy == "bootstrap") {
        options.predictor.policy = PolicyMode::Bootstrap;
      } else if (policy == "core") {
        options.predictor.policy = PolicyMode::Core;
      } else if (policy == "model") {
        options.predictor.policy = PolicyMode::Model;
      } else if (policy == "exhaustive") {
        options.predictor.policy = PolicyMode::Exhaustive;
      } else {
        llvm::errs() << "unknown policy: " << policy << '\n';
        return std::nullopt;
      }
      options.policySpecified = true;
      continue;
    }
    if (argument.starts_with("--model=")) {
      const auto path = argument.substr(std::string_view("--model=").size());
      if (path.empty()) {
        llvm::errs() << "--model requires a path\n";
        return std::nullopt;
      }
      options.predictor.modelPath = path;
      continue;
    }
    if (argument.starts_with("--target=")) {
      const auto path = argument.substr(std::string_view("--target=").size());
      if (path.empty()) {
        llvm::errs() << "--target requires a path\n";
        return std::nullopt;
      }
      options.predictor.targetPath = path;
      continue;
    }
    if (argument.starts_with("--qdmi-device=")) {
      const auto id =
          argument.substr(std::string_view("--qdmi-device=").size());
      if (id.empty()) {
        llvm::errs() << "--qdmi-device requires an ID\n";
        return std::nullopt;
      }
      options.predictor.deviceId = id;
      continue;
    }
    if (argument.starts_with("--target-qubits=")) {
      const auto value =
          argument.substr(std::string_view("--target-qubits=").size());
      const auto parsed = parseSize(value);
      if (!parsed || *parsed < 2) {
        llvm::errs() << "--target-qubits must be at least 2\n";
        return std::nullopt;
      }
      options.predictor.targetQubits = *parsed;
      options.targetQubitsSpecified = true;
      continue;
    }
    if (argument.starts_with("--max-steps=")) {
      const auto value =
          argument.substr(std::string_view("--max-steps=").size());
      const auto parsed = parseSize(value);
      if (!parsed || *parsed == 0) {
        llvm::errs() << "--max-steps must be positive\n";
        return std::nullopt;
      }
      options.predictor.maxSteps = *parsed;
      continue;
    }
    if (argument.starts_with('-')) {
      llvm::errs() << "unknown option: " << argument << '\n';
      return std::nullopt;
    }
    if (!options.input.empty()) {
      llvm::errs() << "only one input file is supported\n";
      return std::nullopt;
    }
    options.input = argument;
  }

  if (options.input.empty()) {
    llvm::errs() << "missing input file\n";
    printHelp(llvm::errs());
    return std::nullopt;
  }
  if (!options.predictor.modelPath.empty()) {
    if (options.policySpecified &&
        options.predictor.policy != PolicyMode::Model) {
      llvm::errs() << "--model cannot be combined with a non-model policy\n";
      return std::nullopt;
    }
    options.predictor.policy = PolicyMode::Model;
  } else if (options.predictor.policy == PolicyMode::Model) {
    llvm::errs() << "--policy=model requires --model=<path>\n";
    return std::nullopt;
  }
  const auto explicitTargets =
      static_cast<unsigned>(!options.predictor.targetPath.empty()) +
      static_cast<unsigned>(!options.predictor.deviceId.empty()) +
      static_cast<unsigned>(options.targetQubitsSpecified);
  if (explicitTargets > 1) {
    llvm::errs() << "--target, --qdmi-device, and --target-qubits are mutually "
                    "exclusive\n";
    return std::nullopt;
  }
  return options;
}

[[nodiscard]] std::optional<::mlir::QCOProgram>
loadProgram(const std::filesystem::path& input) {
  if (input.extension() == ".qasm") {
    auto qc = ::mlir::QCProgram::fromQASMFile(input);
    if (!qc) {
      return std::nullopt;
    }
    return std::move(*qc).intoQCO();
  }
  if (input.extension() == ".mlir") {
    return ::mlir::QCOProgram::fromMLIRFile(input);
  }
  llvm::errs() << "input must use the .qasm or .mlir extension\n";
  return std::nullopt;
}

[[nodiscard]] bool writeOutput(const std::filesystem::path& output,
                               const std::string& mlir) {
  if (output == "-") {
    llvm::outs() << mlir;
    return true;
  }
  std::ofstream stream(output);
  if (!stream) {
    llvm::errs() << "failed to open output file: " << output.string() << '\n';
    return false;
  }
  stream << mlir;
  return stream.good();
}

} // namespace

int main(int argc, char** argv) {
  const llvm::InitLLVM init(argc, argv);
  const auto options = parseArguments(argc, argv);
  if (!options) {
    return argc == 2 && std::string_view(argv[1]) == "--help" ? 0 : 1;
  }

  auto program = loadProgram(options->input);
  if (!program) {
    llvm::errs() << "failed to load input: " << options->input.string() << '\n';
    return 1;
  }

  ::mlir::PassManager passManager(program->module().getContext());
  passManager.addPass(
      mqt::predictor::compiler::createPredictorPass(options->predictor));
  if (failed(passManager.run(program->module()))) {
    llvm::errs() << "compiled predictor pass failed\n";
    return 1;
  }

  return writeOutput(options->output, program->str()) ? 0 : 1;
}
