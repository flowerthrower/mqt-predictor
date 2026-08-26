# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

cmake_minimum_required(VERSION 3.24...4.4)

project(
  mqt-predictor-mlir
  LANGUAGES C CXX
  DESCRIPTION "Experimental MLIR-based MQT Predictor")

# Import the LLVM/MLIR targets at the top level so they remain visible to both
# the embedded Core build and this project's sibling cpp directory.
set(MQT_PREDICTOR_FIND_PACKAGE_TARGETS_GLOBAL
    "${CMAKE_FIND_PACKAGE_TARGETS_GLOBAL}")
set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL ON)
find_package(MLIR REQUIRED CONFIG)
set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL
    "${MQT_PREDICTOR_FIND_PACKAGE_TARGETS_GLOBAL}")
unset(MQT_PREDICTOR_FIND_PACKAGE_TARGETS_GLOBAL)
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}" "${MLIR_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Cap'n Proto, pulled in by Core's jeff dependency, requires GNU-style
# compiler extensions even though the predictor sources themselves use C++20.
set(CMAKE_CXX_EXTENSIONS ON)

include(FetchContent)

set(MQT_CORE_REVISION
    "0c50dd30815638517aa159d20e78290cd449323e"
    CACHE INTERNAL "MQT Core revision used by the experimental MLIR predictor" FORCE)
set(MQT_CORE_SOURCE_DIR
    "$ENV{MQT_CORE_SOURCE_DIR}"
    CACHE PATH "Optional local MQT Core source directory")

set(BUILD_MQT_CORE_TESTS
    OFF
    CACHE BOOL "Build MQT Core tests" FORCE)
set(BUILD_MQT_CORE_BINDINGS
    OFF
    CACHE BOOL "Build MQT Core Python bindings" FORCE)
set(BUILD_MQT_CORE_SHARED_LIBS
    OFF
    CACHE BOOL "Build MQT Core shared libraries" FORCE)
set(BUILD_MQT_CORE_QDMI_DDSIM_DEVICE
    OFF
    CACHE BOOL "Build the MQT Core DDSIM QDMI device" FORCE)
set(BUILD_MQT_CORE_QDMI_SC_DEVICE
    ON
    CACHE BOOL "Build the MQT Core superconducting QDMI device" FORCE)
set(MQT_CORE_INSTALL
    OFF
    CACHE BOOL "Generate MQT Core installation instructions" FORCE)
set(LLVM_INSTALL_TOOLCHAIN_ONLY
    ON
    CACHE BOOL "Disable install rules for the embedded MLIR libraries" FORCE)

if(MQT_CORE_SOURCE_DIR)
  if(NOT EXISTS "${MQT_CORE_SOURCE_DIR}/CMakeLists.txt")
    message(
      FATAL_ERROR
        "MQT_CORE_SOURCE_DIR does not contain CMakeLists.txt: ${MQT_CORE_SOURCE_DIR}")
  endif()
  find_package(Git REQUIRED)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${MQT_CORE_SOURCE_DIR}" rev-parse HEAD
    RESULT_VARIABLE MQT_CORE_GIT_RESULT
    OUTPUT_VARIABLE MQT_CORE_SOURCE_REVISION
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT MQT_CORE_GIT_RESULT EQUAL 0 OR NOT MQT_CORE_SOURCE_REVISION STREQUAL
                                             MQT_CORE_REVISION)
    message(
      FATAL_ERROR
        "MQT_CORE_SOURCE_DIR must be a Git checkout at ${MQT_CORE_REVISION}; found '${MQT_CORE_SOURCE_REVISION}'")
  endif()
  unset(MQT_CORE_GIT_RESULT)
  unset(MQT_CORE_SOURCE_REVISION)
  FetchContent_Declare(mqt-core SOURCE_DIR "${MQT_CORE_SOURCE_DIR}")
else()
  FetchContent_Declare(
    mqt-core
    GIT_REPOSITORY https://github.com/munich-quantum-toolkit/core.git
    GIT_TAG ${MQT_CORE_REVISION})
endif()

FetchContent_MakeAvailable(mqt-core)
set_property(DIRECTORY "${mqt-core_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL YES)

include(CTest)
add_subdirectory(cpp)
