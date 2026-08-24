# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(MQT_PREDICTOR_MLIR_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
add_compile_definitions(MQT_PREDICTOR_CORE_REVISION="${MQT_CORE_REVISION}")

add_mlir_library(
  MQTPredictorMLIR
  DISABLE_INSTALL
  lib/CircuitFeatures.cpp
  lib/Model.cpp
  lib/Policy.cpp
  lib/PredictorPass.cpp
  lib/Target.cpp
  ADDITIONAL_HEADER_DIRS
  ${MQT_PREDICTOR_MLIR_INCLUDE_DIR}
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MQTCompilerPipeline
  MLIRQCOTransforms)

target_include_directories(
  MQTPredictorMLIR SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

target_sources(
  MQTPredictorMLIR
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         ${MQT_PREDICTOR_MLIR_INCLUDE_DIR}
         FILES
         ${MQT_PREDICTOR_MLIR_INCLUDE_DIR}/mqt/predictor/mlir/CircuitFeatures.h
         ${MQT_PREDICTOR_MLIR_INCLUDE_DIR}/mqt/predictor/mlir/Model.h
         ${MQT_PREDICTOR_MLIR_INCLUDE_DIR}/mqt/predictor/mlir/Policy.h
         ${MQT_PREDICTOR_MLIR_INCLUDE_DIR}/mqt/predictor/mlir/PredictorPass.h
         ${MQT_PREDICTOR_MLIR_INCLUDE_DIR}/mqt/predictor/mlir/Target.h)

target_compile_features(MQTPredictorMLIR PUBLIC cxx_std_20)
set_target_properties(MQTPredictorMLIR PROPERTIES UNITY_BUILD OFF)
mqt_mlir_target_use_project_options(MQTPredictorMLIR)
if(TARGET obj.MQTPredictorMLIR)
  set_target_properties(obj.MQTPredictorMLIR PROPERTIES UNITY_BUILD OFF)
endif()

add_executable(mqt-predictor-cc tools/mqt-predictor-cc.cpp)
target_link_libraries(mqt-predictor-cc PRIVATE MQTPredictorMLIR)
target_compile_features(mqt-predictor-cc PRIVATE cxx_std_20)
set_target_properties(mqt-predictor-cc PROPERTIES UNITY_BUILD OFF)
mqt_mlir_target_use_project_options(mqt-predictor-cc)
mlir_check_all_link_libraries(mqt-predictor-cc)

if(BUILD_TESTING)
  add_executable(mqt-predictor-artifact-test test/ArtifactTest.cpp)
  target_link_libraries(mqt-predictor-artifact-test PRIVATE MQTPredictorMLIR)
  target_compile_features(mqt-predictor-artifact-test PRIVATE cxx_std_20)
  set_target_properties(mqt-predictor-artifact-test PROPERTIES UNITY_BUILD OFF)
  mqt_mlir_target_use_project_options(mqt-predictor-artifact-test)

  add_test(
    NAME mqt-predictor-artifact-test
    COMMAND
      mqt-predictor-artifact-test
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/line-4-target.json
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/line-4-policy.json)

  add_executable(mqt-predictor-policy-test test/PolicyTest.cpp)
  target_link_libraries(mqt-predictor-policy-test PRIVATE MQTPredictorMLIR)
  target_compile_features(mqt-predictor-policy-test PRIVATE cxx_std_20)
  set_target_properties(mqt-predictor-policy-test PROPERTIES UNITY_BUILD OFF)
  mqt_mlir_target_use_project_options(mqt-predictor-policy-test)

  add_test(NAME mqt-predictor-policy-test COMMAND mqt-predictor-policy-test)

  add_test(
    NAME mqt-predictor-model-e2e
    COMMAND
      mqt-predictor-cc --trace
      --target=${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/line-4-target.json
      --model=${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/line-4-policy.json -o
      ${CMAKE_CURRENT_BINARY_DIR}/bell-model.mlir
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/bell.qasm)
  set_tests_properties(
    mqt-predictor-model-e2e
    PROPERTIES PASS_REGULAR_EXPRESSION "policy=model.*action=terminate"
               FAIL_REGULAR_EXPRESSION "falling back;error:")

  add_test(
    NAME mqt-predictor-bootstrap-e2e
    COMMAND
      mqt-predictor-cc --trace --target-qubits=4 -o
      ${CMAKE_CURRENT_BINARY_DIR}/bell-bootstrap.mlir
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/bell.qasm)
  set_tests_properties(
    mqt-predictor-bootstrap-e2e
    PROPERTIES PASS_REGULAR_EXPRESSION "action=terminate"
               FAIL_REGULAR_EXPRESSION "falling back;error:")

  add_test(
    NAME mqt-predictor-rotations-e2e
    COMMAND
      mqt-predictor-cc --trace --target-qubits=2 -o
      ${CMAKE_CURRENT_BINARY_DIR}/rotations-bootstrap.mlir
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/rotations.qasm)
  set_tests_properties(
    mqt-predictor-rotations-e2e
    PROPERTIES PASS_REGULAR_EXPRESSION "action=merge-rotations"
               FAIL_REGULAR_EXPRESSION "falling back;error:")

  add_test(
    NAME mqt-predictor-core-e2e
    COMMAND
      mqt-predictor-cc --policy=core --target-qubits=4 -o
      ${CMAKE_CURRENT_BINARY_DIR}/bell-core.mlir
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/bell.qasm)

  add_test(
    NAME mqt-predictor-fallback-e2e
    COMMAND
      mqt-predictor-cc --trace --max-steps=1 --target-qubits=4 -o
      ${CMAKE_CURRENT_BINARY_DIR}/bell-fallback.mlir
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/bell.qasm)
  set_tests_properties(
    mqt-predictor-fallback-e2e PROPERTIES PASS_REGULAR_EXPRESSION
                                              "falling back to Core's canonical target pipeline")

  add_test(
    NAME mqt-predictor-rejects-duplicate-sites
    COMMAND mqt-predictor-cc --target-qubits=4
            ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/duplicate-site.mlir)
  set_tests_properties(mqt-predictor-rejects-duplicate-sites PROPERTIES WILL_FAIL TRUE)

  add_test(
    NAME mqt-predictor-rejects-unterminated-wire
    COMMAND mqt-predictor-cc --target-qubits=4
            ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/unterminated-wire.mlir)
  set_tests_properties(mqt-predictor-rejects-unterminated-wire PROPERTIES WILL_FAIL TRUE)

  add_test(
    NAME mqt-predictor-rejects-unsupported-linear-cast
    COMMAND mqt-predictor-cc --target-qubits=4
            ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/unsupported-linear-cast.mlir)
  set_tests_properties(
    mqt-predictor-rejects-unsupported-linear-cast PROPERTIES WILL_FAIL TRUE)
endif()
