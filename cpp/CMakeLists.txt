# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(MQT_PREDICTOR_MLIR_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
add_compile_definitions(
  MQT_PREDICTOR_CORE_REVISION="${MQT_CORE_COMPATIBILITY_ID}")

option(MQT_PREDICTOR_ENABLE_ONNX "Enable ONNX Runtime policy inference" OFF)
set(ONNXRUNTIME_ROOT
    ""
    CACHE PATH "Root of an ONNX Runtime C/C++ SDK")
if(MQT_PREDICTOR_ENABLE_ONNX)
  find_path(
    ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_c_api.h
    PATHS "${ONNXRUNTIME_ROOT}/include"
    NO_DEFAULT_PATH REQUIRED)
  find_library(
    ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH REQUIRED)
  if(NOT TARGET onnxruntime::onnxruntime)
    add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED GLOBAL)
    set_target_properties(
      onnxruntime::onnxruntime
      PROPERTIES IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
                 INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")
  endif()
endif()

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
  MQTCompilerQDMIAdapter
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
if(MQT_PREDICTOR_ENABLE_ONNX)
  target_compile_definitions(MQTPredictorMLIR
                             PRIVATE MQT_PREDICTOR_ENABLE_ONNX=1)
  target_link_libraries(MQTPredictorMLIR PRIVATE onnxruntime::onnxruntime)
endif()
if(TARGET obj.MQTPredictorMLIR)
  set_target_properties(obj.MQTPredictorMLIR PROPERTIES UNITY_BUILD OFF)
  if(MQT_PREDICTOR_ENABLE_ONNX)
    target_compile_definitions(obj.MQTPredictorMLIR
                               PRIVATE MQT_PREDICTOR_ENABLE_ONNX=1)
  endif()
endif()

add_executable(mqt-predictor-cc tools/mqt-predictor-cc.cpp)
target_link_libraries(mqt-predictor-cc PRIVATE MQTPredictorMLIR)
target_compile_features(mqt-predictor-cc PRIVATE cxx_std_20)
set_target_properties(mqt-predictor-cc PROPERTIES UNITY_BUILD OFF)
mqt_mlir_target_use_project_options(mqt-predictor-cc)
mlir_check_all_link_libraries(mqt-predictor-cc)
export_executable_symbols_for_plugins(mqt-predictor-cc)

mqt_get_qdmi_device_targets(mqt_predictor_qdmi_device_targets)
if(mqt_predictor_qdmi_device_targets)
  mqt_copy_qdmi_runtime(mqt-predictor-cc ${mqt_predictor_qdmi_device_targets})
endif()

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
      mqt-predictor-cc --trace --deterministic-policy
      --target=${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/line-4-target.json
      --model=${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/line-4-policy.json -o
      ${CMAKE_CURRENT_BINARY_DIR}/native-line-deterministic-model.mlir
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/native-line.mlir)
  set_tests_properties(
    mqt-predictor-model-e2e
    PROPERTIES PASS_REGULAR_EXPRESSION "policy=model.*action=terminate"
               FAIL_REGULAR_EXPRESSION "falling back;error:")

  add_test(
    NAME mqt-predictor-stochastic-default-e2e
    COMMAND
      mqt-predictor-cc --trace --max-steps=1
      --target=${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/line-4-target.json
      --model=${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/line-4-policy.json -o
      ${CMAKE_CURRENT_BINARY_DIR}/native-line-stochastic.mlir
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/native-line.mlir)
  set_tests_properties(
    mqt-predictor-stochastic-default-e2e
    PROPERTIES PASS_REGULAR_EXPRESSION
               "sampling=stochastic sampling_seed=entropy"
               FAIL_REGULAR_EXPRESSION "error:")

  if(MQT_PREDICTOR_ENABLE_ONNX)
    add_test(
      NAME mqt-predictor-onnx-rejects-architecture-mismatch
      COMMAND
        mqt-predictor-artifact-test
        ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/line-4-target.json
        ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/line-4-policy.onnx
        ${CMAKE_CURRENT_BINARY_DIR}/line-4-policy-unsupported-architecture.onnx)

    add_test(
      NAME mqt-predictor-onnx-model-e2e
      COMMAND
        mqt-predictor-cc --trace --deterministic-policy
        --target=${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/line-4-target.json
        --model=${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/line-4-policy.onnx -o
        ${CMAKE_CURRENT_BINARY_DIR}/native-line-onnx-model.mlir
        ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/native-line.mlir)
    set_tests_properties(
      mqt-predictor-onnx-model-e2e
      PROPERTIES
        PASS_REGULAR_EXPRESSION
        "policy=model model_schema=mqt-predictor-onnx-policy/1.*action=terminate"
        FAIL_REGULAR_EXPRESSION "falling back;error:")

    add_test(
      NAME mqt-predictor-iqm-garnet-onnx-model-e2e
      COMMAND
        mqt-predictor-cc --trace --sampling-seed=19 --max-steps=20
        --qdmi-device=mqt.sc.iqm.garnet
        --model=${CMAKE_CURRENT_SOURCE_DIR}/models/iqm-garnet-ppo-tanh64x64.onnx
        -o
        ${CMAKE_CURRENT_BINARY_DIR}/controlled-u-iqm-garnet-onnx-model.mlir
        ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/controlled-u-from-qiskit.mlir)
    set_tests_properties(
      mqt-predictor-iqm-garnet-onnx-model-e2e
      PROPERTIES
        PASS_REGULAR_EXPRESSION
        "policy=model model_schema=mqt-predictor-onnx-policy/1.*sampling=stochastic sampling_seed=19.*action=terminate"
        FAIL_REGULAR_EXPRESSION "falling back;truncated;error:")
  endif()

  add_test(
    NAME mqt-predictor-register-model-e2e
    COMMAND
      mqt-predictor-cc --trace --deterministic-policy --max-steps=20
      --target=${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/line-4-target.json
      --model=${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/line-4-policy.json -o
      ${CMAKE_CURRENT_BINARY_DIR}/controlled-u-register-model.mlir
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/controlled-u-from-qiskit.mlir)
  set_tests_properties(
    mqt-predictor-register-model-e2e
    PROPERTIES PASS_REGULAR_EXPRESSION "policy=model.*action=terminate"
               FAIL_REGULAR_EXPRESSION "falling back;error:")

  add_test(
    NAME mqt-predictor-model-native-input-state-e2e
    COMMAND
      mqt-predictor-cc --trace --deterministic-policy
      --target=${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/line-4-target.json
      --model=${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/line-4-policy.json -o
      ${CMAKE_CURRENT_BINARY_DIR}/native-line-model.mlir
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/native-line.mlir)
  set_tests_properties(
    mqt-predictor-model-native-input-state-e2e
    PROPERTIES
      PASS_REGULAR_EXPRESSION
      "step=0.*mapped=0 routed=0 synthesized=0 legal=111110"
      FAIL_REGULAR_EXPRESSION "falling back;error:")

  add_test(
    NAME mqt-predictor-model-rejects-partial-mapping-e2e
    COMMAND
      mqt-predictor-cc --trace
      --target=${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/line-4-target.json
      --model=${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/line-4-policy.json -o
      ${CMAKE_CURRENT_BINARY_DIR}/partial-mapping-model.mlir
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/partially-mapped.mlir)
  set_tests_properties(
    mqt-predictor-model-rejects-partial-mapping-e2e
    PROPERTIES WILL_FAIL TRUE)

  add_test(
    NAME mqt-predictor-model-rejects-qasm-frontend
    COMMAND
      mqt-predictor-cc
      --target=${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/line-4-target.json
      --model=${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/line-4-policy.json
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/bell.qasm)
  set_tests_properties(mqt-predictor-model-rejects-qasm-frontend
                       PROPERTIES WILL_FAIL TRUE)

  add_test(
    NAME mqt-predictor-bootstrap-e2e
    COMMAND
      mqt-predictor-cc --trace --target-qubits=4 -o
      ${CMAKE_CURRENT_BINARY_DIR}/bell-bootstrap.mlir
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/bell.qasm)
  set_tests_properties(
    mqt-predictor-bootstrap-e2e
    PROPERTIES
      PASS_REGULAR_EXPRESSION
      "step=2 action=terminate qubits=4.*liveness=4[.]464286e-01.*num_qubits=1[.]000000e[+]00.*parallelism=4[.]761905e-02.*program_communication=3[.]333333e-01"
      FAIL_REGULAR_EXPRESSION "falling back;error:")

  add_test(
    NAME mqt-predictor-unmapped-qubit-width-feature-e2e
    COMMAND
      mqt-predictor-cc --trace --max-steps=1 --target-qubits=4 -o
      ${CMAKE_CURRENT_BINARY_DIR}/bell-unmapped-width.mlir
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/bell.qasm)
  set_tests_properties(
    mqt-predictor-unmapped-qubit-width-feature-e2e
    PROPERTIES
      PASS_REGULAR_EXPRESSION
      "step=0 action=place-and-route qubits=3.*liveness=6[.]111111e-01.*num_qubits=7[.]500000e-01.*parallelism=0[.]000000e[+]00.*program_communication=1[.]000000e[+]00"
      FAIL_REGULAR_EXPRESSION "unsupported QCO structure;error:")

  add_test(
    NAME mqt-predictor-rotations-e2e
    COMMAND
      mqt-predictor-cc --trace --target-qubits=2 -o
      ${CMAKE_CURRENT_BINARY_DIR}/rotations-bootstrap.mlir
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/rotations.qasm)
  set_tests_properties(
    mqt-predictor-rotations-e2e
    PROPERTIES
      PASS_REGULAR_EXPRESSION "step=0 action=place-and-route"
      FAIL_REGULAR_EXPRESSION "falling back;error:")

  add_test(
    NAME mqt-predictor-wide-e2e
    COMMAND
      mqt-predictor-cc --trace --max-steps=1 --target-qubits=4 -o
      ${CMAKE_CURRENT_BINARY_DIR}/wide-bootstrap.mlir
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/wide.qasm)
  set_tests_properties(
    mqt-predictor-wide-e2e
    PROPERTIES PASS_REGULAR_EXPRESSION "schema=mqt-predictor-core-stages/3"
               FAIL_REGULAR_EXPRESSION "unsupported QCO structure;error:")

  add_test(
    NAME mqt-predictor-qft-feature-e2e
    COMMAND
      mqt-predictor-cc --trace --max-steps=1 --target-qubits=4 -o
      ${CMAKE_CURRENT_BINARY_DIR}/qft-feature.mlir
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/qft-4.qasm)
  set_tests_properties(
    mqt-predictor-qft-feature-e2e
    PROPERTIES
      PASS_REGULAR_EXPRESSION
      "features=\\{.*cp=3[.]750000e-01,critical_depth=5[.]000000e-01.*h=2[.]500000e-01.*measure=2[.]500000e-01.*swap=1[.]250000e-01"
               FAIL_REGULAR_EXPRESSION "unsupported QCO structure;error:")

  add_test(
    NAME mqt-predictor-controlled-u-feature-e2e
    COMMAND
      mqt-predictor-cc --trace --max-steps=1 --target-qubits=2 -o
      ${CMAKE_CURRENT_BINARY_DIR}/controlled-u-feature.mlir
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/controlled-u.qasm)
  set_tests_properties(
    mqt-predictor-controlled-u-feature-e2e
    PROPERTIES
      PASS_REGULAR_EXPRESSION
      "features=\\{.*cu3=2[.]500000e-01.*measure=5[.]000000e-01.*p=2[.]500000e-01"
      FAIL_REGULAR_EXPRESSION "unsupported QCO structure;error:")

  add_test(
    NAME mqt-predictor-from-qiskit-qco-feature-e2e
    COMMAND
      mqt-predictor-cc --trace --target-qubits=4 -o
      ${CMAKE_CURRENT_BINARY_DIR}/controlled-u-from-qiskit-output.mlir
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/controlled-u-from-qiskit.mlir)
  set_tests_properties(
    mqt-predictor-from-qiskit-qco-feature-e2e
    PROPERTIES
      PASS_REGULAR_EXPRESSION
      "features=\\{.*cu3=3[.]333333e-01.*measure=6[.]666667e-01.*p=0[.]000000e[+]00"
      FAIL_REGULAR_EXPRESSION "unsupported QCO structure;error:")

  add_test(
    NAME mqt-predictor-core-e2e
    COMMAND
      mqt-predictor-cc --policy=core --target-qubits=4 -o
      ${CMAKE_CURRENT_BINARY_DIR}/bell-core.mlir
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/bell.qasm)

  add_test(
    NAME mqt-predictor-exhaustive-e2e
    COMMAND
      mqt-predictor-cc --trace --policy=exhaustive --target-qubits=4 -o
      ${CMAKE_CURRENT_BINARY_DIR}/register-exhaustive.mlir
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/register-routing.qasm)
  set_tests_properties(
    mqt-predictor-exhaustive-e2e
    PROPERTIES PASS_REGULAR_EXPRESSION
               "search=exhaustive.*candidates=17;search-result winner="
               FAIL_REGULAR_EXPRESSION "falling back;error:")

  add_test(
    NAME mqt-predictor-iqm-garnet-e2e
    COMMAND
      mqt-predictor-cc --trace --policy=exhaustive
      --qdmi-device=mqt.sc.iqm.garnet -o
      ${CMAKE_CURRENT_BINARY_DIR}/bell-iqm-garnet.mlir
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/bell.qasm)
  set_tests_properties(
    mqt-predictor-iqm-garnet-e2e
    PROPERTIES PASS_REGULAR_EXPRESSION
               "target=IQM Garnet.*candidates=17;search-result winner="
               FAIL_REGULAR_EXPRESSION "falling back;error:")

  add_test(
    NAME mqt-predictor-budget-termination-e2e
    COMMAND
      mqt-predictor-cc --trace --max-steps=1 --target-qubits=4 -o
      ${CMAKE_CURRENT_BINARY_DIR}/bell-fallback.mlir
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/bell.qasm)
  set_tests_properties(
    mqt-predictor-budget-termination-e2e
    PROPERTIES
      PASS_REGULAR_EXPRESSION
      "bootstrap episode truncated after 1 policy decisions"
      FAIL_REGULAR_EXPRESSION "error:")

  add_test(
    NAME mqt-predictor-horizon-truncates-before-termination-e2e
    COMMAND
      mqt-predictor-cc --trace --max-steps=2 --target-qubits=4 -o
      ${CMAKE_CURRENT_BINARY_DIR}/bell-budget-stages.mlir
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/bell.qasm)
  set_tests_properties(
    mqt-predictor-horizon-truncates-before-termination-e2e
    PROPERTIES
      PASS_REGULAR_EXPRESSION
      "bootstrap episode truncated after 2 policy decisions"
      FAIL_REGULAR_EXPRESSION "step=2 action=terminate;error:")

  add_test(
    NAME mqt-predictor-horizon-counts-termination-e2e
    COMMAND
      mqt-predictor-cc --trace --max-steps=3 --target-qubits=4 -o
      ${CMAKE_CURRENT_BINARY_DIR}/bell-budget-termination.mlir
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/bell.qasm)
  set_tests_properties(
    mqt-predictor-horizon-counts-termination-e2e
    PROPERTIES PASS_REGULAR_EXPRESSION "step=2 action=terminate"
               FAIL_REGULAR_EXPRESSION "falling back;truncated;error:")

  add_test(
    NAME mqt-predictor-rejects-pass-budget
    COMMAND mqt-predictor-cc --max-steps=21
            ${CMAKE_CURRENT_SOURCE_DIR}/test/Inputs/bell.qasm)
  set_tests_properties(mqt-predictor-rejects-pass-budget PROPERTIES WILL_FAIL
                                                                       TRUE)

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
