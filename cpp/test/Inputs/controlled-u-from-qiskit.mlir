// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

module {
  func.func @main() -> !cbit.reg<2> attributes {mqt.entry_point} {
    %c2 = arith.constant 2 : index
    %0 = qtensor.alloc(%c2) {mqt.register_name = "q"} : tensor<2x!qco.qubit>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %1 = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "c"} : !cbit.reg<2>
    %cst = arith.constant 0.000000e+00 : f64
    qco.gphase(%cst)
    %out_tensor, %result = qtensor.extract %0[%c0] : tensor<2x!qco.qubit>
    %out_tensor_0, %result_1 = qtensor.extract %out_tensor[%c1] : tensor<2x!qco.qubit>
    %controls_out, %targets_out = qco.ctrl(%result) targets (%arg0 = %result_1) {
      %cst_11 = arith.constant 0.000000e+00 : f64
      qco.gphase(%cst_11)
      %cst_12 = arith.constant 1.000000e-01 : f64
      %cst_13 = arith.constant 2.000000e-01 : f64
      %cst_14 = arith.constant 3.000000e-01 : f64
      %6 = qco.u(%cst_12, %cst_13, %cst_14) %arg0 : !qco.qubit -> !qco.qubit
      qco.yield %6 : !qco.qubit
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    %2 = qtensor.insert %targets_out into %out_tensor_0[%c1] : tensor<2x!qco.qubit>
    %3 = qtensor.insert %controls_out into %2[%c0] : tensor<2x!qco.qubit>
    %out_tensor_2, %result_3 = qtensor.extract %3[%c0] : tensor<2x!qco.qubit>
    %qubit_out, %result_4 = qco.measure %result_3 : !qco.qubit
    %4 = qtensor.insert %qubit_out into %out_tensor_2[%c0] : tensor<2x!qco.qubit>
    %c0_5 = arith.constant 0 : index
    cbit.store %result_4, %1[%c0_5] : !cbit.reg<2>
    %out_tensor_6, %result_7 = qtensor.extract %4[%c1] : tensor<2x!qco.qubit>
    %qubit_out_8, %result_9 = qco.measure %result_7 : !qco.qubit
    %5 = qtensor.insert %qubit_out_8 into %out_tensor_6[%c1] : tensor<2x!qco.qubit>
    %c1_10 = arith.constant 1 : index
    cbit.store %result_9, %1[%c1_10] : !cbit.reg<2>
    qtensor.dealloc %5 : tensor<2x!qco.qubit>
    return %1 : !cbit.reg<2>
  }
}
