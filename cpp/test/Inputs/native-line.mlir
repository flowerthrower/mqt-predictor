// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

module {
  func.func @main() -> !cbit.reg<2> attributes {mqt.entry_point} {
    %theta = arith.constant 5.000000e-01 : f64
    %phi = arith.constant 2.000000e-01 : f64
    %lambda = arith.constant 1.000000e-01 : f64
    %one = arith.constant 1 : index
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : index
    %qubits = qtensor.alloc(%two) {mqt.register_name = "q"} : tensor<2x!qco.qubit>
    %bits = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "c"} : !cbit.reg<2>
    %rest, %q0 = qtensor.extract %qubits[%zero] : tensor<2x!qco.qubit>
    %q0_u = qco.u(%theta, %phi, %lambda) %q0 : !qco.qubit -> !qco.qubit
    %empty, %q1 = qtensor.extract %rest[%one] : tensor<2x!qco.qubit>
    %control, %target = qco.ctrl(%q0_u) targets (%arg0 = %q1) {
      %out = qco.x %arg0 : !qco.qubit -> !qco.qubit
      qco.yield %out : !qco.qubit
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    %q0_out, %bit0 = qco.measure %control : !qco.qubit
    cbit.store %bit0, %bits[%zero] : !cbit.reg<2>
    %partial = qtensor.insert %q0_out into %empty[%zero] : tensor<2x!qco.qubit>
    %q1_out, %bit1 = qco.measure %target : !qco.qubit
    %complete = qtensor.insert %q1_out into %partial[%one] : tensor<2x!qco.qubit>
    cbit.store %bit1, %bits[%one] : !cbit.reg<2>
    qtensor.dealloc %complete : tensor<2x!qco.qubit>
    return %bits : !cbit.reg<2>
  }
}
