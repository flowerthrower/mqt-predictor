// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

OPENQASM 3.0;
include "stdgates.inc";
gate qft _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
  h _gate_q_3;
  cp(pi / 2) _gate_q_3, _gate_q_2;
  cp(pi / 4) _gate_q_3, _gate_q_1;
  cp(pi / 8) _gate_q_3, _gate_q_0;
  h _gate_q_2;
  cp(pi / 2) _gate_q_2, _gate_q_1;
  cp(pi / 4) _gate_q_2, _gate_q_0;
  h _gate_q_1;
  cp(pi / 2) _gate_q_1, _gate_q_0;
  h _gate_q_0;
  swap _gate_q_0, _gate_q_3;
  swap _gate_q_1, _gate_q_2;
}
bit[4] meas;
qubit[4] q;
qft q[0], q[1], q[2], q[3];
barrier q[0], q[1], q[2], q[3];
meas[0] = measure q[0];
meas[1] = measure q[1];
meas[2] = measure q[2];
meas[3] = measure q[3];
