// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

OPENQASM 3.0;
include "stdgates.inc";

qubit q0;
qubit q1;
qubit q2;
bit c0;
bit c1;
bit c2;

h q0;
rz(pi / 8) q0;
rz(pi / 8) q0;
cx q0, q1;
cx q1, q2;
cx q0, q2;
c0 = measure q0;
c1 = measure q1;
c2 = measure q2;
