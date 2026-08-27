// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;
bit[2] c;

ctrl @ U(0.1, 0.2, 0.3) q[0], q[1];
c = measure q;
