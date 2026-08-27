// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[3] c;
ccx q[0], q[1], q[2];
c = measure q;
