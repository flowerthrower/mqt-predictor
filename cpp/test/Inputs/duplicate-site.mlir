// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

module {
  func.func @main() attributes {mqt.entry_point} {
    %q0 = qco.static 0 : !qco.qubit
    %q1 = qco.static 0 : !qco.qubit
    qco.sink %q0 : !qco.qubit
    qco.sink %q1 : !qco.qubit
    return
  }
}
