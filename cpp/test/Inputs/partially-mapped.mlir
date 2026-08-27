// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

module {
  func.func @main() attributes {mqt.entry_point} {
    %dynamic = qco.alloc : !qco.qubit
    %static = qco.static 0 : !qco.qubit
    %dynamic_out = qco.x %dynamic : !qco.qubit -> !qco.qubit
    %static_out = qco.x %static : !qco.qubit -> !qco.qubit
    qco.sink %dynamic_out : !qco.qubit
    qco.sink %static_out : !qco.qubit
    return
  }
}
