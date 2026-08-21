// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

module {
  func.func @main() attributes {mqt.entry_point} {
    %q = qco.alloc : !qco.qubit
    %tensor = builtin.unrealized_conversion_cast %q : !qco.qubit to tensor<1x!qco.qubit>
    %q2 = builtin.unrealized_conversion_cast %tensor : tensor<1x!qco.qubit> to !qco.qubit
    qco.sink %q2 : !qco.qubit
    return
  }
}
