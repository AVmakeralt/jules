//===- SymbolDCEPass.h - Symbol Dead Code Elimination Pass ------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Symbol Dead Code Elimination (SymbolDCE) pass.
//
// After whole-program graph collapsing, many function definitions and
// operations may have their outputs entirely hardcoded into their call
// sites. These functions and operations are now "dead" and can be
// stripped from the binary entirely.
//
// The SymbolDCE pass:
//   1. Identifies all symbols (functions, globals) that are reachable
//      from the entry point (main)
//   2. Removes all unreachable symbols
//   3. Within each function, removes operations whose results have
//      no users (standard DCE)
//
// This pass ensures that the final code given to XLA is a streamlined,
// minimal representation with no dead code bloat.
//
//===----------------------------------------------------------------------===//

#ifndef JULES_PASSES_SYMBOL_DCE_PASS_H
#define JULES_PASSES_SYMBOL_DCE_PASS_H

#include <memory>

namespace mlir {
class Pass;
}

namespace jules {

/// Create the Symbol Dead Code Elimination pass.
std::unique_ptr<mlir::Pass> createSymbolDCEPass();

} // namespace jules

#endif // JULES_PASSES_SYMBOL_DCE_PASS_H
