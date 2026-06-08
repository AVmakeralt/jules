//===- StableHLOLowering.h - Jules to StableHLO Lowering Pass --------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file declares the lowering pass from the Jules MLIR dialect to the
// StableHLO dialect. The mapping is:
//
//   jules.matmul    -> stablehlo.dot_general
//   jules.add       -> stablehlo.add
//   jules.mul       -> stablehlo.multiply
//   jules.sub       -> stablehlo.subtract
//   jules.div       -> stablehlo.divide
//   jules.pow       -> stablehlo.power
//   jules.neg       -> stablehlo.negate
//   jules.relu      -> stablehlo.max(stablehlo.constant(0), x)
//   jules.sigmoid   -> stablehlo.logistic
//   jules.tanh      -> stablehlo.tanh
//   jules.mean      -> stablehlo.reduce + stablehlo.divide
//   jules.sum       -> stablehlo.reduce
//   jules.zeros     -> stablehlo.broadcast_in_dim(stablehlo.constant(0))
//   jules.ones      -> stablehlo.broadcast_in_dim(stablehlo.constant(1))
//   jules.random    -> stablehlo.rng_bit_generator + convert
//   jules.cast      -> stablehlo.convert
//   jules.transpose -> stablehlo.transpose
//   jules.reshape   -> stablehlo.reshape
//   jules.concat    -> stablehlo.concatenate
//   jules.slice     -> stablehlo.slice
//   jules.select    -> stablehlo.select
//   jules.cmp             -> stablehlo.compare
//   jules.constant        -> stablehlo.constant
//   jules.log             -> stablehlo.log
//   jules.pad             -> stablehlo.pad
//   jules.broadcast_in_dim -> stablehlo.broadcast_in_dim
//   jules.reduce          -> stablehlo.reduce
//   jules.while           -> stablehlo.while
//   jules.parallel        -> scf.parallel
//   jules.extern_kernel   -> stablehlo.custom_call
//   jules.func            -> func.func
//   jules.call      -> func.call
//
//===----------------------------------------------------------------------===//

#ifndef JULES_PASSES_STABLEHLOWERING_H
#define JULES_PASSES_STABLEHLOWERING_H

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace jules {

/// Populate conversion patterns from Jules dialect to StableHLO.
void populateJulesToStableHLOPatterns(mlir::RewritePatternSet &patterns,
                                       mlir::TypeConverter &typeConverter);

/// Get the type converter for Jules -> StableHLO.
mlir::TypeConverter getJulesToStableHLOTypeConverter();

} // namespace jules

#endif // JULES_PASSES_STABLEHLOWERING_H
