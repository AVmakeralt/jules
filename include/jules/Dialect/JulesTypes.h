//===- JulesTypes.h - Jules MLIR Types -------------------------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file declares custom types for the Jules MLIR dialect. While most
// tensor types use MLIR's built-in RankedTensorType, Jules also defines:
//
//   - SymbolicDimType: a dimension that is a named symbolic variable
//     (e.g. "Batch", "Features") which gets resolved during shape inference
//
//   - BF16Type: brain floating-point 16-bit type for mixed precision
//   - FP8E4M3Type: 8-bit floating-point with 4 exponent, 3 mantissa bits
//   - FP8E5M2Type: 8-bit floating-point with 5 exponent, 2 mantissa bits
//
//===----------------------------------------------------------------------===//

#ifndef JULES_DIALECT_JULESTYPES_H
#define JULES_DIALECT_JULESTYPES_H

#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "llvm/ADT/StringRef.h"

// TableGen'd type declarations
#include "jules/Dialect/JulesOpsTypes.h.inc"

namespace jules {

// ── BF16 Type ───────────────────────────────────────────────────────────────
//
// Brain floating-point 16-bit type. This is distinct from MLIR's built-in
// BFloat16Type to allow custom handling in the Jules dialect, including
// custom parsing/printing and integration with the Jules type system.
//
// BF16 has 1 sign bit, 8 exponent bits, and 7 mantissa bits. It provides
// the same dynamic range as f32 but with reduced precision, making it ideal
// for deep learning workloads where range matters more than precision.

class BF16Type : public Type::TypeBase<BF16Type, Type, TypeStorage> {
public:
  using Base::Base;

  /// Dialect type name for parsing/printing.
  static constexpr llvm::StringLiteral name = "jules.bf16";

  /// Get or create a BF16Type in the given context.
  static BF16Type get(MLIRContext *ctx);

  /// Return the bit width of this type.
  unsigned getWidth() const { return 16; }

  /// Return the number of exponent bits.
  unsigned getExponentBits() const { return 8; }

  /// Return the number of mantissa bits.
  unsigned getMantissaBits() const { return 7; }
};

// ── FP8 E4M3 Type ───────────────────────────────────────────────────────────
//
// 8-bit floating-point type with 4 exponent bits and 3 mantissa bits.
// This format (E4M3) is designed for forward-pass computations in neural
// networks. It has a smaller dynamic range than E5M2 but higher precision,
// making it suitable for activations and weight storage.
//
// The format uses 1 sign bit, 4 exponent bits (bias=7), and 3 mantissa bits.
// Maximum representable value: 448.0. Does not support NaN or Inf.

class FP8E4M3Type : public Type::TypeBase<FP8E4M3Type, Type, TypeStorage> {
public:
  using Base::Base;

  /// Dialect type name for parsing/printing.
  static constexpr llvm::StringLiteral name = "jules.fp8_e4m3";

  /// Get or create an FP8E4M3Type in the given context.
  static FP8E4M3Type get(MLIRContext *ctx);

  /// Return the bit width of this type.
  unsigned getWidth() const { return 8; }

  /// Return the number of exponent bits.
  unsigned getExponentBits() const { return 4; }

  /// Return the number of mantissa bits.
  unsigned getMantissaBits() const { return 3; }
};

// ── FP8 E5M2 Type ───────────────────────────────────────────────────────────
//
// 8-bit floating-point type with 5 exponent bits and 2 mantissa bits.
// This format (E5M2) is designed for backward-pass computations (gradients)
// in neural networks. It has a larger dynamic range than E4M3 but lower
// precision, making it suitable for gradient representations.
//
// The format uses 1 sign bit, 5 exponent bits (bias=15), and 2 mantissa bits.
// Maximum representable value: 57344.0. Supports NaN and Inf.

class FP8E5M2Type : public Type::TypeBase<FP8E5M2Type, Type, TypeStorage> {
public:
  using Base::Base;

  /// Dialect type name for parsing/printing.
  static constexpr llvm::StringLiteral name = "jules.fp8_e5m2";

  /// Get or create an FP8E5M2Type in the given context.
  static FP8E5M2Type get(MLIRContext *ctx);

  /// Return the bit width of this type.
  unsigned getWidth() const { return 8; }

  /// Return the number of exponent bits.
  unsigned getExponentBits() const { return 5; }

  /// Return the number of mantissa bits.
  unsigned getMantissaBits() const { return 2; }
};

} // namespace jules

#endif // JULES_DIALECT_JULESTYPES_H
