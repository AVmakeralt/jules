//===- LLVMBackend.h - LLVM-Based CPU Backend for Jules ---------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file declares the LLVM-based CPU backend for the Jules MLIR compiler.
// It replaces the regex-based CPUTensorInterpreter with a proper MLIR → LLVM
// dialect lowering + ORC JIT compilation pipeline.
//
// The backend supports two compilation paths:
//
//   Primary path (full MLIR lowering):
//     Jules dialect → arith/math/func dialects → LLVM dialect → LLVM IR → JIT
//
//   Fallback path (direct LLVM IR generation):
//     Parse MLIR text → extract op sequences → generate LLVM IR → JIT
//
// The fallback path ensures the backend works even when the full MLIR
// conversion infrastructure is not linked or when dialect conversion
// passes encounter unsupported operations.
//
//===----------------------------------------------------------------------===//

#ifndef JULES_LLVM_BACKEND_H
#define JULES_LLVM_BACKEND_H

#include "jules/Diagnostics.h"
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlir {
class ModuleOp;
class MLIRContext;
} // namespace mlir

namespace llvm {
class Module;
class LLVMContext;
class Function;
class Value;
class BasicBlock;
class IRBuilderBase;
class Type;
} // namespace llvm

namespace jules {

// ── Tensor Descriptor ────────────────────────────────────────────────────────

/// Runtime descriptor for a tensor passed to/from JIT-compiled functions.
/// This mirrors the memref descriptor layout expected by MLIR's LLVM lowering.
struct TensorDescriptor {
  float   *data;           ///< Pointer to contiguous float data
  int64_t  offset;         ///< Offset from the base pointer
  int64_t  sizes[6];       ///< Size of each dimension (max rank 6)
  int64_t  strides[6];     ///< Stride of each dimension
  int32_t  rank;           ///< Number of dimensions
};

// ── Operation Record for Fallback Path ───────────────────────────────────────

/// Describes a single operation extracted from MLIR text for the fallback
/// compilation path. This is a proper structured representation, not a
/// regex-based hack.
struct LLVMOpsRecord {
  enum OpKind : uint8_t {
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Relu,
    Sigmoid,
    Tanh,
    MatMul,
    Max,
    Min,
    Exp,
    Log,
    Constant,
    Reshape,
    Transpose,
    Reduce,
    Copy,
  };

  OpKind kind;

  /// SSA value indices.
  int input1 = -1;
  int input2 = -1;
  int output = -1;

  /// Shape of the output tensor.
  std::vector<int64_t> outputShape;

  /// For constant ops: the raw float data.
  std::vector<float> constData;
};

// ── Compiled Module Cache Entry ──────────────────────────────────────────────

/// Represents a compiled module cached for reuse.
struct CompiledModuleEntry {
  void *entryPoint = nullptr;       ///< JIT'd function pointer
  std::string moduleKey;            ///< Cache key (hash of MLIR text)
  uint64_t compileTimeMs = 0;       ///< How long compilation took
  uint64_t executionCount = 0;      ///< How many times executed
};

// ── LLVM Backend ─────────────────────────────────────────────────────────────

/// The LLVM-based CPU backend: compiles MLIR modules to native code via
/// MLIR lowering + LLVM ORC JIT, with a fallback path that generates
/// LLVM IR directly from parsed MLIR text.
class LLVMBackend {
public:
  explicit LLVMBackend(DiagnosticsEngine &diag);
  ~LLVMBackend();

  /// Compile an MLIR module to a native function pointer and execute it.
  /// \param module     The MLIR module to compile
  /// \param entryPoint The name of the entry function
  /// \param args       Array of argument pointers (typically TensorDescriptor*)
  /// \return           The return value pointer, or nullptr on failure
  void *compileAndExecute(mlir::ModuleOp module,
                          const std::string &entryPoint,
                          void **args);

  /// Compile an MLIR module and return the JIT'd function address.
  /// \param module     The MLIR module to compile
  /// \param entryPoint The name of the entry function
  /// \return           Function pointer, or nullptr on failure
  void *compile(mlir::ModuleOp module, const std::string &entryPoint);

  /// Compile from an MLIR text string (fallback path).
  /// Parses the MLIR text, generates LLVM IR directly, and JIT compiles.
  void *compileFromText(const std::string &mlirText,
                        const std::string &entryPoint);

  /// Execute a previously compiled function.
  void *executeCached(const std::string &moduleKey,
                      const std::string &entryPoint,
                      void **args);

  /// Check if a module is already compiled and cached.
  bool isCompiled(const std::string &moduleKey) const;

  /// Get the number of cached modules.
  size_t cacheSize() const;

  /// Clear the compiled module cache.
  void clearCache();

  /// Get the diagnostics engine.
  DiagnosticsEngine &getDiag() { return diag_; }

  /// Enable/disable verbose logging.
  void setVerbose(bool v) { verbose_ = v; }

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  DiagnosticsEngine &diag_;
  bool verbose_ = false;
};

} // namespace jules

#endif // JULES_LLVM_BACKEND_H
