//===- Tracing.h - Global Tracing Engine -----------------------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Global Tracing Engine for the Jules JIT compiler.
//
// The tracing engine records the actual sequence of operations across
// multiple user functions, loops, and conditional closures into one
// continuous execution trace. Unlike per-function compilation, global
// tracing captures cross-function dataflow, enabling whole-program
// optimization before handing the trace to MLIR.
//
// Key design principles:
//   - Lazy evaluation: tensor operations record to the trace, never execute
//   - Flat index SSA: use std::vector indices, not heap-allocated IR nodes
//   - Trace barriers: I/O, host sync, and side effects flush the trace
//   - Cross-function inlining: called functions are inlined into the trace
//
//===----------------------------------------------------------------------===//

#ifndef JULES_TRACING_H
#define JULES_TRACING_H

#include "jules/AST.h"
#include "jules/Token.h"
#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace jules {

// ── Trace Value: a lightweight SSA reference ────────────────────────────────
//
// Instead of heap-allocating MLIR Values during tracing, we use a flat
// index into the trace's operation list. This gives O(1) access and
// minimal memory overhead.

using TraceValueId = uint32_t;

static constexpr TraceValueId kInvalidTraceValue = static_cast<TraceValueId>(-1);

/// A reference to a value inside a trace. Carries its type info so that
/// downstream lowering can construct the correct MLIR types without
/// re-computing them.
struct TraceValue {
  TraceValueId id = kInvalidTraceValue;

  /// The Jules type of this value (owned, because the trace owns all types).
  std::unique_ptr<TypeNode> type;

  /// Source location for diagnostics.
  SourceLocation loc;

  /// Human-readable name (for debugging; empty if unnamed).
  std::string name;

  bool isValid() const { return id != kInvalidTraceValue; }
};

// ── Trace Operation ─────────────────────────────────────────────────────────

/// The kind of operation recorded in the trace.
enum class TraceOpKind : uint16_t {
  // Tensor creation
  Constant,       // scalar or tensor constant
  Zeros,          // zeros tensor
  Ones,           // ones tensor
  Random,         // random tensor

  // Element-wise binary
  Add,
  Sub,
  Mul,
  Div,
  Mod,
  Pow,

  // Element-wise unary
  Neg,
  Relu,
  Sigmoid,
  Tanh,
  Sqrt,
  Exp,
  Log,
  Abs,

  // Matrix
  MatMul,

  // Reduction
  Mean,
  Sum,

  // Tensor manipulation
  Transpose,
  Reshape,
  Concat,
  Slice,
  Cast,

  // Comparison / logical
  CmpEq, CmpNeq, CmpLt, CmpGt, CmpLeq, CmpGeq,
  Select,

  // Autodiff
  Grad,

  // Function call (inlined into trace)
  Call,

  // Barrier: forces trace compilation
  Barrier,

  // Return from trace
  Return,
};

/// A single recorded operation in the global trace.
struct TraceOp {
  TraceOpKind       kind;
  SourceLocation    loc;

  /// Input operands (indices into the trace's value table).
  std::vector<TraceValueId> inputs;

  /// Output values produced by this operation.
  std::vector<TraceValueId> outputs;

  /// Optional attributes (e.g., shape for zeros, comparison direction for cmp).
  struct Attr {
    enum class Kind { Int, Float, String, Shape, Type };
    Kind kind;
    std::variant<int64_t, double, std::string, std::vector<int64_t>,
                 std::unique_ptr<TypeNode>>
        value;
  };
  std::vector<Attr> attrs;

  /// Name of the called function (for Call ops).
  std::string calleeName;
};

// ── Trace Barrier ───────────────────────────────────────────────────────────

/// Reasons why a trace must be flushed (compiled and executed).
enum class TraceBarrierKind {
  HostSync,       // User requested data back on CPU
  Print,          // I/O side effect
  ExternalCall,   // Call to an extern function
  ShapeGuard,     // Dynamic shape assertion failed
  UserBarrier,    // Explicit trace_barrier() call
};

// ── Active Trace ────────────────────────────────────────────────────────────

/// The global trace: a flat list of operations and their values.
/// This is the core data structure that accumulates operations lazily.
class ActiveTrace {
public:
  ActiveTrace() = default;

  /// Reset the trace (after compilation).
  void clear();

  /// Number of operations in the trace.
  size_t numOps() const { return ops_.size(); }

  /// Number of values in the trace.
  size_t numValues() const { return values_.size(); }

  /// Check if the trace is empty.
  bool empty() const { return ops_.empty(); }

  // ── Value management ─────────────────────────────────────────────────────

  /// Allocate a new trace value with the given type.
  TraceValueId allocateValue(std::unique_ptr<TypeNode> type,
                              SourceLocation loc = {},
                              const std::string &name = "");

  /// Look up a value by ID.
  const TraceValue &getValue(TraceValueId id) const;
  TraceValue &getValue(TraceValueId id);

  // ── Operation recording ──────────────────────────────────────────────────

  /// Record a unary operation.
  TraceValueId recordUnary(TraceOpKind kind, TraceValueId input,
                            SourceLocation loc = {});

  /// Record a binary operation.
  TraceValueId recordBinary(TraceOpKind kind, TraceValueId lhs,
                             TraceValueId rhs, SourceLocation loc = {});

  /// Record a constant.
  TraceValueId recordConstant(double value, std::unique_ptr<TypeNode> type,
                               SourceLocation loc = {});

  /// Record a tensor creation (zeros/ones/random).
  TraceValueId recordTensorCreate(TraceOpKind kind,
                                   std::vector<int64_t> shape,
                                   std::unique_ptr<TypeNode> type,
                                   SourceLocation loc = {});

  /// Record a matmul.
  TraceValueId recordMatMul(TraceValueId lhs, TraceValueId rhs,
                             SourceLocation loc = {});

  /// Record a reduction.
  TraceValueId recordReduction(TraceOpKind kind, TraceValueId input,
                                SourceLocation loc = {});

  /// Record a transpose.
  TraceValueId recordTranspose(TraceValueId input, SourceLocation loc = {});

  /// Record a reshape.
  TraceValueId recordReshape(TraceValueId input,
                              std::vector<int64_t> newShape,
                              SourceLocation loc = {});

  /// Record a cast.
  TraceValueId recordCast(TraceValueId input, std::unique_ptr<TypeNode> targetType,
                           SourceLocation loc = {});

  /// Record a comparison.
  TraceValueId recordCmp(TraceOpKind kind, TraceValueId lhs, TraceValueId rhs,
                          SourceLocation loc = {});

  /// Record a select.
  TraceValueId recordSelect(TraceValueId cond, TraceValueId trueVal,
                             TraceValueId falseVal, SourceLocation loc = {});

  /// Record a function call (inlined into the trace).
  TraceValueId recordCall(const std::string &callee,
                           std::vector<TraceValueId> args,
                           std::unique_ptr<TypeNode> returnType,
                           SourceLocation loc = {});

  /// Record a grad (autodiff) operation.
  TraceValueId recordGrad(TraceValueId fn, const std::string &diffVar,
                           SourceLocation loc = {});

  /// Record a barrier (forces trace compilation).
  void recordBarrier(TraceBarrierKind kind, SourceLocation loc = {});

  /// Record a return.
  void recordReturn(TraceValueId value, SourceLocation loc = {});

  // ── Access ───────────────────────────────────────────────────────────────

  const std::vector<TraceOp> &getOps() const { return ops_; }
  const std::vector<TraceValue> &getValues() const { return values_; }

  /// Get the barrier reason (if a barrier was recorded).
  std::optional<TraceBarrierKind> getBarrierKind() const { return barrierKind_; }

  /// Get the return value (if a return was recorded).
  std::optional<TraceValueId> getReturnValue() const { return returnValue_; }

private:
  std::vector<TraceOp>    ops_;
  std::vector<TraceValue> values_;
  std::optional<TraceBarrierKind> barrierKind_;
  std::optional<TraceValueId>     returnValue_;

  /// Add an operation and return the index.
  size_t addOp(TraceOp op);
};

// ── Global Tracer ───────────────────────────────────────────────────────────

/// The top-level tracing engine. Manages the active trace and provides
/// the interface for the lazy evaluation runtime.
class GlobalTracer {
public:
  GlobalTracer();

  /// Get the active trace.
  ActiveTrace &getActiveTrace() { return activeTrace_; }

  /// Begin a new trace (clears any existing trace).
  void beginTrace();

  /// End the current trace and return it for compilation.
  /// After this call, the active trace is empty.
  ActiveTrace endTrace();

  /// Check if a trace is currently being recorded.
  bool isTracing() const { return tracing_; }

  /// Record a trace barrier. This triggers compilation of the current trace.
  void emitBarrier(TraceBarrierKind kind);

  /// Look up a function by name in the function table.
  FunctionDecl *lookupFunction(const std::string &name);

  /// Register a function in the function table.
  void registerFunction(std::unique_ptr<FunctionDecl> fn);

  /// The number of times the current trace has been executed (for PGO).
  uint64_t getTraceExecutionCount() const { return executionCount_; }
  void incrementExecutionCount() { ++executionCount_; }

private:
  ActiveTrace  activeTrace_;
  bool         tracing_ = false;
  uint64_t     executionCount_ = 0;

  /// Function table for cross-function tracing.
  std::unordered_map<std::string, std::unique_ptr<FunctionDecl>> functions_;
};

} // namespace jules

#endif // JULES_TRACING_H
