//===- Runtime.h - Lazy Evaluation Runtime ---------------------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines the lazy evaluation runtime for the Jules JIT compiler.
//
// In the Jules runtime, tensor objects are **placeholders** (lazy tensors)
// that hold a virtual TraceValueId. When operations occur, they don't execute
// on the GPU — they record to the global trace instead. Only when a barrier
// is hit (e.g., host sync, I/O) does the trace get compiled and executed.
//
// This design gives the JIT maximum visibility into the computation graph,
// enabling cross-function optimization, kernel fusion, and PGO.
//
//===----------------------------------------------------------------------===//

#ifndef JULES_RUNTIME_H
#define JULES_RUNTIME_H

#include "jules/Tracing.h"
#include "jules/AST.h"
#include "jules/Diagnostics.h"
#include <cassert>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace jules {

// ── Lazy Tensor ─────────────────────────────────────────────────────────────

/// A lazy tensor: holds a TraceValueId instead of actual data.
/// Operations on LazyTensors record to the global trace.
class LazyTensor {
public:
  /// Default constructor (for use in containers like std::tuple).
  LazyTensor() = default;

  /// Create a lazy tensor from a trace value.
  explicit LazyTensor(TraceValueId traceId, GlobalTracer &tracer)
      : traceId_(traceId), tracer_(&tracer) {}

  /// Create a lazy tensor for a constant scalar.
  static LazyTensor constant(double value, std::unique_ptr<TypeNode> type,
                              GlobalTracer &tracer, SourceLocation loc = {});

  /// Create a zero-filled tensor.
  static LazyTensor zeros(std::vector<int64_t> shape,
                           std::unique_ptr<TypeNode> type,
                           GlobalTracer &tracer, SourceLocation loc = {});

  /// Create a one-filled tensor.
  static LazyTensor ones(std::vector<int64_t> shape,
                          std::unique_ptr<TypeNode> type,
                          GlobalTracer &tracer, SourceLocation loc = {});

  /// Create a random tensor.
  static LazyTensor random(std::vector<int64_t> shape,
                            std::unique_ptr<TypeNode> type,
                            GlobalTracer &tracer, SourceLocation loc = {});

  // ── Operations (all lazy — record to trace) ──────────────────────────────

  LazyTensor add(const LazyTensor &other, SourceLocation loc = {}) const;
  LazyTensor sub(const LazyTensor &other, SourceLocation loc = {}) const;
  LazyTensor mul(const LazyTensor &other, SourceLocation loc = {}) const;
  LazyTensor div(const LazyTensor &other, SourceLocation loc = {}) const;
  LazyTensor pow(const LazyTensor &other, SourceLocation loc = {}) const;
  LazyTensor neg(SourceLocation loc = {}) const;
  LazyTensor matmul(const LazyTensor &other, SourceLocation loc = {}) const;
  LazyTensor relu(SourceLocation loc = {}) const;
  LazyTensor sigmoid(SourceLocation loc = {}) const;
  LazyTensor tanh(SourceLocation loc = {}) const;
  LazyTensor sqrt(SourceLocation loc = {}) const;
  LazyTensor exp(SourceLocation loc = {}) const;
  LazyTensor log(SourceLocation loc = {}) const;
  LazyTensor abs(SourceLocation loc = {}) const;
  LazyTensor mean(SourceLocation loc = {}) const;
  LazyTensor sum(SourceLocation loc = {}) const;
  LazyTensor transpose(SourceLocation loc = {}) const;
  LazyTensor reshape(std::vector<int64_t> newShape,
                      SourceLocation loc = {}) const;
  LazyTensor castTo(std::unique_ptr<TypeNode> targetType,
                     SourceLocation loc = {}) const;

  // ── Accessors ────────────────────────────────────────────────────────────

  TraceValueId getTraceId() const { return traceId_; }
  const TypeNode *getType() const;

  /// Force materialization of this tensor (triggers trace compilation).
  /// Returns a handle to the computed data.
  void materialize();

private:
  TraceValueId   traceId_;
  GlobalTracer  *tracer_;
};

// ── Runtime Environment ─────────────────────────────────────────────────────

/// The runtime environment: manages the global tracer, variable bindings,
/// and the JIT compilation/execution pipeline.
class RuntimeEnvironment {
public:
  explicit RuntimeEnvironment(DiagnosticsEngine &diag);

  /// Get the global tracer.
  GlobalTracer &getTracer() { return tracer_; }

  // ── Variable bindings ────────────────────────────────────────────────────

  /// Bind a name to a lazy tensor value.
  void bind(const std::string &name, LazyTensor value);

  /// Look up a variable by name.
  std::optional<LazyTensor> lookup(const std::string &name) const;

  /// Push a new scope.
  void pushScope();

  /// Pop the current scope.
  void popScope();

  // ── JIT execution ────────────────────────────────────────────────────────

  /// Force compilation and execution of the current trace.
  /// This is called when a barrier is hit.
  void flushTrace();

  /// Check if there is an active trace that needs compilation.
  bool hasPendingTrace() const;

  // ── Function registration ────────────────────────────────────────────────

  /// Register a user-defined function.
  void registerFunction(std::unique_ptr<FunctionDecl> fn);

  /// Call a registered function (inlines into the current trace).
  LazyTensor callFunction(const std::string &name,
                           std::vector<LazyTensor> args,
                           SourceLocation loc = {});

private:
  DiagnosticsEngine &diag_;
  GlobalTracer       tracer_;

  /// Scoped variable table.
  std::vector<std::unordered_map<std::string, LazyTensor>> scopes_;
};

} // namespace jules

#endif // JULES_RUNTIME_H
