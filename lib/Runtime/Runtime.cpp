//===- Runtime.cpp - Lazy Evaluation Runtime Implementation ----------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the lazy evaluation runtime. All tensor operations
// record to the global trace instead of executing immediately. Only when
// a barrier is hit does the trace get compiled and executed.
//
//===----------------------------------------------------------------------===//

#include "jules/Runtime.h"
#include "jules/Diagnostics.h"
#include <cassert>
#include <utility>

namespace jules {

// ── LazyTensor implementation ───────────────────────────────────────────────

LazyTensor LazyTensor::constant(double value, std::unique_ptr<TypeNode> type,
                                 GlobalTracer &tracer, SourceLocation loc) {
  auto id = tracer.getActiveTrace().recordConstant(value, std::move(type), loc);
  return LazyTensor(id, tracer);
}

LazyTensor LazyTensor::zeros(std::vector<int64_t> shape,
                              std::unique_ptr<TypeNode> type,
                              GlobalTracer &tracer, SourceLocation loc) {
  auto id = tracer.getActiveTrace().recordTensorCreate(
      TraceOpKind::Zeros, std::move(shape), std::move(type), loc);
  return LazyTensor(id, tracer);
}

LazyTensor LazyTensor::ones(std::vector<int64_t> shape,
                             std::unique_ptr<TypeNode> type,
                             GlobalTracer &tracer, SourceLocation loc) {
  auto id = tracer.getActiveTrace().recordTensorCreate(
      TraceOpKind::Ones, std::move(shape), std::move(type), loc);
  return LazyTensor(id, tracer);
}

LazyTensor LazyTensor::random(std::vector<int64_t> shape,
                               std::unique_ptr<TypeNode> type,
                               GlobalTracer &tracer, SourceLocation loc) {
  auto id = tracer.getActiveTrace().recordTensorCreate(
      TraceOpKind::Random, std::move(shape), std::move(type), loc);
  return LazyTensor(id, tracer);
}

LazyTensor LazyTensor::add(const LazyTensor &other, SourceLocation loc) const {
  auto id = tracer_->getActiveTrace().recordBinary(
      TraceOpKind::Add, traceId_, other.traceId_, loc);
  return LazyTensor(id, *tracer_);
}

LazyTensor LazyTensor::sub(const LazyTensor &other, SourceLocation loc) const {
  auto id = tracer_->getActiveTrace().recordBinary(
      TraceOpKind::Sub, traceId_, other.traceId_, loc);
  return LazyTensor(id, *tracer_);
}

LazyTensor LazyTensor::mul(const LazyTensor &other, SourceLocation loc) const {
  auto id = tracer_->getActiveTrace().recordBinary(
      TraceOpKind::Mul, traceId_, other.traceId_, loc);
  return LazyTensor(id, *tracer_);
}

LazyTensor LazyTensor::div(const LazyTensor &other, SourceLocation loc) const {
  auto id = tracer_->getActiveTrace().recordBinary(
      TraceOpKind::Div, traceId_, other.traceId_, loc);
  return LazyTensor(id, *tracer_);
}

LazyTensor LazyTensor::pow(const LazyTensor &other, SourceLocation loc) const {
  auto id = tracer_->getActiveTrace().recordBinary(
      TraceOpKind::Pow, traceId_, other.traceId_, loc);
  return LazyTensor(id, *tracer_);
}

LazyTensor LazyTensor::neg(SourceLocation loc) const {
  auto id = tracer_->getActiveTrace().recordUnary(
      TraceOpKind::Neg, traceId_, loc);
  return LazyTensor(id, *tracer_);
}

LazyTensor LazyTensor::matmul(const LazyTensor &other,
                               SourceLocation loc) const {
  auto id = tracer_->getActiveTrace().recordMatMul(
      traceId_, other.traceId_, loc);
  return LazyTensor(id, *tracer_);
}

LazyTensor LazyTensor::relu(SourceLocation loc) const {
  auto id = tracer_->getActiveTrace().recordUnary(
      TraceOpKind::Relu, traceId_, loc);
  return LazyTensor(id, *tracer_);
}

LazyTensor LazyTensor::sigmoid(SourceLocation loc) const {
  auto id = tracer_->getActiveTrace().recordUnary(
      TraceOpKind::Sigmoid, traceId_, loc);
  return LazyTensor(id, *tracer_);
}

LazyTensor LazyTensor::tanh(SourceLocation loc) const {
  auto id = tracer_->getActiveTrace().recordUnary(
      TraceOpKind::Tanh, traceId_, loc);
  return LazyTensor(id, *tracer_);
}

LazyTensor LazyTensor::sqrt(SourceLocation loc) const {
  auto id = tracer_->getActiveTrace().recordUnary(
      TraceOpKind::Sqrt, traceId_, loc);
  return LazyTensor(id, *tracer_);
}

LazyTensor LazyTensor::exp(SourceLocation loc) const {
  auto id = tracer_->getActiveTrace().recordUnary(
      TraceOpKind::Exp, traceId_, loc);
  return LazyTensor(id, *tracer_);
}

LazyTensor LazyTensor::log(SourceLocation loc) const {
  auto id = tracer_->getActiveTrace().recordUnary(
      TraceOpKind::Log, traceId_, loc);
  return LazyTensor(id, *tracer_);
}

LazyTensor LazyTensor::abs(SourceLocation loc) const {
  auto id = tracer_->getActiveTrace().recordUnary(
      TraceOpKind::Abs, traceId_, loc);
  return LazyTensor(id, *tracer_);
}

LazyTensor LazyTensor::mean(SourceLocation loc) const {
  auto id = tracer_->getActiveTrace().recordReduction(
      TraceOpKind::Mean, traceId_, loc);
  return LazyTensor(id, *tracer_);
}

LazyTensor LazyTensor::sum(SourceLocation loc) const {
  auto id = tracer_->getActiveTrace().recordReduction(
      TraceOpKind::Sum, traceId_, loc);
  return LazyTensor(id, *tracer_);
}

LazyTensor LazyTensor::transpose(SourceLocation loc) const {
  auto id = tracer_->getActiveTrace().recordTranspose(traceId_, loc);
  return LazyTensor(id, *tracer_);
}

LazyTensor LazyTensor::reshape(std::vector<int64_t> newShape,
                                SourceLocation loc) const {
  auto id = tracer_->getActiveTrace().recordReshape(
      traceId_, std::move(newShape), loc);
  return LazyTensor(id, *tracer_);
}

LazyTensor LazyTensor::castTo(std::unique_ptr<TypeNode> targetType,
                                SourceLocation loc) const {
  auto id = tracer_->getActiveTrace().recordCast(
      traceId_, std::move(targetType), loc);
  return LazyTensor(id, *tracer_);
}

const TypeNode *LazyTensor::getType() const {
  auto &val = tracer_->getActiveTrace().getValue(traceId_);
  return val.type.get();
}

void LazyTensor::materialize() {
  // Force a host sync barrier, which triggers trace compilation.
  tracer_->emitBarrier(TraceBarrierKind::HostSync);
}

// ── RuntimeEnvironment implementation ────────────────────────────────────────

RuntimeEnvironment::RuntimeEnvironment(DiagnosticsEngine &diag)
    : diag_(diag) {
  scopes_.emplace_back();
}

void RuntimeEnvironment::bind(const std::string &name, LazyTensor value) {
  if (!scopes_.empty()) {
    scopes_.back()[name] = std::move(value);
  }
}

std::optional<LazyTensor> RuntimeEnvironment::lookup(
    const std::string &name) const {
  for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
    auto valIt = it->find(name);
    if (valIt != it->end()) {
      return valIt->second;
    }
  }
  return std::nullopt;
}

void RuntimeEnvironment::pushScope() { scopes_.emplace_back(); }

void RuntimeEnvironment::popScope() {
  if (scopes_.size() > 1) {
    scopes_.pop_back();
  }
}

void RuntimeEnvironment::flushTrace() {
  // The JIT compiler will handle this.
  // For now, we just increment the execution count.
  tracer_.incrementExecutionCount();
}

bool RuntimeEnvironment::hasPendingTrace() const {
  return tracer_.isTracing() && !tracer_.getActiveTrace().empty();
}

void RuntimeEnvironment::registerFunction(
    std::unique_ptr<FunctionDecl> fn) {
  tracer_.registerFunction(std::move(fn));
}

LazyTensor RuntimeEnvironment::callFunction(const std::string &name,
                                             std::vector<LazyTensor> args,
                                             SourceLocation loc) {
  // Look up the function in the tracer's function table.
  auto *fnDecl = tracer_.lookupFunction(name);
  if (!fnDecl) {
    diag_.error(loc, "undefined function '" + name + "'");
    // Return an invalid lazy tensor.
    return LazyTensor(kInvalidTraceValue, tracer_);
  }

  // Inline the function into the current trace by recording a Call op.
  // A full implementation would walk the function body and inline each
  // operation, but for cross-function tracing we use Call ops that the
  // MLIR lowering will later inline.
  std::vector<TraceValueId> argIds;
  argIds.reserve(args.size());
  for (const auto &arg : args) {
    argIds.push_back(arg.getTraceId());
  }

  // Determine the return type from the function declaration.
  auto returnType = fnDecl->getType()->getResult()
                        ? fnDecl->getType()->getResult()->clone()
                        : std::make_unique<ScalarType>(ScalarType::SK_F32);

  auto resultId = tracer_.getActiveTrace().recordCall(
      name, std::move(argIds), std::move(returnType), loc);

  return LazyTensor(resultId, tracer_);
}

} // namespace jules
