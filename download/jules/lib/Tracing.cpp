//===- Tracing.cpp - Global Tracing Engine Implementation ------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the Global Tracing Engine. The engine records
// operations into a flat trace using std::vector indices (SSA values),
// minimizing allocation overhead during tracing.
//
//===----------------------------------------------------------------------===//

#include "jules/Tracing.h"
#include "jules/Diagnostics.h"
#include <cassert>
#include <utility>

namespace jules {

// ── ActiveTrace implementation ──────────────────────────────────────────────

void ActiveTrace::clear() {
  ops_.clear();
  values_.clear();
  barrierKind_.reset();
  returnValue_.reset();
}

TraceValueId ActiveTrace::allocateValue(std::unique_ptr<TypeNode> type,
                                         SourceLocation loc,
                                         const std::string &name) {
  TraceValueId id = static_cast<TraceValueId>(values_.size());
  values_.push_back({id, std::move(type), loc, name});
  return id;
}

const TraceValue &ActiveTrace::getValue(TraceValueId id) const {
  assert(id < values_.size() && "TraceValueId out of range");
  return values_[id];
}

TraceValue &ActiveTrace::getValue(TraceValueId id) {
  assert(id < values_.size() && "TraceValueId out of range");
  return values_[id];
}

size_t ActiveTrace::addOp(TraceOp op) {
  size_t idx = ops_.size();
  ops_.push_back(std::move(op));
  return idx;
}

// ── Operation recording ─────────────────────────────────────────────────────

TraceValueId ActiveTrace::recordUnary(TraceOpKind kind, TraceValueId input,
                                       SourceLocation loc) {
  // Infer result type from input type.
  auto &inputVal = getValue(input);
  auto resultType = inputVal.type ? inputVal.type->clone()
                                  : std::make_unique<ScalarType>(ScalarType::SK_F32);

  auto resultId = allocateValue(std::move(resultType), loc);

  TraceOp op;
  op.kind = kind;
  op.loc = loc;
  op.inputs = {input};
  op.outputs = {resultId};
  addOp(std::move(op));

  return resultId;
}

TraceValueId ActiveTrace::recordBinary(TraceOpKind kind, TraceValueId lhs,
                                        TraceValueId rhs, SourceLocation loc) {
  // Infer result type: for element-wise ops, use broadcasting.
  auto &lhsVal = getValue(lhs);
  auto resultType = lhsVal.type ? lhsVal.type->clone()
                                : std::make_unique<ScalarType>(ScalarType::SK_F32);

  auto resultId = allocateValue(std::move(resultType), loc);

  TraceOp op;
  op.kind = kind;
  op.loc = loc;
  op.inputs = {lhs, rhs};
  op.outputs = {resultId};
  addOp(std::move(op));

  return resultId;
}

TraceValueId ActiveTrace::recordConstant(double value,
                                          std::unique_ptr<TypeNode> type,
                                          SourceLocation loc) {
  auto resultId = allocateValue(std::move(type), loc);

  TraceOp op;
  op.kind = TraceOpKind::Constant;
  op.loc = loc;
  op.outputs = {resultId};
  op.attrs.push_back({TraceOp::Attr::Kind::Float, value});
  addOp(std::move(op));

  return resultId;
}

TraceValueId ActiveTrace::recordTensorCreate(TraceOpKind kind,
                                              std::vector<int64_t> shape,
                                              std::unique_ptr<TypeNode> type,
                                              SourceLocation loc) {
  auto resultId = allocateValue(std::move(type), loc);

  TraceOp op;
  op.kind = kind;
  op.loc = loc;
  op.outputs = {resultId};
  op.attrs.push_back({TraceOp::Attr::Kind::Shape, shape});
  addOp(std::move(op));

  return resultId;
}

TraceValueId ActiveTrace::recordMatMul(TraceValueId lhs, TraceValueId rhs,
                                        SourceLocation loc) {
  // Infer matmul result shape from lhs and rhs shapes.
  auto &lhsVal = getValue(lhs);
  auto &rhsVal = getValue(rhs);

  std::unique_ptr<TypeNode> resultType;

  if (lhsVal.type && lhsVal.type->getKind() == TypeNode::TensorType &&
      rhsVal.type && rhsVal.type->getKind() == TypeNode::TensorType) {
    auto &lhsTensor = static_cast<const TensorType &>(*lhsVal.type);
    auto &rhsTensor = static_cast<const TensorType &>(*rhsVal.type);
    const auto &lhsDims = lhsTensor.getDims();
    const auto &rhsDims = rhsTensor.getDims();

    // [M, K] ** [K, N] -> [M, N]
    if (lhsDims.size() == 2 && rhsDims.size() == 2) {
      std::vector<Dimension> resultDims = {lhsDims[0], rhsDims[1]};
      resultType = std::make_unique<TensorType>(std::move(resultDims),
                                                 lhsTensor.getElementKind(), loc);
    }
    // [B, M, K] ** [K, N] -> [B, M, N]
    else if (lhsDims.size() == 3 && rhsDims.size() == 2) {
      std::vector<Dimension> resultDims = {lhsDims[0], lhsDims[1], rhsDims[1]};
      resultType = std::make_unique<TensorType>(std::move(resultDims),
                                                 lhsTensor.getElementKind(), loc);
    }
    // [B, M, K] ** [B, K, N] -> [B, M, N]
    else if (lhsDims.size() == 3 && rhsDims.size() == 3) {
      std::vector<Dimension> resultDims = {lhsDims[0], lhsDims[1], rhsDims[2]};
      resultType = std::make_unique<TensorType>(std::move(resultDims),
                                                 lhsTensor.getElementKind(), loc);
    } else {
      resultType = std::make_unique<ScalarType>(ScalarType::SK_F32, loc);
    }
  } else {
    resultType = std::make_unique<ScalarType>(ScalarType::SK_F32, loc);
  }

  auto resultId = allocateValue(std::move(resultType), loc);

  TraceOp op;
  op.kind = TraceOpKind::MatMul;
  op.loc = loc;
  op.inputs = {lhs, rhs};
  op.outputs = {resultId};
  addOp(std::move(op));

  return resultId;
}

TraceValueId ActiveTrace::recordReduction(TraceOpKind kind, TraceValueId input,
                                           SourceLocation loc) {
  // Reduction produces a scalar tensor.
  auto &inputVal = getValue(input);
  std::unique_ptr<TypeNode> resultType;

  if (inputVal.type && inputVal.type->getKind() == TypeNode::TensorType) {
    auto &tensorTy = static_cast<const TensorType &>(*inputVal.type);
    resultType = std::make_unique<TensorType>(
        std::vector<Dimension>{}, tensorTy.getElementKind(), loc);
  } else {
    resultType = std::make_unique<ScalarType>(ScalarType::SK_F32, loc);
  }

  auto resultId = allocateValue(std::move(resultType), loc);

  TraceOp op;
  op.kind = kind;
  op.loc = loc;
  op.inputs = {input};
  op.outputs = {resultId};
  addOp(std::move(op));

  return resultId;
}

TraceValueId ActiveTrace::recordTranspose(TraceValueId input,
                                           SourceLocation loc) {
  auto &inputVal = getValue(input);
  std::unique_ptr<TypeNode> resultType;

  if (inputVal.type && inputVal.type->getKind() == TypeNode::TensorType) {
    auto &tensorTy = static_cast<const TensorType &>(*inputVal.type);
    std::vector<Dimension> reversedDims(tensorTy.getDims().rbegin(),
                                         tensorTy.getDims().rend());
    resultType = std::make_unique<TensorType>(std::move(reversedDims),
                                               tensorTy.getElementKind(), loc);
  } else {
    resultType = inputVal.type ? inputVal.type->clone()
                               : std::make_unique<ScalarType>(ScalarType::SK_F32);
  }

  auto resultId = allocateValue(std::move(resultType), loc);

  TraceOp op;
  op.kind = TraceOpKind::Transpose;
  op.loc = loc;
  op.inputs = {input};
  op.outputs = {resultId};
  addOp(std::move(op));

  return resultId;
}

TraceValueId ActiveTrace::recordReshape(TraceValueId input,
                                         std::vector<int64_t> newShape,
                                         SourceLocation loc) {
  auto &inputVal = getValue(input);
  std::unique_ptr<TypeNode> resultType;

  if (inputVal.type && inputVal.type->getKind() == TypeNode::TensorType) {
    auto &tensorTy = static_cast<const TensorType &>(*inputVal.type);
    std::vector<Dimension> dims;
    dims.reserve(newShape.size());
    for (int64_t d : newShape) {
      dims.push_back(d >= 0 ? Dimension::concrete(d) : Dimension::dynamic());
    }
    resultType = std::make_unique<TensorType>(std::move(dims),
                                               tensorTy.getElementKind(), loc);
  } else {
    resultType = std::make_unique<ScalarType>(ScalarType::SK_F32, loc);
  }

  auto resultId = allocateValue(std::move(resultType), loc);

  TraceOp op;
  op.kind = TraceOpKind::Reshape;
  op.loc = loc;
  op.inputs = {input};
  op.outputs = {resultId};
  op.attrs.push_back({TraceOp::Attr::Kind::Shape, newShape});
  addOp(std::move(op));

  return resultId;
}

TraceValueId ActiveTrace::recordCast(TraceValueId input,
                                      std::unique_ptr<TypeNode> targetType,
                                      SourceLocation loc) {
  auto resultId = allocateValue(std::move(targetType), loc);

  TraceOp op;
  op.kind = TraceOpKind::Cast;
  op.loc = loc;
  op.inputs = {input};
  op.outputs = {resultId};
  addOp(std::move(op));

  return resultId;
}

TraceValueId ActiveTrace::recordCmp(TraceOpKind kind, TraceValueId lhs,
                                     TraceValueId rhs, SourceLocation loc) {
  auto resultType = std::make_unique<ScalarType>(ScalarType::SK_Bool, loc);
  auto resultId = allocateValue(std::move(resultType), loc);

  TraceOp op;
  op.kind = kind;
  op.loc = loc;
  op.inputs = {lhs, rhs};
  op.outputs = {resultId};
  addOp(std::move(op));

  return resultId;
}

TraceValueId ActiveTrace::recordSelect(TraceValueId cond, TraceValueId trueVal,
                                        TraceValueId falseVal,
                                        SourceLocation loc) {
  // Result type matches the true/false value type.
  auto &trueValEntry = getValue(trueVal);
  auto resultType = trueValEntry.type ? trueValEntry.type->clone()
                                      : std::make_unique<ScalarType>(ScalarType::SK_F32);

  auto resultId = allocateValue(std::move(resultType), loc);

  TraceOp op;
  op.kind = TraceOpKind::Select;
  op.loc = loc;
  op.inputs = {cond, trueVal, falseVal};
  op.outputs = {resultId};
  addOp(std::move(op));

  return resultId;
}

TraceValueId ActiveTrace::recordCall(const std::string &callee,
                                      std::vector<TraceValueId> args,
                                      std::unique_ptr<TypeNode> returnType,
                                      SourceLocation loc) {
  auto resultId = allocateValue(std::move(returnType), loc);

  TraceOp op;
  op.kind = TraceOpKind::Call;
  op.loc = loc;
  op.inputs = std::move(args);
  op.outputs = {resultId};
  op.calleeName = callee;
  addOp(std::move(op));

  return resultId;
}

TraceValueId ActiveTrace::recordGrad(TraceValueId fn,
                                      const std::string &diffVar,
                                      SourceLocation loc) {
  auto &fnVal = getValue(fn);
  auto resultType = fnVal.type ? fnVal.type->clone()
                               : std::make_unique<ScalarType>(ScalarType::SK_F32);

  auto resultId = allocateValue(std::move(resultType), loc);

  TraceOp op;
  op.kind = TraceOpKind::Grad;
  op.loc = loc;
  op.inputs = {fn};
  op.outputs = {resultId};
  op.attrs.push_back({TraceOp::Attr::Kind::String, diffVar});
  addOp(std::move(op));

  return resultId;
}

void ActiveTrace::recordBarrier(TraceBarrierKind kind, SourceLocation loc) {
  barrierKind_ = kind;

  TraceOp op;
  op.kind = TraceOpKind::Barrier;
  op.loc = loc;
  addOp(std::move(op));
}

void ActiveTrace::recordReturn(TraceValueId value, SourceLocation loc) {
  returnValue_ = value;

  TraceOp op;
  op.kind = TraceOpKind::Return;
  op.loc = loc;
  op.inputs = {value};
  addOp(std::move(op));
}

// ── GlobalTracer implementation ─────────────────────────────────────────────

GlobalTracer::GlobalTracer() = default;

void GlobalTracer::beginTrace() {
  activeTrace_.clear();
  tracing_ = true;
  executionCount_ = 0;
}

ActiveTrace GlobalTracer::endTrace() {
  tracing_ = false;
  auto trace = std::move(activeTrace_);
  activeTrace_ = ActiveTrace{};
  return trace;
}

void GlobalTracer::emitBarrier(TraceBarrierKind kind) {
  activeTrace_.recordBarrier(kind);
}

FunctionDecl *GlobalTracer::lookupFunction(const std::string &name) {
  auto it = functions_.find(name);
  return it != functions_.end() ? it->second.get() : nullptr;
}

void GlobalTracer::registerFunction(std::unique_ptr<FunctionDecl> fn) {
  std::string name = fn->getName();
  functions_[std::move(name)] = std::move(fn);
}

} // namespace jules
