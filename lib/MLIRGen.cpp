//===- MLIRGen.cpp - AST to MLIR Lowering Implementation -------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the AST-to-MLIR lowering. It walks the Jules AST
// and emits MLIR operations in the Jules dialect, creating a valid ModuleOp
// with FuncOps for each top-level function declaration.
//
//===----------------------------------------------------------------------===//

#include "jules/MLIRGen.h"
#include "jules/AST.h"
#include "jules/Diagnostics.h"
#include "jules/Dialect/JulesDialect.h"
#include "jules/Dialect/JulesOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/Casting.h"

#include <cassert>
#include <cstddef>
#include <string>
#include <unordered_map>

using namespace mlir;
using namespace jules;

//===----------------------------------------------------------------------===//
// MLIRGenerator implementation
//===----------------------------------------------------------------------===//

MLIRGenerator::MLIRGenerator(MLIRContext &context, DiagnosticsEngine &diag)
    : context_(context), diag_(diag),
      module_(ModuleOp::create(UnknownLoc::get(&context))),
      builder_(&context),
      currentFunc_() {}

OwningOpRef<ModuleOp> MLIRGenerator::lowerProgram(Program &program) {
  // Register the Jules dialect.
  context_.getOrLoadDialect<JulesDialect>();
  context_.getOrLoadDialect<func::FuncDialect>();

  // Lower each function declaration.
  for (auto &fn : program.getFunctions()) {
    lowerFunctionDecl(*fn);
  }

  // Lower extern declarations as function declarations without bodies.
  for (auto &ext : program.getExterns()) {
    auto funcType = lowerType(*ext->getType()).dyn_cast<::mlir::FunctionType>();
    if (!funcType) {
      diag_.error(ext->getLocation(),
                  "extern '" + ext->getName() + "' must have a function type");
      continue;
    }

    auto funcOp = func::FuncOp::create(
        builder_.getUnknownLoc(), ext->getName(), funcType);
    funcOp.setPrivate();
    module_->push_back(funcOp);
  }

  // Verify the module.
  if (failed(::mlir::verify(*module_))) {
    diag_.error(SourceLocation{},
                "MLIR module verification failed after AST lowering");
    return nullptr;
  }

  return std::move(module_);
}

void MLIRGenerator::lowerFunctionDecl(FunctionDecl &fn) {
  // Convert the function type.
  auto funcType = lowerType(*fn.getType()).dyn_cast<::mlir::FunctionType>();
  if (!funcType) {
    diag_.error(fn.getLocation(),
                "function '" + fn.getName() + "' must have a function type");
    return;
  }

  // Create the function operation.
  auto loc = builder_.getUnknownLoc();
  auto funcOp = func::FuncOp::create(loc, fn.getName(), funcType);

  // Annotate arguments with their Jules names for the autodiff pass.
  auto &params = fn.getParams();
  for (unsigned i = 0; i < funcOp.getNumArguments() && i < params.size(); ++i) {
    funcOp.setArgAttr(i, "jules.name",
                      builder_.getStringAttr(params[i].first));
  }

  // Create the entry block.
  auto *entryBlock = funcOp.addEntryBlock();
  builder_.setInsertionPointToStart(entryBlock);

  currentFunc_ = funcOp;

  // Declare parameters as variables.
  pushScope();
  for (unsigned i = 0; i < funcOp.getNumArguments() && i < params.size(); ++i) {
    declareVariable(params[i].first, funcOp.getArgument(i));
  }

  // Lower the function body.
  auto bodyValue = lowerExpr(*const_cast<Expr *>(fn.getBody()));

  // Create a return operation.
  if (bodyValue) {
    builder_.create<func::ReturnOp>(loc, bodyValue);
  } else {
    // If lowering failed, create a return with a zero value.
    auto zeroType = funcType.getResult(0);
    auto zero = builder_.create<ConstantOp>(loc,
        builder_.getFloatAttr(FloatType::getF32(&context_), 0.0), zeroType);
    builder_.create<func::ReturnOp>(loc, zero.getResult());
  }

  popScope();

  module_->push_back(funcOp);
  currentFunc_ = func::FuncOp();
}

mlir::Value MLIRGenerator::lowerExpr(Expr &expr) {
  switch (expr.getKind()) {
  case Expr::FloatLiteralExpr: {
    auto &floatExpr = static_cast<FloatLiteralExpr &>(expr);
    auto f64Type = builder_.getF64Type();
    return builder_.create<ConstantOp>(
        expr.getLocation() == SourceLocation{} ? builder_.getUnknownLoc()
            : builder_.getUnknownLoc(),
        builder_.getFloatAttr(f64Type, floatExpr.getValue()), f64Type).getResult();
  }

  case Expr::IntLiteralExpr: {
    auto &intExpr = static_cast<IntLiteralExpr &>(expr);
    auto i64Type = builder_.getIntegerType(64);
    return builder_.create<ConstantOp>(
        builder_.getUnknownLoc(),
        builder_.getIntegerAttr(i64Type, intExpr.getValue()), i64Type).getResult();
  }

  case Expr::BoolLiteralExpr: {
    auto &boolExpr = static_cast<BoolLiteralExpr &>(expr);
    auto i1Type = builder_.getIntegerType(1);
    return builder_.create<ConstantOp>(
        builder_.getUnknownLoc(),
        builder_.getIntegerAttr(i1Type, boolExpr.getValue() ? 1 : 0), i1Type).getResult();
  }

  case Expr::UnitLiteralExpr: {
    // Unit values are represented as zero-element tuples or just a constant.
    auto f32Type = builder_.getF32Type();
    return builder_.create<ConstantOp>(
        builder_.getUnknownLoc(),
        builder_.getFloatAttr(f32Type, 0.0), f32Type).getResult();
  }

  case Expr::IdentifierExpr: {
    auto &idExpr = static_cast<IdentifierExpr &>(expr);
    auto val = lookupVariable(idExpr.getName());
    if (!val) {
      diag_.error(idExpr.getLocation(),
                  "undefined variable '" + idExpr.getName() + "'");
      return Value();
    }
    return val;
  }

  case Expr::TensorLiteralExpr: {
    auto &tensorExpr = static_cast<TensorLiteralExpr &>(expr);
    // Lower each element and collect their values.
    auto elementType = FloatType::getF32(&context_);
    SmallVector<Attribute, 8> elements;

    for (const auto &el : tensorExpr.getElements()) {
      auto elVal = lowerExpr(*const_cast<Expr *>(el.get()));
      if (!elVal) return Value();

      // Extract the constant value if possible.
      if (auto constOp = elVal.getDefiningOp<ConstantOp>()) {
        elements.push_back(constOp.getValueAttr());
      } else {
        // Non-constant element — we'll need to construct this differently.
        elements.push_back(builder_.getFloatAttr(elementType, 0.0));
      }
    }

    auto tensorType = RankedTensorType::get(
        {static_cast<int64_t>(elements.size())}, elementType);
    auto denseAttr = DenseFPElementsAttr::get(tensorType, elements);
    return builder_.create<ConstantOp>(builder_.getUnknownLoc(),
                                       denseAttr, tensorType).getResult();
  }

  case Expr::BinaryExpr: {
    auto &binExpr = static_cast<BinaryExpr &>(expr);
    auto lhs = lowerExpr(*const_cast<Expr *>(binExpr.getLHS()));
    auto rhs = lowerExpr(*const_cast<Expr *>(binExpr.getRHS()));
    if (!lhs || !rhs) return Value();

    auto loc = builder_.getUnknownLoc();

    switch (binExpr.getOp()) {
    case BinaryExpr::Add:    return builder_.create<AddOp>(loc, lhs, rhs).getResult();
    case BinaryExpr::Sub:    return builder_.create<SubOp>(loc, lhs, rhs).getResult();
    case BinaryExpr::Mul:    return builder_.create<MulOp>(loc, lhs, rhs).getResult();
    case BinaryExpr::Div:    return builder_.create<DivOp>(loc, lhs, rhs).getResult();
    case BinaryExpr::Mod:    return builder_.create<ModOp>(loc, lhs, rhs).getResult();
    case BinaryExpr::Pow:    return builder_.create<PowOp>(loc, lhs, rhs).getResult();
    case BinaryExpr::MatMul: return builder_.create<MatMulOp>(loc, lhs, rhs,
                                          builder_.getStringAttr(""),
                                          builder_.getStringAttr("")).getResult();
    case BinaryExpr::Eq:     return builder_.create<CmpOp>(loc, builder_.getI1Type(), lhs, rhs,
                                              builder_.getStringAttr("EQ")).getResult();
    case BinaryExpr::Neq:    return builder_.create<CmpOp>(loc, builder_.getI1Type(), lhs, rhs,
                                              builder_.getStringAttr("NE")).getResult();
    case BinaryExpr::Lt:     return builder_.create<CmpOp>(loc, builder_.getI1Type(), lhs, rhs,
                                              builder_.getStringAttr("LT")).getResult();
    case BinaryExpr::Gt:     return builder_.create<CmpOp>(loc, builder_.getI1Type(), lhs, rhs,
                                              builder_.getStringAttr("GT")).getResult();
    case BinaryExpr::Leq:    return builder_.create<CmpOp>(loc, builder_.getI1Type(), lhs, rhs,
                                              builder_.getStringAttr("LE")).getResult();
    case BinaryExpr::Geq:    return builder_.create<CmpOp>(loc, builder_.getI1Type(), lhs, rhs,
                                              builder_.getStringAttr("GE")).getResult();
    case BinaryExpr::And:
    case BinaryExpr::Or:
      // Logical and/or are more complex; use select for now.
      return lhs; // Placeholder
    }
    return Value();
  }

  case Expr::UnaryExpr: {
    auto &unaryExpr = static_cast<UnaryExpr &>(expr);
    auto operand = lowerExpr(*const_cast<Expr *>(unaryExpr.getOperand()));
    if (!operand) return Value();

    switch (unaryExpr.getOp()) {
    case UnaryExpr::Negate:
      return builder_.create<NegOp>(builder_.getUnknownLoc(), operand).getResult();
    case UnaryExpr::Not: {
      auto one = builder_.create<ConstantOp>(
          builder_.getUnknownLoc(),
          builder_.getIntegerAttr(builder_.getIntegerType(1), 1)).getResult();
      return builder_.create<SubOp>(builder_.getUnknownLoc(), one, operand).getResult();
    }
    }
    return Value();
  }

  case Expr::CallExpr: {
    auto &callExpr = static_cast<CallExpr &>(expr);

    // Lower arguments.
    SmallVector<Value, 4> args;
    for (const auto &arg : callExpr.getArgs()) {
      auto argVal = lowerExpr(*const_cast<Expr *>(arg.get()));
      if (!argVal) return Value();
      args.push_back(argVal);
    }

    auto loc = builder_.getUnknownLoc();

    // Handle built-in function calls.
    if (callExpr.getCallee()->getKind() == Expr::IdentifierExpr) {
      auto &callee = static_cast<const IdentifierExpr &>(*callExpr.getCallee());
      const std::string &name = callee.getName();

      if (name == "relu")    return builder_.create<ReluOp>(loc, args[0]).getResult();
      if (name == "sigmoid") return builder_.create<SigmoidOp>(loc, args[0]).getResult();
      if (name == "tanh")    return builder_.create<TanhOp>(loc, args[0]).getResult();
      if (name == "mean")    return builder_.create<MeanOp>(loc, args[0]).getResult();
      if (name == "sum")     return builder_.create<SumOp>(loc, args[0]).getResult();
      if (name == "sqrt")    return builder_.create<SqrtOp>(loc, args[0]).getResult();
      if (name == "exp")     return builder_.create<ExpOp>(loc, args[0]).getResult();
      if (name == "log")     return builder_.create<LogOp>(loc, args[0]).getResult();
      if (name == "abs")     return builder_.create<AbsOp>(loc, args[0]).getResult();
      if (name == "transpose") return builder_.create<TransposeOp>(loc, args[0]).getResult();
      if (name == "cross_entropy") {
        return builder_.create<CrossEntropyOp>(loc, args[0].getType(), args[0], args[1]).getResult();
      }

      // Tensor creation functions.
      if (name == "zeros" || name == "ones" || name == "random") {
        // The argument should be a shape specification.
        // For now, interpret the first argument as determining the output shape.
        auto argType = args[0].getType().dyn_cast<RankedTensorType>();
        if (argType) {
          auto resultType = RankedTensorType::get(argType.getShape(),
                                                    FloatType::getF32(&context_));
          if (name == "zeros")  return builder_.create<ZerosOp>(loc, resultType).getResult();
          if (name == "ones")   return builder_.create<OnesOp>(loc, resultType).getResult();
          if (name == "random") return builder_.create<RandomOp>(loc, resultType).getResult();
        }
        // If the arg is not a tensor, create a 1D result.
        auto resultType = RankedTensorType::get({1}, FloatType::getF32(&context_));
        if (name == "zeros")  return builder_.create<ZerosOp>(loc, resultType).getResult();
        if (name == "ones")   return builder_.create<OnesOp>(loc, resultType).getResult();
        return builder_.create<RandomOp>(loc, resultType).getResult();
      }

      // User-defined function call.
      auto funcOp = module_->lookupSymbol<func::FuncOp>(name);
      if (funcOp) {
        auto callOp = builder_.create<func::CallOp>(loc, funcOp, args);
        if (callOp.getNumResults() > 0) {
          return callOp.getResult(0);
        }
        return Value();
      }

      diag_.error(callExpr.getLocation(),
                  "undefined function '" + name + "'");
      return Value();
    }

    // If the callee is a lambda or other expression, lower it.
    auto calleeVal = lowerExpr(*const_cast<Expr *>(callExpr.getCallee()));
    if (!calleeVal) return Value();

    // We can't call a value directly in the Jules dialect.
    // This would require a jules.call_indirect op.
    diag_.error(callExpr.getLocation(),
                "indirect calls are not yet supported");
    return Value();
  }

  case Expr::IndexExpr: {
    auto &indexExpr = static_cast<IndexExpr &>(expr);
    auto object = lowerExpr(*const_cast<Expr *>(indexExpr.getObject()));
    if (!object) return Value();

    SmallVector<int64_t, 4> startIndices, limitIndices, strides;
    for (const auto &idx : indexExpr.getIndices()) {
      // For now, treat each index as a full-dimension slice.
      startIndices.push_back(0);
      if (auto intExpr = dynamic_cast<const IntLiteralExpr *>(idx.get())) {
        limitIndices.push_back(intExpr->getValue());
      } else {
        limitIndices.push_back(ShapedType::kDynamic);
      }
      strides.push_back(1);
    }

    auto inputType = object.getType().dyn_cast<RankedTensorType>();
    if (!inputType) return object;

    auto startAttr = builder_.getI64ArrayAttr(startIndices);
    auto limitAttr = builder_.getI64ArrayAttr(limitIndices);
    auto stridesAttr = builder_.getI64ArrayAttr(strides);

    return builder_.create<SliceOp>(
        builder_.getUnknownLoc(), object.getType(),
        object,
        builder_.getI64ArrayAttr(startIndices),
        builder_.getI64ArrayAttr(limitIndices),
        builder_.getI64ArrayAttr(strides)).getResult();
  }

  case Expr::LetExpr: {
    auto &letExpr = static_cast<LetExpr &>(expr);
    auto value = lowerExpr(*const_cast<Expr *>(letExpr.getValue()));
    if (!value) return Value();

    pushScope();
    declareVariable(letExpr.getName(), value);

    auto body = lowerExpr(*const_cast<Expr *>(letExpr.getBody()));
    popScope();

    return body;
  }

  case Expr::LambdaExpr: {
    // Lambdas are lowered as nested function definitions.
    auto &lambdaExpr = static_cast<LambdaExpr &>(expr);

    // Build the function type.
    SmallVector<Type, 4> paramTypes;
    for (const auto &param : lambdaExpr.getParams()) {
      if (param.type) {
        paramTypes.push_back(lowerType(*param.type));
      } else {
        paramTypes.push_back(FloatType::getF32(&context_));
      }
    }

    // Lower the body to determine the return type.
    // For now, use f32 as a default.
    auto resultType = FloatType::getF32(&context_);
    auto funcType = builder_.getFunctionType(paramTypes, resultType);

    // Create a unique name for the lambda.
    static int lambdaCounter = 0;
    auto lambdaName = "__lambda_" + std::to_string(lambdaCounter++);

    auto funcOp = func::FuncOp::create(builder_.getUnknownLoc(),
                                        lambdaName, funcType);
    auto *entryBlock = funcOp.addEntryBlock();

    // Save the current insertion point.
    auto savedInsertionPoint = builder_.saveInsertionPoint();
    builder_.setInsertionPointToStart(entryBlock);

    pushScope();
    for (unsigned i = 0; i < lambdaExpr.getParams().size() &&
                         i < funcOp.getNumArguments(); ++i) {
      declareVariable(lambdaExpr.getParams()[i].name,
                      funcOp.getArgument(i));
    }

    auto bodyValue = lowerExpr(*const_cast<Expr *>(lambdaExpr.getBody()));
    if (bodyValue) {
      builder_.create<func::ReturnOp>(builder_.getUnknownLoc(), bodyValue);
    }

    popScope();
    builder_.restoreInsertionPoint(savedInsertionPoint);

    module_->push_back(funcOp);

    // Return a reference to the function (as a symbol ref).
    // This is a simplification; a full implementation would return a
    // callable value.
    return Value(); // Lambda calls are handled through func::CallOp
  }

  case Expr::IfExpr: {
    auto &ifExpr = static_cast<IfExpr &>(expr);
    auto cond = lowerExpr(*const_cast<Expr *>(ifExpr.getCondition()));
    if (!cond) return Value();

    // Lower if-then-else as a jules.select.
    auto trueVal = lowerExpr(*const_cast<Expr *>(ifExpr.getTrueBranch()));
    auto falseVal = lowerExpr(*const_cast<Expr *>(ifExpr.getFalseBranch()));
    if (!trueVal || !falseVal) return Value();

    return builder_.create<SelectOp>(
        builder_.getUnknownLoc(), trueVal.getType(), cond, trueVal, falseVal).getResult();
  }

  case Expr::BlockExpr: {
    auto &blockExpr = static_cast<BlockExpr &>(expr);
    pushScope();

    for (const auto &binding : blockExpr.getBindings()) {
      auto val = lowerExpr(*const_cast<Expr *>(binding.value.get()));
      if (val) {
        declareVariable(binding.name, val);
      }
    }

    auto result = lowerExpr(*const_cast<Expr *>(blockExpr.getResult()));
    popScope();
    return result;
  }

  case Expr::CastExpr: {
    auto &castExpr = static_cast<CastExpr &>(expr);
    auto input = lowerExpr(*const_cast<Expr *>(castExpr.getExpr()));
    if (!input) return Value();

    auto targetType = lowerType(*castExpr.getTargetType());
    return builder_.create<CastOp>(
        builder_.getUnknownLoc(), targetType, input,
        TypeAttr::get(targetType)).getResult();
  }

  case Expr::GRADExpr: {
    auto &gradExpr = static_cast<GRADExpr &>(expr);
    auto fnVal = lowerExpr(*const_cast<Expr *>(gradExpr.getFunction()));
    if (!fnVal) return Value();

    return builder_.create<GradOp>(
        builder_.getUnknownLoc(), fnVal.getType(), fnVal,
        builder_.getStringAttr(gradExpr.getDiffVar())).getResult();
  }

  default:
    diag_.error(SourceLocation{},
                "unsupported expression kind in MLIR lowering");
    return Value();
  }
}

// ── Type lowering ───────────────────────────────────────────────────────────

Type MLIRGenerator::lowerType(const TypeNode &type) {
  switch (type.getKind()) {
  case TypeNode::ScalarType:
    return lowerScalarType(static_cast<const ScalarType &>(type));
  case TypeNode::TensorType:
    return lowerTensorType(static_cast<const TensorType &>(type));
  case TypeNode::FunctionType:
    return lowerFunctionType(static_cast<const FunctionType &>(type));
  default:
    return builder_.getF32Type();
  }
}

Type MLIRGenerator::lowerScalarType(const ScalarType &type) {
  switch (type.getScalarKind()) {
  case ScalarType::SK_F32:  return builder_.getF32Type();
  case ScalarType::SK_F64:  return builder_.getF64Type();
  case ScalarType::SK_I32:  return builder_.getIntegerType(32);
  case ScalarType::SK_I64:  return builder_.getIntegerType(64);
  case ScalarType::SK_Bool: return builder_.getIntegerType(1);
  case ScalarType::SK_Unit: return builder_.getF32Type(); // Unit represented as f32
  }
  return builder_.getF32Type();
}

Type MLIRGenerator::lowerTensorType(const TensorType &type) {
  auto elemType = lowerScalarType(ScalarType(type.getElementKind()));

  SmallVector<int64_t, 4> shape;
  for (const auto &dim : type.getDims()) {
    switch (dim.kind) {
    case Dimension::DK_Concrete:
      shape.push_back(dim.size);
      break;
    case Dimension::DK_Symbolic:
      // Symbolic dimensions are lowered as dynamic dimensions.
      shape.push_back(ShapedType::kDynamic);
      break;
    case Dimension::DK_Dynamic:
      shape.push_back(ShapedType::kDynamic);
      break;
    }
  }

  return RankedTensorType::get(shape, elemType);
}

Type MLIRGenerator::lowerFunctionType(const FunctionType &type) {
  SmallVector<Type, 4> inputTypes;
  for (const auto &param : type.getParams()) {
    inputTypes.push_back(lowerType(*param));
  }

  SmallVector<Type, 1> resultTypes;
  if (type.getResult()) {
    resultTypes.push_back(lowerType(*type.getResult()));
  }

  return builder_.getFunctionType(inputTypes, resultTypes);
}

// ── Variable table ─────────────────────────────────────────────────────────

void MLIRGenerator::declareVariable(const std::string &name, Value value) {
  if (!scopeStack_.empty()) {
    scopeStack_.back()[name] = value;
  }
}

Value MLIRGenerator::lookupVariable(const std::string &name) const {
  for (auto it = scopeStack_.rbegin(); it != scopeStack_.rend(); ++it) {
    auto valIt = it->find(name);
    if (valIt != it->end()) {
      return valIt->second;
    }
  }
  return Value();
}

void MLIRGenerator::pushScope() { scopeStack_.emplace_back(); }

void MLIRGenerator::popScope() {
  if (!scopeStack_.empty()) scopeStack_.pop_back();
}

//===----------------------------------------------------------------------===//
// Public entry point
//===----------------------------------------------------------------------===//

OwningOpRef<ModuleOp> jules::lowerASTToMLIR(MLIRContext &context,
                                                Program &program,
                                                DiagnosticsEngine &diag) {
  MLIRGenerator generator(context, diag);
  return generator.lowerProgram(program);
}
