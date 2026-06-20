//===- TypeSystem.cpp - Jules Type Checking & Shape Inference ---------------===//
//
// This file implements the TypeChecker, which walks the AST and:
//   1. Resolves variable references
//   2. Infers types for all expressions
//   3. Verifies shape compatibility for tensor operations
//   4. Performs symbolic dimension unification
//   5. Derives gradient types for autodiff
//
//===----------------------------------------------------------------------===//

#include "jules/TypeSystem.h"
#include <algorithm>
#include <cassert>
#include <sstream>

namespace jules {

// ── DimSubstitution ─────────────────────────────────────────────────────────

bool DimSubstitution::bind(const std::string &name, int64_t size) {
  auto it = bindings_.find(name);
  if (it != bindings_.end()) {
    return it->second == size; // Consistent with existing binding
  }
  bindings_[name] = size;
  return true;
}

int64_t DimSubstitution::lookup(const std::string &name) const {
  auto it = bindings_.find(name);
  return it != bindings_.end() ? it->second : -1;
}

bool DimSubstitution::merge(const DimSubstitution &other) {
  for (const auto &[name, size] : other.bindings_) {
    if (!bind(name, size)) return false;
  }
  return true;
}

// ── TypeChecker ─────────────────────────────────────────────────────────────

bool TypeChecker::checkProgram(Program &program) {
  hasError_ = false;

  // First pass: register all function and extern declarations.
  for (const auto &fn : program.getFunctions()) {
    functionDecls_[fn->getName()] = fn.get();
  }
  for (const auto &ext : program.getExterns()) {
    externDecls_[ext->getName()] = ext.get();
  }

  // Second pass: type-check each function body.
  pushScope();
  for (auto &fn : program.getFunctions()) {
    // FunctionDecl is not an ASTNode, so it has no accept(); call
    // visitFunctionDecl directly.
    visitFunctionDecl(*fn);
  }
  popScope();

  return !hasError_;
}

bool TypeChecker::checkExpr(Expr &expr) {
  expr.accept(*this);
  return !hasError_;
}

// ── Scope management ────────────────────────────────────────────────────────

void TypeChecker::pushScope() { scopes_.emplace_back(); }

void TypeChecker::popScope() {
  if (!scopes_.empty()) scopes_.pop_back();
}

void TypeChecker::addBinding(const std::string &name,
                              std::unique_ptr<TypeNode> type) {
  if (!scopes_.empty()) {
    scopes_.back().push_back({name, std::move(type)});
  }
}

const TypeNode *TypeChecker::lookupBinding(const std::string &name) const {
  for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
    for (const auto &binding : *it) {
      if (binding.name == name) {
        return binding.type.get();
      }
    }
  }
  return nullptr;
}

// ── Visitor implementations ─────────────────────────────────────────────────

void TypeChecker::visitFloatLiteralExpr(FloatLiteralExpr &expr) {
  expr.setResolvedType(std::make_unique<ScalarType>(ScalarType::SK_F64,
                                                      expr.getLocation()));
}

void TypeChecker::visitIntLiteralExpr(IntLiteralExpr &expr) {
  expr.setResolvedType(std::make_unique<ScalarType>(ScalarType::SK_I64,
                                                      expr.getLocation()));
}

void TypeChecker::visitBoolLiteralExpr(BoolLiteralExpr &expr) {
  expr.setResolvedType(std::make_unique<ScalarType>(ScalarType::SK_Bool,
                                                      expr.getLocation()));
}

void TypeChecker::visitUnitLiteralExpr(UnitLiteralExpr &expr) {
  expr.setResolvedType(std::make_unique<ScalarType>(ScalarType::SK_Unit,
                                                      expr.getLocation()));
}

void TypeChecker::visitIdentifierExpr(IdentifierExpr &expr) {
  // Look up in local bindings.
  const TypeNode *ty = lookupBinding(expr.getName());
  if (ty) {
    expr.setResolvedType(ty->clone());
    return;
  }

  // Look up in function declarations.
  auto fnIt = functionDecls_.find(expr.getName());
  if (fnIt != functionDecls_.end()) {
    expr.setResolvedType(fnIt->second->getType()->clone());
    return;
  }

  // Look up in extern declarations.
  auto extIt = externDecls_.find(expr.getName());
  if (extIt != externDecls_.end()) {
    expr.setResolvedType(extIt->second->getType()->clone());
    return;
  }

  // Check built-in functions.
  const TypeNode *builtinTy = getBuiltinType(expr.getName());
  if (builtinTy) {
    expr.setResolvedType(builtinTy->clone());
    return;
  }

  diag_.error(expr.getLocation(),
              "undefined variable '" + expr.getName() + "'");
  hasError_ = true;
}

void TypeChecker::visitTensorLiteralExpr(TensorLiteralExpr &expr) {
  // Type-check each element and collect element types.
  ScalarType::ScalarKind elemKind = ScalarType::SK_F64;
  for (auto &el : expr.getElements()) {
    el->accept(*this);
    if (auto *resolved = el->getResolvedType()) {
      if (resolved->getKind() == TypeNode::ScalarType) {
        auto &scalar = static_cast<const ScalarType &>(*resolved);
        // Promote to the widest scalar type.
        if (scalar.getScalarKind() == ScalarType::SK_F64 ||
            scalar.getScalarKind() == ScalarType::SK_F32) {
          elemKind = scalar.getScalarKind();
        }
      } else if (resolved->getKind() == TypeNode::TensorType) {
        // Nested tensor literal — not yet supported for full rank inference.
        elemKind = ScalarType::SK_F64;
      }
    }
  }

  auto tensorType = std::make_unique<TensorType>(
      std::vector<Dimension>{Dimension::concrete(
          static_cast<int64_t>(expr.getElements().size()))},
      elemKind, expr.getLocation());
  expr.setResolvedType(std::move(tensorType));
}

void TypeChecker::visitBinaryExpr(BinaryExpr &expr) {
  // getLHS()/getRHS() return const Expr*; const_cast to call accept()
  // (visitor pattern requires non-const this for in-place mutation).
  const_cast<Expr *>(expr.getLHS())->accept(*this);
  const_cast<Expr *>(expr.getRHS())->accept(*this);

  const TypeNode *lhsType = expr.getLHS()->getResolvedType();
  const TypeNode *rhsType = expr.getRHS()->getResolvedType();

  if (!lhsType || !rhsType) {
    hasError_ = true;
    return;
  }

  auto resultType = inferBinaryType(expr.getOp(), lhsType, rhsType,
                                     expr.getLocation());
  if (resultType) {
    expr.setResolvedType(std::move(resultType));
  } else {
    hasError_ = true;
    diag_.error(expr.getLocation(),
                std::string("type error: cannot apply '") +
                BinaryExpr::opToString(expr.getOp()) + "' to " +
                lhsType->toString() + " and " + rhsType->toString());
  }
}

void TypeChecker::visitUnaryExpr(UnaryExpr &expr) {
  const_cast<Expr *>(expr.getOperand())->accept(*this);

  const TypeNode *operandType = expr.getOperand()->getResolvedType();
  if (!operandType) {
    hasError_ = true;
    return;
  }

  switch (expr.getOp()) {
  case UnaryExpr::Negate:
    expr.setResolvedType(operandType->clone());
    break;
  case UnaryExpr::Not:
    expr.setResolvedType(std::make_unique<ScalarType>(ScalarType::SK_Bool,
                                                        expr.getLocation()));
    break;
  }
}

void TypeChecker::visitCallExpr(CallExpr &expr) {
  // Type-check callee
  const_cast<Expr *>(expr.getCallee())->accept(*this);

  // Type-check arguments
  for (auto &arg : expr.getArgs()) {
    const_cast<Expr &>(*arg).accept(*this);
  }

  const TypeNode *calleeType = expr.getCallee()->getResolvedType();
  if (!calleeType) {
    hasError_ = true;
    return;
  }

  // If the callee is a function type, the result type is the function's return type.
  if (calleeType->getKind() == TypeNode::FunctionType) {
    auto &fnType = static_cast<const FunctionType &>(*calleeType);
    expr.setResolvedType(fnType.getResult()->clone());
    return;
  }

  // Built-in operations that return based on their arguments.
  if (expr.getCallee()->getKind() == Expr::IdentifierExpr) {
    auto &idExpr = static_cast<const IdentifierExpr &>(*expr.getCallee());
    const std::string &name = idExpr.getName();

    if (name == "relu" || name == "sigmoid" || name == "tanh") {
      // Activation functions preserve the input type.
      if (!expr.getArgs().empty() && expr.getArgs()[0]->getResolvedType()) {
        expr.setResolvedType(expr.getArgs()[0]->getResolvedType()->clone());
        return;
      }
    }

    if (name == "zeros" || name == "ones" || name == "random") {
      // These create tensors from shape arguments. The type must be inferred
      // from the arguments. For now, default to the first argument's type
      // if it's a tensor, or create a new tensor type.
      if (expr.getNumArgs() >= 1 && expr.getArgs()[0]->getResolvedType()) {
        expr.setResolvedType(expr.getArgs()[0]->getResolvedType()->clone());
        return;
      }
      // Default to f32 tensor if no shape info available.
      expr.setResolvedType(std::make_unique<TensorType>(
          std::vector<Dimension>{Dimension::dynamic()},
          ScalarType::SK_F32, expr.getLocation()));
      return;
    }

    if (name == "mean" || name == "sum") {
      // Reduction: collapses all dimensions to scalar.
      if (!expr.getArgs().empty() && expr.getArgs()[0]->getResolvedType()) {
        auto *argType = expr.getArgs()[0]->getResolvedType();
        if (argType->getKind() == TypeNode::TensorType) {
          auto &tensorTy = static_cast<const TensorType &>(*argType);
          expr.setResolvedType(std::make_unique<TensorType>(
              std::vector<Dimension>{}, tensorTy.getElementKind(),
              expr.getLocation()));
          return;
        }
      }
      expr.setResolvedType(std::make_unique<ScalarType>(ScalarType::SK_F32,
                                                          expr.getLocation()));
      return;
    }

    if (name == "cross_entropy") {
      // Loss function returns a scalar.
      expr.setResolvedType(std::make_unique<TensorType>(
          std::vector<Dimension>{}, ScalarType::SK_F32, expr.getLocation()));
      return;
    }

    if (name == "sqrt" || name == "exp" || name == "log" || name == "abs") {
      // Math functions preserve input type.
      if (!expr.getArgs().empty() && expr.getArgs()[0]->getResolvedType()) {
        expr.setResolvedType(expr.getArgs()[0]->getResolvedType()->clone());
        return;
      }
    }

    if (name == "transpose") {
      if (!expr.getArgs().empty() && expr.getArgs()[0]->getResolvedType()) {
        auto *argType = expr.getArgs()[0]->getResolvedType();
        if (argType->getKind() == TypeNode::TensorType) {
          auto &tensorTy = static_cast<const TensorType &>(*argType);
          std::vector<Dimension> reversedDims(tensorTy.getDims().rbegin(),
                                               tensorTy.getDims().rend());
          expr.setResolvedType(std::make_unique<TensorType>(
              std::move(reversedDims), tensorTy.getElementKind(),
              expr.getLocation()));
          return;
        }
      }
    }

    if (name == "reshape") {
      // reshape(tensor, [dim1, dim2, ...]) — type from second arg.
      if (expr.getNumArgs() >= 2) {
        if (expr.getArgs()[0]->getResolvedType()) {
          auto *argType = expr.getArgs()[0]->getResolvedType();
          if (argType->getKind() == TypeNode::TensorType) {
            auto &tensorTy = static_cast<const TensorType &>(*argType);
            // For now, we just propagate the element type.
            // A full implementation would parse the shape argument.
            expr.setResolvedType(argType->clone());
            return;
          }
        }
      }
    }

    if (name == "concat") {
      // concat(t1, t2, axis) — infer from first arg.
      if (!expr.getArgs().empty() && expr.getArgs()[0]->getResolvedType()) {
        expr.setResolvedType(expr.getArgs()[0]->getResolvedType()->clone());
        return;
      }
    }
  }

  // Fallback: if we couldn't determine the type, mark as error.
  if (!expr.getResolvedType()) {
    diag_.error(expr.getLocation(), "cannot determine return type of call");
    hasError_ = true;
  }
}

void TypeChecker::visitIndexExpr(IndexExpr &expr) {
  const_cast<Expr *>(expr.getObject())->accept(*this);
  for (auto &idx : expr.getIndices()) {
    const_cast<Expr &>(*idx).accept(*this);
  }

  const TypeNode *objType = expr.getObject()->getResolvedType();
  if (!objType) {
    hasError_ = true;
    return;
  }

  if (objType->getKind() == TypeNode::TensorType) {
    auto &tensorTy = static_cast<const TensorType &>(*objType);
    // Each index reduces one dimension.
    size_t newIndexCount = expr.getIndices().size();
    if (newIndexCount >= tensorTy.getRank()) {
      // Fully indexed — result is scalar.
      expr.setResolvedType(std::make_unique<ScalarType>(
          tensorTy.getElementKind(), expr.getLocation()));
    } else {
      // Partially indexed — remove leading dimensions.
      std::vector<Dimension> newDims(
          tensorTy.getDims().begin() + newIndexCount,
          tensorTy.getDims().end());
      expr.setResolvedType(std::make_unique<TensorType>(
          std::move(newDims), tensorTy.getElementKind(),
          expr.getLocation()));
    }
  } else {
    expr.setResolvedType(objType->clone());
  }
}

void TypeChecker::visitLetExpr(LetExpr &expr) {
  const_cast<Expr *>(expr.getValue())->accept(*this);

  const TypeNode *valueType = expr.getValue()->getResolvedType();
  if (!valueType) {
    hasError_ = true;
    return;
  }

  // If there's a type annotation, verify it matches.
  if (expr.getTypeAnnotation()) {
    if (*valueType != *expr.getTypeAnnotation()) {
      diag_.error(expr.getLocation(),
                  "type annotation mismatch: let " + expr.getName() + " : " +
                  expr.getTypeAnnotation()->toString() + " = ... but value has type " +
                  valueType->toString());
      hasError_ = true;
    }
  }

  // Bind the variable in the current scope.
  pushScope();
  addBinding(expr.getName(), valueType->clone());

  // Type-check the body.
  const_cast<Expr *>(expr.getBody())->accept(*this);

  const TypeNode *bodyType = expr.getBody()->getResolvedType();
  if (bodyType) {
    expr.setResolvedType(bodyType->clone());
  }

  popScope();
}

void TypeChecker::visitLambdaExpr(LambdaExpr &expr) {
  pushScope();

  // Add parameter bindings.
  std::vector<std::unique_ptr<TypeNode>> paramTypes;
  for (const auto &param : expr.getParams()) {
    std::unique_ptr<TypeNode> paramType;
    if (param.type) {
      paramType = param.type->clone();
    } else {
      // Default parameter type to f32 if not annotated.
      paramType = std::make_unique<ScalarType>(ScalarType::SK_F32);
    }
    addBinding(param.name, paramType->clone());
    paramTypes.push_back(std::move(paramType));
  }

  // Type-check the body.
  const_cast<Expr *>(expr.getBody())->accept(*this);

  const TypeNode *bodyType = expr.getBody()->getResolvedType();
  if (bodyType) {
    auto resultType = bodyType->clone();
    expr.setResolvedType(std::make_unique<FunctionType>(
        std::move(paramTypes), std::move(resultType), expr.getLocation()));
  } else {
    hasError_ = true;
  }

  popScope();
}

void TypeChecker::visitIfExpr(IfExpr &expr) {
  const_cast<Expr *>(expr.getCondition())->accept(*this);
  const_cast<Expr *>(expr.getTrueBranch())->accept(*this);
  const_cast<Expr *>(expr.getFalseBranch())->accept(*this);

  const TypeNode *trueType = expr.getTrueBranch()->getResolvedType();
  const TypeNode *falseType = expr.getFalseBranch()->getResolvedType();

  if (!trueType || !falseType) {
    hasError_ = true;
    return;
  }

  if (*trueType != *falseType) {
    diag_.error(expr.getLocation(),
                "if-then-else type mismatch: " + trueType->toString() +
                " vs " + falseType->toString());
    hasError_ = true;
  }

  expr.setResolvedType(trueType->clone());
}

void TypeChecker::visitBlockExpr(BlockExpr &expr) {
  pushScope();

  for (auto &binding : expr.getBindings()) {
    const_cast<Expr &>(*binding.value).accept(*this);
    const TypeNode *valueType = binding.value->getResolvedType();
    if (valueType) {
      addBinding(binding.name, valueType->clone());
    }
  }

  const_cast<Expr *>(expr.getResult())->accept(*this);
  const TypeNode *resultType = expr.getResult()->getResolvedType();
  if (resultType) {
    expr.setResolvedType(resultType->clone());
  }

  popScope();
}

void TypeChecker::visitCastExpr(CastExpr &expr) {
  const_cast<Expr *>(expr.getExpr())->accept(*this);
  expr.setResolvedType(expr.getTargetType()->clone());
}

void TypeChecker::visitGRADExpr(GRADExpr &expr) {
  const_cast<Expr *>(expr.getFunction())->accept(*this);

  const TypeNode *fnType = expr.getFunction()->getResolvedType();
  if (!fnType) {
    hasError_ = true;
    return;
  }

  if (fnType->getKind() == TypeNode::FunctionType) {
    auto &funcType = static_cast<const FunctionType &>(*fnType);
    auto gradType = inferGradType(&funcType, expr.getDiffVar());
    if (gradType) {
      expr.setResolvedType(std::move(gradType));
    } else {
      diag_.error(expr.getLocation(),
                  "cannot compute gradient with respect to '" +
                  expr.getDiffVar() + "'");
      hasError_ = true;
    }
  } else {
    diag_.error(expr.getLocation(),
                "grad requires a function argument, got " +
                fnType->toString());
    hasError_ = true;
  }
}

void TypeChecker::visitFunctionDecl(FunctionDecl &fn) {
  pushScope();

  // Add parameter bindings from the function type.
  auto &params = fn.getParams();
  auto &typeParams = fn.getType()->getParams();

  for (size_t i = 0; i < params.size(); ++i) {
    std::unique_ptr<TypeNode> paramType;
    if (params[i].second) {
      paramType = params[i].second->clone();
    } else if (i < typeParams.size()) {
      paramType = typeParams[i]->clone();
    } else {
      paramType = std::make_unique<ScalarType>(ScalarType::SK_F32);
    }
    addBinding(params[i].first, std::move(paramType));
  }

  // Type-check the body.
  const_cast<Expr *>(fn.getBody())->accept(*this);

  // Verify the body type matches the declared return type.
  const TypeNode *bodyType = fn.getBody()->getResolvedType();
  const TypeNode *declaredResult = fn.getType()->getResult();
  if (bodyType && declaredResult && *bodyType != *declaredResult) {
    diag_.warning(fn.getLocation(),
                  "function '" + fn.getName() + "' declared return type " +
                  declaredResult->toString() + " but body has type " +
                  bodyType->toString());
  }

  popScope();
}

void TypeChecker::visitExternDecl(ExternDecl &ext) {
  // Nothing to check — externs are declarations only.
}

// ── Shape arithmetic ────────────────────────────────────────────────────────

std::unique_ptr<TypeNode> TypeChecker::inferBinaryType(BinaryExpr::Op op,
    const TypeNode *lhsType, const TypeNode *rhsType, SourceLocation loc) {
  switch (op) {
  case BinaryExpr::MatMul: {
    // Matrix multiply: [A, B] ** [B, C] -> [A, C]
    if (lhsType->getKind() == TypeNode::TensorType &&
        rhsType->getKind() == TypeNode::TensorType) {
      auto &lhsTensor = static_cast<const TensorType &>(*lhsType);
      auto &rhsTensor = static_cast<const TensorType &>(*rhsType);
      auto result = inferMatMulShape(&lhsTensor, &rhsTensor, loc);
      if (result) return result;
    }
    // Scalar fallback
    return std::make_unique<ScalarType>(ScalarType::SK_F32, loc);
  }

  case BinaryExpr::Pow: {
    // Power: element-wise, result type is the base type.
    if (lhsType->getKind() == TypeNode::TensorType) {
      return lhsType->clone();
    }
    return std::make_unique<ScalarType>(ScalarType::SK_F64, loc);
  }

  case BinaryExpr::Add:
  case BinaryExpr::Sub:
  case BinaryExpr::Mul:
  case BinaryExpr::Div:
  case BinaryExpr::Mod: {
    // Element-wise operations: broadcast if needed.
    return inferBroadcast(lhsType, rhsType);
  }

  case BinaryExpr::Eq:
  case BinaryExpr::Neq:
  case BinaryExpr::Lt:
  case BinaryExpr::Gt:
  case BinaryExpr::Leq:
  case BinaryExpr::Geq: {
    // Comparison returns bool (or tensor of bools with broadcast shape).
    auto broadcastType = inferBroadcast(lhsType, rhsType);
    if (broadcastType) {
      if (broadcastType->getKind() == TypeNode::TensorType) {
        auto &tensorTy = static_cast<const TensorType &>(*broadcastType);
        return std::make_unique<TensorType>(
            std::vector<Dimension>(tensorTy.getDims().begin(),
                                    tensorTy.getDims().end()),
            ScalarType::SK_Bool, loc);
      }
    }
    return std::make_unique<ScalarType>(ScalarType::SK_Bool, loc);
  }

  case BinaryExpr::And:
  case BinaryExpr::Or:
    return std::make_unique<ScalarType>(ScalarType::SK_Bool, loc);
  }

  return nullptr;
}

std::unique_ptr<TensorType> TypeChecker::inferMatMulShape(
    const TensorType *lhs, const TensorType *rhs, SourceLocation loc) {
  const auto &lhsDims = lhs->getDims();
  const auto &rhsDims = rhs->getDims();

  // 2D @ 2D: [M, K] ** [K, N] -> [M, N]
  if (lhsDims.size() == 2 && rhsDims.size() == 2) {
    // Verify the inner dimension matches.
    const Dimension &lhsInner = lhsDims[1];
    const Dimension &rhsInner = rhsDims[0];

    bool dimsCompatible = false;
    if (lhsInner == rhsInner) {
      dimsCompatible = true;
    } else if (lhsInner.kind == Dimension::DK_Dynamic ||
               rhsInner.kind == Dimension::DK_Dynamic) {
      dimsCompatible = true; // Dynamic dims are always compatible
    } else if (lhsInner.kind == Dimension::DK_Symbolic &&
               rhsInner.kind == Dimension::DK_Symbolic &&
               lhsInner.name == rhsInner.name) {
      dimsCompatible = true; // Same symbolic variable
    }

    if (!dimsCompatible) {
      diag_.warning(loc, "matmul inner dimension mismatch: " +
                    lhsInner.toString() + " vs " + rhsInner.toString());
    }

    std::vector<Dimension> resultDims = {lhsDims[0], rhsDims[1]};
    return std::make_unique<TensorType>(std::move(resultDims),
                                         lhs->getElementKind(), loc);
  }

  // Batched matmul: [B, M, K] ** [K, N] -> [B, M, N]
  // or [B, M, K] ** [B, K, N] -> [B, M, N]
  if (lhsDims.size() == 3 && rhsDims.size() == 2) {
    std::vector<Dimension> resultDims = {lhsDims[0], lhsDims[1], rhsDims[1]};
    return std::make_unique<TensorType>(std::move(resultDims),
                                         lhs->getElementKind(), loc);
  }

  if (lhsDims.size() == 3 && rhsDims.size() == 3) {
    std::vector<Dimension> resultDims = {lhsDims[0], lhsDims[1], rhsDims[2]};
    return std::make_unique<TensorType>(std::move(resultDims),
                                         lhs->getElementKind(), loc);
  }

  // 1D @ 1D (dot product): [N] ** [N] -> [] (scalar)
  if (lhsDims.size() == 1 && rhsDims.size() == 1) {
    return std::make_unique<TensorType>(
        std::vector<Dimension>{}, lhs->getElementKind(), loc);
  }

  // 2D @ 1D: [M, K] ** [K] -> [M]
  if (lhsDims.size() == 2 && rhsDims.size() == 1) {
    std::vector<Dimension> resultDims = {lhsDims[0]};
    return std::make_unique<TensorType>(std::move(resultDims),
                                         lhs->getElementKind(), loc);
  }

  diag_.error(loc, "unsupported matmul shape: " + lhs->toString() +
              " ** " + rhs->toString());
  return nullptr;
}

std::unique_ptr<TypeNode> TypeChecker::inferBroadcast(const TypeNode *lhs,
                                                       const TypeNode *rhs) {
  if (!lhs || !rhs) return nullptr;

  // Scalar + Scalar -> Scalar (promote to wider type)
  if (lhs->getKind() == TypeNode::ScalarType &&
      rhs->getKind() == TypeNode::ScalarType) {
    auto &lScalar = static_cast<const ScalarType &>(*lhs);
    auto &rScalar = static_cast<const ScalarType &>(*rhs);

    // Promote: f64 > f32 > i64 > i32 > bool
    auto promote = [](ScalarType::ScalarKind a,
                      ScalarType::ScalarKind b) -> ScalarType::ScalarKind {
      if (a == ScalarType::SK_F64 || b == ScalarType::SK_F64) return ScalarType::SK_F64;
      if (a == ScalarType::SK_F32 || b == ScalarType::SK_F32) return ScalarType::SK_F32;
      if (a == ScalarType::SK_I64 || b == ScalarType::SK_I64) return ScalarType::SK_I64;
      if (a == ScalarType::SK_I32 || b == ScalarType::SK_I32) return ScalarType::SK_I32;
      return ScalarType::SK_Bool;
    };

    return std::make_unique<ScalarType>(promote(lScalar.getScalarKind(),
                                                 rScalar.getScalarKind()));
  }

  // Tensor + Scalar -> Tensor (broadcast scalar)
  if (lhs->getKind() == TypeNode::TensorType &&
      rhs->getKind() == TypeNode::ScalarType) {
    return lhs->clone();
  }

  // Scalar + Tensor -> Tensor (broadcast scalar)
  if (lhs->getKind() == TypeNode::ScalarType &&
      rhs->getKind() == TypeNode::TensorType) {
    return rhs->clone();
  }

  // Tensor + Tensor -> broadcast
  if (lhs->getKind() == TypeNode::TensorType &&
      rhs->getKind() == TypeNode::TensorType) {
    auto &lTensor = static_cast<const TensorType &>(*lhs);
    auto &rTensor = static_cast<const TensorType &>(*rhs);
    const auto &lDims = lTensor.getDims();
    const auto &rDims = rTensor.getDims();

    // Simple broadcasting: if ranks match, compare dimension by dimension.
    // If one tensor has lower rank, prepend 1s.
    size_t maxRank = std::max(lDims.size(), rDims.size());
    std::vector<Dimension> resultDims(maxRank);

    auto getDim = [](const std::vector<Dimension> &dims, size_t idx) -> Dimension {
      if (idx < dims.size()) return dims[idx];
      return Dimension::concrete(1);
    };

    for (size_t i = 0; i < maxRank; ++i) {
      // Align from the right (broadcasting rules).
      size_t lIdx = lDims.size() >= maxRank ? i : i + maxRank - lDims.size();
      size_t rIdx = rDims.size() >= maxRank ? i : i + maxRank - rDims.size();

      Dimension lDim = lIdx < lDims.size() ? lDims[lIdx] : Dimension::concrete(1);
      Dimension rDim = rIdx < rDims.size() ? rDims[rIdx] : Dimension::concrete(1);

      if (lDim == rDim) {
        resultDims[i] = lDim;
      } else if (lDim.kind == Dimension::DK_Concrete && lDim.size == 1) {
        resultDims[i] = rDim;
      } else if (rDim.kind == Dimension::DK_Concrete && rDim.size == 1) {
        resultDims[i] = lDim;
      } else if (lDim.kind == Dimension::DK_Dynamic || rDim.kind == Dimension::DK_Dynamic) {
        resultDims[i] = Dimension::dynamic();
      } else if (lDim.kind == Dimension::DK_Symbolic) {
        resultDims[i] = lDim;
      } else if (rDim.kind == Dimension::DK_Symbolic) {
        resultDims[i] = rDim;
      } else {
        resultDims[i] = Dimension::dynamic();
      }
    }

    // Element type promotion.
    auto promote = [](ScalarType::ScalarKind a,
                      ScalarType::ScalarKind b) -> ScalarType::ScalarKind {
      if (a == ScalarType::SK_F64 || b == ScalarType::SK_F64) return ScalarType::SK_F64;
      if (a == ScalarType::SK_F32 || b == ScalarType::SK_F32) return ScalarType::SK_F32;
      if (a == ScalarType::SK_I64 || b == ScalarType::SK_I64) return ScalarType::SK_I64;
      if (a == ScalarType::SK_I32 || b == ScalarType::SK_I32) return ScalarType::SK_I32;
      return ScalarType::SK_Bool;
    };

    return std::make_unique<TensorType>(
        std::move(resultDims),
        promote(lTensor.getElementKind(), rTensor.getElementKind()));
  }

  return nullptr;
}

std::unique_ptr<FunctionType> TypeChecker::inferGradType(
    const FunctionType *fnType, const std::string &diffVar) {
  // grad(f, x) produces a function with the same parameter types as f,
  // but the return type is the same as the type of x (the gradient
  // has the same shape as the differentiated variable).
  //
  // For example, if f : [B,I]f32 -> [I,O]f32 -> [O]f32 -> []f32
  // and diffVar is the second parameter, then
  // grad(f, W) : [B,I]f32 -> [I,O]f32 -> [O]f32 -> [I,O]f32

  // Find the parameter matching diffVar.
  std::unique_ptr<TypeNode> gradResultType;
  for (const auto &param : fnType->getParams()) {
    if (param->getKind() == TypeNode::TensorType) {
      auto &tensorTy = static_cast<const TensorType &>(*param);
      // We don't have parameter names in FunctionType, so we use
      // the type itself. A full implementation would track names.
      gradResultType = param->clone();
      break; // Use the first tensor parameter as a reasonable default
    }
  }

  if (!gradResultType) {
    gradResultType = fnType->getResult()->clone();
  }

  std::vector<std::unique_ptr<TypeNode>> paramTypes;
  for (const auto &p : fnType->getParams()) {
    paramTypes.push_back(p->clone());
  }

  return std::make_unique<FunctionType>(std::move(paramTypes),
                                         std::move(gradResultType));
}

const TypeNode *TypeChecker::getBuiltinType(const std::string &name) const {
  // Built-in functions with their types.
  // These are lazily created and cached.
  static std::unordered_map<std::string, std::unique_ptr<FunctionType>> builtins;

  if (builtins.empty()) {
    // relu : [D]f32 -> [D]f32
    {
      std::vector<std::unique_ptr<TypeNode>> params;
      params.push_back(std::make_unique<TensorType>(
          std::vector<Dimension>{Dimension::dynamic()}, ScalarType::SK_F32));
      auto result = std::make_unique<TensorType>(
          std::vector<Dimension>{Dimension::dynamic()}, ScalarType::SK_F32);
      builtins["relu"] = std::make_unique<FunctionType>(
          std::move(params), std::move(result));
    }
    // sigmoid : [D]f32 -> [D]f32
    {
      std::vector<std::unique_ptr<TypeNode>> params;
      params.push_back(std::make_unique<TensorType>(
          std::vector<Dimension>{Dimension::dynamic()}, ScalarType::SK_F32));
      auto result = std::make_unique<TensorType>(
          std::vector<Dimension>{Dimension::dynamic()}, ScalarType::SK_F32);
      builtins["sigmoid"] = std::make_unique<FunctionType>(
          std::move(params), std::move(result));
    }
    // tanh : [D]f32 -> [D]f32
    {
      std::vector<std::unique_ptr<TypeNode>> params;
      params.push_back(std::make_unique<TensorType>(
          std::vector<Dimension>{Dimension::dynamic()}, ScalarType::SK_F32));
      auto result = std::make_unique<TensorType>(
          std::vector<Dimension>{Dimension::dynamic()}, ScalarType::SK_F32);
      builtins["tanh"] = std::make_unique<FunctionType>(
          std::move(params), std::move(result));
    }
    // mean : [D]f32 -> []f32
    {
      std::vector<std::unique_ptr<TypeNode>> params;
      params.push_back(std::make_unique<TensorType>(
          std::vector<Dimension>{Dimension::dynamic()}, ScalarType::SK_F32));
      auto result = std::make_unique<TensorType>(
          std::vector<Dimension>{}, ScalarType::SK_F32);
      builtins["mean"] = std::make_unique<FunctionType>(
          std::move(params), std::move(result));
    }
    // sum : [D]f32 -> []f32
    {
      std::vector<std::unique_ptr<TypeNode>> params;
      params.push_back(std::make_unique<TensorType>(
          std::vector<Dimension>{Dimension::dynamic()}, ScalarType::SK_F32));
      auto result = std::make_unique<TensorType>(
          std::vector<Dimension>{}, ScalarType::SK_F32);
      builtins["sum"] = std::make_unique<FunctionType>(
          std::move(params), std::move(result));
    }
    // zeros : [D]f32 -> [D]f32
    {
      std::vector<std::unique_ptr<TypeNode>> params;
      params.push_back(std::make_unique<TensorType>(
          std::vector<Dimension>{Dimension::dynamic()}, ScalarType::SK_F32));
      auto result = std::make_unique<TensorType>(
          std::vector<Dimension>{Dimension::dynamic()}, ScalarType::SK_F32);
      builtins["zeros"] = std::make_unique<FunctionType>(
          std::move(params), std::move(result));
    }
    // ones : [D]f32 -> [D]f32
    {
      std::vector<std::unique_ptr<TypeNode>> params;
      params.push_back(std::make_unique<TensorType>(
          std::vector<Dimension>{Dimension::dynamic()}, ScalarType::SK_F32));
      auto result = std::make_unique<TensorType>(
          std::vector<Dimension>{Dimension::dynamic()}, ScalarType::SK_F32);
      builtins["ones"] = std::make_unique<FunctionType>(
          std::move(params), std::move(result));
    }
    // random : [D]f32 -> [D]f32
    {
      std::vector<std::unique_ptr<TypeNode>> params;
      params.push_back(std::make_unique<TensorType>(
          std::vector<Dimension>{Dimension::dynamic()}, ScalarType::SK_F32));
      auto result = std::make_unique<TensorType>(
          std::vector<Dimension>{Dimension::dynamic()}, ScalarType::SK_F32);
      builtins["random"] = std::make_unique<FunctionType>(
          std::move(params), std::move(result));
    }
    // cross_entropy : [D]f32 -> [D]f32 -> []f32
    {
      std::vector<std::unique_ptr<TypeNode>> params;
      params.push_back(std::make_unique<TensorType>(
          std::vector<Dimension>{Dimension::dynamic()}, ScalarType::SK_F32));
      params.push_back(std::make_unique<TensorType>(
          std::vector<Dimension>{Dimension::dynamic()}, ScalarType::SK_F32));
      auto result = std::make_unique<TensorType>(
          std::vector<Dimension>{}, ScalarType::SK_F32);
      builtins["cross_entropy"] = std::make_unique<FunctionType>(
          std::move(params), std::move(result));
    }
    // sqrt : [D]f32 -> [D]f32
    {
      std::vector<std::unique_ptr<TypeNode>> params;
      params.push_back(std::make_unique<TensorType>(
          std::vector<Dimension>{Dimension::dynamic()}, ScalarType::SK_F32));
      auto result = std::make_unique<TensorType>(
          std::vector<Dimension>{Dimension::dynamic()}, ScalarType::SK_F32);
      builtins["sqrt"] = std::make_unique<FunctionType>(
          std::move(params), std::move(result));
    }
    // exp : [D]f32 -> [D]f32
    {
      std::vector<std::unique_ptr<TypeNode>> params;
      params.push_back(std::make_unique<TensorType>(
          std::vector<Dimension>{Dimension::dynamic()}, ScalarType::SK_F32));
      auto result = std::make_unique<TensorType>(
          std::vector<Dimension>{Dimension::dynamic()}, ScalarType::SK_F32);
      builtins["exp"] = std::make_unique<FunctionType>(
          std::move(params), std::move(result));
    }
    // log : [D]f32 -> [D]f32
    {
      std::vector<std::unique_ptr<TypeNode>> params;
      params.push_back(std::make_unique<TensorType>(
          std::vector<Dimension>{Dimension::dynamic()}, ScalarType::SK_F32));
      auto result = std::make_unique<TensorType>(
          std::vector<Dimension>{Dimension::dynamic()}, ScalarType::SK_F32);
      builtins["log"] = std::make_unique<FunctionType>(
          std::move(params), std::move(result));
    }
    // abs : [D]f32 -> [D]f32
    {
      std::vector<std::unique_ptr<TypeNode>> params;
      params.push_back(std::make_unique<TensorType>(
          std::vector<Dimension>{Dimension::dynamic()}, ScalarType::SK_F32));
      auto result = std::make_unique<TensorType>(
          std::vector<Dimension>{Dimension::dynamic()}, ScalarType::SK_F32);
      builtins["abs"] = std::make_unique<FunctionType>(
          std::move(params), std::move(result));
    }
    // transpose : [D1, D2]f32 -> [D2, D1]f32
    {
      std::vector<std::unique_ptr<TypeNode>> params;
      params.push_back(std::make_unique<TensorType>(
          std::vector<Dimension>{Dimension::dynamic(), Dimension::dynamic()},
          ScalarType::SK_F32));
      auto result = std::make_unique<TensorType>(
          std::vector<Dimension>{Dimension::dynamic(), Dimension::dynamic()},
          ScalarType::SK_F32);
      builtins["transpose"] = std::make_unique<FunctionType>(
          std::move(params), std::move(result));
    }
  }

  auto it = builtins.find(name);
  return it != builtins.end() ? it->second.get() : nullptr;
}

} // namespace jules
