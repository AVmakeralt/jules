//===- AST.cpp - Jules AST Implementation ----------------------------------===//

#include "jules/AST.h"
#include <sstream>

namespace jules {

// ── Dimension ───────────────────────────────────────────────────────────────

std::string Dimension::toString() const {
  switch (kind) {
  case DK_Symbolic: return name;
  case DK_Concrete: return std::to_string(size);
  case DK_Dynamic:  return "?";
  }
  return "?";
}

// ── ScalarType ──────────────────────────────────────────────────────────────

std::string ScalarType::toString() const {
  switch (kind_) {
  case SK_F32:    return "f32";
  case SK_F64:    return "f64";
  case SK_I32:    return "i32";
  case SK_I64:    return "i64";
  case SK_Bool:   return "bool";
  case SK_Unit:   return "unit";
  case SK_BF16:   return "bf16";
  case SK_FP8E4M3: return "fp8e4m3";
  case SK_FP8E5M2: return "fp8e5m2";
  }
  return "unknown";
}

bool ScalarType::operator==(const TypeNode &other) const {
  if (other.getKind() != Kind::ScalarType) return false;
  auto &o = static_cast<const ScalarType &>(other);
  return kind_ == o.kind_;
}

std::unique_ptr<TypeNode> ScalarType::clone() const {
  return std::make_unique<ScalarType>(kind_, loc_);
}

ScalarType::ScalarKind ScalarType::keywordToScalarKind(TokenKind kw) {
  switch (kw) {
  case TokenKind::KwF32:  return SK_F32;
  case TokenKind::KwF64:  return SK_F64;
  case TokenKind::KwI32:  return SK_I32;
  case TokenKind::KwI64:  return SK_I64;
  case TokenKind::KwBool: return SK_Bool;
  case TokenKind::KwUnit: return SK_Unit;
  default:
    return SK_F32; // Should not happen, but provide a safe default
  }
}

// ── TensorType ──────────────────────────────────────────────────────────────

std::string TensorType::toString() const {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < dims_.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << dims_[i].toString();
  }
  oss << "]" << jules::ScalarType(element_).toString();
  return oss.str();
}

bool TensorType::operator==(const TypeNode &other) const {
  if (other.getKind() != Kind::TensorType) return false;
  auto &o = static_cast<const TensorType &>(other);
  if (dims_.size() != o.dims_.size()) return false;
  if (element_ != o.element_) return false;
  for (size_t i = 0; i < dims_.size(); ++i) {
    if (dims_[i] != o.dims_[i]) return false;
  }
  return true;
}

std::unique_ptr<TypeNode> TensorType::clone() const {
  return std::make_unique<TensorType>(dims_, element_, loc_);
}

std::unique_ptr<TensorType> TensorType::resolve(
    const std::vector<std::pair<std::string, int64_t>> &bindings) const {
  std::vector<Dimension> resolvedDims;
  resolvedDims.reserve(dims_.size());

  for (const auto &dim : dims_) {
    if (dim.kind == Dimension::DK_Symbolic) {
      bool found = false;
      for (const auto &[name, size] : bindings) {
        if (name == dim.name) {
          resolvedDims.push_back(Dimension::concrete(size));
          found = true;
          break;
        }
      }
      if (!found) {
        resolvedDims.push_back(dim); // Keep symbolic if not resolved
      }
    } else {
      resolvedDims.push_back(dim);
    }
  }

  return std::make_unique<TensorType>(std::move(resolvedDims), element_, loc_);
}

// ── FunctionType ────────────────────────────────────────────────────────────

std::string FunctionType::toString() const {
  std::ostringstream oss;
  for (size_t i = 0; i < params_.size(); ++i) {
    if (i > 0) oss << " -> ";
    oss << params_[i]->toString();
  }
  if (result_) {
    oss << " -> " << result_->toString();
  }
  return oss.str();
}

bool FunctionType::operator==(const TypeNode &other) const {
  if (other.getKind() != Kind::FunctionType) return false;
  auto &o = static_cast<const FunctionType &>(other);
  if (params_.size() != o.params_.size()) return false;
  for (size_t i = 0; i < params_.size(); ++i) {
    if (*params_[i] != *o.params_[i]) return false;
  }
  if (result_ && o.result_) {
    return *result_ == *o.result_;
  }
  return !result_ && !o.result_;
}

std::unique_ptr<TypeNode> FunctionType::clone() const {
  std::vector<std::unique_ptr<TypeNode>> clonedParams;
  clonedParams.reserve(params_.size());
  for (const auto &p : params_) {
    clonedParams.push_back(p->clone());
  }
  auto clonedResult = result_ ? result_->clone() : nullptr;
  return std::make_unique<FunctionType>(std::move(clonedParams),
                                         std::move(clonedResult), loc_);
}

// ── Expression clone() implementations ──────────────────────────────────────

std::unique_ptr<Expr> FloatLiteralExpr::clone() const {
  auto e = std::make_unique<FloatLiteralExpr>(value_, loc_);
  if (resolvedType_) e->setResolvedType(resolvedType_->clone());
  return e;
}

std::unique_ptr<Expr> IntLiteralExpr::clone() const {
  auto e = std::make_unique<IntLiteralExpr>(value_, loc_);
  if (resolvedType_) e->setResolvedType(resolvedType_->clone());
  return e;
}

std::unique_ptr<Expr> BoolLiteralExpr::clone() const {
  auto e = std::make_unique<BoolLiteralExpr>(value_, loc_);
  if (resolvedType_) e->setResolvedType(resolvedType_->clone());
  return e;
}

std::unique_ptr<Expr> UnitLiteralExpr::clone() const {
  auto e = std::make_unique<UnitLiteralExpr>(loc_);
  if (resolvedType_) e->setResolvedType(resolvedType_->clone());
  return e;
}

std::unique_ptr<Expr> IdentifierExpr::clone() const {
  auto e = std::make_unique<IdentifierExpr>(name_, loc_);
  if (resolvedType_) e->setResolvedType(resolvedType_->clone());
  return e;
}

std::unique_ptr<Expr> TensorLiteralExpr::clone() const {
  std::vector<std::unique_ptr<Expr>> clonedElems;
  clonedElems.reserve(elements_.size());
  for (const auto &el : elements_) {
    clonedElems.push_back(el->clone());
  }
  auto e = std::make_unique<TensorLiteralExpr>(std::move(clonedElems), loc_);
  if (resolvedType_) e->setResolvedType(resolvedType_->clone());
  return e;
}

std::unique_ptr<Expr> BinaryExpr::clone() const {
  auto e = std::make_unique<BinaryExpr>(op_, lhs_->clone(), rhs_->clone(), loc_);
  if (resolvedType_) e->setResolvedType(resolvedType_->clone());
  return e;
}

std::unique_ptr<Expr> UnaryExpr::clone() const {
  auto e = std::make_unique<UnaryExpr>(op_, operand_->clone(), loc_);
  if (resolvedType_) e->setResolvedType(resolvedType_->clone());
  return e;
}

std::unique_ptr<Expr> CallExpr::clone() const {
  auto clonedCallee = callee_->clone();
  std::vector<std::unique_ptr<Expr>> clonedArgs;
  clonedArgs.reserve(args_.size());
  for (const auto &a : args_) {
    clonedArgs.push_back(a->clone());
  }
  auto e = std::make_unique<CallExpr>(std::move(clonedCallee),
                                       std::move(clonedArgs), loc_);
  if (resolvedType_) e->setResolvedType(resolvedType_->clone());
  return e;
}

std::unique_ptr<Expr> IndexExpr::clone() const {
  auto clonedObj = object_->clone();
  std::vector<std::unique_ptr<Expr>> clonedIndices;
  clonedIndices.reserve(indices_.size());
  for (const auto &idx : indices_) {
    clonedIndices.push_back(idx->clone());
  }
  auto e = std::make_unique<IndexExpr>(std::move(clonedObj),
                                        std::move(clonedIndices), loc_);
  if (resolvedType_) e->setResolvedType(resolvedType_->clone());
  return e;
}

std::unique_ptr<Expr> LetExpr::clone() const {
  auto e = std::make_unique<LetExpr>(name_, value_->clone(), body_->clone(), loc_);
  if (typeAnnotation_) e->setTypeAnnotation(typeAnnotation_->clone());
  if (resolvedType_) e->setResolvedType(resolvedType_->clone());
  return e;
}

std::unique_ptr<Expr> LambdaExpr::clone() const {
  std::vector<Param> clonedParams;
  clonedParams.reserve(params_.size());
  for (const auto &p : params_) {
    Param cp;
    cp.name = p.name;
    cp.type = p.type ? p.type->clone() : nullptr;
    clonedParams.push_back(std::move(cp));
  }
  auto e = std::make_unique<LambdaExpr>(std::move(clonedParams),
                                          body_->clone(), loc_);
  if (resolvedType_) e->setResolvedType(resolvedType_->clone());
  return e;
}

std::unique_ptr<Expr> IfExpr::clone() const {
  auto e = std::make_unique<IfExpr>(cond_->clone(), trueBranch_->clone(),
                                     falseBranch_->clone(), loc_);
  if (resolvedType_) e->setResolvedType(resolvedType_->clone());
  return e;
}

std::unique_ptr<Expr> BlockExpr::clone() const {
  std::vector<Binding> clonedBindings;
  clonedBindings.reserve(bindings_.size());
  for (const auto &b : bindings_) {
    Binding cb;
    cb.name = b.name;
    cb.value = b.value->clone();
    cb.typeAnnotation = b.typeAnnotation ? b.typeAnnotation->clone() : nullptr;
    clonedBindings.push_back(std::move(cb));
  }
  auto e = std::make_unique<BlockExpr>(std::move(clonedBindings),
                                        result_->clone(), loc_);
  if (resolvedType_) e->setResolvedType(resolvedType_->clone());
  return e;
}

std::unique_ptr<Expr> CastExpr::clone() const {
  auto e = std::make_unique<CastExpr>(expr_->clone(), targetType_->clone(), loc_);
  if (resolvedType_) e->setResolvedType(resolvedType_->clone());
  return e;
}

std::unique_ptr<Expr> GRADExpr::clone() const {
  auto e = std::make_unique<GRADExpr>(fn_->clone(), diffVar_, loc_);
  if (resolvedType_) e->setResolvedType(resolvedType_->clone());
  return e;
}

// ── BinaryExpr helpers ──────────────────────────────────────────────────────

const char *BinaryExpr::opToString(Op op) {
  switch (op) {
  case Add:    return "+";
  case Sub:    return "-";
  case Mul:    return "*";
  case Div:    return "/";
  case Mod:    return "%";
  case Pow:    return "^";
  case MatMul: return "**";
  case Eq:     return "==";
  case Neq:    return "!=";
  case Lt:     return "<";
  case Gt:     return ">";
  case Leq:    return "<=";
  case Geq:    return ">=";
  case And:    return "&&";
  case Or:     return "||";
  }
  return "?";
}

const char *UnaryExpr::opToString(Op op) {
  switch (op) {
  case Negate: return "-";
  case Not:    return "!";
  }
  return "?";
}

// ── Program ─────────────────────────────────────────────────────────────────

const FunctionDecl *Program::findFunction(const std::string &name) const {
  for (const auto &fn : functions_) {
    if (fn->getName() == name) return fn.get();
  }
  return nullptr;
}

const ExternDecl *Program::findExtern(const std::string &name) const {
  for (const auto &ext : externs_) {
    if (ext->getName() == name) return ext.get();
  }
  return nullptr;
}

} // namespace jules
