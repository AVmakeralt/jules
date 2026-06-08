//===- AST.h - Jules Abstract Syntax Tree ----------------------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Abstract Syntax Tree nodes for the Jules language.
// The AST is a fully-typed, immutable tree produced by the Parser and consumed
// by the type-checker and MLIR lowering passes.
//
// The tree is designed around expressions (the language is expression-based).
// Every node carries a SourceLocation for diagnostics.
//
//===----------------------------------------------------------------------===//

#ifndef JULES_AST_H
#define JULES_AST_H

#include "jules/Token.h"
#include "jules/Diagnostics.h"
#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace jules {

class ASTVisitor;

// ── Forward declarations ────────────────────────────────────────────────────

#define JULES_AST_NODE_KINDS(F)                                               \
  F(FloatLiteralExpr)                                                         \
  F(IntLiteralExpr)                                                           \
  F(BoolLiteralExpr)                                                          \
  F(UnitLiteralExpr)                                                          \
  F(IdentifierExpr)                                                           \
  F(TensorLiteralExpr)                                                        \
  F(BinaryExpr)                                                               \
  F(UnaryExpr)                                                                \
  F(CallExpr)                                                                 \
  F(IndexExpr)                                                                \
  F(LetExpr)                                                                  \
  F(LambdaExpr)                                                               \
  F(IfExpr)                                                                   \
  F(BlockExpr)                                                                \
  F(CastExpr)                                                                 \
  F(GRADExpr)                                                                 \
  F(FunctionDecl)                                                             \
  F(ExternDecl)                                                               \
  F(TypeAnnotation)                                                           \
  F(TensorType)                                                               \
  F(ScalarType)                                                               \
  F(FunctionType)                                                             \
  F(Program)

#define FORWARD_DECLARE(KIND) class KIND;
JULES_AST_NODE_KINDS(FORWARD_DECLARE)
#undef FORWARD_DECLARE

// ── Type AST ────────────────────────────────────────────────────────────────

/// Base class for all type AST nodes.
class TypeNode {
public:
  enum Kind {
#define TYPE_KIND(KIND) KIND,
    JULES_AST_NODE_KINDS(TYPE_KIND)
#undef TYPE_KIND
  };

  virtual ~TypeNode() = default;
  virtual Kind getKind() const = 0;
  virtual std::string toString() const = 0;
  virtual bool operator==(const TypeNode &other) const = 0;
  bool operator!=(const TypeNode &other) const { return !(*this == other); }

  /// Deep-clone this type node.
  virtual std::unique_ptr<TypeNode> clone() const = 0;
};

/// Scalar types: f32, f64, i32, i64, bool, unit.
class ScalarType : public TypeNode {
public:
  enum ScalarKind { SK_F32, SK_F64, SK_I32, SK_I64, SK_Bool, SK_Unit };

  explicit ScalarType(ScalarKind kind, SourceLocation loc = {})
      : kind_(kind), loc_(loc) {}

  Kind getKind() const override { return Kind::ScalarType; }
  ScalarKind getScalarKind() const { return kind_; }
  SourceLocation getLocation() const { return loc_; }

  std::string toString() const override;
  bool operator==(const TypeNode &other) const override;
  std::unique_ptr<TypeNode> clone() const override;

  static ScalarKind keywordToScalarKind(TokenKind kw);

private:
  ScalarKind     kind_;
  SourceLocation loc_;
};

/// Tensor type: [D1, D2, ...]ElementType
/// Dimensions can be symbolic (identifiers) or concrete (integers) or dynamic
/// (represented by a question mark).
struct Dimension {
  enum DimKind { DK_Symbolic, DK_Concrete, DK_Dynamic };

  DimKind     kind;
  std::string name;    // for DK_Symbolic
  int64_t     size;    // for DK_Concrete

  static Dimension symbolic(const std::string &n) {
    return {DK_Symbolic, n, 0};
  }
  static Dimension concrete(int64_t s) {
    return {DK_Concrete, "", s};
  }
  static Dimension dynamic() {
    return {DK_Dynamic, "?", 0};
  }

  bool operator==(const Dimension &o) const {
    if (kind != o.kind) return false;
    switch (kind) {
    case DK_Symbolic: return name == o.name;
    case DK_Concrete: return size == o.size;
    case DK_Dynamic:  return true;
    }
    return false;
  }
  bool operator!=(const Dimension &o) const { return !(*this == o); }

  std::string toString() const;
};

class TensorType : public TypeNode {
public:
  TensorType(std::vector<Dimension> dims, ScalarType::ScalarKind element,
             SourceLocation loc = {})
      : dims_(std::move(dims)), element_(element), loc_(loc) {}

  Kind getKind() const override { return Kind::TensorType; }
  const std::vector<Dimension> &getDims() const { return dims_; }
  ScalarType::ScalarKind getElementKind() const { return element_; }
  SourceLocation getLocation() const { return loc_; }
  unsigned getRank() const { return static_cast<unsigned>(dims_.size()); }

  std::string toString() const override;
  bool operator==(const TypeNode &other) const override;
  std::unique_ptr<TypeNode> clone() const override;

  /// Resolve symbolic dimensions against a set of bindings.
  /// Returns a new TensorType with concrete dimensions where possible.
  std::unique_ptr<TensorType> resolve(
      const std::vector<std::pair<std::string, int64_t>> &bindings) const;

private:
  std::vector<Dimension>  dims_;
  ScalarType::ScalarKind  element_;
  SourceLocation          loc_;
};

/// Function type: T1 -> T2 -> ... -> TResult
class FunctionType : public TypeNode {
public:
  FunctionType(std::vector<std::unique_ptr<TypeNode>> params,
               std::unique_ptr<TypeNode> result, SourceLocation loc = {})
      : params_(std::move(params)), result_(std::move(result)), loc_(loc) {}

  Kind getKind() const override { return Kind::FunctionType; }
  const std::vector<std::unique_ptr<TypeNode>> &getParams() const { return params_; }
  const TypeNode *getResult() const { return result_.get(); }
  SourceLocation getLocation() const { return loc_; }

  std::string toString() const override;
  bool operator==(const TypeNode &other) const override;
  std::unique_ptr<TypeNode> clone() const override;

private:
  std::vector<std::unique_ptr<TypeNode>> params_;
  std::unique_ptr<TypeNode>              result_;
  SourceLocation                          loc_;
};

// ── Expression AST ──────────────────────────────────────────────────────────

/// Base class for all expression AST nodes.
class Expr {
public:
  enum Kind {
#define EXPR_KIND(KIND) KIND,
    JULES_AST_NODE_KINDS(EXPR_KIND)
#undef EXPR_KIND
  };

  virtual ~Expr() = default;
  virtual Kind getKind() const = 0;
  virtual SourceLocation getLocation() const = 0;
  virtual std::unique_ptr<Expr> clone() const = 0;
  virtual void accept(ASTVisitor &visitor) = 0;

  /// After type-checking, every expression carries a resolved type.
  const TypeNode *getResolvedType() const { return resolvedType_.get(); }
  void setResolvedType(std::unique_ptr<TypeNode> ty) {
    resolvedType_ = std::move(ty);
  }

protected:
  std::unique_ptr<TypeNode> resolvedType_;
};

/// Float literal: 3.14, 1.0e-5
class FloatLiteralExpr : public Expr {
public:
  FloatLiteralExpr(double value, SourceLocation loc)
      : value_(value), loc_(loc) {}

  Kind getKind() const override { return Kind::FloatLiteralExpr; }
  SourceLocation getLocation() const override { return loc_; }
  double getValue() const { return value_; }
  std::unique_ptr<Expr> clone() const override;
  void accept(ASTVisitor &visitor) override;

private:
  double         value_;
  SourceLocation loc_;
};

/// Integer literal: 42
class IntLiteralExpr : public Expr {
public:
  IntLiteralExpr(int64_t value, SourceLocation loc)
      : value_(value), loc_(loc) {}

  Kind getKind() const override { return Kind::IntLiteralExpr; }
  SourceLocation getLocation() const override { return loc_; }
  int64_t getValue() const { return value_; }
  std::unique_ptr<Expr> clone() const override;
  void accept(ASTVisitor &visitor) override;

private:
  int64_t        value_;
  SourceLocation loc_;
};

/// Boolean literal: true | false
class BoolLiteralExpr : public Expr {
public:
  BoolLiteralExpr(bool value, SourceLocation loc)
      : value_(value), loc_(loc) {}

  Kind getKind() const override { return Kind::BoolLiteralExpr; }
  SourceLocation getLocation() const override { return loc_; }
  bool getValue() const { return value_; }
  std::unique_ptr<Expr> clone() const override;
  void accept(ASTVisitor &visitor) override;

private:
  bool           value_;
  SourceLocation loc_;
};

/// Unit literal: ()
class UnitLiteralExpr : public Expr {
public:
  explicit UnitLiteralExpr(SourceLocation loc) : loc_(loc) {}

  Kind getKind() const override { return Kind::UnitLiteralExpr; }
  SourceLocation getLocation() const override { return loc_; }
  std::unique_ptr<Expr> clone() const override;
  void accept(ASTVisitor &visitor) override;

private:
  SourceLocation loc_;
};

/// Identifier reference: X, W, b
class IdentifierExpr : public Expr {
public:
  IdentifierExpr(std::string name, SourceLocation loc)
      : name_(std::move(name)), loc_(loc) {}

  Kind getKind() const override { return Kind::IdentifierExpr; }
  SourceLocation getLocation() const override { return loc_; }
  const std::string &getName() const { return name_; }
  std::unique_ptr<Expr> clone() const override;
  void accept(ASTVisitor &visitor) override;

private:
  std::string    name_;
  SourceLocation loc_;
};

/// Tensor literal: [1.0, 2.0, 3.0]
class TensorLiteralExpr : public Expr {
public:
  TensorLiteralExpr(std::vector<std::unique_ptr<Expr>> elements,
                    SourceLocation loc)
      : elements_(std::move(elements)), loc_(loc) {}

  Kind getKind() const override { return Kind::TensorLiteralExpr; }
  SourceLocation getLocation() const override { return loc_; }
  const std::vector<std::unique_ptr<Expr>> &getElements() const { return elements_; }
  std::unique_ptr<Expr> clone() const override;
  void accept(ASTVisitor &visitor) override;

private:
  std::vector<std::unique_ptr<Expr>> elements_;
  SourceLocation loc_;
};

/// Binary expression: lhs op rhs
class BinaryExpr : public Expr {
public:
  enum Op {
    Add, Sub, Mul, Div, Mod,        // arithmetic
    Pow,                             // ^ power
    MatMul,                          // ** matrix multiply
    Eq, Neq, Lt, Gt, Leq, Geq,      // comparison
    And, Or,                          // logical
  };

  BinaryExpr(Op op, std::unique_ptr<Expr> lhs, std::unique_ptr<Expr> rhs,
             SourceLocation loc)
      : op_(op), lhs_(std::move(lhs)), rhs_(std::move(rhs)), loc_(loc) {}

  Kind getKind() const override { return Kind::BinaryExpr; }
  SourceLocation getLocation() const override { return loc_; }
  Op getOp() const { return op_; }
  const Expr *getLHS() const { return lhs_.get(); }
  const Expr *getRHS() const { return rhs_.get(); }
  std::unique_ptr<Expr> clone() const override;
  void accept(ASTVisitor &visitor) override;

  static const char *opToString(Op op);

private:
  Op                          op_;
  std::unique_ptr<Expr>       lhs_;
  std::unique_ptr<Expr>       rhs_;
  SourceLocation              loc_;
};

/// Unary expression: op operand
class UnaryExpr : public Expr {
public:
  enum Op { Negate, Not };

  UnaryExpr(Op op, std::unique_ptr<Expr> operand, SourceLocation loc)
      : op_(op), operand_(std::move(operand)), loc_(loc) {}

  Kind getKind() const override { return Kind::UnaryExpr; }
  SourceLocation getLocation() const override { return loc_; }
  Op getOp() const { return op_; }
  const Expr *getOperand() const { return operand_.get(); }
  std::unique_ptr<Expr> clone() const override;
  void accept(ASTVisitor &visitor) override;

  static const char *opToString(Op op);

private:
  Op                    op_;
  std::unique_ptr<Expr> operand_;
  SourceLocation        loc_;
};

/// Function call: callee(args...)
class CallExpr : public Expr {
public:
  CallExpr(std::unique_ptr<Expr> callee,
           std::vector<std::unique_ptr<Expr>> args, SourceLocation loc)
      : callee_(std::move(callee)), args_(std::move(args)), loc_(loc) {}

  Kind getKind() const override { return Kind::CallExpr; }
  SourceLocation getLocation() const override { return loc_; }
  const Expr *getCallee() const { return callee_.get(); }
  const std::vector<std::unique_ptr<Expr>> &getArgs() const { return args_; }
  size_t getNumArgs() const { return args_.size(); }
  std::unique_ptr<Expr> clone() const override;
  void accept(ASTVisitor &visitor) override;

private:
  std::unique_ptr<Expr>                callee_;
  std::vector<std::unique_ptr<Expr>>   args_;
  SourceLocation                       loc_;
};

/// Index expression: expr[dims...]
class IndexExpr : public Expr {
public:
  IndexExpr(std::unique_ptr<Expr> object,
            std::vector<std::unique_ptr<Expr>> indices, SourceLocation loc)
      : object_(std::move(object)), indices_(std::move(indices)), loc_(loc) {}

  Kind getKind() const override { return Kind::IndexExpr; }
  SourceLocation getLocation() const override { return loc_; }
  const Expr *getObject() const { return object_.get(); }
  const std::vector<std::unique_ptr<Expr>> &getIndices() const { return indices_; }
  std::unique_ptr<Expr> clone() const override;
  void accept(ASTVisitor &visitor) override;

private:
  std::unique_ptr<Expr>                object_;
  std::vector<std::unique_ptr<Expr>>   indices_;
  SourceLocation                        loc_;
};

/// Let expression: let name = value in body
class LetExpr : public Expr {
public:
  LetExpr(std::string name, std::unique_ptr<Expr> value,
          std::unique_ptr<Expr> body, SourceLocation loc)
      : name_(std::move(name)), value_(std::move(value)),
        body_(std::move(body)), loc_(loc) {}

  Kind getKind() const override { return Kind::LetExpr; }
  SourceLocation getLocation() const override { return loc_; }
  const std::string &getName() const { return name_; }
  const Expr *getValue() const { return value_.get(); }
  const Expr *getBody() const { return body_.get(); }
  std::unique_ptr<Expr> clone() const override;
  void accept(ASTVisitor &visitor) override;

  /// Optional type annotation on the binding.
  void setTypeAnnotation(std::unique_ptr<TypeNode> ty) { typeAnnotation_ = std::move(ty); }
  const TypeNode *getTypeAnnotation() const { return typeAnnotation_.get(); }

private:
  std::string                   name_;
  std::unique_ptr<Expr>         value_;
  std::unique_ptr<Expr>         body_;
  std::unique_ptr<TypeNode>     typeAnnotation_;
  SourceLocation                loc_;
};

/// Lambda expression: \param -> body
class LambdaExpr : public Expr {
public:
  struct Param {
    std::string                   name;
    std::unique_ptr<TypeNode>     type;
  };

  LambdaExpr(std::vector<Param> params, std::unique_ptr<Expr> body,
             SourceLocation loc)
      : params_(std::move(params)), body_(std::move(body)), loc_(loc) {}

  Kind getKind() const override { return Kind::LambdaExpr; }
  SourceLocation getLocation() const override { return loc_; }
  const std::vector<Param> &getParams() const { return params_; }
  const Expr *getBody() const { return body_.get(); }
  std::unique_ptr<Expr> clone() const override;
  void accept(ASTVisitor &visitor) override;

private:
  std::vector<Param>      params_;
  std::unique_ptr<Expr>   body_;
  SourceLocation          loc_;
};

/// If expression: if cond then trueBranch else falseBranch
class IfExpr : public Expr {
public:
  IfExpr(std::unique_ptr<Expr> cond, std::unique_ptr<Expr> trueBranch,
         std::unique_ptr<Expr> falseBranch, SourceLocation loc)
      : cond_(std::move(cond)), trueBranch_(std::move(trueBranch)),
        falseBranch_(std::move(falseBranch)), loc_(loc) {}

  Kind getKind() const override { return Kind::IfExpr; }
  SourceLocation getLocation() const override { return loc_; }
  const Expr *getCondition() const { return cond_.get(); }
  const Expr *getTrueBranch() const { return trueBranch_.get(); }
  const Expr *getFalseBranch() const { return falseBranch_.get(); }
  std::unique_ptr<Expr> clone() const override;
  void accept(ASTVisitor &visitor) override;

private:
  std::unique_ptr<Expr> cond_;
  std::unique_ptr<Expr> trueBranch_;
  std::unique_ptr<Expr> falseBranch_;
  SourceLocation        loc_;
};

/// Block expression: a sequence of let-bindings ending in an expression.
/// Used for multi-line function bodies.
class BlockExpr : public Expr {
public:
  struct Binding {
    std::string               name;
    std::unique_ptr<Expr>     value;
    std::unique_ptr<TypeNode> typeAnnotation;
  };

  BlockExpr(std::vector<Binding> bindings, std::unique_ptr<Expr> result,
            SourceLocation loc)
      : bindings_(std::move(bindings)), result_(std::move(result)), loc_(loc) {}

  Kind getKind() const override { return Kind::BlockExpr; }
  SourceLocation getLocation() const override { return loc_; }
  const std::vector<Binding> &getBindings() const { return bindings_; }
  const Expr *getResult() const { return result_.get(); }
  std::unique_ptr<Expr> clone() const override;
  void accept(ASTVisitor &visitor) override;

private:
  std::vector<Binding>    bindings_;
  std::unique_ptr<Expr>   result_;
  SourceLocation          loc_;
};

/// Cast expression: cast(expr, Type)
class CastExpr : public Expr {
public:
  CastExpr(std::unique_ptr<Expr> expr, std::unique_ptr<TypeNode> targetType,
           SourceLocation loc)
      : expr_(std::move(expr)), targetType_(std::move(targetType)), loc_(loc) {}

  Kind getKind() const override { return Kind::CastExpr; }
  SourceLocation getLocation() const override { return loc_; }
  const Expr *getExpr() const { return expr_.get(); }
  const TypeNode *getTargetType() const { return targetType_.get(); }
  std::unique_ptr<Expr> clone() const override;
  void accept(ASTVisitor &visitor) override;

private:
  std::unique_ptr<Expr>     expr_;
  std::unique_ptr<TypeNode> targetType_;
  SourceLocation            loc_;
};

/// Autodiff expression: grad(fn, var)
class GRADExpr : public Expr {
public:
  GRADExpr(std::unique_ptr<Expr> fn, std::string diffVar,
           SourceLocation loc)
      : fn_(std::move(fn)), diffVar_(std::move(diffVar)), loc_(loc) {}

  Kind getKind() const override { return Kind::GRADExpr; }
  SourceLocation getLocation() const override { return loc_; }
  const Expr *getFunction() const { return fn_.get(); }
  const std::string &getDiffVar() const { return diffVar_; }
  std::unique_ptr<Expr> clone() const override;
  void accept(ASTVisitor &visitor) override;

private:
  std::unique_ptr<Expr> fn_;
  std::string           diffVar_;
  SourceLocation        loc_;
};

// ── Declaration AST ─────────────────────────────────────────────────────────

/// Function declaration:
///   name : Type -> Type -> Type
///   name params = body
class FunctionDecl {
public:
  FunctionDecl(std::string name, std::unique_ptr<FunctionType> type,
               std::vector<std::pair<std::string, std::unique_ptr<TypeNode>>> params,
               std::unique_ptr<Expr> body, SourceLocation loc)
      : name_(std::move(name)), type_(std::move(type)),
        params_(std::move(params)), body_(std::move(body)), loc_(loc) {}

  const std::string &getName() const { return name_; }
  const FunctionType *getType() const { return type_.get(); }
  const std::vector<std::pair<std::string, std::unique_ptr<TypeNode>>> &getParams() const {
    return params_;
  }
  const Expr *getBody() const { return body_.get(); }
  SourceLocation getLocation() const { return loc_; }

private:
  std::string   name_;
  std::unique_ptr<FunctionType> type_;
  std::vector<std::pair<std::string, std::unique_ptr<TypeNode>>> params_;
  std::unique_ptr<Expr> body_;
  SourceLocation loc_;
};

/// Extern declaration: extern "name" : Type
class ExternDecl {
public:
  ExternDecl(std::string name, std::unique_ptr<TypeNode> type,
             SourceLocation loc)
      : name_(std::move(name)), type_(std::move(type)), loc_(loc) {}

  const std::string &getName() const { return name_; }
  const TypeNode *getType() const { return type_.get(); }
  SourceLocation getLocation() const { return loc_; }

private:
  std::string               name_;
  std::unique_ptr<TypeNode> type_;
  SourceLocation            loc_;
};

/// Top-level program: a collection of function and extern declarations.
class Program {
public:
  Program() = default;

  void addFunction(std::unique_ptr<FunctionDecl> fn) {
    functions_.push_back(std::move(fn));
  }
  void addExtern(std::unique_ptr<ExternDecl> ext) {
    externs_.push_back(std::move(ext));
  }

  const std::vector<std::unique_ptr<FunctionDecl>> &getFunctions() const {
    return functions_;
  }
  const std::vector<std::unique_ptr<ExternDecl>> &getExterns() const {
    return externs_;
  }

  /// Look up a function by name. Returns nullptr if not found.
  const FunctionDecl *findFunction(const std::string &name) const;

  /// Look up an extern by name. Returns nullptr if not found.
  const ExternDecl *findExtern(const std::string &name) const;

private:
  std::vector<std::unique_ptr<FunctionDecl>> functions_;
  std::vector<std::unique_ptr<ExternDecl>>   externs_;
};

// ── Visitor ─────────────────────────────────────────────────────────────────

class ASTVisitor {
public:
  virtual ~ASTVisitor() = default;

#define DEFINE_VISIT(KIND)                                                    \
  virtual void visit##KIND(KIND &) {}
  JULES_AST_NODE_KINDS(DEFINE_VISIT)
#undef DEFINE_VISIT
};

// ── Inline implementations ──────────────────────────────────────────────────

inline void FloatLiteralExpr::accept(ASTVisitor &v) { v.visitFloatLiteralExpr(*this); }
inline void IntLiteralExpr::accept(ASTVisitor &v) { v.visitIntLiteralExpr(*this); }
inline void BoolLiteralExpr::accept(ASTVisitor &v) { v.visitBoolLiteralExpr(*this); }
inline void UnitLiteralExpr::accept(ASTVisitor &v) { v.visitUnitLiteralExpr(*this); }
inline void IdentifierExpr::accept(ASTVisitor &v) { v.visitIdentifierExpr(*this); }
inline void TensorLiteralExpr::accept(ASTVisitor &v) { v.visitTensorLiteralExpr(*this); }
inline void BinaryExpr::accept(ASTVisitor &v) { v.visitBinaryExpr(*this); }
inline void UnaryExpr::accept(ASTVisitor &v) { v.visitUnaryExpr(*this); }
inline void CallExpr::accept(ASTVisitor &v) { v.visitCallExpr(*this); }
inline void IndexExpr::accept(ASTVisitor &v) { v.visitIndexExpr(*this); }
inline void LetExpr::accept(ASTVisitor &v) { v.visitLetExpr(*this); }
inline void LambdaExpr::accept(ASTVisitor &v) { v.visitLambdaExpr(*this); }
inline void IfExpr::accept(ASTVisitor &v) { v.visitIfExpr(*this); }
inline void BlockExpr::accept(ASTVisitor &v) { v.visitBlockExpr(*this); }
inline void CastExpr::accept(ASTVisitor &v) { v.visitCastExpr(*this); }
inline void GRADExpr::accept(ASTVisitor &v) { v.visitGRADExpr(*this); }

} // namespace jules

#endif // JULES_AST_H
