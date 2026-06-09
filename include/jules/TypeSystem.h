//===- TypeSystem.h - Jules Type System & Shape Inference -------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines the type-checking and shape-inference engine for the
// Jules language. The TypeChecker walks the AST, resolves symbolic dimension
// variables, verifies type compatibility, and annotates every expression with
// its resolved type.
//
// Key features:
//   - Compile-time shape verification (e.g. [B, I] @ [I, O] -> [B, O])
//   - Symbolic dimension unification (e.g. matching dimensions across args)
//   - Automatic broadcasting inference for element-wise ops
//   - Gradient type derivation for autodiff (grad(f, x) produces a function
//     with the same input type and a result type matching x's type)
//
//===----------------------------------------------------------------------===//

#ifndef JULES_TYPESYSTEM_H
#define JULES_TYPESYSTEM_H

#include "jules/AST.h"
#include "jules/Diagnostics.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace jules {

/// A dimension substitution mapping symbolic names to concrete sizes.
class DimSubstitution {
public:
  /// Try to bind a symbolic dimension to a concrete size.
  /// Returns false if there is a conflict with an existing binding.
  bool bind(const std::string &name, int64_t size);

  /// Look up the concrete size for a symbolic name.
  /// Returns -1 if not bound.
  int64_t lookup(const std::string &name) const;

  /// Merge another substitution into this one.
  /// Returns false on conflict.
  bool merge(const DimSubstitution &other);

  void clear() { bindings_.clear(); }

  const std::unordered_map<std::string, int64_t> &getBindings() const {
    return bindings_;
  }

private:
  std::unordered_map<std::string, int64_t> bindings_;
};

/// The type-checker: walks the AST, resolves types, and annotates nodes.
class TypeChecker : public ASTVisitor {
public:
  explicit TypeChecker(DiagnosticsEngine &diag) : diag_(diag), hasError_(false) {}

  /// Type-check an entire program. Returns true on success.
  bool checkProgram(Program &program);

  /// Type-check a single expression in the given context.
  bool checkExpr(Expr &expr);

  bool hasErrors() const { return hasError_; }

  // ── Visitor overrides ────────────────────────────────────────────────────
  void visitFloatLiteralExpr(FloatLiteralExpr &) override;
  void visitIntLiteralExpr(IntLiteralExpr &) override;
  void visitBoolLiteralExpr(BoolLiteralExpr &) override;
  void visitUnitLiteralExpr(UnitLiteralExpr &) override;
  void visitIdentifierExpr(IdentifierExpr &) override;
  void visitTensorLiteralExpr(TensorLiteralExpr &) override;
  void visitBinaryExpr(BinaryExpr &) override;
  void visitUnaryExpr(UnaryExpr &) override;
  void visitCallExpr(CallExpr &) override;
  void visitIndexExpr(IndexExpr &) override;
  void visitLetExpr(LetExpr &) override;
  void visitLambdaExpr(LambdaExpr &) override;
  void visitIfExpr(IfExpr &) override;
  void visitBlockExpr(BlockExpr &) override;
  void visitCastExpr(CastExpr &) override;
  void visitGRADExpr(GRADExpr &) override;
  void visitFunctionDecl(FunctionDecl &) override;
  void visitExternDecl(ExternDecl &) override;
  void visitTypeAnnotation(TypeAnnotation &) override {}
  void visitTensorType(TensorType &) override {}
  void visitScalarType(ScalarType &) override {}
  void visitFunctionType(FunctionType &) override {}
  void visitProgram(Program &) override {}

private:
  // ── Environment for name resolution ──────────────────────────────────────
  struct Binding {
    std::string               name;
    std::unique_ptr<TypeNode> type;
  };

  /// Push a new scope.
  void pushScope();
  /// Pop the top scope.
  void popScope();
  /// Add a binding to the current scope.
  void addBinding(const std::string &name, std::unique_ptr<TypeNode> type);
  /// Look up a binding by name, searching from innermost scope outward.
  const TypeNode *lookupBinding(const std::string &name) const;

  // ── Shape arithmetic ─────────────────────────────────────────────────────

  /// Infer the result type of a binary operation.
  std::unique_ptr<TypeNode> inferBinaryType(BinaryExpr::Op op,
                                             const TypeNode *lhsType,
                                             const TypeNode *rhsType,
                                             SourceLocation loc);

  /// Compute matrix multiplication result shape: [B,M] @ [M,N] -> [B,N]
  std::unique_ptr<TensorType> inferMatMulShape(const TensorType *lhs,
                                                const TensorType *rhs,
                                                SourceLocation loc);

  /// Compute element-wise broadcasting: [A,B] + [B] -> [A,B]
  std::unique_ptr<TypeNode> inferBroadcast(const TypeNode *lhs,
                                            const TypeNode *rhs);

  /// Compute the gradient type for autodiff: given a function type and the
  /// differentiation variable, return the type of the gradient function.
  std::unique_ptr<FunctionType> inferGradType(const FunctionType *fnType,
                                               const std::string &diffVar);

  // ── Built-in function types ──────────────────────────────────────────────
  const TypeNode *getBuiltinType(const std::string &name) const;

  // ── Members ──────────────────────────────────────────────────────────────
  DiagnosticsEngine                                    &diag_;
  bool                                                  hasError_;
  std::vector<std::vector<Binding>>                     scopes_;
  std::unordered_map<std::string, FunctionDecl *>       functionDecls_;
  std::unordered_map<std::string, ExternDecl *>         externDecls_;
  DimSubstitution                                       dimSubst_;
};

} // namespace jules

#endif // JULES_TYPESYSTEM_H
