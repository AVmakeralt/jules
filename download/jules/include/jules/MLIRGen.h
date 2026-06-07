//===- MLIRGen.h - AST to MLIR Lowering ------------------------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file declares the AST-to-MLIR lowering pass. It walks the Jules AST
// and emits MLIR operations in the Jules dialect, creating a valid MLIR
// ModuleOp with FuncOps for each top-level function declaration.
//
//===----------------------------------------------------------------------===//

#ifndef JULES_MLIRGEN_H
#define JULES_MLIRGEN_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include <memory>
#include <string>
#include <unordered_map>

namespace jules {

class Program;
class Expr;
class FunctionDecl;
class TypeNode;
class TensorType;
class ScalarType;
class FunctionType;
class DiagnosticsEngine;

/// Lower a Jules AST Program into an MLIR ModuleOp.
std::unique_ptr<mlir::ModuleOp> lowerASTToMLIR(mlir::MLIRContext &context,
                                                 Program &program,
                                                 DiagnosticsEngine &diag);

/// The implementation of the AST -> MLIR lowering.
class MLIRGenerator {
public:
  explicit MLIRGenerator(mlir::MLIRContext &context, DiagnosticsEngine &diag);

  /// Lower an entire Program.
  std::unique_ptr<mlir::ModuleOp> lowerProgram(Program &program);

private:
  /// Lower a FunctionDecl to a func::FuncOp.
  void lowerFunctionDecl(FunctionDecl &fn);

  /// Lower an Expr to an MLIR Value.
  mlir::Value lowerExpr(Expr &expr);

  /// Map a Jules TypeNode to an MLIR Type.
  mlir::Type lowerType(const TypeNode &type);
  mlir::Type lowerScalarType(const ScalarType &type);
  mlir::Type lowerTensorType(const TensorType &type);
  mlir::Type lowerFunctionType(const FunctionType &type);

  /// Get or create a constant of the given value and type.
  mlir::Value createConstant(double value, mlir::Type type);

  // ── Scoped variable table ────────────────────────────────────────────────
  void declareVariable(const std::string &name, mlir::Value value);
  mlir::Value lookupVariable(const std::string &name) const;
  void pushScope();
  void popScope();

  // ── Members ──────────────────────────────────────────────────────────────
  mlir::MLIRContext                                       &context_;
  DiagnosticsEngine                                       &diag_;
  std::unique_ptr<mlir::ModuleOp>                          module_;
  mlir::OpBuilder                                          builder_;

  /// Scoped variable table (stack of name->value maps).
  std::vector<std::unordered_map<std::string, mlir::Value>> scopeStack_;

  /// Current function being lowered.
  mlir::func::FuncOp                                       currentFunc_;
};

} // namespace jules

#endif // JULES_MLIRGEN_H
