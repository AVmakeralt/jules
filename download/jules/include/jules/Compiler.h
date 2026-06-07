//===- Compiler.h - Jules Compiler Driver ----------------------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines the top-level Compiler driver that orchestrates the
// full compilation pipeline:
//
//   Source -> Lexer -> Parser -> TypeChecker -> MLIR Gen -> Passes -> Output
//
// The compiler can produce:
//   - An AST dump (for debugging)
//   - MLIR in the Jules dialect
//   - MLIR after autodiff transformation
//   - MLIR lowered to StableHLO
//   - An XLA executable (when XLA runtime is available)
//
//===----------------------------------------------------------------------===//

#ifndef JULES_COMPILER_H
#define JULES_COMPILER_H

#include "jules/Diagnostics.h"
#include <memory>
#include <string>

namespace mlir {
class MLIRContext;
class ModuleOp;
class OpPrintingFlags;
} // namespace mlir

namespace jules {

class Program;

/// Configuration for the compiler pipeline.
struct CompilerOptions {
  /// Input source file path (or "-" for stdin).
  std::string inputFile;

  /// Output file path (or "-" for stdout).
  std::string outputFile;

  /// What emission mode to use.
  enum EmissionMode {
    EmitAST,          // Dump the parsed AST
    EmitMLIR,         // Emit MLIR in the Jules dialect
    EmitMLIRAfterAD,  // Emit MLIR after autodiff pass
    EmitStableHLO,    // Emit MLIR lowered to StableHLO
    EmitExecutable,   // Compile through XLA to an executable
  };

  EmissionMode emissionMode = EmitMLIR;

  /// Whether to print verbose diagnostics.
  bool verbose = false;

  /// Whether to enable the autodiff pass.
  bool enableAutodiff = true;

  /// Optimization level (0 = none, 1 = basic, 2 = aggressive).
  int optLevel = 1;

  /// Target backend for XLA compilation.
  std::string xlaTarget = "cpu";  // "cpu", "cuda", "rocm", "tpu"
};

/// The main compiler driver.
class Compiler {
public:
  explicit Compiler(CompilerOptions options);

  /// Run the compilation pipeline. Returns true on success.
  bool compile();

  /// Get the diagnostics engine.
  DiagnosticsEngine &getDiagnostics() { return diag_; }

private:
  // ── Pipeline stages ──────────────────────────────────────────────────────

  /// Read the source file into a string.
  std::string readSource();

  /// Parse the source into an AST.
  std::unique_ptr<Program> parseSource(const std::string &source);

  /// Type-check the AST.
  bool typeCheck(Program &program);

  /// Lower the AST to MLIR in the Jules dialect.
  bool lowerToMLIR(Program &program);

  /// Run the autodiff pass on the MLIR module.
  bool runAutodiffPass();

  /// Lower from Jules dialect to StableHLO.
  bool lowerToStableHLO();

  /// Compile through XLA to an executable.
  bool compileXLA();

  /// Emit the current MLIR module to the output.
  bool emitOutput();

  // ── Members ──────────────────────────────────────────────────────────────
  CompilerOptions                          options_;
  DiagnosticsEngine                        diag_;
  std::unique_ptr<mlir::MLIRContext>       mlirContext_;
  std::unique_ptr<mlir::ModuleOp>          mlirModule_;
};

} // namespace jules

#endif // JULES_COMPILER_H
