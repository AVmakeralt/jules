//===- AOTCompiler.h - AOT Compilation Engine for Tier 1 -------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines the AOT compilation engine for Tier 1 of the two-tier
// hybrid execution system.
//
// The AOT compiler produces generic, dynamic-shape MLIR binaries that
// execute instantly with zero startup delay. These binaries include
// telemetry hooks that profile actual shapes during execution, feeding
// data to the PGO system for Tier 2 JIT compilation.
//
// AOT Compilation Pipeline:
//
//   Source → Lexer → Parser → TypeChecker → MLIR Gen
//          → Shape Inference → Autodiff → Autodiff Pruning
//          → Graph Collapsing → SCCP → SymbolDCE
//          → Algebraic Simplification → SIMD Layout
//          → Polyhedral Optimization → Canonicalization + CSE
//          → StableHLO Lowering → XLA Compilation (dynamic shapes)
//          → Tier 1 Executable
//
// Key differences from the standard compiler:
//   - Dynamic shapes are preserved (tensor<?x?xf32>)
//   - Telemetry hooks are injected for shape profiling
//   - The polyhedral engine operates on generic loop bounds
//   - The output is a generic binary that handles any input shape
//
//===----------------------------------------------------------------------===//

#ifndef JULES_AOT_COMPILER_H
#define JULES_AOT_COMPILER_H

#include "jules/Compiler.h"
#include "jules/DispatchTable.h"
#include "jules/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include <memory>
#include <string>

namespace mlir {
class MLIRContext;
class ModuleOp;
class PassManager;
} // namespace mlir

namespace jules {

class Program;

// ── AOT Compiler Configuration ──────────────────────────────────────────────

struct AOTCompilerConfig {
  /// Optimization level (0 = none, 1 = basic, 2 = aggressive).
  int optLevel = 2;

  /// Whether to enable autodiff.
  bool enableAutodiff = true;

  /// Whether to enable graph collapsing.
  bool enableGraphCollapsing = true;

  /// Whether to enable autodiff pruning.
  bool enableAutodiffPruning = true;

  /// Whether to enable SCCP (constant propagation).
  bool enableSCCP = true;

  /// Whether to enable SymbolDCE.
  bool enableSymbolDCE = true;

  /// Whether to enable SIMD layout optimization.
  bool enableSIMDLayout = true;

  /// Whether to enable polyhedral (affine) optimization.
  bool enablePolyhedral = true;

  /// Whether to inject telemetry hooks for PGO profiling.
  bool injectTelemetryHooks = true;

  /// Target device: "cpu", "cuda", "rocm", "tpu".
  std::string targetDevice = "cpu";

  /// Whether to emit verbose output.
  bool verbose = false;
};

// ── AOT Compiler ────────────────────────────────────────────────────────────

/// The AOT compilation engine: produces Tier 1 dynamic-shape executables.
class AOTCompiler {
public:
  explicit AOTCompiler(DiagnosticsEngine &diag, AOTCompilerConfig config = {});
  ~AOTCompiler();

  // ── Compilation ──────────────────────────────────────────────────────────

  /// Compile a source file to a Tier 1 executable.
  /// Returns the executable handle on success, nullptr on failure.
  std::shared_ptr<ExecutableHandle> compile(const std::string &source,
                                             const std::string &sourceName = "<input>");

  /// Compile an AST program to a Tier 1 executable.
  std::shared_ptr<ExecutableHandle> compileFromAST(Program &program);

  /// Compile an MLIR module to a Tier 1 executable.
  std::shared_ptr<ExecutableHandle> compileFromMLIR(mlir::ModuleOp module);

  // ── Registration ─────────────────────────────────────────────────────────

  /// Compile and register a function as a Tier 1 entry in the dispatch table.
  bool compileAndRegister(const std::string &source,
                           const std::string &functionName,
                           DispatchTable &dispatchTable);

  // ── Accessors ────────────────────────────────────────────────────────────

  DiagnosticsEngine &getDiag() { return diag_; }
  const AOTCompilerConfig &getConfig() const { return config_; }

private:
  /// Parse and type-check source into an AST.
  std::unique_ptr<Program> parseSource(const std::string &source,
                                        const std::string &sourceName);

  /// Lower an AST to MLIR in the Jules dialect.
  mlir::OwningOpRef<mlir::ModuleOp> lowerToMLIR(Program &program);

  /// Run the full AOT optimization pipeline on an MLIR module.
  bool runAOTPipeline(mlir::ModuleOp module);

  /// Inject telemetry hooks into the MLIR module for PGO profiling.
  bool injectTelemetry(mlir::ModuleOp module);

  /// Lower the optimized MLIR to StableHLO.
  bool lowerToStableHLO(mlir::ModuleOp module);

  /// Serialize the compiled module into a Tier 1 executable handle.
  std::shared_ptr<ExecutableHandle> serializeModule(mlir::ModuleOp module);

  DiagnosticsEngine                        &diag_;
  AOTCompilerConfig                         config_;
  std::unique_ptr<mlir::MLIRContext>        mlirContext_;
};

} // namespace jules

#endif // JULES_AOT_COMPILER_H
