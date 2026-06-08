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
//   AOT Mode (Tier 1):
//     Source → Lexer → Parser → TypeChecker → MLIR Gen → Passes → StableHLO → XLA
//
//   JIT Mode (Two-Tier Hybrid):
//     Source → AOT Compile (Tier 1) → Runtime Execution → PGO Profiling
//            → Background JIT Compile (Tier 2) → Tier 2 Dispatch
//
// The compiler can produce:
//   - An AST dump (for debugging)
//   - MLIR in the Jules dialect
//   - MLIR after autodiff transformation
//   - MLIR lowered to StableHLO
//   - A Tier 1 AOT executable (dynamic shapes, zero cold-start)
//   - A Tiered execution engine (AOT + JIT hybrid)
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
class TieredExecution;
struct TieredExecutionConfig;

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
    EmitTiered,       // Compile with two-tier AOT/JIT hybrid
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

  // ── Two-Tier AOT/JIT Options ────────────────────────────────────────────

  /// Whether to enable Tier 2 JIT compilation.
  bool enableTier2 = true;

  /// Whether to enable PGO profiling.
  bool enablePGO = true;

  /// Number of warmup iterations before PGO recompilation.
  uint64_t pgoWarmupThreshold = 10;

  /// Whether to enable graph collapsing passes.
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

  /// Whether to enable producer-consumer kernel fusion.
  bool enableKernelFusion = true;

  /// Whether to enable memory planning (buffer reuse, in-place ops).
  bool enableMemoryPlanning = true;

  /// Whether to enable mixed precision (bf16/fp8) optimization.
  bool enableMixedPrecision = false;

  /// Target precision for mixed precision ("bf16" or "fp8").
  std::string mixedPrecisionTarget = "bf16";

  /// Whether to enable shape polymorphism (symbolic dimension constraints).
  bool enableShapePolymorphism = true;

  /// Whether to enable quantization pass.
  bool enableQuantization = false;

  /// Whether to use the fused autodiff pass (autodiff+pruning+collapsing in one).
  bool enableFusedAutodiff = true;

  /// Whether to enable polyhedral optimization on the backward pass.
  bool enableBackwardPolyhedral = true;

  /// Whether to inject telemetry hooks for PGO profiling.
  bool injectTelemetryHooks = true;

  /// Number of JIT worker threads.
  unsigned jitWorkers = 2;

  /// Memory budget in bytes for memory planning (0 = unlimited).
  uint64_t memoryBudgetBytes = 0;

  /// Whether to use the LLVM backend instead of the interpreter.
  bool useLLVMBackend = true;
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

  /// Run the full AOT optimization pipeline (including new passes).
  bool runAOTOptimizations();

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
