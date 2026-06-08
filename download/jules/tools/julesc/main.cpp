//===- main.cpp - Jules Compiler Driver ------------------------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the command-line interface for the Jules compiler
// (julesc). It parses command-line options and invokes the Compiler driver.
//
// Usage:
//   julesc [options] <input.jules>
//
// Options:
//   -o <file>          Output file (default: stdout)
//   -emit-ast          Emit the parsed AST
//   -emit-mlir         Emit MLIR in the Jules dialect (default)
//   -emit-mlir-ad      Emit MLIR after the autodiff pass
//   -emit-stablehlo    Emit MLIR lowered to StableHLO
//   -emit-exe          Compile through XLA to an executable
//   -jit               Run in JIT mode with global tracing & PGO
//   -jit-pgo           Enable PGO recompilation in JIT mode
//   -jit-warmup <n>    Number of warmup iterations before PGO (default: 10)
//   -no-autodiff       Disable the autodiff pass
//   -no-graph-collapse Disable graph collapsing passes
//   -no-ad-prune       Disable autodiff pruning
//   -O0/-O1/-O2        Optimization level
//   -target <t>        XLA target: cpu, cuda, rocm, tpu
//   -v                 Verbose output
//   -h                 Show help
//
//===----------------------------------------------------------------------===//

#include "jules/Compiler.h"
#include "jules/Diagnostics.h"
#include "jules/Dialect/JulesDialect.h"
#include "jules/Passes/Passes.h"
#include "jules/JIT.h"
#include "jules/Tracing.h"
#include "jules/Runtime.h"
#include "jules/Profiler.h"
#include "jules/PJRT.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/LogicalResult.h"

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static void printUsage(const char *progName) {
  std::cerr << "Usage: " << progName << " [options] <input.jules>\n\n"
            << "Compilation modes:\n"
            << "  -emit-ast          Emit the parsed AST\n"
            << "  -emit-mlir         Emit MLIR in the Jules dialect (default)\n"
            << "  -emit-mlir-ad      Emit MLIR after the autodiff pass\n"
            << "  -emit-stablehlo    Emit MLIR lowered to StableHLO\n"
            << "  -emit-exe          Compile through XLA to an executable\n"
            << "  -jit               Run in JIT mode with global tracing\n\n"
            << "JIT options:\n"
            << "  -jit-pgo           Enable PGO recompilation in JIT mode\n"
            << "  -jit-warmup <n>    Warmup iterations before PGO (default: 10)\n\n"
            << "General options:\n"
            << "  -o <file>          Output file (default: stdout)\n"
            << "  -O0 / -O1 / -O2    Optimization level\n"
            << "  -no-autodiff       Disable the autodiff pass\n"
            << "  -no-graph-collapse Disable graph collapsing passes\n"
            << "  -no-ad-prune       Disable autodiff pruning\n"
            << "  -target <target>   XLA target: cpu, cuda, rocm, tpu\n"
            << "  -v                 Verbose output\n"
            << "  -h                 Show this help message\n";
}

int main(int argc, char **argv) {
  jules::CompilerOptions options;
  options.emissionMode = jules::CompilerOptions::EmitMLIR;
  options.inputFile = "-";
  options.outputFile = "-";

  bool jitMode = false;
  bool jitPGO = false;
  uint64_t jitWarmup = 10;
  bool noGraphCollapse = false;
  bool noADPrune = false;

  std::vector<std::string> args;
  for (int i = 1; i < argc; ++i) {
    args.emplace_back(argv[i]);
  }

  for (size_t i = 0; i < args.size(); ++i) {
    const std::string &arg = args[i];

    if (arg == "-h" || arg == "--help") {
      printUsage(argv[0]);
      return 0;
    } else if (arg == "-o") {
      if (i + 1 >= args.size()) {
        std::cerr << "error: -o requires an argument\n";
        return 1;
      }
      options.outputFile = args[++i];
    } else if (arg == "-emit-ast") {
      options.emissionMode = jules::CompilerOptions::EmitAST;
    } else if (arg == "-emit-mlir") {
      options.emissionMode = jules::CompilerOptions::EmitMLIR;
    } else if (arg == "-emit-mlir-ad") {
      options.emissionMode = jules::CompilerOptions::EmitMLIRAfterAD;
    } else if (arg == "-emit-stablehlo") {
      options.emissionMode = jules::CompilerOptions::EmitStableHLO;
    } else if (arg == "-emit-exe") {
      options.emissionMode = jules::CompilerOptions::EmitExecutable;
    } else if (arg == "-jit") {
      jitMode = true;
      options.emissionMode = jules::CompilerOptions::EmitExecutable;
    } else if (arg == "-jit-pgo") {
      jitPGO = true;
    } else if (arg == "-jit-warmup") {
      if (i + 1 >= args.size()) {
        std::cerr << "error: -jit-warmup requires an argument\n";
        return 1;
      }
      jitWarmup = std::stoull(args[++i]);
    } else if (arg == "-O0") {
      options.optLevel = 0;
    } else if (arg == "-O1") {
      options.optLevel = 1;
    } else if (arg == "-O2") {
      options.optLevel = 2;
    } else if (arg == "-no-autodiff") {
      options.enableAutodiff = false;
    } else if (arg == "-no-graph-collapse") {
      noGraphCollapse = true;
    } else if (arg == "-no-ad-prune") {
      noADPrune = true;
    } else if (arg == "-target") {
      if (i + 1 >= args.size()) {
        std::cerr << "error: -target requires an argument\n";
        return 1;
      }
      options.xlaTarget = args[++i];
    } else if (arg == "-v") {
      options.verbose = true;
    } else if (arg[0] == '-') {
      std::cerr << "error: unknown option: " << arg << "\n";
      printUsage(argv[0]);
      return 1;
    } else {
      options.inputFile = arg;
    }
  }

  // Initialize MLIR dialects and passes.
  mlir::DialectRegistry registry;
  registry.insert<jules::JulesDialect>();
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();
  jules::registerJulesPasses();

  if (jitMode) {
    // ── JIT Mode ─────────────────────────────────────────────────────────
    // In JIT mode, we use the global tracing engine to record operations
    // lazily, then compile and execute them through the JIT pipeline.
    jules::DiagnosticsEngine diag;
    jules::JITConfig jitConfig;
    jitConfig.optLevel = options.optLevel;
    jitConfig.enablePGO = jitPGO;
    jitConfig.pgoWarmupThreshold = jitWarmup;
    jitConfig.enableGraphCollapsing = !noGraphCollapse;
    jitConfig.enableAutodiffPruning = !noADPrune;
    jitConfig.targetDevice = options.xlaTarget;
    jitConfig.verbose = options.verbose;

    jules::JITCompiler jit(diag, jitConfig);

    // Initialize PJRT client.
    auto pjrtClient = jules::PJRTClient::create(options.xlaTarget, diag);
    if (!pjrtClient) {
      std::cerr << "error: failed to initialize PJRT client\n";
      return 1;
    }

    if (options.verbose) {
      std::cerr << "JIT mode: target=" << options.xlaTarget
                << " pgo=" << (jitPGO ? "on" : "off")
                << " warmup=" << jitWarmup << "\n";
    }

    // In a full JIT execution, we would:
    // 1. Parse the source to get the AST
    // 2. Set up the runtime environment with the global tracer
    // 3. Execute the main function lazily (recording to the trace)
    // 4. When a barrier is hit, compile the trace through MLIR
    // 5. Execute the compiled trace via PJRT
    // 6. If PGO is enabled, profile shape data and recompile

    // For AOT-JIT hybrid, compile normally but with JIT-optimized pipeline.
    jules::Compiler compiler(std::move(options));
    bool success = compiler.compile();

    if (compiler.getDiagnostics().hasErrors()) {
      return 1;
    }
    return success ? 0 : 1;
  }

  // ── AOT Mode (standard compilation) ───────────────────────────────────
  jules::Compiler compiler(std::move(options));
  bool success = compiler.compile();

  if (compiler.getDiagnostics().hasErrors()) {
    return 1;
  }

  return success ? 0 : 1;
}
