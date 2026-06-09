//===- main.cpp - Jules Compiler Driver ------------------------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the command-line interface for the Jules compiler
// (julesc). It supports both AOT and two-tier AOT/JIT hybrid compilation.
//
// Usage:
//   julesc [options] <input.jules>
//
// Compilation modes:
//   -emit-ast          Emit the parsed AST
//   -emit-mlir         Emit MLIR in the Jules dialect (default)
//   -emit-mlir-ad      Emit MLIR after the autodiff pass
//   -emit-stablehlo    Emit MLIR lowered to StableHLO
//   -emit-exe          Compile through XLA to an executable
//   -emit-tiered       Compile with two-tier AOT/JIT hybrid
//
// AOT/JIT hybrid options:
//   -jit               Run in JIT mode with global tracing & PGO
//   -jit-pgo           Enable PGO recompilation in JIT mode
//   -jit-warmup <n>    Number of warmup iterations before PGO (default: 10)
//   -jit-workers <n>   Number of background JIT worker threads (default: 2)
//   -no-tier2          Disable Tier 2 JIT compilation
//   -no-pgo            Disable PGO profiling
//
// Optimization options:
//   -no-autodiff       Disable the autodiff pass
//   -no-graph-collapse Disable graph collapsing passes
//   -no-ad-prune       Disable autodiff pruning
//   -no-sccp           Disable SCCP constant propagation
//   -no-symbol-dce     Disable SymbolDCE
//   -no-simd           Disable SIMD layout optimization
//   -no-poly           Disable polyhedral (affine) optimization
//   -no-telemetry      Disable PGO telemetry hooks injection
//
// General options:
//   -o <file>          Output file (default: stdout)
//   -O0 / -O1 / -O2    Optimization level
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
#include "jules/TieredExecution.h"
#include "jules/DispatchTable.h"
#include "jules/JITQueue.h"
#include "jules/CachePolicy.h"
#include "jules/BailoutHandler.h"
#include "jules/AOTCompiler.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/LogicalResult.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
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
            << "  -emit-tiered       Compile with two-tier AOT/JIT hybrid\n\n"

            << "AOT/JIT hybrid options:\n"
            << "  -jit               Run in JIT mode with global tracing\n"
            << "  -jit-pgo           Enable PGO recompilation in JIT mode\n"
            << "  -jit-warmup <n>    Warmup iterations before PGO (default: 10)\n"
            << "  -jit-workers <n>   Background JIT worker threads (default: 2)\n"
            << "  -no-tier2          Disable Tier 2 JIT compilation\n"
            << "  -no-pgo            Disable PGO profiling\n\n"

            << "Optimization options:\n"
            << "  -no-autodiff       Disable the autodiff pass\n"
            << "  -no-graph-collapse Disable graph collapsing passes\n"
            << "  -no-ad-prune       Disable autodiff pruning\n"
            << "  -no-sccp           Disable SCCP constant propagation\n"
            << "  -no-symbol-dce     Disable SymbolDCE\n"
            << "  -no-simd           Disable SIMD layout optimization\n"
            << "  -no-poly           Disable polyhedral optimization\n"
            << "  -no-telemetry      Disable PGO telemetry hooks\n\n"

            << "General options:\n"
            << "  -o <file>          Output file (default: stdout)\n"
            << "  -O0 / -O1 / -O2    Optimization level\n"
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
  unsigned jitWorkers = 2;

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
    } else if (arg == "-emit-tiered") {
      options.emissionMode = jules::CompilerOptions::EmitTiered;
    } else if (arg == "-jit") {
      jitMode = true;
      options.emissionMode = jules::CompilerOptions::EmitExecutable;
    } else if (arg == "-jit-pgo") {
      jitPGO = true;
      options.enablePGO = true;
    } else if (arg == "-jit-warmup") {
      if (i + 1 >= args.size()) {
        std::cerr << "error: -jit-warmup requires an argument\n";
        return 1;
      }
      jitWarmup = std::stoull(args[++i]);
      options.pgoWarmupThreshold = jitWarmup;
    } else if (arg == "-jit-workers") {
      if (i + 1 >= args.size()) {
        std::cerr << "error: -jit-workers requires an argument\n";
        return 1;
      }
      jitWorkers = static_cast<unsigned>(std::stoul(args[++i]));
      options.jitWorkers = jitWorkers;
    } else if (arg == "-no-tier2") {
      options.enableTier2 = false;
    } else if (arg == "-no-pgo") {
      options.enablePGO = false;
    } else if (arg == "-O0") {
      options.optLevel = 0;
    } else if (arg == "-O1") {
      options.optLevel = 1;
    } else if (arg == "-O2") {
      options.optLevel = 2;
    } else if (arg == "-no-autodiff") {
      options.enableAutodiff = false;
    } else if (arg == "-no-graph-collapse") {
      options.enableGraphCollapsing = false;
    } else if (arg == "-no-ad-prune") {
      options.enableAutodiffPruning = false;
    } else if (arg == "-no-sccp") {
      options.enableSCCP = false;
    } else if (arg == "-no-symbol-dce") {
      options.enableSymbolDCE = false;
    } else if (arg == "-no-simd") {
      options.enableSIMDLayout = false;
    } else if (arg == "-no-poly") {
      options.enablePolyhedral = false;
    } else if (arg == "-no-telemetry") {
      options.injectTelemetryHooks = false;
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

  // ── Tiered Execution Mode ────────────────────────────────────────────────
  if (options.emissionMode == jules::CompilerOptions::EmitTiered || jitMode) {
    jules::DiagnosticsEngine diag;

    // Configure the tiered execution engine.
    jules::TieredExecutionConfig tieredConfig;
    tieredConfig.aotConfig.optLevel = options.optLevel;
    tieredConfig.aotConfig.enableAutodiff = options.enableAutodiff;
    tieredConfig.aotConfig.enableGraphCollapsing = options.enableGraphCollapsing;
    tieredConfig.aotConfig.enableAutodiffPruning = options.enableAutodiffPruning;
    tieredConfig.aotConfig.enableSCCP = options.enableSCCP;
    tieredConfig.aotConfig.enableSymbolDCE = options.enableSymbolDCE;
    tieredConfig.aotConfig.enableSIMDLayout = options.enableSIMDLayout;
    tieredConfig.aotConfig.enablePolyhedral = options.enablePolyhedral;
    tieredConfig.aotConfig.injectTelemetryHooks = options.injectTelemetryHooks;
    tieredConfig.aotConfig.targetDevice = options.xlaTarget;
    tieredConfig.aotConfig.verbose = options.verbose;

    tieredConfig.jitQueueConfig.numWorkers = jitWorkers;
    tieredConfig.jitQueueConfig.verbose = options.verbose;

    tieredConfig.cachePolicyConfig.warmupThreshold = jitWarmup;
    tieredConfig.cachePolicyConfig.verbose = options.verbose;

    tieredConfig.pjrtPlatform = options.xlaTarget;
    tieredConfig.enableTier2 = options.enableTier2;
    tieredConfig.enablePGO = options.enablePGO || jitPGO;
    tieredConfig.verbose = options.verbose;

    jules::TieredExecution engine(diag, tieredConfig);

    if (!engine.initialize()) {
      std::cerr << "error: failed to initialize tiered execution engine\n";
      return 1;
    }

    // Read the source file.
    std::ifstream file(options.inputFile);
    if (!file.is_open() && options.inputFile != "-") {
      std::cerr << "error: cannot open file: " << options.inputFile << "\n";
      return 1;
    }
    std::ostringstream ss;
    if (options.inputFile == "-") {
      ss << std::cin.rdbuf();
    } else {
      ss << file.rdbuf();
    }
    std::string source = ss.str();

    // Load the program (compiles to Tier 1 AOT).
    if (!engine.loadProgram(source, options.inputFile)) {
      std::cerr << "error: failed to load program\n";
      return 1;
    }

    if (options.verbose) {
      std::cerr << "=== Tiered Execution Engine ===\n"
                << "  Platform:  " << options.xlaTarget << "\n"
                << "  Tier 2:    " << (options.enableTier2 ? "enabled" : "disabled") << "\n"
                << "  PGO:       " << (tieredConfig.enablePGO ? "enabled" : "disabled") << "\n"
                << "  Workers:   " << jitWorkers << "\n"
                << "  Warmup:    " << jitWarmup << "\n"
                << "  SCCP:      " << (options.enableSCCP ? "enabled" : "disabled") << "\n"
                << "  SymbolDCE: " << (options.enableSymbolDCE ? "enabled" : "disabled") << "\n"
                << "  SIMD:      " << (options.enableSIMDLayout ? "enabled" : "disabled") << "\n"
                << "  Polyhedral:" << (options.enablePolyhedral ? "enabled" : "disabled") << "\n"
                << "  Telemetry: " << (options.injectTelemetryHooks ? "enabled" : "disabled") << "\n";
    }

    // Execute the main function.
    auto result = engine.execute("main", {});

    if (options.verbose) {
      std::cerr << "Execution result: tier=" << result.tierUsed
                << " success=" << result.success
                << " bailout=" << result.bailedOut
                << " time=" << result.executionTimeMs << "ms\n";
      std::cerr << "Stats: total=" << engine.totalExecutions()
                << " tier1=" << engine.tier1Executions()
                << " tier2=" << engine.tier2Executions()
                << " bailouts=" << engine.totalBailouts() << "\n";
    }

    // Wait for any background compilations to finish.
    engine.waitForCompilations();
    engine.shutdown();

    return result.success ? 0 : 1;
  }

  // ── Standard AOT Compilation Mode ────────────────────────────────────────
  jules::Compiler compiler(std::move(options));
  bool success = compiler.compile();

  if (compiler.getDiagnostics().hasErrors()) {
    return 1;
  }

  return success ? 0 : 1;
}
