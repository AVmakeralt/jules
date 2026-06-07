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
//   -o <file>      Output file (default: stdout)
//   -emit-ast      Emit the parsed AST
//   -emit-mlir     Emit MLIR in the Jules dialect (default)
//   -emit-mlir-ad  Emit MLIR after the autodiff pass
//   -emit-stablehlo Emit MLIR lowered to StableHLO
//   -emit-exe      Compile through XLA to an executable
//   -O0/-O1/-O2    Optimization level
//   -no-autodiff   Disable the autodiff pass
//   -target <t>    XLA target: cpu, cuda, rocm, tpu
//   -v             Verbose output
//   -h             Show help
//
//===----------------------------------------------------------------------===//

#include "jules/Compiler.h"
#include "jules/Diagnostics.h"
#include "jules/Dialect/JulesDialect.h"
#include "jules/Passes/Passes.h"

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
            << "Options:\n"
            << "  -o <file>         Output file (default: stdout)\n"
            << "  -emit-ast         Emit the parsed AST\n"
            << "  -emit-mlir        Emit MLIR in the Jules dialect (default)\n"
            << "  -emit-mlir-ad     Emit MLIR after the autodiff pass\n"
            << "  -emit-stablehlo   Emit MLIR lowered to StableHLO\n"
            << "  -emit-exe         Compile through XLA to an executable\n"
            << "  -O0 / -O1 / -O2   Optimization level\n"
            << "  -no-autodiff      Disable the autodiff pass\n"
            << "  -target <target>  XLA target: cpu, cuda, rocm, tpu\n"
            << "  -v                Verbose output\n"
            << "  -h                Show this help message\n";
}

int main(int argc, char **argv) {
  jules::CompilerOptions options;
  options.emissionMode = jules::CompilerOptions::EmitMLIR;
  options.inputFile = "-";
  options.outputFile = "-";

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
    } else if (arg == "-O0") {
      options.optLevel = 0;
    } else if (arg == "-O1") {
      options.optLevel = 1;
    } else if (arg == "-O2") {
      options.optLevel = 2;
    } else if (arg == "-no-autodiff") {
      options.enableAutodiff = false;
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

  // Create and run the compiler.
  jules::Compiler compiler(std::move(options));
  bool success = compiler.compile();

  if (compiler.getDiagnostics().hasErrors()) {
    return 1;
  }

  return success ? 0 : 1;
}
