//===- Compiler.cpp - Jules Compiler Driver Implementation ------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the top-level Compiler driver that orchestrates the
// full compilation pipeline, including the two-tier AOT/JIT hybrid system.
//
//===----------------------------------------------------------------------===//

#include "jules/Compiler.h"
#include "jules/Lexer.h"
#include "jules/Parser.h"
#include "jules/AST.h"
#include "jules/TypeSystem.h"
#include "jules/MLIRGen.h"
#include "jules/Diagnostics.h"
#include "jules/Dialect/JulesDialect.h"
#include "jules/Dialect/JulesOps.h"
#include "jules/Passes/Passes.h"
#include "jules/PJRT.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace mlir;
using namespace jules;

Compiler::Compiler(CompilerOptions options)
    : options_(std::move(options)),
      diag_(DiagnosticsEngine::defaultEmit) {}

bool Compiler::compile() {
  // Step 1: Read source.
  std::string source = readSource();
  if (source.empty() && options_.inputFile != "-") {
    diag_.fatal(SourceLocation{},
                "could not read input file: " + options_.inputFile);
    return false;
  }

  // Step 2: Parse into AST.
  auto program = parseSource(source);
  if (!program || diag_.hasErrors()) {
    return false;
  }

  // Step 3: Type-check.
  if (!typeCheck(*program)) {
    return false;
  }

  // If we're only emitting the AST, dump it and stop.
  if (options_.emissionMode == CompilerOptions::EmitAST) {
    std::ostream &out = (options_.outputFile == "-" || options_.outputFile.empty())
                            ? std::cout
                            : *(new std::ofstream(options_.outputFile));
    for (const auto &fn : program->getFunctions()) {
      out << fn->getName() << " : " << fn->getType()->toString() << "\n";
    }
    if (&out != &std::cout) delete &out;
    return true;
  }

  // Step 4: Lower AST to MLIR.
  if (!lowerToMLIR(*program)) {
    return false;
  }

  // Step 5: Run autodiff pass if enabled.
  if (options_.enableAutodiff &&
      options_.emissionMode >= CompilerOptions::EmitMLIRAfterAD) {
    if (!runAutodiffPass()) {
      return false;
    }
  }

  // Step 6: Run the full AOT optimization pipeline (if emitting executable or tiered).
  if (options_.emissionMode >= CompilerOptions::EmitExecutable) {
    if (!runAOTOptimizations()) {
      return false;
    }
  }

  // Step 7: Lower to StableHLO if requested.
  if (options_.emissionMode >= CompilerOptions::EmitStableHLO) {
    if (!lowerToStableHLO()) {
      return false;
    }
  }

  // Step 8: Compile through XLA if requested.
  if (options_.emissionMode == CompilerOptions::EmitExecutable) {
    if (!compileXLA()) {
      return false;
    }
  }

  // Step 9: Emit output.
  return emitOutput();
}

std::string Compiler::readSource() {
  if (options_.inputFile == "-" || options_.inputFile.empty()) {
    std::ostringstream ss;
    ss << std::cin.rdbuf();
    return ss.str();
  }

  std::ifstream file(options_.inputFile);
  if (!file.is_open()) {
    diag_.error(SourceLocation{},
                "cannot open file: " + options_.inputFile);
    return "";
  }

  std::ostringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

std::unique_ptr<Program> Compiler::parseSource(const std::string &source) {
  Lexer lexer(source, diag_, options_.inputFile);
  Parser parser(lexer, diag_);
  auto program = parser.parseProgram();

  if (diag_.hasErrors()) {
    diag_.error(SourceLocation{}, "parsing failed with errors");
    return nullptr;
  }

  return program;
}

bool Compiler::typeCheck(Program &program) {
  TypeChecker checker(diag_);
  bool ok = checker.checkProgram(program);

  if (!ok || diag_.hasErrors()) {
    diag_.error(SourceLocation{}, "type checking failed");
    return false;
  }

  return true;
}

bool Compiler::lowerToMLIR(Program &program) {
  mlirContext_ = std::make_unique<MLIRContext>();
  mlirContext_->getOrLoadDialect<JulesDialect>();
  mlirContext_->getOrLoadDialect<func::FuncDialect>();

  auto module = lowerASTToMLIR(*mlirContext_, program, diag_);
  if (!module) {
    diag_.error(SourceLocation{}, "MLIR lowering failed");
    return false;
  }

  mlirModule_ = std::move(module);
  return true;
}

bool Compiler::runAutodiffPass() {
  if (!mlirModule_) return false;

  PassManager pm(mlirContext_.get());
  pm.addPass(createShapeInferencePass());

  if (options_.enableFusedAutodiff) {
    // Use the fused autodiff+pruning+collapsing pass for minimal IR bloat.
    pm.addPass(createFusedAutodiffPass());
  } else {
    pm.addPass(createAutodiffPass());

    if (options_.enableAutodiffPruning) {
      pm.addPass(createAutodiffPruningPass());
    }
  }

  if (failed(pm.run(*mlirModule_))) {
    diag_.error(SourceLocation{}, "autodiff pass failed");
    return false;
  }

  return true;
}

bool Compiler::runAOTOptimizations() {
  if (!mlirModule_) return false;

  PassManager pm(mlirContext_.get());

  // ── Phase 1: Shape Inference + Polymorphism ──────────────────────────────
  pm.addPass(createShapeInferencePass());

  if (options_.enableShapePolymorphism) {
    pm.addPass(createShapePolymorphismPass());
  }

  // ── Phase 2: Graph Collapsing ───────────────────────────────────────────
  if (options_.enableGraphCollapsing) {
    pm.addPass(createGraphCollapsingPass());
  }

  // ── Phase 3: Whole-Program Collapsing ───────────────────────────────────
  pm.addPass(createWholeProgramCollapsingPass());

  // ── Phase 4: SCCP (Constant Propagation with Tensor Consteval) ──────────
  if (options_.enableSCCP) {
    pm.addPass(createSCCPPass());
  }

  // ── Phase 5: SymbolDCE ──────────────────────────────────────────────────
  if (options_.enableSymbolDCE) {
    pm.addPass(createSymbolDCEPass());
  }

  // ── Phase 6: Algebraic Simplification ───────────────────────────────────
  pm.addPass(createAlgebraicSimplificationPass());

  // ── Phase 7: Mixed Precision (bf16/fp8) ─────────────────────────────────
  if (options_.enableMixedPrecision) {
    pm.addPass(createMixedPrecisionPass(options_.mixedPrecisionTarget));
  }

  // ── Phase 8: SIMD Layout Optimization ───────────────────────────────────
  if (options_.enableSIMDLayout) {
    pm.addPass(createSIMDLayoutPass());
  }

  // ── Phase 9: Kernel Fusion ──────────────────────────────────────────────
  if (options_.enableKernelFusion) {
    pm.addPass(createProducerConsumerFusionPass());
  }

  // ── Phase 10: Memory Planning ───────────────────────────────────────────
  if (options_.enableMemoryPlanning) {
    if (options_.memoryBudgetBytes > 0) {
      pm.addPass(createMemoryPlanningPass(options_.memoryBudgetBytes));
    } else {
      pm.addPass(createMemoryPlanningPass());
    }
  }

  // ── Phase 11: Polyhedral Optimization ───────────────────────────────────
  if (options_.enablePolyhedral) {
    pm.addPass(createPolyhedralOptPass());
  }

  // ── Phase 12: Quantization ──────────────────────────────────────────────
  if (options_.enableQuantization) {
    pm.addPass(createQuantizePass());
  }

  // ── Phase 13: Final Cleanup ─────────────────────────────────────────────
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  if (failed(pm.run(*mlirModule_))) {
    diag_.error(SourceLocation{}, "AOT optimization pipeline failed");
    return false;
  }

  return true;
}

bool Compiler::lowerToStableHLO() {
  if (!mlirModule_) return false;

  PassManager pm(mlirContext_.get());
  pm.addPass(createJulesToStableHLOLoweringPass());
  pm.addPass(createCanonicalizerPass());

  if (failed(pm.run(*mlirModule_))) {
    diag_.error(SourceLocation{}, "StableHLO lowering failed");
    return false;
  }

  return true;
}

bool Compiler::compileXLA() {
  if (!mlirModule_) return false;

  if (options_.verbose) {
    std::cerr << "Compiling through XLA for target: " << options_.xlaTarget
              << std::endl;
  }

  // Serialize the MLIR module to string.
  std::string mlirString;
  {
    llvm::raw_string_ostream os(mlirString);
    mlirModule_->print(os);
  }

  // Create PJRT client for the target platform.
  DiagnosticsEngine pjrtDiag(DiagnosticsEngine::defaultEmit);
  auto pjrtClient = PJRTClient::create(options_.xlaTarget, pjrtDiag);
  if (!pjrtClient) {
    diag_.error(SourceLocation{},
                "failed to create PJRT client for target: " + options_.xlaTarget);
    return false;
  }

  // Compile and load the program through PJRT.
  auto executable = pjrtClient->compileAndLoad(mlirString);
  if (!executable) {
    diag_.error(SourceLocation{}, "XLA compilation failed");
    return false;
  }

  if (options_.verbose) {
    std::cerr << "XLA compilation successful" << std::endl;
  }

  return true;
}

bool Compiler::emitOutput() {
  if (!mlirModule_ && options_.emissionMode != CompilerOptions::EmitAST) {
    diag_.error(SourceLocation{}, "no MLIR module to emit");
    return false;
  }

  std::ostream *out = &std::cout;
  std::ofstream outFile;
  if (options_.outputFile != "-" && !options_.outputFile.empty()) {
    outFile.open(options_.outputFile);
    if (!outFile.is_open()) {
      diag_.error(SourceLocation{},
                  "cannot open output file: " + options_.outputFile);
      return false;
    }
    out = &outFile;
  }

  if (mlirModule_) {
    auto printingFlags = OpPrintingFlags();
    printingFlags.useLocalScope();
    printingFlags.enableDebugInfo();

    mlirModule_->print(*out, printingFlags);
    *out << "\n";
  }

  return true;
}
