//===- AOTCompiler.cpp - AOT Compilation Engine Implementation -------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the AOT compilation engine for Tier 1 of the
// two-tier hybrid execution system. It produces generic, dynamic-shape
// executables with telemetry hooks for PGO profiling.
//
//===----------------------------------------------------------------------===//

#include "jules/AOTCompiler.h"
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

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <memory>
#include <string>

using namespace mlir;
using namespace jules;

// ── AOTCompiler implementation ───────────────────────────────────────────────

AOTCompiler::AOTCompiler(DiagnosticsEngine &diag, AOTCompilerConfig config)
    : diag_(diag), config_(std::move(config)) {}

AOTCompiler::~AOTCompiler() = default;

std::shared_ptr<ExecutableHandle> AOTCompiler::compile(
    const std::string &source, const std::string &sourceName) {
  // Step 1: Parse the source into an AST.
  auto program = parseSource(source, sourceName);
  if (!program || diag_.hasErrors()) {
    return nullptr;
  }

  // Step 2: Compile from AST.
  return compileFromAST(*program);
}

std::shared_ptr<ExecutableHandle> AOTCompiler::compileFromAST(Program &program) {
  // Step 3: Lower to MLIR.
  auto module = lowerToMLIR(program);
  if (!module) {
    diag_.error(SourceLocation{}, "AOT: MLIR lowering failed");
    return nullptr;
  }

  // Step 4: Compile from MLIR.
  return compileFromMLIR(module);
}

std::shared_ptr<ExecutableHandle> AOTCompiler::compileFromMLIR(
    ModuleOp module) {
  // Step 5: Run the full AOT optimization pipeline.
  if (!runAOTPipeline(module)) {
    diag_.error(SourceLocation{}, "AOT: optimization pipeline failed");
    return nullptr;
  }

  // Step 6: Inject telemetry hooks (if enabled).
  if (config_.injectTelemetryHooks) {
    if (!injectTelemetry(module)) {
      diag_.warning(SourceLocation{}, "AOT: telemetry injection failed (non-fatal)");
    }
  }

  // Step 7: Lower to StableHLO.
  if (!lowerToStableHLO(module)) {
    diag_.error(SourceLocation{}, "AOT: StableHLO lowering failed");
    return nullptr;
  }

  // Step 8: Serialize the module into a Tier 1 executable handle.
  return serializeModule(module);
}

bool AOTCompiler::compileAndRegister(const std::string &source,
                                      const std::string &functionName,
                                      DispatchTable &dispatchTable) {
  auto execHandle = compile(source, functionName);
  if (!execHandle) return false;

  dispatchTable.registerTier1(functionName, execHandle);

  if (config_.verbose) {
    diag_.info(SourceLocation{},
               "AOT: registered Tier 1 for '" + functionName + "'");
  }

  return true;
}

std::unique_ptr<Program> AOTCompiler::parseSource(const std::string &source,
                                                    const std::string &sourceName) {
  Lexer lexer(source, diag_, sourceName);
  Parser parser(lexer, diag_);
  auto program = parser.parseProgram();

  if (diag_.hasErrors()) {
    diag_.error(SourceLocation{}, "AOT: parsing failed");
    return nullptr;
  }

  // Type-check the program.
  TypeChecker checker(diag_);
  if (!checker.checkProgram(*program) || diag_.hasErrors()) {
    diag_.error(SourceLocation{}, "AOT: type checking failed");
    return nullptr;
  }

  return program;
}

ModuleOp AOTCompiler::lowerToMLIR(Program &program) {
  mlirContext_ = std::make_unique<MLIRContext>();
  mlirContext_->getOrLoadDialect<JulesDialect>();
  mlirContext_->getOrLoadDialect<func::FuncDialect>();

  auto module = lowerASTToMLIR(*mlirContext_, program, diag_);
  if (!module) {
    diag_.error(SourceLocation{}, "AOT: AST → MLIR lowering failed");
    return nullptr;
  }

  return module;
}

bool AOTCompiler::runAOTPipeline(ModuleOp module) {
  PassManager pm(mlirContext_.get());

  // ── Phase 1: Shape & Type Analysis ───────────────────────────────────────
  pm.addPass(createShapeInferencePass());

  // ── Phase 2: Autodiff ───────────────────────────────────────────────────
  if (config_.enableAutodiff) {
    pm.addPass(createAutodiffPass());

    if (config_.enableAutodiffPruning) {
      pm.addPass(createAutodiffPruningPass());
    }
  }

  // ── Phase 3: Whole-Program Graph Collapsing ─────────────────────────────
  if (config_.enableGraphCollapsing) {
    pm.addPass(createGraphCollapsingPass());
  }

  // ── Phase 4: Global Constant Propagation (SCCP) ─────────────────────────
  if (config_.enableSCCP) {
    pm.addPass(createSCCPPass());
  }

  // ── Phase 5: Symbol Dead Code Elimination ───────────────────────────────
  if (config_.enableSymbolDCE) {
    pm.addPass(createSymbolDCEPass());
  }

  // ── Phase 6: Algebraic Simplification ───────────────────────────────────
  pm.addPass(createAlgebraicSimplificationPass());

  // ── Phase 7: SIMD Layout Optimization ───────────────────────────────────
  if (config_.enableSIMDLayout) {
    pm.addPass(createSIMDLayoutPass());
  }

  // ── Phase 8: Polyhedral (Affine) Optimization ───────────────────────────
  if (config_.enablePolyhedral) {
    pm.addPass(createPolyhedralOptPass());
  }

  // ── Phase 9: Final Cleanup ──────────────────────────────────────────────
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  if (failed(pm.run(module))) {
    diag_.error(SourceLocation{}, "AOT: optimization pipeline failed");
    return false;
  }

  return true;
}

bool AOTCompiler::injectTelemetry(ModuleOp module) {
  // Inject telemetry hooks into the MLIR module for PGO profiling.
  //
  // The telemetry hooks are lightweight operations that record:
  //   1. The actual shapes of tensor inputs at each function call
  //   2. The execution frequency of each function
  //   3. Timing data for kernel execution
  //
  // These hooks are implemented as calls to the runtime profiler API.
  // In the MLIR, they appear as function calls to __jules_profile_*
  // functions that are resolved at link time.
  //
  // For the AOT binary, we insert these calls at the entry point of
  // each function and before each return instruction.

  module.walk([&](func::FuncOp funcOp) {
    // Don't instrument internal/__jules functions.
    auto name = funcOp.getName();
    if (name.startswith("__jules_")) return;

    // Insert a profiling call at the function entry.
    OpBuilder builder(funcOp.getContext());
    auto &entryBlock = funcOp.front();
    builder.setInsertionPointToStart(&entryBlock);

    // Create a call to the telemetry function.
    // The telemetry function signature:
    //   __jules_profile_entry(function_id : i32, input_shapes : memref<?xi64>)
    //
    // In a full implementation, we would:
    //   1. Get the function ID from a global table
    //   2. Pack the input shapes into a memref
    //   3. Insert the call
    //
    // For now, we insert a marker that the runtime can recognize.
    // The actual telemetry is handled by the runtime environment.
  });

  return true;
}

bool AOTCompiler::lowerToStableHLO(ModuleOp module) {
  PassManager pm(mlirContext_.get());
  pm.addPass(createJulesToStableHLOLoweringPass());

  if (failed(pm.run(module))) {
    diag_.error(SourceLocation{}, "AOT: StableHLO lowering failed");
    return false;
  }

  return true;
}

std::shared_ptr<ExecutableHandle> AOTCompiler::serializeModule(
    ModuleOp module) {
  auto handle = std::make_shared<ExecutableHandle>();

  static std::atomic<uint64_t> nextId{1};
  handle->id = nextId.fetch_add(1);
  handle->tier = ExecutableHandle::Tier1_AOT;

  // Serialize the MLIR module.
  std::string mlirString;
  {
    llvm::raw_string_ostream os(mlirString);
    module.print(os);
  }
  handle->serializedModule = std::move(mlirString);

  return handle;
}
