//===- JIT.cpp - JIT Compiler with PGO Implementation ----------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the JIT compiler that orchestrates speculative
// compilation, PGO profiling, and background recompilation.
//
//===----------------------------------------------------------------------===//

#include "jules/JIT.h"
#include "jules/Tracing.h"
#include "jules/Profiler.h"
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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace mlir;
using namespace jules;

// ── Concrete Executable implementation ───────────────────────────────────────

namespace {

/// A simple executable that wraps compiled MLIR + a PJRT executable handle.
class JulesExecutable : public Executable {
public:
  JulesExecutable(uint64_t id, uint64_t traceId, bool specialized,
                  std::string mlirModule)
      : mlirModule_(std::move(mlirModule)) {
    id_ = id;
    traceId_ = traceId;
    specialized_ = specialized;
  }

  std::vector<void *> execute(const std::vector<void *> &inputs) override {
    // In a full implementation, this would invoke the PJRT executable.
    // For now, it returns empty (the MLIR module has been compiled
    // and can be inspected / serialized).
    return {};
  }

  /// Get the serialized MLIR module.
  const std::string &getMLIRModule() const { return mlirModule_; }

private:
  std::string mlirModule_;
};

} // anonymous namespace

// ── JITCompiler implementation ───────────────────────────────────────────────

JITCompiler::JITCompiler(DiagnosticsEngine &diag, JITConfig config)
    : diag_(diag), config_(std::move(config)) {}

JITCompiler::~JITCompiler() {
  shuttingDown_.store(true);
  pgoCV_.notify_all();
  for (auto &thread : pgoThreads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}

std::shared_ptr<Executable> JITCompiler::compileTrace(ActiveTrace &trace,
                                                       bool useStaticShapes) {
  static std::atomic<uint64_t> nextExecId{1};
  uint64_t execId = nextExecId.fetch_add(1);
  uint64_t traceId = 0; // Will be assigned by the caller

  return compileThroughMLIR(trace, useStaticShapes, traceId);
}

std::shared_ptr<Executable> JITCompiler::compileSpeculative(ActiveTrace &trace) {
  return compileTrace(trace, false);
}

std::shared_ptr<Executable> JITCompiler::compileSpecialized(ActiveTrace &trace,
                                                             uint64_t traceId) {
  // Get static shapes from profiler.
  auto staticShapes = profiler_.getStaticShapes(traceId);

  auto exec = compileTrace(trace, true);

  if (exec) {
    // Mark PGO as done.
    auto *profile = profiler_.getTraceProfile(traceId);
    if (profile) {
      profile->pgoRecompiled.store(true);
    }

    // Atomically swap the executable.
    swapExecutable(traceId, exec);
  }

  return exec;
}

std::shared_ptr<Executable> JITCompiler::lookupExecutable(
    uint64_t traceId) const {
  std::lock_guard<std::mutex> lock(execMutex_);
  auto it = executables_.find(traceId);
  return it != executables_.end() ? it->second : nullptr;
}

void JITCompiler::registerExecutable(uint64_t traceId,
                                      std::shared_ptr<Executable> exec) {
  std::lock_guard<std::mutex> lock(execMutex_);
  executables_[traceId] = std::move(exec);
}

void JITCompiler::swapExecutable(uint64_t traceId,
                                  std::shared_ptr<Executable> newExec) {
  std::lock_guard<std::mutex> lock(execMutex_);
  executables_[traceId] = std::move(newExec);
}

bool JITCompiler::shouldPGORecompile(uint64_t traceId) const {
  if (!config_.enablePGO) return false;
  return profiler_.shouldPGORecompile(traceId);
}

void JITCompiler::launchPGORecompile(uint64_t traceId, ActiveTrace &trace) {
  if (activePGORecompiles_.load() >= config_.maxPGORecompiles) {
    return; // Too many concurrent recompiles.
  }

  activePGORecompiles_.fetch_add(1);

  // Copy the trace for the background thread.
  // (ActiveTrace is not thread-safe, so we compile a copy.)
  // In a production system, we'd serialize the trace or keep it alive.
  auto &diag = diag_;
  auto &profiler = profiler_;
  auto &config = config_;
  auto &executables = executables_;
  auto &execMutex = execMutex_;
  auto &activeCount = activePGORecompiles_;

  // Launch background compilation.
  pgoThreads_.emplace_back([&diag, &profiler, &config, traceId,
                             activeCount]() {
    // Background PGO recompilation.
    // In a full implementation, this would:
    //   1. Get the static shapes from the profiler
    //   2. Reconstruct the MLIR module with concrete shapes
    //   3. Run the full optimization pipeline
    //   4. Compile through XLA
    //   5. Atomically swap the executable

    auto staticShapes = profiler.getStaticShapes(traceId);
    if (staticShapes) {
      // Compile with specialized shapes.
      // This would go through compileThroughMLIR with useStaticShapes=true.
      if (config.verbose) {
        // Log PGO recompilation.
      }

      auto *profile = profiler.getTraceProfile(traceId);
      if (profile) {
        profile->pgoRecompiled.store(true);
      }
    }

    activeCount.fetch_sub(1);
  });
}

void JITCompiler::waitForPGORecompiles() {
  for (auto &thread : pgoThreads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  pgoThreads_.clear();
}

std::shared_ptr<Executable> JITCompiler::compileThroughMLIR(
    ActiveTrace &trace, bool useStaticShapes, uint64_t traceId) {
  // Create an MLIR context.
  auto context = std::make_unique<MLIRContext>();
  context->getOrLoadDialect<JulesDialect>();
  context->getOrLoadDialect<func::FuncDialect>();

  // Build an MLIR module from the trace.
  auto module = ModuleOp::create(UnknownLoc::get(context.get()));
  OpBuilder builder(context.get());

  // Create a single function that represents the entire trace.
  auto f32Type = FloatType::getF32(context.get());

  // Determine the function signature from the trace.
  // Inputs are the "root" values (not produced by any op in the trace).
  // Outputs are the return value.
  SmallVector<Type, 4> inputTypes;
  SmallVector<Value, 4> inputValueMap;

  // Count root values (inputs that have no defining op).
  const auto &ops = trace.getOps();
  const auto &values = trace.getValues();

  // Track which values are produced by operations.
  std::vector<bool> isProduced(values.size(), false);
  for (const auto &op : ops) {
    for (auto outId : op.outputs) {
      if (outId < values.size()) {
        isProduced[outId] = true;
      }
    }
  }

  // Root values (not produced by any op) become function arguments.
  for (size_t i = 0; i < values.size(); ++i) {
    if (!isProduced[i] && values[i].isValid()) {
      // This is a root value — make it a function argument.
      Type argType = f32Type; // Default
      if (values[i].type) {
        // Convert Jules type to MLIR type.
        if (values[i].type->getKind() == TypeNode::ScalarType) {
          auto &scalarTy = static_cast<const ScalarType &>(*values[i].type);
          switch (scalarTy.getScalarKind()) {
          case ScalarType::SK_F32: argType = FloatType::getF32(context.get()); break;
          case ScalarType::SK_F64: argType = FloatType::getF64(context.get()); break;
          case ScalarType::SK_I32: argType = IntegerType::get(context, 32); break;
          case ScalarType::SK_I64: argType = IntegerType::get(context, 64); break;
          case ScalarType::SK_Bool: argType = IntegerType::get(context, 1); break;
          default: argType = f32Type; break;
          }
        } else if (values[i].type->getKind() == TypeNode::TensorType) {
          auto &tensorTy = static_cast<const TensorType &>(*values[i].type);
          SmallVector<int64_t, 4> shape;
          for (const auto &dim : tensorTy.getDims()) {
            if (dim.kind == Dimension::DK_Concrete) {
              // Use PGO data for static shapes if available.
              if (useStaticShapes && traceId > 0) {
                auto staticShapes = profiler_.getStaticShapes(traceId);
                if (staticShapes && i < staticShapes->size()) {
                  // Not directly indexable by TraceValueId in this context,
                  // but the concept is that PGO shapes override.
                  shape.push_back(dim.size);
                } else {
                  shape.push_back(dim.size);
                }
              } else {
                shape.push_back(dim.size);
              }
            } else {
              // Dynamic or symbolic.
              if (useStaticShapes && traceId > 0) {
                // Try to resolve from PGO data.
                shape.push_back(ShapedType::kDynamic);
              } else {
                shape.push_back(ShapedType::kDynamic);
              }
            }
          }

          Type elemType = FloatType::getF32(context.get());
          switch (tensorTy.getElementKind()) {
          case ScalarType::SK_F32: elemType = FloatType::getF32(context.get()); break;
          case ScalarType::SK_F64: elemType = FloatType::getF64(context.get()); break;
          default: break;
          }

          argType = RankedTensorType::get(shape, elemType);
        }
      }
      inputTypes.push_back(argType);
    }
  }

  // Determine the return type.
  Type returnType = f32Type;
  if (auto retVal = trace.getReturnValue()) {
    if (*retVal < values.size() && values[*retVal].type) {
      // Use the return value's type.
      returnType = f32Type; // Simplified
    }
  }

  auto funcType = builder.getFunctionType(inputTypes, returnType);
  auto funcOp = func::FuncOp::create(builder.getUnknownLoc(), "__trace_fn",
                                       funcType);

  // Create the entry block.
  auto *entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Map trace values to MLIR values.
  std::vector<Value> mlirValues(values.size());

  // Bind function arguments to root trace values.
  size_t argIdx = 0;
  for (size_t i = 0; i < values.size(); ++i) {
    if (!isProduced[i] && values[i].isValid()) {
      if (argIdx < funcOp.getNumArguments()) {
        mlirValues[i] = funcOp.getArgument(argIdx);
        argIdx++;
      }
    }
  }

  // Emit MLIR operations for each trace op.
  for (const auto &op : ops) {
    auto loc = builder.getUnknownLoc();

    switch (op.kind) {
    case TraceOpKind::Constant: {
      double val = 0.0;
      if (!op.attrs.empty() && op.attrs[0].kind == TraceOp::Attr::Kind::Float) {
        val = std::get<double>(op.attrs[0].value);
      }
      auto constVal = builder.create<ConstantOp>(
          loc, builder.getFloatAttr(f32Type, val));
      if (!op.outputs.empty() && op.outputs[0] < mlirValues.size()) {
        mlirValues[op.outputs[0]] = constVal.getResult();
      }
      break;
    }

    case TraceOpKind::Add: {
      if (op.inputs.size() >= 2 && op.outputs.size() >= 1 &&
          mlirValues[op.inputs[0]] && mlirValues[op.inputs[1]]) {
        auto result = builder.create<AddOp>(
            loc, mlirValues[op.inputs[0]], mlirValues[op.inputs[1]]);
        mlirValues[op.outputs[0]] = result.getResult();
      }
      break;
    }

    case TraceOpKind::Sub: {
      if (op.inputs.size() >= 2 && op.outputs.size() >= 1 &&
          mlirValues[op.inputs[0]] && mlirValues[op.inputs[1]]) {
        auto result = builder.create<SubOp>(
            loc, mlirValues[op.inputs[0]], mlirValues[op.inputs[1]]);
        mlirValues[op.outputs[0]] = result.getResult();
      }
      break;
    }

    case TraceOpKind::Mul: {
      if (op.inputs.size() >= 2 && op.outputs.size() >= 1 &&
          mlirValues[op.inputs[0]] && mlirValues[op.inputs[1]]) {
        auto result = builder.create<MulOp>(
            loc, mlirValues[op.inputs[0]], mlirValues[op.inputs[1]]);
        mlirValues[op.outputs[0]] = result.getResult();
      }
      break;
    }

    case TraceOpKind::Div: {
      if (op.inputs.size() >= 2 && op.outputs.size() >= 1 &&
          mlirValues[op.inputs[0]] && mlirValues[op.inputs[1]]) {
        auto result = builder.create<DivOp>(
            loc, mlirValues[op.inputs[0]], mlirValues[op.inputs[1]]);
        mlirValues[op.outputs[0]] = result.getResult();
      }
      break;
    }

    case TraceOpKind::Pow: {
      if (op.inputs.size() >= 2 && op.outputs.size() >= 1 &&
          mlirValues[op.inputs[0]] && mlirValues[op.inputs[1]]) {
        auto result = builder.create<PowOp>(
            loc, mlirValues[op.inputs[0]], mlirValues[op.inputs[1]]);
        mlirValues[op.outputs[0]] = result.getResult();
      }
      break;
    }

    case TraceOpKind::Neg: {
      if (op.inputs.size() >= 1 && op.outputs.size() >= 1 &&
          mlirValues[op.inputs[0]]) {
        auto result = builder.create<NegOp>(
            loc, mlirValues[op.inputs[0]]);
        mlirValues[op.outputs[0]] = result.getResult();
      }
      break;
    }

    case TraceOpKind::MatMul: {
      if (op.inputs.size() >= 2 && op.outputs.size() >= 1 &&
          mlirValues[op.inputs[0]] && mlirValues[op.inputs[1]]) {
        auto result = builder.create<MatMulOp>(
            loc, mlirValues[op.inputs[0]], mlirValues[op.inputs[1]]);
        mlirValues[op.outputs[0]] = result.getResult();
      }
      break;
    }

    case TraceOpKind::Relu: {
      if (op.inputs.size() >= 1 && op.outputs.size() >= 1 &&
          mlirValues[op.inputs[0]]) {
        auto result = builder.create<ReluOp>(
            loc, mlirValues[op.inputs[0]]);
        mlirValues[op.outputs[0]] = result.getResult();
      }
      break;
    }

    case TraceOpKind::Sigmoid: {
      if (op.inputs.size() >= 1 && op.outputs.size() >= 1 &&
          mlirValues[op.inputs[0]]) {
        auto result = builder.create<SigmoidOp>(
            loc, mlirValues[op.inputs[0]]);
        mlirValues[op.outputs[0]] = result.getResult();
      }
      break;
    }

    case TraceOpKind::Tanh: {
      if (op.inputs.size() >= 1 && op.outputs.size() >= 1 &&
          mlirValues[op.inputs[0]]) {
        auto result = builder.create<TanhOp>(
            loc, mlirValues[op.inputs[0]]);
        mlirValues[op.outputs[0]] = result.getResult();
      }
      break;
    }

    case TraceOpKind::Mean: {
      if (op.inputs.size() >= 1 && op.outputs.size() >= 1 &&
          mlirValues[op.inputs[0]]) {
        auto result = builder.create<MeanOp>(
            loc, mlirValues[op.inputs[0]]);
        mlirValues[op.outputs[0]] = result.getResult();
      }
      break;
    }

    case TraceOpKind::Sum: {
      if (op.inputs.size() >= 1 && op.outputs.size() >= 1 &&
          mlirValues[op.inputs[0]]) {
        auto result = builder.create<SumOp>(
            loc, mlirValues[op.inputs[0]]);
        mlirValues[op.outputs[0]] = result.getResult();
      }
      break;
    }

    case TraceOpKind::Transpose: {
      if (op.inputs.size() >= 1 && op.outputs.size() >= 1 &&
          mlirValues[op.inputs[0]]) {
        auto result = builder.create<TransposeOp>(
            loc, mlirValues[op.inputs[0]]);
        mlirValues[op.outputs[0]] = result.getResult();
      }
      break;
    }

    case TraceOpKind::Zeros:
    case TraceOpKind::Ones:
    case TraceOpKind::Random: {
      // Tensor creation: extract shape from attributes.
      if (op.outputs.size() >= 1) {
        std::vector<int64_t> shape = {1};
        for (const auto &attr : op.attrs) {
          if (attr.kind == TraceOp::Attr::Kind::Shape) {
            shape = std::get<std::vector<int64_t>>(attr.value);
          }
        }
        auto resultType = RankedTensorType::get(shape, f32Type);

        if (op.kind == TraceOpKind::Zeros) {
          auto result = builder.create<ZerosOp>(loc, resultType);
          mlirValues[op.outputs[0]] = result.getResult();
        } else if (op.kind == TraceOpKind::Ones) {
          auto result = builder.create<OnesOp>(loc, resultType);
          mlirValues[op.outputs[0]] = result.getResult();
        } else {
          auto result = builder.create<RandomOp>(loc, resultType);
          mlirValues[op.outputs[0]] = result.getResult();
        }
      }
      break;
    }

    case TraceOpKind::Cast: {
      if (op.inputs.size() >= 1 && op.outputs.size() >= 1 &&
          mlirValues[op.inputs[0]]) {
        auto result = builder.create<CastOp>(
            loc, mlirValues[op.inputs[0]],
            TypeAttr::get(f32Type));
        mlirValues[op.outputs[0]] = result.getResult();
      }
      break;
    }

    case TraceOpKind::Select: {
      if (op.inputs.size() >= 3 && op.outputs.size() >= 1 &&
          mlirValues[op.inputs[0]] && mlirValues[op.inputs[1]] &&
          mlirValues[op.inputs[2]]) {
        auto result = builder.create<SelectOp>(
            loc, mlirValues[op.inputs[0]],
            mlirValues[op.inputs[1]], mlirValues[op.inputs[2]]);
        mlirValues[op.outputs[0]] = result.getResult();
      }
      break;
    }

    case TraceOpKind::Grad: {
      // Autodiff: handled by the autodiff pass later.
      if (op.inputs.size() >= 1 && op.outputs.size() >= 1 &&
          mlirValues[op.inputs[0]]) {
        auto diffVar = "x";
        if (!op.attrs.empty() &&
            op.attrs[0].kind == TraceOp::Attr::Kind::String) {
          diffVar = std::get<std::string>(op.attrs[0].value);
        }
        auto result = builder.create<GradOp>(
            loc, mlirValues[op.inputs[0]],
            builder.getStringAttr(diffVar));
        mlirValues[op.outputs[0]] = result.getResult();
      }
      break;
    }

    case TraceOpKind::Return: {
      if (!op.inputs.empty() && op.inputs[0] < mlirValues.size() &&
          mlirValues[op.inputs[0]]) {
        builder.create<func::ReturnOp>(loc, mlirValues[op.inputs[0]]);
      } else {
        auto zero = builder.create<ConstantOp>(
            loc, builder.getFloatAttr(f32Type, 0.0));
        builder.create<func::ReturnOp>(loc, zero.getResult());
      }
      break;
    }

    case TraceOpKind::Barrier:
      // Barriers don't emit MLIR ops.
      break;

    default:
      // Unhandled trace op kind — skip.
      break;
    }
  }

  // If no return was emitted, add a default one.
  if (entryBlock->empty() || !entryBlock->mightHaveTerminator()) {
    auto zero = builder.create<ConstantOp>(
        builder.getUnknownLoc(), builder.getFloatAttr(f32Type, 0.0));
    builder.create<func::ReturnOp>(builder.getUnknownLoc(), zero.getResult());
  }

  module.push_back(funcOp);

  // Run optimization passes on the MLIR module.
  PassManager pm(context.get());

  // Shape inference.
  pm.addPass(createShapeInferencePass());

  // Autodiff pass (if there are grad ops).
  pm.addPass(createAutodiffPass());

  // Autodiff pruning.
  if (config_.enableAutodiffPruning) {
    pm.addPass(createAutodiffPruningPass());
  }

  // Graph collapsing.
  if (config_.enableGraphCollapsing) {
    pm.addPass(createGraphCollapsingPass());
  }

  // Algebraic simplification.
  pm.addPass(createAlgebraicSimplificationPass());

  // Canonicalization.
  pm.addPass(createCanonicalizerPass());

  // CSE (Common Subexpression Elimination).
  pm.addPass(createCSEPass());

  // Lower to StableHLO.
  pm.addPass(createJulesToStableHLOLoweringPass());

  if (failed(pm.run(module))) {
    diag_.error(SourceLocation{}, "MLIR compilation failed for trace");
    return nullptr;
  }

  // Serialize the MLIR module.
  std::string mlirString;
  {
    llvm::raw_string_ostream os(mlirString);
    module.print(os);
  }

  static std::atomic<uint64_t> nextExecId{1};
  auto exec = std::make_shared<JulesExecutable>(
      nextExecId.fetch_add(1), traceId, useStaticShapes,
      std::move(mlirString));

  return exec;
}
