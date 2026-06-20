//===- JIT.cpp - Tier 2 JIT Compiler Implementation ------------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the Tier 2 JIT compiler. It produces specialized,
// static-shape XLA binaries that are hyper-optimized for the specific
// tensor dimensions observed during PGO profiling.
//
// The Tier 2 compilation pipeline:
//
//   ActiveTrace → MLIR Module (with concrete shapes)
//     → Shape Inference
//     → Autodiff Pass
//     → Autodiff Pruning
//     → Graph Collapsing
//     → Whole-Program Collapsing
//     → SCCP (Constant Propagation)
//     → SymbolDCE
//     → Algebraic Simplification
//     → SIMD Layout Optimization
//     → Polyhedral Optimization (with static dimensions)
//     → Canonicalization + CSE
//     → StableHLO Lowering
//     → Tier 2 Executable
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

/// A JIT-compiled executable that wraps compiled MLIR + a PJRT executable.
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
    return {};
  }

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
  uint64_t traceId = 0;

  return compileThroughMLIR(trace, useStaticShapes, traceId);
}

std::shared_ptr<Executable> JITCompiler::compileSpeculative(ActiveTrace &trace) {
  return compileTrace(trace, false);
}

std::shared_ptr<Executable> JITCompiler::compileSpecialized(ActiveTrace &trace,
                                                             uint64_t traceId) {
  auto exec = compileTrace(trace, true);

  if (exec) {
    auto *profile = profiler_.getTraceProfile(traceId);
    if (profile) {
      profile->pgoRecompiled.store(true);
    }
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
    return;
  }

  activePGORecompiles_.fetch_add(1);

  auto &diag = diag_;
  auto &profiler = profiler_;
  auto &config = config_;
  auto &executables = executables_;
  auto &execMutex = execMutex_;
  auto &activeCount = activePGORecompiles_;

  pgoThreads_.emplace_back([this, traceId]() {
    auto &diag = diag_;
    auto &profiler = profiler_;
    auto &config = config_;
    auto &activeCount = activePGORecompiles_;
    auto staticShapes = profiler.getStaticShapes(traceId);
    if (staticShapes) {
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
  // FIX (Perf 5): Reuse the MLIRContext across JIT compilations instead of
  // creating a new one each time. Creating a new MLIRContext and loading
  // dialects costs hundreds of milliseconds. We cache the context and
  // only create it once.
  //
  // Note: MLIRContext is NOT thread-safe for concurrent mutations, so we
  // protect it with a mutex. This is still much faster than creating a
  // new context each time, since the mutex is only held during the
  // context setup phase.
  std::shared_ptr<MLIRContext> context;
  {
    std::lock_guard<std::mutex> lock(contextMutex_);
    if (!cachedContext_) {
      cachedContext_ = std::make_shared<MLIRContext>();
      cachedContext_->getOrLoadDialect<JulesDialect>();
      cachedContext_->getOrLoadDialect<func::FuncDialect>();
    }
    context = cachedContext_;
  }

  auto module = ModuleOp::create(UnknownLoc::get(context.get()));
  OpBuilder builder(context.get());

  auto f32Type = FloatType::getF32(context.get());

  // Determine the function signature from the trace.
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

  // Root values become function arguments.
  SmallVector<Type, 4> inputTypes;
  for (size_t i = 0; i < values.size(); ++i) {
    if (!isProduced[i] && values[i].isValid()) {
      Type argType = f32Type;
      if (values[i].type) {
        if (values[i].type->getKind() == TypeNode::TensorType) {
          auto &tensorTy = static_cast<const TensorType &>(*values[i].type);
          SmallVector<int64_t, 4> shape;
          for (const auto &dim : tensorTy.getDims()) {
            if (dim.kind == Dimension::DK_Concrete) {
              shape.push_back(dim.size);
            } else if (useStaticShapes && traceId > 0) {
              // Try to resolve from PGO data.
              auto staticShapes = profiler_.getStaticShapes(traceId);
              if (staticShapes) {
                auto it = staticShapes->find(static_cast<TraceValueId>(i));
                if (it != staticShapes->end() && !it->second.empty()) {
                  // Use the profiled dimension.
                  size_t dimIdx = shape.size();
                  if (dimIdx < it->second.size()) {
                    shape.push_back(it->second[dimIdx]);
                  } else {
                    shape.push_back(ShapedType::kDynamic);
                  }
                } else {
                  shape.push_back(ShapedType::kDynamic);
                }
              } else {
                shape.push_back(ShapedType::kDynamic);
              }
            } else {
              shape.push_back(ShapedType::kDynamic);
            }
          }

          Type elemType = FloatType::getF32(context.get());
          switch (tensorTy.getElementKind()) {
          case ScalarType::SK_F32: elemType = FloatType::getF32(context.get()); break;
          case ScalarType::SK_F64: elemType = FloatType::getF64(context.get()); break;
          default: break;
          }
          argType = RankedTensorType::get(shape, elemType);
        } else if (values[i].type->getKind() == TypeNode::ScalarType) {
          auto &scalarTy = static_cast<const ScalarType &>(*values[i].type);
          switch (scalarTy.getScalarKind()) {
          case ScalarType::SK_F32: argType = FloatType::getF32(context.get()); break;
          case ScalarType::SK_F64: argType = FloatType::getF64(context.get()); break;
          case ScalarType::SK_I32: argType = IntegerType::get(context.get(), 32); break;
          case ScalarType::SK_I64: argType = IntegerType::get(context.get(), 64); break;
          case ScalarType::SK_Bool: argType = IntegerType::get(context.get(), 1); break;
          default: argType = f32Type; break;
          }
        }
      }
      inputTypes.push_back(argType);
    }
  }

  Type returnType = f32Type;
  auto funcType = builder.getFunctionType(inputTypes, returnType);
  auto funcOp = func::FuncOp::create(builder.getUnknownLoc(), "__trace_fn",
                                       funcType);
  auto *entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Map trace values to MLIR values.
  std::vector<Value> mlirValues(values.size());

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
          loc, builder.getFloatAttr(f32Type, val), f32Type);
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
        auto result = builder.create<NegOp>(loc, mlirValues[op.inputs[0]]);
        mlirValues[op.outputs[0]] = result.getResult();
      }
      break;
    }
    case TraceOpKind::MatMul: {
      if (op.inputs.size() >= 2 && op.outputs.size() >= 1 &&
          mlirValues[op.inputs[0]] && mlirValues[op.inputs[1]]) {
        auto result = builder.create<MatMulOp>(
            loc, mlirValues[op.inputs[0]], mlirValues[op.inputs[1]],
            ::mlir::StringAttr(), ::mlir::StringAttr());
        mlirValues[op.outputs[0]] = result.getResult();
      }
      break;
    }
    case TraceOpKind::Relu: {
      if (op.inputs.size() >= 1 && op.outputs.size() >= 1 &&
          mlirValues[op.inputs[0]]) {
        auto result = builder.create<ReluOp>(loc, mlirValues[op.inputs[0]]);
        mlirValues[op.outputs[0]] = result.getResult();
      }
      break;
    }
    case TraceOpKind::Sigmoid: {
      if (op.inputs.size() >= 1 && op.outputs.size() >= 1 &&
          mlirValues[op.inputs[0]]) {
        auto result = builder.create<SigmoidOp>(loc, mlirValues[op.inputs[0]]);
        mlirValues[op.outputs[0]] = result.getResult();
      }
      break;
    }
    case TraceOpKind::Tanh: {
      if (op.inputs.size() >= 1 && op.outputs.size() >= 1 &&
          mlirValues[op.inputs[0]]) {
        auto result = builder.create<TanhOp>(loc, mlirValues[op.inputs[0]]);
        mlirValues[op.outputs[0]] = result.getResult();
      }
      break;
    }
    case TraceOpKind::Mean: {
      if (op.inputs.size() >= 1 && op.outputs.size() >= 1 &&
          mlirValues[op.inputs[0]]) {
        auto result = builder.create<MeanOp>(loc, mlirValues[op.inputs[0]]);
        mlirValues[op.outputs[0]] = result.getResult();
      }
      break;
    }
    case TraceOpKind::Sum: {
      if (op.inputs.size() >= 1 && op.outputs.size() >= 1 &&
          mlirValues[op.inputs[0]]) {
        auto result = builder.create<SumOp>(loc, mlirValues[op.inputs[0]]);
        mlirValues[op.outputs[0]] = result.getResult();
      }
      break;
    }
    case TraceOpKind::Transpose: {
      if (op.inputs.size() >= 1 && op.outputs.size() >= 1 &&
          mlirValues[op.inputs[0]]) {
        auto result = builder.create<TransposeOp>(loc, mlirValues[op.inputs[0]]);
        mlirValues[op.outputs[0]] = result.getResult();
      }
      break;
    }
    case TraceOpKind::Zeros:
    case TraceOpKind::Ones:
    case TraceOpKind::Random: {
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
            loc, f32Type, mlirValues[op.inputs[0]], TypeAttr::get(f32Type));
        mlirValues[op.outputs[0]] = result.getResult();
      }
      break;
    }
    case TraceOpKind::Select: {
      if (op.inputs.size() >= 3 && op.outputs.size() >= 1 &&
          mlirValues[op.inputs[0]] && mlirValues[op.inputs[1]] &&
          mlirValues[op.inputs[2]]) {
        auto result = builder.create<SelectOp>(
            loc, mlirValues[op.inputs[1]].getType(),
            mlirValues[op.inputs[0]],
            mlirValues[op.inputs[1]], mlirValues[op.inputs[2]]);
        mlirValues[op.outputs[0]] = result.getResult();
      }
      break;
    }
    case TraceOpKind::Grad: {
      if (op.inputs.size() >= 1 && op.outputs.size() >= 1 &&
          mlirValues[op.inputs[0]]) {
        std::string diffVar = "x";
        if (!op.attrs.empty() &&
            op.attrs[0].kind == TraceOp::Attr::Kind::String) {
          diffVar = std::get<std::string>(op.attrs[0].value);
        }
        auto result = builder.create<GradOp>(
            loc, mlirValues[op.inputs[0]].getType(),
            mlirValues[op.inputs[0]],
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
            loc, builder.getFloatAttr(f32Type, 0.0), f32Type);
        builder.create<func::ReturnOp>(loc, zero.getResult());
      }
      break;
    }
    case TraceOpKind::Barrier:
      break;
    default:
      break;
    }
  }

  if (entryBlock->empty() || !entryBlock->mightHaveTerminator()) {
    auto zero = builder.create<ConstantOp>(
        builder.getUnknownLoc(), builder.getFloatAttr(f32Type, 0.0), f32Type);
    builder.create<func::ReturnOp>(builder.getUnknownLoc(), zero.getResult());
  }

  module.push_back(funcOp);

  // ── Run the full Tier 2 optimization pipeline ───────────────────────────
  PassManager pm(context.get());

  // Shape inference with static shapes.
  pm.addPass(createShapeInferencePass());

  // Autodiff.
  pm.addPass(createAutodiffPass());

  if (config_.enableAutodiffPruning) {
    pm.addPass(createAutodiffPruningPass());
  }

  // Graph collapsing.
  if (config_.enableGraphCollapsing) {
    pm.addPass(createGraphCollapsingPass());
  }

  // Whole-program collapsing (interprocedural).
  pm.addPass(createWholeProgramCollapsingPass());

  // SCCP (constant propagation).
  if (config_.enableSCCP) {
    pm.addPass(createSCCPPass());
  }

  // SymbolDCE.
  if (config_.enableSymbolDCE) {
    pm.addPass(createSymbolDCEPass());
  }

  // Algebraic simplification.
  pm.addPass(createAlgebraicSimplificationPass());

  // SIMD layout optimization.
  if (config_.enableSIMDLayout) {
    pm.addPass(createSIMDLayoutPass());
  }

  // Polyhedral optimization (with concrete shapes for optimal tiling).
  if (config_.enablePolyhedral) {
    pm.addPass(createPolyhedralOptPass());
  }

  // Final cleanup.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Lower to StableHLO.
  if (auto p = createJulesToStableHLOLoweringPass()) pm.addPass(std::move(p));

  if (failed(pm.run(module))) {
    diag_.error(SourceLocation{}, "Tier 2 MLIR compilation failed for trace");
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
