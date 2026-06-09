//===- Passes.h - Jules MLIR Passes ----------------------------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MLIR transformation passes for the Jules compiler:
//
//   Core passes:
//     - AutodiffPass: Reverse-mode automatic differentiation at the MLIR level
//     - JulesToStableHLOLoweringPass: Lower Jules dialect to StableHLO
//     - ShapeInferencePass: Resolve symbolic dimensions
//
//   Optimization passes:
//     - GraphCollapsingPass: Aggressive structural pruning
//     - AlgebraicSimplificationPass: Mathematical identity simplification
//     - AutodiffPruningPass: Clean up after autodiff
//     - PGOPass: Profile-guided shape specialization
//
//   Two-Tier AOT/JIT passes:
//     - WholeProgramCollapsingPass: Interprocedural constant folding
//     - SCCPPass: Sparse Conditional Constant Propagation
//     - SymbolDCEPass: Symbol Dead Code Elimination
//     - SIMDLayoutPass: SIMD vectorization layout optimization
//     - PolyhedralOptPass: Affine/polyhedral loop optimization
//     - ProducerConsumerFusionPass: Kernel fusion for elementwise chains,
//       matmul+activation, and elementwise+reduction
//     - MemoryPlanningPass: Buffer lifetime analysis and in-place reuse
//     - MixedPrecisionPass: Mixed precision (bf16/fp8) optimization
//
//===----------------------------------------------------------------------===//

#ifndef JULES_PASSES_PASSES_H
#define JULES_PASSES_PASSES_H

#include <cstdint>
#include <memory>

namespace mlir {
class Pass;
class Operation;
} // namespace mlir

namespace jules {

class Profiler;

// ── Core Passes ─────────────────────────────────────────────────────────────

/// Create the Autodiff pass.
std::unique_ptr<mlir::Pass> createAutodiffPass();

/// Create the shape-inference pass.
std::unique_ptr<mlir::Pass> createShapeInferencePass();

/// Create the Jules-to-StableHLO lowering pass.
std::unique_ptr<mlir::Pass> createJulesToStableHLOLoweringPass();

// ── Optimization Passes ─────────────────────────────────────────────────────

/// Create the graph collapsing pass.
std::unique_ptr<mlir::Pass> createGraphCollapsingPass();

/// Create the algebraic simplification pass.
std::unique_ptr<mlir::Pass> createAlgebraicSimplificationPass();

/// Create the autodiff pruning pass.
std::unique_ptr<mlir::Pass> createAutodiffPruningPass();

/// Create the fused autodiff pass (autodiff + pruning + collapsing in one).
std::unique_ptr<mlir::Pass> createFusedAutodiffPass();

/// Create the PGO-informed optimization pass.
std::unique_ptr<mlir::Pass> createPGOPass(const Profiler &profiler);

// ── Two-Tier AOT/JIT Passes ─────────────────────────────────────────────────

/// Create the whole-program graph collapsing pass.
/// Performs interprocedural constant folding and partial evaluation.
std::unique_ptr<mlir::Pass> createWholeProgramCollapsingPass();

/// Create the Sparse Conditional Constant Propagation pass.
/// Discovers and propagates constant values throughout the module.
std::unique_ptr<mlir::Pass> createSCCPPass();

/// Create the Symbol Dead Code Elimination pass.
/// Removes unreachable functions and unused operations.
std::unique_ptr<mlir::Pass> createSymbolDCEPass();

/// Create the SIMD vectorization layout optimization pass.
/// Structures MLIR code for optimal XLA SIMD generation.
std::unique_ptr<mlir::Pass> createSIMDLayoutPass();

/// Create the polyhedral (affine) optimization pass.
/// Applies loop tiling, fusion, and skewing for cache locality.
std::unique_ptr<mlir::Pass> createPolyhedralOptPass();

/// Create the polyhedral optimization pass with a specific cache size.
std::unique_ptr<mlir::Pass> createPolyhedralOptPass(uint64_t cacheSizeBytes);

/// Create the Producer-Consumer Fusion pass.
/// Fuses elementwise chains, matmul+activation, and elementwise+reduction
/// into fusion clusters for reduced kernel launch overhead.
std::unique_ptr<mlir::Pass> createProducerConsumerFusionPass();

/// Create the Memory Planning pass.
/// Analyzes buffer lifetimes and annotates operations with in-place reuse
/// hints, alignment, and alias information.
std::unique_ptr<mlir::Pass> createMemoryPlanningPass();

/// Create the Memory Planning pass with a specific memory budget limit.
std::unique_ptr<mlir::Pass> createMemoryPlanningPass(uint64_t memoryBudgetBytes);

/// Create the mixed precision optimization pass with default settings
/// (target precision = bf16, min op count = 5).
std::unique_ptr<mlir::Pass> createMixedPrecisionPass();

/// Create the mixed precision optimization pass with a specific target
/// precision ("bf16" or "fp8").
std::unique_ptr<mlir::Pass> createMixedPrecisionPass(std::string targetPrecision);

// ── Shape Polymorphism ──────────────────────────────────────────────────────

/// Create the Shape Polymorphism pass.
/// Carries symbolic dimension constraints through the pipeline instead of
/// lowering everything to ShapedType::kDynamic, enabling XLA to specialize.
std::unique_ptr<mlir::Pass> createShapePolymorphismPass();

// ── Quantization ────────────────────────────────────────────────────────────

/// Create the Quantization pass with default options.
/// Inserts fake-quantization nodes for inference optimization.
std::unique_ptr<mlir::Pass> createQuantizePass();

/// Create the Quantization pass with explicit options.
std::unique_ptr<mlir::Pass> createQuantizePass(
    bool quantizeWeights,
    bool quantizeActivations,
    unsigned numBits,
    bool perChannel);

// ── Kernel Routing ──────────────────────────────────────────────────────────

/// Create the Kernel Routing pass.
/// Recognizes operation patterns (matmul+relu, softmax, layernorm, etc.)
/// and routes them to fused kernel implementations via extern_kernel ops.
/// This is THE bridge between the kernel layer and the MLIR compiler.
std::unique_ptr<mlir::Pass> createKernelRoutingPass();

// ── Registration ────────────────────────────────────────────────────────────

/// Register all Jules passes with the MLIR pass pipeline.
void registerJulesPasses();

} // namespace jules

#endif // JULES_PASSES_PASSES_H
