//===- KernelRoutingPass.h - Route fused kernels to MLIR ops -----*- C++ -*-===//
//
// Part of the Jules Project, under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//
//
// This pass bridges the kernel layer (jules/Kernel/) with the MLIR compiler
// pipeline. It recognizes operation patterns that match fused kernels and
// replaces them with jules.extern_kernel calls that route to the actual
// fused kernel implementations at runtime.
//
// Pattern recognition:
//   jules.matmul + jules.relu           → extern_kernel "fusedMatmulRelu"
//   jules.matmul + add(bias) + jules.relu → extern_kernel "fusedMatmulBiasRelu"
//   jules.matmul + jules.sigmoid        → extern_kernel "fusedMatmulSigmoid"
//   jules.matmul + jules.tanh           → extern_kernel "fusedMatmulTanh"
//   jules.matmul + jules.neg (GELU)     → extern_kernel "fusedMatmulGelu"
//   softmax pattern                      → extern_kernel "fusedSoftmax"
//   layernorm pattern                    → extern_kernel "fusedLayerNorm"
//   MLP forward pattern                  → extern_kernel "fusedMLPForward"
//   MLP forward+backward pattern         → extern_kernel "interleavedMLPForwardBackward"
//   attention pattern                    → extern_kernel "flashAttention"
//   int8 matmul pattern                  → extern_kernel "matmulInt8"
//
// The pass runs BEFORE the ProducerConsumerFusionPass, so that fused kernel
// calls are already in place when fusion considers the remaining ops.
//
//===----------------------------------------------------------------------===//

#ifndef JULES_PASSES_KERNEL_ROUTING_PASS_H
#define JULES_PASSES_KERNEL_ROUTING_PASS_H

#include <memory>

namespace mlir {
class Pass;
}

namespace jules {

/// Create the Kernel Routing pass.
/// This pass recognizes operation patterns in the Jules MLIR dialect
/// and routes them to the appropriate fused kernel implementations
/// via jules.extern_kernel ops.
std::unique_ptr<mlir::Pass> createKernelRoutingPass();

} // namespace jules

#endif // JULES_PASSES_KERNEL_ROUTING_PASS_H
