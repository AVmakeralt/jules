//===- JulesOps.td - Jules Dialect Operation Definitions --------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Jules dialect operations using MLIR's TableGen
// declarative specification format. The C++ op classes are auto-generated
// from these definitions.
//
//===----------------------------------------------------------------------===//

#ifndef JULES_OPS
#define JULES_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def Jules_Dialect : Dialect {
  let name = "jules";
  let summary = "Jules AI language dialect";
  let description = [{
    The Jules dialect provides operations for tensor computation, automatic
    differentiation, and neural network primitives. It maps directly to the
    Jules functional programming language and serves as the IR before lowering
    to StableHLO for XLA compilation.
  }];
  let cppNamespace = "::jules";
  let hasConstantMaterializer = 1;
}

//===----------------------------------------------------------------------===//
// Base class for Jules operations
//===----------------------------------------------------------------------===//

class JulesOp<string mnemonic, list<Trait> traits = []> :
    Op<Jules_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// Constant
//===----------------------------------------------------------------------===//

def ConstantOp : JulesOp<"constant", [NoSideEffect, ConstantLike]> {
  let summary = "Create a constant value";
  let description = [{
    Creates a constant scalar or tensor value. The value is specified as a
    FloatAttr, IntegerAttr, or DenseElementsAttr.
  }];
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let builders = [
    OpBuilder<(ins "double":$value, "Type":$type)>
  ];
}

//===----------------------------------------------------------------------===//
// Element-wise binary operations
//===----------------------------------------------------------------------===//

def AddOp : JulesOp<"add", [NoSideEffect, Commutative, SameOperandsAndResultType]> {
  let summary = "Element-wise addition";
  let description = [{
    Computes element-wise addition of two tensors or scalars.
    Supports broadcasting.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
}

def SubOp : JulesOp<"sub", [NoSideEffect, SameOperandsAndResultType, InferTypeOpInterface]> {
  let summary = "Element-wise subtraction";
  let description = [{
    Computes element-wise subtraction of two tensors or scalars.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
}

def MulOp : JulesOp<"mul", [NoSideEffect, Commutative, InferTypeOpInterface]> {
  let summary = "Element-wise multiplication";
  let description = [{
    Computes element-wise multiplication of two tensors or scalars.
    Supports broadcasting.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
  let extraClassDeclaration = [{
    static ::llvm::LogicalResult inferReturnTypes(
        ::mlir::MLIRContext *context,
        ::llvm::Optional<::mlir::Location> location,
        ::mlir::ValueRange operands,
        ::mlir::DictionaryAttr attributes,
        ::mlir::RegionRange regions,
        ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes);
  }];
}

def DivOp : JulesOp<"div", [NoSideEffect]> {
  let summary = "Element-wise division";
  let description = [{
    Computes element-wise division of two tensors or scalars.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def PowOp : JulesOp<"pow", [NoSideEffect]> {
  let summary = "Element-wise power";
  let description = [{
    Computes element-wise power operation: lhs ^ rhs.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def NegOp : JulesOp<"neg", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Negation";
  let description = [{
    Computes the negation of a tensor or scalar.
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Matrix operations
//===----------------------------------------------------------------------===//

def MatMulOp : JulesOp<"matmul", [NoSideEffect, InferTypeOpInterface]> {
  let summary = "Matrix multiplication";
  let description = [{
    Computes matrix multiplication. Supports 2D and batched variants:
      [M, K] ** [K, N] -> [M, N]
      [B, M, K] ** [K, N] -> [B, M, N]
      [B, M, K] ** [B, K, N] -> [B, M, N]
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs,
                       OptionalAttr<StrAttr>:$compute_precision,
                       OptionalAttr<StrAttr>:$accumulation_precision);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let extraClassDeclaration = [{
    static ::llvm::LogicalResult inferReturnTypes(
        ::mlir::MLIRContext *context,
        ::llvm::Optional<::mlir::Location> location,
        ::mlir::ValueRange operands,
        ::mlir::DictionaryAttr attributes,
        ::mlir::RegionRange regions,
        ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes);
  }];
}

//===----------------------------------------------------------------------===//
// Activation functions
//===----------------------------------------------------------------------===//

def ReluOp : JulesOp<"relu", [NoSideEffect, SameOperandsAndResultType, InferTypeOpInterface]> {
  let summary = "ReLU activation";
  let description = [{
    Computes the Rectified Linear Unit activation: max(0, x).
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
}

def SigmoidOp : JulesOp<"sigmoid", [NoSideEffect, SameOperandsAndResultType, InferTypeOpInterface]> {
  let summary = "Sigmoid activation";
  let description = [{
    Computes the sigmoid activation: 1 / (1 + exp(-x)).
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
}

def TanhOp : JulesOp<"tanh", [NoSideEffect, SameOperandsAndResultType, InferTypeOpInterface]> {
  let summary = "Tanh activation";
  let description = [{
    Computes the hyperbolic tangent activation.
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
}

def GeluOp : JulesOp<"gelu", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "GELU activation";
  let description = [{
    Computes the Gaussian Error Linear Unit activation:
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
}

def ExpOp : JulesOp<"exp", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Element-wise exponential";
  let description = [{
    Computes the element-wise exponential function: exp(x).
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Reduction operations
//===----------------------------------------------------------------------===//

def MeanOp : JulesOp<"mean", [NoSideEffect, InferTypeOpInterface]> {
  let summary = "Mean reduction";
  let description = [{
    Computes the mean of all elements in a tensor, producing a scalar tensor.
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let extraClassDeclaration = [{
    static ::llvm::LogicalResult inferReturnTypes(
        ::mlir::MLIRContext *context,
        ::llvm::Optional<::mlir::Location> location,
        ::mlir::ValueRange operands,
        ::mlir::DictionaryAttr attributes,
        ::mlir::RegionRange regions,
        ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes);
  }];
}

def SumOp : JulesOp<"sum", [NoSideEffect, InferTypeOpInterface]> {
  let summary = "Sum reduction";
  let description = [{
    Computes the sum of all elements in a tensor, producing a scalar tensor.
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let extraClassDeclaration = [{
    static ::llvm::LogicalResult inferReturnTypes(
        ::mlir::MLIRContext *context,
        ::llvm::Optional<::mlir::Location> location,
        ::mlir::ValueRange operands,
        ::mlir::DictionaryAttr attributes,
        ::mlir::RegionRange regions,
        ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes);
  }];
}

//===----------------------------------------------------------------------===//
// Tensor creation
//===----------------------------------------------------------------------===//

def ZerosOp : JulesOp<"zeros", [NoSideEffect]> {
  let summary = "Zero-filled tensor";
  let description = [{
    Creates a tensor filled with zeros of the specified shape and element type.
  }];
  let arguments = (ins I64ArrayAttr:$shape);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$shape attr-dict `:` type($result)";
  let builders = [
    OpBuilder<(ins "RankedTensorType":$resultType)>
  ];
}

def OnesOp : JulesOp<"ones", [NoSideEffect]> {
  let summary = "One-filled tensor";
  let description = [{
    Creates a tensor filled with ones of the specified shape and element type.
  }];
  let arguments = (ins I64ArrayAttr:$shape);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$shape attr-dict `:` type($result)";
  let builders = [
    OpBuilder<(ins "RankedTensorType":$resultType)>
  ];
}

def RandomOp : JulesOp<"random"> {
  let summary = "Random tensor";
  let description = [{
    Creates a tensor filled with random values of the specified shape
    and element type.
  }];
  let arguments = (ins I64ArrayAttr:$shape);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$shape attr-dict `:` type($result)";
  let builders = [
    OpBuilder<(ins "RankedTensorType":$resultType)>
  ];
}

//===----------------------------------------------------------------------===//
// Type conversion
//===----------------------------------------------------------------------===//

def CastOp : JulesOp<"cast", [NoSideEffect]> {
  let summary = "Type cast";
  let description = [{
    Casts a tensor or scalar from one type to another.
  }];
  let arguments = (ins AnyType:$input, TypeAttr:$target_type);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input `to` $target_type attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Autodiff
//===----------------------------------------------------------------------===//

def GradOp : JulesOp<"grad"> {
  let summary = "Automatic differentiation";
  let description = [{
    Computes the gradient of a function with respect to a specified variable.
    This operation is replaced by the autodiff pass with the actual forward
    + backward computation graph.
  }];
  let arguments = (ins AnyType:$fn, StrAttr:$diff_var);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$fn `,` $diff_var attr-dict `:` type($fn) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Tensor manipulation
//===----------------------------------------------------------------------===//

def TransposeOp : JulesOp<"transpose", [NoSideEffect, InferTypeOpInterface]> {
  let summary = "Tensor transpose";
  let description = [{
    Transposes a 2D tensor by swapping its dimensions.
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let extraClassDeclaration = [{
    static ::llvm::LogicalResult inferReturnTypes(
        ::mlir::MLIRContext *context,
        ::llvm::Optional<::mlir::Location> location,
        ::mlir::ValueRange operands,
        ::mlir::DictionaryAttr attributes,
        ::mlir::RegionRange regions,
        ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes);
  }];
}

def ReshapeOp : JulesOp<"reshape", [NoSideEffect]> {
  let summary = "Tensor reshape";
  let description = [{
    Reshapes a tensor to a new shape with the same total number of elements.
  }];
  let arguments = (ins AnyType:$input, I64ArrayAttr:$shape);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input `to` $shape attr-dict `:` type($input) `->` type($result)";
  let builders = [
    OpBuilder<(ins "Value":$input, "RankedTensorType":$resultType)>
  ];
}

def ConcatOp : JulesOp<"concat", [NoSideEffect]> {
  let summary = "Tensor concatenation";
  let description = [{
    Concatenates tensors along a specified axis.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs, I64Attr:$axis);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$inputs `,` $axis attr-dict `:` type($inputs) `->` type($result)";
}

def SliceOp : JulesOp<"slice", [NoSideEffect]> {
  let summary = "Tensor slice";
  let description = [{
    Extracts a slice from a tensor along each dimension.
  }];
  let arguments = (ins AnyType:$input,
                       I64ArrayAttr:$start_indices,
                       I64ArrayAttr:$limit_indices,
                       I64ArrayAttr:$strides);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input `[` $start_indices `:` $limit_indices `:` $strides `]` attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Conditional
//===----------------------------------------------------------------------===//

def SelectOp : JulesOp<"select", [NoSideEffect]> {
  let summary = "Conditional select";
  let description = [{
    Selects between two values based on a condition (element-wise ternary).
  }];
  let arguments = (ins AnyType:$condition, AnyType:$true_value, AnyType:$false_value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$condition `,` $true_value `,` $false_value attr-dict `:` type($condition) `,` type($true_value) `,` type($false_value) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Comparison
//===----------------------------------------------------------------------===//

def CmpOp : JulesOp<"cmp", [NoSideEffect]> {
  let summary = "Comparison operation";
  let description = [{
    Compares two tensors or scalars element-wise.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, StrAttr:$comparison_direction);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs `,` $comparison_direction attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Element-wise unary math
//===----------------------------------------------------------------------===//

def LogOp : JulesOp<"log", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Element-wise natural logarithm";
  let description = [{
    Computes the element-wise natural logarithm of a tensor or scalar.
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Tensor padding
//===----------------------------------------------------------------------===//

def PadOp : JulesOp<"pad", [NoSideEffect]> {
  let summary = "Tensor padding";
  let description = [{
    Pads a tensor with edge values or a constant along each dimension.
    The padding_low and padding_high specify the number of elements to
    pad on the low and high side of each dimension. The interior_padding
    specifies the number of elements to insert between existing elements.
  }];
  let arguments = (ins AnyType:$input,
                       AnyType:$padding_value,
                       I64ArrayAttr:$padding_low,
                       I64ArrayAttr:$padding_high,
                       I64ArrayAttr:$interior_padding);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input `,` $padding_value `low` $padding_low `high` $padding_high `interior` $interior_padding attr-dict `:` type($input) `,` type($padding_value) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Broadcasting
//===----------------------------------------------------------------------===//

def BroadcastInDimOp : JulesOp<"broadcast_in_dim", [NoSideEffect]> {
  let summary = "Broadcast tensor with explicit dimension mapping";
  let description = [{
    Broadcasts a tensor to a target shape using the given broadcast_dimensions.
    The broadcast_dimensions array maps each dimension of the operand to a
    dimension of the result. This is the same semantics as StableHLO's
    broadcast_in_dim.
  }];
  let arguments = (ins AnyType:$input, I64ArrayAttr:$broadcast_dimensions);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input `,` $broadcast_dimensions attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Reduction with body
//===----------------------------------------------------------------------===//

def ReduceOp : JulesOp<"reduce", [NoSideEffect]> {
  let summary = "Reduction with a body region";
  let description = [{
    Reduces a tensor along the given dimensions using a reduction body region.
    The body region takes two arguments (accumulator, element) and returns
    the updated accumulator.
  }];
  let arguments = (ins AnyType:$input, AnyType:$init_value, I64ArrayAttr:$dimensions);
  let results = (outs AnyType:$result);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$input `,` $init_value `dims` $dimensions attr-dict `:` type($input) `,` type($init_value) `->` type($result) $body";
}

//===----------------------------------------------------------------------===//
// Control flow
//===----------------------------------------------------------------------===//

def WhileOp : JulesOp<"while"> {
  let summary = "While loop with carried variables";
  let description = [{
    Executes a while loop: while(condition(carried_vars)) { body(carried_vars) }
    The condition and body are regions. Carried variables are passed through
    both regions and updated each iteration.
  }];
  let arguments = (ins Variadic<AnyType>:$carried_vars);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$cond, AnyRegion:$body);
  let assemblyFormat = [{
    `(` $carried_vars `:` type($carried_vars) `)` attr-dict
    `cond` $cond `do` $body `:` type($results)
  }];
}

//===----------------------------------------------------------------------===//
// Parallel execution
//===----------------------------------------------------------------------===//

def ParallelOp : JulesOp<"parallel"> {
  let summary = "Data-parallel execution primitive";
  let description = [{
    Executes a computation in parallel over a range of indices.
    Maps to XLA's SPMD partitioner or shardy dialect.
    Each index is independent — no cross-iteration dependencies.
  }];
  let arguments = (ins I64Attr:$lowerBound, I64Attr:$upperBound, I64Attr:$step);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    $lowerBound `to` $upperBound `step` $step attr-dict `:` type($results) $body
  }];
}

//===----------------------------------------------------------------------===//
// Fake quantization (quantization-aware training)
//===----------------------------------------------------------------------===//

def FakeQuantOp : JulesOp<"fake_quant", [NoSideEffect]> {
  let summary = "Fake quantization for quantization-aware training";
  let description = [{
    Simulates the effect of quantization by quantizing to a lower-precision
    integer type and dequantizing back. This allows the model to learn weights
    that are robust to quantization noise during training.

    Compute: round(input / scale + zero_point) * scale - zero_point * scale

    In inference mode, this op is replaced with actual int8 CastOp + MatMulInt8.
  }];
  let arguments = (ins AnyType:$input,
                       F64Attr:$scale,
                       I64Attr:$zero_point,
                       I32Attr:$num_bits);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input `,` $scale `,` $zero_point `,` $num_bits attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Custom kernel injection
//===----------------------------------------------------------------------===//

def ExternKernelOp : JulesOp<"extern_kernel", [NoSideEffect]> {
  let summary = "External kernel call";
  let description = [{
    Calls an external kernel function by name, bypassing the StableHLO path.
    The kernel must be available as a shared library symbol or PJRT plugin.
    Used for flash attention, sparse ops, custom reductions, etc.
  }];
  let arguments = (ins StrAttr:$kernel_name, Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $kernel_name `(` $inputs `:` type($inputs) `)` attr-dict `:` type($results)
  }];
}

//===----------------------------------------------------------------------===//
// Convolution
//===----------------------------------------------------------------------===//

def Conv2DOp : JulesOp<"conv2d", [NoSideEffect]> {
  let summary = "2D convolution";
  let description = [{
    Computes a 2D convolution over an input tensor with a kernel tensor.
    Supports standard convolution with padding, stride, and dilation.
      Input:  [batch, in_channels, height, width]
      Kernel: [out_channels, in_channels, kH, kW]
      Output: [batch, out_channels, outH, outW]
  }];
  let arguments = (ins AnyType:$input, AnyType:$kernel,
                       I64ArrayAttr:$padding,
                       I64ArrayAttr:$strides,
                       I64ArrayAttr:$dilation);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $input `,` $kernel `padding` $padding `strides` $strides `dilation` $dilation
    attr-dict `:` type($input) `,` type($kernel) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Distributed collective operations
//===----------------------------------------------------------------------===//

def AllReduceOp : JulesOp<"all_reduce", [NoSideEffect]> {
  let summary = "All-reduce collective operation";
  let description = [{
    Performs an all-reduce operation across all devices in a device group.
    Combines values from all devices using the specified reduction operation
    (sum, max, min, prod) and returns the result to all devices.
    This is the fundamental primitive for distributed training (e.g.,
    gradient synchronization in data-parallel SGD).
  }];
  let arguments = (ins AnyType:$input, StrAttr:$reduction);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input `,` $reduction attr-dict `:` type($input) `->` type($result)";
}

def AllGatherOp : JulesOp<"all_gather", [NoSideEffect]> {
  let summary = "All-gather collective operation";
  let description = [{
    Gathers values from all devices along a specified dimension and
    returns the concatenated result to all devices. Used for tensor
    parallelism where each device holds a shard of a tensor.
  }];
  let arguments = (ins AnyType:$input, I64Attr:$gather_dim);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input `,` $gather_dim attr-dict `:` type($input) `->` type($result)";
}

#endif // JULES_OPS
