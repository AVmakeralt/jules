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
  let parser = [{ return parseJulesOp(parser, result); }];
  let printer = [{ printJulesOp(p, *this); }];
}

//===----------------------------------------------------------------------===//
// Constant
//===----------------------------------------------------------------------===//

def ConstantOp : JulesOp<"constant", [Pure, ConstantLike]> {
  let summary = "Create a constant value";
  let description = [{
    Creates a constant scalar or tensor value. The value is specified as a
    FloatAttr, IntegerAttr, or DenseElementsAttr.
  }];
  let arguments = (ins AttrAttr:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
  let hasFolder = 1;
  let builders = [
    OpBuilder<(ins "double":$value, "Type":$type)>
  ];
}

//===----------------------------------------------------------------------===//
// Element-wise binary operations
//===----------------------------------------------------------------------===//

def AddOp : JulesOp<"add", [Pure, Commutative, SameOperandsAndResultType]> {
  let summary = "Element-wise addition";
  let description = [{
    Computes element-wise addition of two tensors or scalars.
    Supports broadcasting.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let hasFolder = 1;
}

def SubOp : JulesOp<"sub", [Pure, SameOperandsAndResultType]> {
  let summary = "Element-wise subtraction";
  let description = [{
    Computes element-wise subtraction of two tensors or scalars.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let hasFolder = 1;
}

def MulOp : JulesOp<"mul", [Pure, Commutative]> {
  let summary = "Element-wise multiplication";
  let description = [{
    Computes element-wise multiplication of two tensors or scalars.
    Supports broadcasting.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let hasFolder = 1;
}

def DivOp : JulesOp<"div", [Pure]> {
  let summary = "Element-wise division";
  let description = [{
    Computes element-wise division of two tensors or scalars.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def PowOp : JulesOp<"pow", [Pure]> {
  let summary = "Element-wise power";
  let description = [{
    Computes element-wise power operation: lhs ^ rhs.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def NegOp : JulesOp<"neg", [Pure, SameOperandsAndResultType]> {
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

def MatMulOp : JulesOp<"matmul", [Pure]> {
  let summary = "Matrix multiplication";
  let description = [{
    Computes matrix multiplication. Supports 2D and batched variants:
      [M, K] ** [K, N] -> [M, N]
      [B, M, K] ** [K, N] -> [B, M, N]
      [B, M, K] ** [B, K, N] -> [B, M, N]
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Activation functions
//===----------------------------------------------------------------------===//

def ReluOp : JulesOp<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "ReLU activation";
  let description = [{
    Computes the Rectified Linear Unit activation: max(0, x).
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
}

def SigmoidOp : JulesOp<"sigmoid", [Pure, SameOperandsAndResultType]> {
  let summary = "Sigmoid activation";
  let description = [{
    Computes the sigmoid activation: 1 / (1 + exp(-x)).
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
}

def TanhOp : JulesOp<"tanh", [Pure, SameOperandsAndResultType]> {
  let summary = "Tanh activation";
  let description = [{
    Computes the hyperbolic tangent activation.
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Reduction operations
//===----------------------------------------------------------------------===//

def MeanOp : JulesOp<"mean", [Pure]> {
  let summary = "Mean reduction";
  let description = [{
    Computes the mean of all elements in a tensor, producing a scalar tensor.
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def SumOp : JulesOp<"sum", [Pure]> {
  let summary = "Sum reduction";
  let description = [{
    Computes the sum of all elements in a tensor, producing a scalar tensor.
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Tensor creation
//===----------------------------------------------------------------------===//

def ZerosOp : JulesOp<"zeros", [Pure]> {
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

def OnesOp : JulesOp<"ones", [Pure]> {
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

def CastOp : JulesOp<"cast", [Pure]> {
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
  let assemblyFormat = "$fn `,` $diff_var attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Tensor manipulation
//===----------------------------------------------------------------------===//

def TransposeOp : JulesOp<"transpose", [Pure]> {
  let summary = "Tensor transpose";
  let description = [{
    Transposes a 2D tensor by swapping its dimensions.
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def ReshapeOp : JulesOp<"reshape", [Pure]> {
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

def ConcatOp : JulesOp<"concat", [Pure]> {
  let summary = "Tensor concatenation";
  let description = [{
    Concatenates tensors along a specified axis.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs, I64Attr:$axis);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$inputs `,` $axis attr-dict `:` type($result)";
}

def SliceOp : JulesOp<"slice", [Pure]> {
  let summary = "Tensor slice";
  let description = [{
    Extracts a slice from a tensor along each dimension.
  }];
  let arguments = (ins AnyType:$input,
                       I64ArrayAttr:$start_indices,
                       I64ArrayAttr:$limit_indices,
                       I64ArrayAttr:$strides);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input `[` $start_indices `:` $limit_indices `:` $strides `]` attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Conditional
//===----------------------------------------------------------------------===//

def SelectOp : JulesOp<"select", [Pure]> {
  let summary = "Conditional select";
  let description = [{
    Selects between two values based on a condition (element-wise ternary).
  }];
  let arguments = (ins AnyType:$condition, AnyType:$true_value, AnyType:$false_value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$condition `,` $true_value `,` $false_value attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Comparison
//===----------------------------------------------------------------------===//

def CmpOp : JulesOp<"cmp", [Pure]> {
  let summary = "Comparison operation";
  let description = [{
    Compares two tensors or scalars element-wise.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, StrAttr:$comparison_direction);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs `,` $comparison_direction attr-dict `:` type($result)";
}

#endif // JULES_OPS
