#ifndef BUDDYGRAPH_IR_BGRAP_OPS_TD
#define BUDDYGRAPH_IR_BGRAP_OPS_TD

include "BuddyGraph/IR/BGraphDialect.td"
include "BuddyGraph/IR/BGraphEnums.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def BGraph_Conv2DOp : BGraph_Op<"conv2d", [Pure]> {
  let summary = "Static f32 two-dimensional convolution";
  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$filter,
                       Variadic<AnyRankedTensor>:$bias,
                       DenseI64ArrayAttr:$strides,
                       DenseI64ArrayAttr:$pads,
                       DenseI64ArrayAttr:$dilations,
                       I64Attr:$groups,
                       BGraph_LayoutAttr:$layout);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def BGraph_BatchNormOp : BGraph_Op<"batch_norm", [Pure]> {
  let summary = "Inference-mode batch normalization";
  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$scale,
                       AnyRankedTensor:$bias,
                       AnyRankedTensor:$mean,
                       AnyRankedTensor:$variance,
                       F64Attr:$epsilon,
                       BGraph_LayoutAttr:$layout);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def BGraph_ReluOp : BGraph_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "Elementwise rectified linear unit";
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def BGraph_ClampOp : BGraph_Op<"clamp", [Pure]> {
  let summary = "Elementwise clamp with constant scalar bounds";
  let arguments = (ins AnyRankedTensor:$input,
                       F32Attr:$min_value,
                       F32Attr:$max_value);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

class BGraph_BinaryOp<string mnemonic>
    : BGraph_Op<mnemonic, [Pure]> {
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def BGraph_AddOp : BGraph_BinaryOp<"add"> {
  let summary = "Broadcasting elementwise addition";
}
def BGraph_SubOp : BGraph_BinaryOp<"sub"> {
  let summary = "Broadcasting elementwise subtraction";
}
def BGraph_MulOp : BGraph_BinaryOp<"mul"> {
  let summary = "Broadcasting elementwise multiplication";
}
def BGraph_DivOp : BGraph_BinaryOp<"div"> {
  let summary = "Broadcasting elementwise division";
}

def BGraph_ReshapeOp : BGraph_Op<"reshape", [Pure]> {
  let summary = "Reshape to a constant shape";
  let arguments = (ins AnyRankedTensor:$input, DenseI64ArrayAttr:$shape);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def BGraph_TransposeOp : BGraph_Op<"transpose", [Pure]> {
  let summary = "Transpose by a constant permutation";
  let arguments = (ins AnyRankedTensor:$input, DenseI64ArrayAttr:$permutation);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def BGraph_ReduceMeanOp : BGraph_Op<"reduce_mean", [Pure]> {
  let summary = "Mean reduction over constant axes";
  let arguments = (ins AnyRankedTensor:$input,
                       DenseI64ArrayAttr:$axes,
                       BoolAttr:$keep_dims);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def BGraph_FusedElementwiseOp
    : BGraph_Op<"fused_elementwise", [Pure]> {
  let summary = "A broadcast elementwise chain represented by a scalar region";
  let arguments = (ins Variadic<AnyRankedTensor>:$inputs);
  let results = (outs AnyRankedTensor:$result);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}

def BGraph_YieldOp : BGraph_Op<"yield", [Pure, Terminator,
    HasParent<"FusedElementwiseOp">]> {
  let summary = "Yield a scalar from fused_elementwise";
  let arguments = (ins AnyType:$value);
  let hasVerifier = 1;
}

#endif // BUDDYGRAPH_IR_BGRAP_OPS_TD
