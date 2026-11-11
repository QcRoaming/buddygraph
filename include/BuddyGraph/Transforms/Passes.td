#ifndef BUDDYGRAPH_TRANSFORMS_PASSES_TD
#define BUDDYGRAPH_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def BGraphInferShapes
    : Pass<"bgraph-infer-shapes", "::mlir::ModuleOp"> {
  let summary = "Refine BGraph result tensor shapes from operands and attributes";
  let constructor = "::buddy::bgraph::createInferShapesPass()";
  let dependentDialects = ["::buddy::bgraph::BGraphDialect"];
}

def BGraphFoldBatchNormIntoConv
    : Pass<"bgraph-fold-bn-into-conv", "::mlir::ModuleOp"> {
  let summary = "Fold constant inference BatchNorm parameters into Conv2D";
  let constructor = "::buddy::bgraph::createFoldBatchNormIntoConvPass()";
  let dependentDialects = [
    "::buddy::bgraph::BGraphDialect",
    "::mlir::arith::ArithDialect"
  ];
}

def BGraphFuseElementwise
    : Pass<"bgraph-fuse-elementwise", "::mlir::ModuleOp"> {
  let summary = "Fuse single-use BGraph elementwise chains into a scalar region";
  let constructor = "::buddy::bgraph::createFuseElementwisePass()";
  let dependentDialects = [
    "::buddy::bgraph::BGraphDialect",
    "::mlir::arith::ArithDialect"
  ];
}

def ConvertBGraphToLinalg
    : Pass<"convert-bgraph-to-linalg", "::mlir::ModuleOp"> {
  let summary = "Fully convert BGraph operations to Linalg, Tensor, Arith, and Math";
  let constructor = "::buddy::bgraph::createConvertBGraphToLinalgPass()";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::math::MathDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

def BGraphVectorizeConv
    : Pass<"bgraph-vectorize-conv", "::mlir::ModuleOp"> {
  let summary = "Tile output channels and vectorize NCHW convolution width";
  let constructor = "::buddy::bgraph::createVectorizeConvPass()";
  let options = [
    Option<"vectorWidth", "vector-width", "int64_t", "8",
           "Number of adjacent output pixels per vector">,
    Option<"channelTile", "channel-tile", "int64_t", "4",
           "Number of output channels sharing each input vector">
  ];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::vector::VectorDialect"
  ];
}

#endif // BUDDYGRAPH_TRANSFORMS_PASSES_TD
