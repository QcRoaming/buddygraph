// RUN: buddygraph-opt --convert-bgraph-to-linalg %S/../../Dialect/BGraph/canonicalize.mlir | FileCheck %s --check-prefix=BASIC
// RUN: buddygraph-opt --bgraph-infer-shapes --convert-bgraph-to-linalg %S/../../Dialect/BGraph/infer-shapes.mlir | FileCheck %s --check-prefix=INFER
// RUN: buddygraph-opt --bgraph-fuse-elementwise --convert-bgraph-to-linalg %S/../../Dialect/BGraph/fuse-elementwise.mlir | FileCheck %s --check-prefix=FUSED
// RUN: buddygraph-opt --convert-bgraph-to-linalg %S/../../Dialect/BGraph/fold-bn.mlir | FileCheck %s --check-prefix=CONVBN
// RUN: buddygraph-opt --pass-pipeline='builtin.module(bgraph-fold-bn-into-conv,canonicalize,cse,convert-bgraph-to-linalg)' %S/../../Dialect/BGraph/fold-bn.mlir | FileCheck %s --check-prefix=PIPELINE

// BASIC-NOT: bgraph.
// BASIC: linalg.generic
// BASIC: tensor.reshape
// BASIC: linalg.transpose
// BASIC: arith.cmpf olt
// BASIC: arith.select

// INFER-NOT: bgraph.
// INFER: iterator_types = ["reduction", "parallel", "reduction"]
// INFER: linalg.conv_2d_nchw_fchw

// FUSED-NOT: bgraph.
// FUSED-LABEL: func.func @fuse
// FUSED-COUNT-1: linalg.generic
// FUSED: arith.addf
// FUSED: arith.maximumf
// FUSED: arith.mulf
// FUSED: arith.cmpf olt
// FUSED: arith.select

// CONVBN-NOT: bgraph.
// CONVBN: linalg.conv_2d_nchw_fchw
// CONVBN: math.sqrt

// PIPELINE-NOT: bgraph.
// PIPELINE: linalg.conv_2d_nchw_fchw
