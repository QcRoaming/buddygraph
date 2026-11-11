// RUN: buddygraph-opt --bgraph-vectorize-conv="vector-width=4 channel-tile=2" %s | FileCheck %s --implicit-check-not=vector.fma

// Both the output width (11) and channel count (5) have remainders.
// CHECK-LABEL: func.func @tiled_tail(
// CHECK-NOT: linalg.conv_2d_nchw_fchw
// CHECK: scf.for
// CHECK: vector.load {{.*}} : memref<1x5x3x11xf32>, vector<4xf32>
// CHECK: vector.broadcast {{.*}} : f32 to vector<4xf32>
// CHECK: arith.mulf {{.*}} : vector<4xf32>
// CHECK: arith.addf {{.*}} : vector<4xf32>
// CHECK: vector.store
// CHECK: memref.load {{.*}} : memref<1x5x3x11xf32>
// CHECK: arith.mulf {{.*}} : f32
// CHECK: memref.store
// CHECK-NOT: linalg.conv_2d_nchw_fchw
// CHECK: return
func.func @tiled_tail(%input: memref<1x3x5x13xf32>,
                      %filter: memref<5x3x3x3xf32>,
                      %output: memref<1x5x3x11xf32>) {
  linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter : memref<1x3x5x13xf32>, memref<5x3x3x3xf32>)
    outs(%output : memref<1x5x3x11xf32>)
  return
}

// Width smaller than the vector width takes only the scalar tail.
// CHECK-LABEL: func.func @small_width(
// CHECK-NOT: vector.
// CHECK-NOT: linalg.conv_2d_nchw_fchw
// CHECK: memref.load
// CHECK-NOT: vector.
// CHECK-NOT: linalg.conv_2d_nchw_fchw
// CHECK: return
func.func @small_width(%input: memref<1x3x4x5xf32>,
                       %filter: memref<3x3x3x3xf32>,
                       %output: memref<1x3x2x3xf32>) {
  linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter : memref<1x3x4x5xf32>, memref<3x3x3x3xf32>)
    outs(%output : memref<1x3x2x3xf32>)
  return
}

// Stride two remains available to the standard Linalg lowering.
// CHECK-LABEL: func.func @stride_two(
// CHECK-NOT: vector.
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK: return
func.func @stride_two(%input: memref<1x3x7x9xf32>,
                      %filter: memref<5x3x3x3xf32>,
                      %output: memref<1x5x3x4xf32>) {
  linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%input, %filter : memref<1x3x7x9xf32>, memref<5x3x3x3xf32>)
    outs(%output : memref<1x5x3x4xf32>)
  return
}
