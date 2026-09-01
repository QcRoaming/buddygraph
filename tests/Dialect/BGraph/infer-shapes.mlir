// RUN: buddygraph-opt --bgraph-infer-shapes %s | FileCheck %s

func.func @broadcast(%lhs: tensor<2x1xf32>, %rhs: tensor<1x3xf32>) {
  %0 = "bgraph.add"(%lhs, %rhs) : (tensor<2x1xf32>, tensor<1x3xf32>) -> tensor<?x?xf32>
  return
}
// CHECK-LABEL: func.func @broadcast
// CHECK: -> tensor<2x3xf32>

func.func @clamp(%arg0: tensor<2x3xf32>) {
  %0 = "bgraph.clamp"(%arg0) {min_value = -1.0 : f32, max_value = 1.0 : f32} : (tensor<2x3xf32>) -> tensor<?x?xf32>
  return
}
// CHECK-LABEL: func.func @clamp
// CHECK: -> tensor<2x3xf32>

func.func @reshape(%arg0: tensor<2x3x4xf32>) {
  %0 = "bgraph.reshape"(%arg0) {shape = array<i64: 6, -1>} : (tensor<2x3x4xf32>) -> tensor<?x?xf32>
  return
}
// CHECK-LABEL: func.func @reshape
// CHECK: -> tensor<6x4xf32>

func.func @transpose(%arg0: tensor<2x3x4xf32>) {
  %0 = "bgraph.transpose"(%arg0) {permutation = array<i64: 2, 0, 1>} : (tensor<2x3x4xf32>) -> tensor<?x?x?xf32>
  return
}
// CHECK-LABEL: func.func @transpose
// CHECK: -> tensor<4x2x3xf32>

func.func @reduce(%arg0: tensor<2x3x4xf32>) {
  %0 = "bgraph.reduce_mean"(%arg0) {axes = array<i64: 0, 2>, keep_dims = false} : (tensor<2x3x4xf32>) -> tensor<?xf32>
  return
}
// CHECK-LABEL: func.func @reduce
// CHECK: -> tensor<3xf32>

func.func @conv(%arg0: tensor<1x2x5x5xf32>, %filter: tensor<4x2x3x3xf32>) {
  %0 = "bgraph.conv2d"(%arg0, %filter) {
    dilations = array<i64: 1, 1>,
    groups = 1 : i64,
    layout = #bgraph.layout<nchw>,
    pads = array<i64: 1, 1, 1, 1>,
    strides = array<i64: 2, 2>
  } : (tensor<1x2x5x5xf32>, tensor<4x2x3x3xf32>) -> tensor<?x?x?x?xf32>
  return
}
// CHECK-LABEL: func.func @conv
// CHECK: -> tensor<1x4x3x3xf32>
