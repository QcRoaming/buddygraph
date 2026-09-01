// RUN: not buddygraph-opt --convert-bgraph-to-linalg %s 2>&1 | FileCheck %s

func.func @dynamic_conv(%input: tensor<1x1x?x?xf32>) -> tensor<1x1x4x4xf32> {
  %filter = arith.constant dense<1.0> : tensor<1x1x3x3xf32>
  %0 = "bgraph.conv2d"(%input, %filter) {
    dilations = array<i64: 1, 1>, groups = 1 : i64,
    layout = #bgraph.layout<nchw>, pads = array<i64: 1, 1, 1, 1>,
    strides = array<i64: 1, 1>
  } : (tensor<1x1x?x?xf32>, tensor<1x1x3x3xf32>) -> tensor<1x1x4x4xf32>
  return %0 : tensor<1x1x4x4xf32>
}

// CHECK: failed to legalize operation 'bgraph.conv2d'
