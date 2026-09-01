// RUN: buddygraph-opt %s | buddygraph-opt | FileCheck %s

module {
  func.func @relu(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32> {
    // CHECK: bgraph.relu
    %0 = "bgraph.relu"(%arg0) : (tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
    return %0 : tensor<1x2x3x4xf32>
  }

  func.func @clamp(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
    // CHECK: bgraph.clamp
    // CHECK-SAME: max_value = 6.000000e+00 : f32
    // CHECK-SAME: min_value = 0.000000e+00 : f32
    %0 = "bgraph.clamp"(%arg0) {min_value = 0.0 : f32, max_value = 6.0 : f32} : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
