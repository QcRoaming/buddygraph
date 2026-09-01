// RUN: buddygraph-opt --canonicalize %s | FileCheck %s

func.func @canonicalize(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %zero = arith.constant dense<0.0> : tensor<2x3xf32>
  %one = arith.constant dense<1.0> : tensor<2x3xf32>
  %0 = "bgraph.relu"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %1 = "bgraph.relu"(%0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %2 = "bgraph.add"(%1, %zero) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %3 = "bgraph.mul"(%2, %one) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %3 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @canonicalize
// CHECK-COUNT-1: "bgraph.relu"
// CHECK-COUNT-1: "bgraph.add"
// CHECK-NOT: "bgraph.mul"

func.func @reshape_chain(%arg0: tensor<2x3xf32>) -> tensor<6xf32> {
  %0 = "bgraph.reshape"(%arg0) {shape = array<i64: 3, 2>} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  %1 = "bgraph.reshape"(%0) {shape = array<i64: 6>} : (tensor<3x2xf32>) -> tensor<6xf32>
  return %1 : tensor<6xf32>
}

// CHECK-LABEL: func.func @reshape_chain
// CHECK-COUNT-1: "bgraph.reshape"

func.func @transpose_inverse(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "bgraph.transpose"(%arg0) {permutation = array<i64: 1, 0>} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  %1 = "bgraph.transpose"(%0) {permutation = array<i64: 1, 0>} : (tensor<3x2xf32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @transpose_inverse
// CHECK-NOT: "bgraph.transpose"
// CHECK: return %arg0

func.func @mul_left_identity(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %one = arith.constant dense<1.0> : tensor<2x3xf32>
  %0 = "bgraph.mul"(%one, %arg0) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @mul_left_identity
// CHECK-NOT: "bgraph.mul"
// CHECK: return %arg0

func.func @do_not_elide_broadcast_one(%arg0: tensor<1xf32>) -> tensor<2x3xf32> {
  %one = arith.constant dense<1.0> : tensor<2x3xf32>
  %0 = "bgraph.mul"(%one, %arg0) : (tensor<2x3xf32>, tensor<1xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @do_not_elide_broadcast_one
// CHECK: "bgraph.mul"

func.func @nested_clamp(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "bgraph.clamp"(%arg0) {min_value = -1.0 : f32, max_value = 1.0 : f32} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %1 = "bgraph.clamp"(%0) {min_value = -1.0 : f32, max_value = 1.0 : f32} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @nested_clamp
// CHECK-COUNT-1: "bgraph.clamp"

func.func @different_clamp_bounds(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "bgraph.clamp"(%arg0) {min_value = -2.0 : f32, max_value = 2.0 : f32} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %1 = "bgraph.clamp"(%0) {min_value = -1.0 : f32, max_value = 1.0 : f32} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @different_clamp_bounds
// CHECK-COUNT-2: "bgraph.clamp"
