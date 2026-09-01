// RUN: buddygraph-opt --convert-bgraph-to-linalg %s | FileCheck %s

func.func @sub_div(%lhs: tensor<2x3xf32>, %rhs: tensor<3xf32>) -> tensor<2x3xf32> {
  %0 = "bgraph.sub"(%lhs, %rhs) : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  %1 = "bgraph.div"(%0, %rhs) : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

func.func @clamp(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "bgraph.clamp"(%arg0) {min_value = -1.0 : f32, max_value = 1.0 : f32} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-NOT: bgraph.
// CHECK: linalg.generic
// CHECK: arith.subf
// CHECK: linalg.generic
// CHECK: arith.divf
// CHECK-LABEL: func.func @clamp
// CHECK: linalg.generic
// CHECK: arith.cmpf olt
// CHECK: arith.select
// CHECK: arith.cmpf olt
// CHECK: arith.select
