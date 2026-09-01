// RUN: buddygraph-opt --bgraph-fuse-elementwise %s | FileCheck %s
// RUN: buddygraph-opt --bgraph-fuse-elementwise -mlir-print-debuginfo %s | FileCheck %s --check-prefix=LOC

// LOC-DAG: #loc[[ADD:[0-9]+]] = loc("add")
// LOC-DAG: #loc[[RELU:[0-9]+]] = loc("relu")
// LOC-DAG: #loc[[MUL:[0-9]+]] = loc("mul")
// LOC: loc(fused[#loc[[MUL]], #loc[[RELU]], #loc[[ADD]]])

func.func @fuse(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>,
                %arg2: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "bgraph.add"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32> loc("add")
  %1 = "bgraph.relu"(%0) : (tensor<2x3xf32>) -> tensor<2x3xf32> loc("relu")
  %2 = "bgraph.mul"(%1, %arg2) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32> loc("mul")
  return %2 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @fuse
// CHECK-NOT: "bgraph.add"
// CHECK-NOT: "bgraph.relu"
// CHECK-NOT: "bgraph.mul"
// CHECK-COUNT-1: "bgraph.fused_elementwise"
// CHECK: ^bb0(%{{.*}}: f32, %{{.*}}: f32, %{{.*}}: f32):
// CHECK: arith.addf
// CHECK: arith.maximumf
// CHECK: arith.mulf
// CHECK: "bgraph.yield"

func.func @do_not_duplicate(%arg0: tensor<2x3xf32>,
                            %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "bgraph.add"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %1 = "bgraph.relu"(%0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %2 = "bgraph.mul"(%0, %1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %2 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @do_not_duplicate
// CHECK: "bgraph.add"

func.func @fuse_clamp(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>,
                      %arg2: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "bgraph.add"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %1 = "bgraph.clamp"(%0) {min_value = -1.0 : f32, max_value = 1.0 : f32} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %2 = "bgraph.mul"(%1, %arg2) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %2 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @fuse_clamp
// CHECK-NOT: "bgraph.clamp"
// CHECK-COUNT-1: "bgraph.fused_elementwise"
// CHECK: arith.cmpf olt
// CHECK: arith.select
// CHECK: arith.cmpf olt
// CHECK: arith.select
