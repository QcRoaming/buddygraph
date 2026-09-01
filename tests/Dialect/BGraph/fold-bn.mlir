// RUN: buddygraph-opt --bgraph-fold-bn-into-conv %s | FileCheck %s
// RUN: buddygraph-opt --bgraph-fold-bn-into-conv -mlir-print-debuginfo %s | FileCheck %s --check-prefix=LOC

// LOC-DAG: #loc[[CONV:[0-9]+]] = loc("conv")
// LOC-DAG: #loc[[BN:[0-9]+]] = loc("batch_norm")
// LOC: loc(fused[#loc[[CONV]], #loc[[BN]]])

func.func @fold(%input: tensor<1x1x2x2xf32>) -> tensor<1x2x2x2xf32> {
  %filter = arith.constant dense<[[[[1.0]]], [[[2.0]]]]> : tensor<2x1x1x1xf32>
  %scale = arith.constant dense<[2.0, 4.0]> : tensor<2xf32>
  %beta = arith.constant dense<[0.5, -0.5]> : tensor<2xf32>
  %mean = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %variance = arith.constant dense<[3.0, 15.0]> : tensor<2xf32>
  %conv = "bgraph.conv2d"(%input, %filter) {
    dilations = array<i64: 1, 1>, groups = 1 : i64,
    layout = #bgraph.layout<nchw>, pads = array<i64: 0, 0, 0, 0>,
    strides = array<i64: 1, 1>
  } : (tensor<1x1x2x2xf32>, tensor<2x1x1x1xf32>) -> tensor<1x2x2x2xf32> loc("conv")
  %normalized = "bgraph.batch_norm"(%conv, %scale, %beta, %mean, %variance) {
    epsilon = 1.0 : f64, layout = #bgraph.layout<nchw>
  } : (tensor<1x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<1x2x2x2xf32> loc("batch_norm")
  return %normalized : tensor<1x2x2x2xf32>
}

// CHECK-LABEL: func.func @fold
// CHECK-NOT: "bgraph.batch_norm"
// CHECK: arith.constant dense<[-5.000000e-01, -2.500000e+00]>
// CHECK: "bgraph.conv2d"(%arg0, %{{.*}}, %{{.*}})

func.func @fold_nonunit_with_bias(%input: tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32> {
  %filter = arith.constant dense<2.0> : tensor<1x1x1x1xf32>
  %conv_bias = arith.constant dense<1.0> : tensor<1xf32>
  %scale = arith.constant dense<4.0> : tensor<1xf32>
  %beta = arith.constant dense<0.5> : tensor<1xf32>
  %mean = arith.constant dense<2.0> : tensor<1xf32>
  %variance = arith.constant dense<3.0> : tensor<1xf32>
  %conv = "bgraph.conv2d"(%input, %filter, %conv_bias) {
    dilations = array<i64: 1, 1>, groups = 1 : i64,
    layout = #bgraph.layout<nchw>, pads = array<i64: 0, 0, 0, 0>,
    strides = array<i64: 1, 1>
  } : (tensor<1x1x2x2xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x2x2xf32>
  %normalized = "bgraph.batch_norm"(%conv, %scale, %beta, %mean, %variance) {
    epsilon = 1.0 : f64, layout = #bgraph.layout<nchw>
  } : (tensor<1x1x2x2xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x2x2xf32>
  return %normalized : tensor<1x1x2x2xf32>
}

// CHECK-LABEL: func.func @fold_nonunit_with_bias
// CHECK-NOT: "bgraph.batch_norm"
// CHECK: arith.constant dense<4.000000e+00> : tensor<1x1x1x1xf32>
// CHECK: arith.constant dense<-1.500000e+00> : tensor<1xf32>
// CHECK: "bgraph.conv2d"(%arg0, %{{.*}}, %{{.*}})

func.func @multiple_users(%input: tensor<1x1x2x2xf32>) -> tensor<1x2x2x2xf32> {
  %filter = arith.constant dense<[[[[1.0]]], [[[2.0]]]]> : tensor<2x1x1x1xf32>
  %scale = arith.constant dense<[2.0, 4.0]> : tensor<2xf32>
  %beta = arith.constant dense<[0.5, -0.5]> : tensor<2xf32>
  %mean = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %variance = arith.constant dense<[3.0, 15.0]> : tensor<2xf32>
  %conv = "bgraph.conv2d"(%input, %filter) {
    dilations = array<i64: 1, 1>, groups = 1 : i64,
    layout = #bgraph.layout<nchw>, pads = array<i64: 0, 0, 0, 0>,
    strides = array<i64: 1, 1>
  } : (tensor<1x1x2x2xf32>, tensor<2x1x1x1xf32>) -> tensor<1x2x2x2xf32>
  %side = "bgraph.relu"(%conv) : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %normalized = "bgraph.batch_norm"(%conv, %scale, %beta, %mean, %variance) {
    epsilon = 1.0 : f64, layout = #bgraph.layout<nchw>
  } : (tensor<1x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<1x2x2x2xf32>
  %result = "bgraph.add"(%normalized, %side) : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  return %result : tensor<1x2x2x2xf32>
}

// CHECK-LABEL: func.func @multiple_users
// CHECK: "bgraph.batch_norm"

func.func @layout_mismatch(%input: tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32> {
  %filter = arith.constant dense<1.0> : tensor<2x2x1x1xf32>
  %parameter = arith.constant dense<1.0> : tensor<2xf32>
  %variance = arith.constant dense<3.0> : tensor<2xf32>
  %conv = "bgraph.conv2d"(%input, %filter) {
    dilations = array<i64: 1, 1>, groups = 1 : i64,
    layout = #bgraph.layout<nchw>, pads = array<i64: 0, 0, 0, 0>,
    strides = array<i64: 1, 1>
  } : (tensor<1x2x2x2xf32>, tensor<2x2x1x1xf32>) -> tensor<1x2x2x2xf32>
  %normalized = "bgraph.batch_norm"(%conv, %parameter, %parameter, %parameter, %variance) {
    epsilon = 1.0 : f64, layout = #bgraph.layout<nhwc>
  } : (tensor<1x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<1x2x2x2xf32>
  return %normalized : tensor<1x2x2x2xf32>
}

// CHECK-LABEL: func.func @layout_mismatch
// CHECK: "bgraph.batch_norm"

func.func @epsilon_rounding_boundary(%input: tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32> {
  %filter = arith.constant dense<1.0> : tensor<1x1x1x1xf32>
  %scale = arith.constant dense<1.0> : tensor<1xf32>
  %zero = arith.constant dense<0.0> : tensor<1xf32>
  %variance = arith.constant dense<-1.0> : tensor<1xf32>
  %conv = "bgraph.conv2d"(%input, %filter) {
    dilations = array<i64: 1, 1>, groups = 1 : i64,
    layout = #bgraph.layout<nchw>, pads = array<i64: 0, 0, 0, 0>,
    strides = array<i64: 1, 1>
  } : (tensor<1x1x2x2xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x2x2xf32>
  %normalized = "bgraph.batch_norm"(%conv, %scale, %zero, %zero, %variance) {
    epsilon = 1.00000001 : f64, layout = #bgraph.layout<nchw>
  } : (tensor<1x1x2x2xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x2x2xf32>
  return %normalized : tensor<1x1x2x2xf32>
}

// CHECK-LABEL: func.func @epsilon_rounding_boundary
// CHECK: "bgraph.batch_norm"
