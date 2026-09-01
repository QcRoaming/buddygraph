// RUN: buddygraph-opt --verify-diagnostics --split-input-file %s

func.func @wrong_element_type(%arg0: tensor<2x3xi32>) {
  // expected-error@+1 {{'bgraph.relu' op input must have f32 elements}}
  %0 = "bgraph.relu"(%arg0) : (tensor<2x3xi32>) -> tensor<2x3xi32>
  return
}

// -----

func.func @invalid_broadcast(%arg0: tensor<2x3xf32>, %arg1: tensor<4xf32>) {
  // expected-error@+1 {{'bgraph.add' op operands are not broadcast compatible}}
  %0 = "bgraph.add"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<4xf32>) -> tensor<2x3xf32>
  return
}

// -----

func.func @unsupported_groups(%input: tensor<1x4x5x5xf32>,
                              %filter: tensor<4x2x3x3xf32>) {
  // expected-error@+1 {{'bgraph.conv2d' op BuddyGraph MVP currently supports groups=1 only}}
  %0 = "bgraph.conv2d"(%input, %filter) {dilations = array<i64: 1, 1>, groups = 2 : i64, layout = #bgraph.layout<nchw>, pads = array<i64: 1, 1, 1, 1>, strides = array<i64: 1, 1>} : (tensor<1x4x5x5xf32>, tensor<4x2x3x3xf32>) -> tensor<1x4x5x5xf32>
  return
}

// -----

func.func @kernel_larger_than_input(%input: tensor<1x1x2x2xf32>,
                                    %filter: tensor<1x1x5x5xf32>) {
  // expected-error@+1 {{'bgraph.conv2d' op kernel does not fit within the padded input or dimensions overflow}}
  %0 = "bgraph.conv2d"(%input, %filter) {dilations = array<i64: 1, 1>, groups = 1 : i64, layout = #bgraph.layout<nchw>, pads = array<i64: 0, 0, 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x1x2x2xf32>, tensor<1x1x5x5xf32>) -> tensor<1x1x?x?xf32>
  return
}

// -----

func.func @invalid_conv_channels(%input: tensor<1x2x2x2xf32>,
                                 %filter: tensor<1x1x1x1xf32>) {
  // expected-error@+1 {{'bgraph.conv2d' op filter input channels do not match input/groups}}
  %0 = "bgraph.conv2d"(%input, %filter) {dilations = array<i64: 1, 1>, groups = 1 : i64, layout = #bgraph.layout<nchw>, pads = array<i64: 0, 0, 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x2x2x2xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x2x2xf32>
  return
}

// -----

func.func @invalid_batch_norm_channels(
    %input: tensor<1x2x2x2xf32>, %parameter: tensor<1xf32>) {
  // expected-error@+1 {{'bgraph.batch_norm' op scale must be rank 1 with the channel length}}
  %0 = "bgraph.batch_norm"(%input, %parameter, %parameter, %parameter, %parameter) {epsilon = 1.0e-5 : f64, layout = #bgraph.layout<nchw>} : (tensor<1x2x2x2xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x2x2x2xf32>
  return
}

// -----

func.func @invalid_transpose(%arg0: tensor<2x3xf32>) {
  // expected-error@+1 {{'bgraph.transpose' op permutation must contain each input axis exactly once}}
  %0 = "bgraph.transpose"(%arg0) {permutation = array<i64: 0, 0>} : (tensor<2x3xf32>) -> tensor<2x2xf32>
  return
}

// -----

func.func @invalid_reduce(%arg0: tensor<2x3xf32>) {
  // expected-error@+1 {{'bgraph.reduce_mean' op axes must be unique and within the input rank}}
  %0 = "bgraph.reduce_mean"(%arg0) {axes = array<i64: 1, 1>, keep_dims = false} : (tensor<2x3xf32>) -> tensor<2xf32>
  return
}

// -----

func.func @invalid_reshape(%arg0: tensor<2x3xf32>) {
  // expected-error@+1 {{'bgraph.reshape' op shape must contain positive dimensions and at most one valid -1}}
  %0 = "bgraph.reshape"(%arg0) {shape = array<i64: -1, -1>} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return
}

// -----

func.func @invalid_fused_body(%arg0: tensor<2x3xf32>) {
  // expected-error@+1 {{'bgraph.fused_elementwise' op body contains unsupported scalar operation arith.negf}}
  %0 = "bgraph.fused_elementwise"(%arg0) ({
  ^bb0(%value: f32):
    %negated = arith.negf %value : f32
    "bgraph.yield"(%negated) : (f32) -> ()
  }) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return
}

// -----

func.func @invalid_yield_type(%arg0: tensor<2x3xf32>) {
  // expected-error@+1 {{'bgraph.fused_elementwise' op body must terminate with bgraph.yield of f32}}
  %0 = "bgraph.fused_elementwise"(%arg0) ({
  ^bb0(%value: f32):
    %wrong = arith.constant 0 : i32
    "bgraph.yield"(%wrong) : (i32) -> ()
  }) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return
}

// -----

func.func @invalid_clamp_bounds(%arg0: tensor<2x3xf32>) {
  // expected-error@+1 {{'bgraph.clamp' op min_value must be less than or equal to max_value}}
  %0 = "bgraph.clamp"(%arg0) {min_value = 2.0 : f32, max_value = 1.0 : f32} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return
}

// -----

func.func @nonfinite_clamp_bound(%arg0: tensor<2x3xf32>) {
  // expected-error@+1 {{'bgraph.clamp' op bounds must be finite}}
  %0 = "bgraph.clamp"(%arg0) {min_value = 0.0 : f32, max_value = 0x7F800000 : f32} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return
}

// -----

func.func @invalid_clamp_shape(%arg0: tensor<2x3xf32>) {
  // expected-error@+1 {{'bgraph.clamp' op input and result shapes must match}}
  %0 = "bgraph.clamp"(%arg0) {min_value = 0.0 : f32, max_value = 1.0 : f32} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  return
}
