// RUN: buddygraph-opt --bgraph-bufferize %s | FileCheck %s --implicit-check-not=memref.alloc --implicit-check-not=memref.copy --implicit-check-not=func.call

// The result is computed into the caller's buffer even with rank-1 and scalar
// broadcasting. The private tensor function must be inlined before empty-tensor
// elimination can propagate that destination.
#identity = affine_map<(d0, d1) -> (d0, d1)>
#bias = affine_map<(d0, d1) -> (d1)>
#scalar = affine_map<(d0, d1) -> ()>

func.func private @compute(%x: tensor<2x17xf32>, %bias: tensor<17xf32>,
                           %factor: tensor<f32>) -> tensor<2x17xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x17xf32>
  %result = linalg.generic {
    indexing_maps = [#identity, #bias, #scalar, #identity],
    iterator_types = ["parallel", "parallel"]
  } ins(%x, %bias, %factor : tensor<2x17xf32>, tensor<17xf32>, tensor<f32>)
    outs(%empty : tensor<2x17xf32>) {
    ^bb0(%xv: f32, %bv: f32, %fv: f32, %unused: f32):
      %sum = arith.addf %xv, %bv : f32
      %relu = arith.maximumf %sum, %zero : f32
      %scaled = arith.mulf %relu, %fv : f32
      linalg.yield %scaled : f32
  } -> tensor<2x17xf32>
  return %result : tensor<2x17xf32>
}

// CHECK-NOT: @compute
// CHECK-LABEL: func.func @direct(
// CHECK-SAME: %[[X:.*]]: memref<2x17xf32>, %[[B:.*]]: memref<17xf32>, %[[F:.*]]: memref<f32>, %[[OUT:.*]]: memref<2x17xf32>
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[X]], %[[B]], %[[F]]
// CHECK-SAME: outs(%[[OUT]] : memref<2x17xf32>)
// CHECK: arith.maximumf
// CHECK: return
func.func @direct(%xm: memref<2x17xf32>, %bm: memref<17xf32>,
                  %fm: memref<f32>, %out: memref<2x17xf32>) {
  %x = bufferization.to_tensor %xm restrict : memref<2x17xf32> to tensor<2x17xf32>
  %bias = bufferization.to_tensor %bm restrict : memref<17xf32> to tensor<17xf32>
  %factor = bufferization.to_tensor %fm restrict : memref<f32> to tensor<f32>
  %result = func.call @compute(%x, %bias, %factor)
      : (tensor<2x17xf32>, tensor<17xf32>, tensor<f32>) -> tensor<2x17xf32>
  bufferization.materialize_in_destination %result in restrict writable %out
      : (tensor<2x17xf32>, memref<2x17xf32>) -> ()
  return
}
