// RUN: buddygraph-opt --canonicalize %s | FileCheck %s --check-prefix=IR
// RUN: buddygraph-opt --canonicalize --convert-bgraph-to-linalg \
// RUN:   '--one-shot-bufferize=bufferize-function-boundaries' \
// RUN:   --convert-linalg-to-loops --lower-affine --convert-scf-to-cf \
// RUN:   --convert-cf-to-llvm --convert-math-to-llvm --convert-arith-to-llvm \
// RUN:   --finalize-memref-to-llvm --convert-func-to-llvm \
// RUN:   --reconcile-unrealized-casts %s -o %t
// RUN: mlir-runner %t -e signed_zero -entry-point-result=f32 \
// RUN:   | FileCheck %s --check-prefix=RUNNER

// Without an explicit no-signed-zeros contract, replacing x + +0.0 with x
// changes -0.0 into the wrong signed result. Keep the Add and check the value
// after the complete CPU lowering pipeline.
module {
  func.func @signed_zero() -> f32 {
    %negative_zero = arith.constant dense<-0.0> : tensor<f32>
    %positive_zero = arith.constant dense<0.0> : tensor<f32>
    %sum = "bgraph.add"(%negative_zero, %positive_zero)
        : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %value = tensor.extract %sum[] : tensor<f32>
    return %value : f32
  }
}

// IR-LABEL: func.func @signed_zero
// IR: %[[SUM:.*]] = "bgraph.add"
// IR: tensor.extract %[[SUM]][]

// RUNNER: {{^0\.000000e\+00$}}
