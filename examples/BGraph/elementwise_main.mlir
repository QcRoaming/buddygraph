module {
  func.func @main() -> f32 {
    %a = arith.constant dense<[-2.0, 1.0, 3.0, -4.0]> : tensor<4xf32>
    %b = arith.constant dense<[1.0, 2.0, -5.0, 6.0]> : tensor<4xf32>
    %scale = arith.constant dense<[2.0, 4.0, 6.0, 8.0]> : tensor<4xf32>
    %sum = "bgraph.add"(%a, %b) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %relu = "bgraph.relu"(%sum) : (tensor<4xf32>) -> tensor<4xf32>
    %scaled = "bgraph.mul"(%relu, %scale) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %index = arith.constant 3 : index
    %result = tensor.extract %scaled[%index] : tensor<4xf32>
    return %result : f32
  }
}
