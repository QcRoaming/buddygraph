# BuddyGraph examples

`elementwise_main.mlir` 是不依赖 ONNX 的最小 runner 闭环。它计算两个常量
tensor 的 Add → Relu → Mul，并用 `tensor.extract` 取下标 3 的元素作为 `f32`
入口返回值。

```bash
cd /home/jlq/project/buddygraph
export BUDDYGRAPH_TMP=/home/jlq/project/buddygraph/tmp
mkdir -p "$BUDDYGRAPH_TMP"
build/bin/buddygraph-opt \
  --bgraph-fuse-elementwise \
  --convert-bgraph-to-linalg \
  '--one-shot-bufferize=bufferize-function-boundaries' \
  --convert-linalg-to-loops --lower-affine \
  --convert-scf-to-cf --convert-cf-to-llvm \
  --convert-math-to-llvm --convert-arith-to-llvm \
  --finalize-memref-to-llvm --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  examples/BGraph/elementwise_main.mlir \
  -o "$BUDDYGRAPH_TMP/elementwise_main.llvm.mlir"

/buddy-mlir/llvm/build/bin/mlir-runner \
  "$BUDDYGRAPH_TMP/elementwise_main.llvm.mlir" \
  -e main -entry-point-result=f32
```

预期输出为 `1.600000e+01`。ONNX 端到端示例由
`tests/E2E/BGraph/onnx_elementwise.py` 自动生成并检查。
