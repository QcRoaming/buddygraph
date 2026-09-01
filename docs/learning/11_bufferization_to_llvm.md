# 11｜从 tensor/Linalg 到 MemRef、LLVM Dialect 与 LLVM IR

## 1. 本章目标

你将保存 Linalg、bufferized、LLVM Dialect 和 LLVM IR 四个真实文件，解释
destination-passing style、allocation 和 runner ABI，并指出当前 deallocation 边界。

## 2. 先运行

```bash
cd /buddy-mlir/jlq/projects/buddygraph
export BUDDYGRAPH_TMP=/buddy-mlir/jlq/projects/buddygraph/tmp
mkdir -p "$BUDDYGRAPH_TMP"

build/bin/buddygraph-opt \
  --bgraph-fuse-elementwise --convert-bgraph-to-linalg \
  examples/BGraph/elementwise_main.mlir \
  -o "$BUDDYGRAPH_TMP/bgraph-11-linalg.mlir"

build/bin/buddygraph-opt \
  --bgraph-fuse-elementwise --convert-bgraph-to-linalg \
  '--one-shot-bufferize=bufferize-function-boundaries' \
  examples/BGraph/elementwise_main.mlir \
  -o "$BUDDYGRAPH_TMP/bgraph-11-buffer.mlir"

build/bin/buddygraph-opt \
  --bgraph-fuse-elementwise --convert-bgraph-to-linalg \
  '--one-shot-bufferize=bufferize-function-boundaries' \
  --convert-linalg-to-loops --lower-affine \
  --convert-scf-to-cf --convert-cf-to-llvm \
  --convert-math-to-llvm --convert-arith-to-llvm \
  --finalize-memref-to-llvm --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  examples/BGraph/elementwise_main.mlir \
  -o "$BUDDYGRAPH_TMP/bgraph-11-llvm.mlir"

/buddy-mlir/llvm/build/bin/mlir-translate --mlir-to-llvmir \
  "$BUDDYGRAPH_TMP/bgraph-11-llvm.mlir" -o "$BUDDYGRAPH_TMP/bgraph-11.ll"

/buddy-mlir/llvm/build/bin/mlir-runner "$BUDDYGRAPH_TMP/bgraph-11-llvm.mlir" \
  -e main -entry-point-result=f32
```

预期输出 `1.600000e+01`。

## 3. 真实代码位置

- `BGraphToLinalg.cpp::createElementwiseGeneric()`：`tensor.empty` + `outs`。
- `Conv2DLowering`、`ReduceMeanLowering`：DPS-style Linalg init/output。
- `buddygraph-opt.cpp`：注册 Arith/Func/CF/Linalg/SCF/Tensor/Vector 的
  BufferizableOpInterface external models。
- `tests/Conversion/BGraphToLinalg/bufferize.mlir`。
- `tests/E2E/BGraph/onnx_elementwise.py::compile_and_run()` 中构造的实际 lowering 参数列表。
- `scripts/benchmark.py::LOWERING`。

## 4. 调用链

```text
BGraph FullConversion
→ tensor.empty + linalg.generic / named linalg ops
→ one-shot-bufferize{bufferize-function-boundaries}
   → analysis: in-place/out-of-place decisions
   → tensor values / function boundaries → memref descriptors
   → tensor.empty destinations → allocations
→ convert-linalg-to-loops
→ lower-affine
→ convert-scf-to-cf
→ convert-cf/math/arith/memref/func-to-llvm
→ reconcile-unrealized-casts
→ LLVM Dialect module
→ mlir-runner JIT 或 mlir-translate → LLVM IR
```

## 5. IR 前后变化

### Tensor/DPS

```mlir
%empty = tensor.empty() : tensor<4xf32>
%result = linalg.generic ... outs(%empty : tensor<4xf32>) {
  ...
} -> tensor<4xf32>
```

`outs` 是 destination-passing style：结构化计算显式接收 init/output value。tensor
仍是 SSA 值，不代表原地可变数组。

### Bufferized

```mlir
%alloc = memref.alloc() : memref<4xf32>
linalg.generic ... outs(%alloc : memref<4xf32>) { ... }
```

One-Shot Bufferize 分析 alias/read-write，决定能否复用 buffer；不能原地时创建新
allocation。`tensor.empty` 本身没有数值初始化，真正初值由 fill/generic body定义。

### Loops / LLVM Dialect

Linalg loops 具体化为 induction/control-flow，加上 `memref.load/store`；MemRef
conversion 再变成 LLVM descriptor、pointer arithmetic、`llvm.load/store`：

```mlir
llvm.func @main() -> f32 {
  %ptr = llvm.call @malloc(...)
  ...
  %v = llvm.load %elementPtr : !llvm.ptr -> f32
  llvm.return %v : f32
}
```

### LLVM IR

`mlir-translate --mlir-to-llvmir` 后才是：

```llvm
define float @main() {
  ...
  ret float %value
}
```

LLVM Dialect 与 LLVM IR 使用不同 parser、type spelling 和 verification pipeline。

## 6. 核心机制

One-Shot Bufferize 需要各 Dialect 的 BufferizableOpInterface models。driver 逐一注册
external models；若承诺 interface 却未注册，bufferization 会报 promised interface
错误。

`bufferize-function-boundaries` 把 tensor function ABI 转成 memref-compatible ABI。
本项目 E2E 为了 runner 简化，用 importer `--scalar-return` 把唯一 rank-0 tensor
`tensor.extract` 成 f32；一般 tensor-return 函数需要 wrapper 或正确的 memref ABI。

当前 pipeline 会生成 `memref.alloc`，但没有运行 ownership-based deallocation 或
显式 `memref.dealloc` pass。测试/runner 是短进程，进程结束回收内存；教程不能据此
声称完整长期运行内存生命周期已解决。第 13 章的 alloc count 只描述 IR 结构。

标准 Linalg 文件已经没有 BGraph，可交给外部 tools 验证：

```bash
/buddy-mlir/llvm/build/bin/mlir-opt "$BUDDYGRAPH_TMP/bgraph-11-linalg.mlir" -o /dev/null
/buddy-mlir/build/bin/buddy-opt "$BUDDYGRAPH_TMP/bgraph-11-linalg.mlir" -o /dev/null
```

但项目当前仍用 `buddygraph-opt` 承载完整后续 pipeline，因为它已经注册必要 passes
和 bufferization interfaces。

## 7. 为什么这样设计

先在 tensor/Linalg 层完成图优化，避免每个 BGraph op 自己实现 buffer interface；
标准结构化 IR 能复用成熟 bufferization 和 LLVM conversions。独立 scalar-return
adapter 把数值 E2E 与一般 tensor ABI 问题分开。

## 8. 常见错误

- BGraph 尚未消除就 One-Shot Bufferize；BGraph 没有 buffer interface。
- 漏注册 external models，出现 promised interface failure。
- 把 `tensor.empty` 当全零；它只是未初始化 destination。
- 漏 `--convert-math-to-llvm`，BN 的 `math.sqrt` 留在 LLVM Dialect文件。
- 漏 `reconcile-unrealized-casts`，残留桥接 cast。
- 看到 `malloc` 就声称有 deallocation；当前没有对应 dealloc pass。
- 用 tensor return 直接套 `-entry-point-result=f32`。

## 9. 动手练习

对四个文件运行：

```bash
for f in "$BUDDYGRAPH_TMP/bgraph-11-linalg.mlir" \
         "$BUDDYGRAPH_TMP/bgraph-11-buffer.mlir" \
         "$BUDDYGRAPH_TMP/bgraph-11-llvm.mlir" \
         "$BUDDYGRAPH_TMP/bgraph-11.ll"; do
  echo "$f"
  rg -c 'tensor.empty|memref.alloc|linalg.generic|llvm.func|define .*@main' "$f" || true
done
```

解释每个计数为什么出现或消失。再查找 `dealloc|free`，区分 translator/runtime
declaration 与真正调用。

## 10. 验收标准

- 四层文件都成功生成，runner 输出 16。
- Linalg 文件可由外部 `mlir-opt` 和 `buddy-opt` 解析。
- 能解释 tensor value、DPS destination 和 memref buffer 的差异。
- 能准确说明当前 allocation/deallocation 边界和 runner ABI。

## 11. 面试追问

**问：tensor 如何变成 memref？**

答：One-Shot Bufferize 基于 read/write/alias 和 destination-style 信息选择原地或新
buffer，重写 tensor operands/results 和函数边界为 memref；随后低层 conversion
再把 descriptor 和 load/store 映射到 LLVM Dialect。

**问：LLVM Dialect 与 LLVM IR 有何区别？**

答：前者仍是 MLIR Dialect，可与其他 MLIR ops 共存并由 MLIR verifier/pass 处理；
后者是 LLVM 自身 IR，由 `mlir-translate` 导出。
