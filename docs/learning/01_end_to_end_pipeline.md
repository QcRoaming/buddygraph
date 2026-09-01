# 01｜端到端 Pipeline：每层 IR 到底是什么

## 1. 本章目标

你将保存同一模型在 BGraph、BN folding、fusion、Linalg、bufferized、LLVM Dialect
和 LLVM IR 七个阶段的快照，并能说出每层保留和消除了什么。

## 2. 先运行

```bash
cd /buddy-mlir/jlq/projects/buddygraph
export BUDDYGRAPH_TMP=/buddy-mlir/jlq/projects/buddygraph/tmp
mkdir -p "$BUDDYGRAPH_TMP/buddygraph-learning"
export PYTHONPATH="$PWD/.deps:/buddy-mlir/llvm/build/tools/mlir/python_packages/mlir_core${PYTHONPATH:+:$PYTHONPATH}"

python3 frontend/BGraph/generate_test_models.py "$BUDDYGRAPH_TMP/buddygraph-learning/models"
python3 frontend/BGraph/import_onnx.py \
  "$BUDDYGRAPH_TMP/buddygraph-learning/models/elementwise_constant.onnx" \
  --function-name main --scalar-return \
  -o "$BUDDYGRAPH_TMP/buddygraph-learning/00-imported.mlir"

build/bin/buddygraph-opt --bgraph-fold-bn-into-conv \
  "$BUDDYGRAPH_TMP/buddygraph-learning/00-imported.mlir" \
  -o "$BUDDYGRAPH_TMP/buddygraph-learning/01-bn.mlir"

build/bin/buddygraph-opt --bgraph-fuse-elementwise \
  "$BUDDYGRAPH_TMP/buddygraph-learning/00-imported.mlir" \
  -o "$BUDDYGRAPH_TMP/buddygraph-learning/02-fused.mlir"

build/bin/buddygraph-opt \
  --bgraph-fold-bn-into-conv --canonicalize --cse \
  --bgraph-fuse-elementwise --convert-bgraph-to-linalg \
  "$BUDDYGRAPH_TMP/buddygraph-learning/00-imported.mlir" \
  -o "$BUDDYGRAPH_TMP/buddygraph-learning/03-linalg.mlir"

build/bin/buddygraph-opt \
  --bgraph-fold-bn-into-conv --canonicalize --cse \
  --bgraph-fuse-elementwise --convert-bgraph-to-linalg \
  '--one-shot-bufferize=bufferize-function-boundaries' \
  "$BUDDYGRAPH_TMP/buddygraph-learning/00-imported.mlir" \
  -o "$BUDDYGRAPH_TMP/buddygraph-learning/04-buffer.mlir"

build/bin/buddygraph-opt \
  --bgraph-fold-bn-into-conv --canonicalize --cse \
  --bgraph-fuse-elementwise --convert-bgraph-to-linalg \
  '--one-shot-bufferize=bufferize-function-boundaries' \
  --convert-linalg-to-loops --lower-affine \
  --convert-scf-to-cf --convert-cf-to-llvm \
  --convert-math-to-llvm --convert-arith-to-llvm \
  --finalize-memref-to-llvm --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  "$BUDDYGRAPH_TMP/buddygraph-learning/00-imported.mlir" \
  -o "$BUDDYGRAPH_TMP/buddygraph-learning/05-llvm.mlir"

/buddy-mlir/llvm/build/bin/mlir-translate --mlir-to-llvmir \
  "$BUDDYGRAPH_TMP/buddygraph-learning/05-llvm.mlir" \
  -o "$BUDDYGRAPH_TMP/buddygraph-learning/06-llvm.ll"

/buddy-mlir/llvm/build/bin/mlir-runner \
  "$BUDDYGRAPH_TMP/buddygraph-learning/05-llvm.mlir" \
  -e main -entry-point-result=f32

python3 tests/E2E/BGraph/onnx_elementwise.py \
  "$PWD" "$PWD/build" /buddy-mlir/llvm/build/bin \
  "$BUDDYGRAPH_TMP/buddygraph-learning/e2e"
```

runner 预期输出 `7.000000e+00`；最后一条命令应直接打印 reference、unoptimized、
optimized 均为 `7.000000e+00` 和 `status=PASS`。再运行：

```bash
rg -c '"bgraph\.' "$BUDDYGRAPH_TMP/buddygraph-learning/00-imported.mlir"
rg -c '"bgraph\.' "$BUDDYGRAPH_TMP/buddygraph-learning/03-linalg.mlir" || true
```

第一项为 6，第二项无命中。

`01-bn.mlir` 与 `02-fused.mlir` 都直接由 `00-imported.mlir` 生成，是为了分别观察
单个 Pass 的**兄弟分支**，不是“先 BN 再 fusion”的连续阶段；`03-linalg.mlir` 才按
命令中列出的组合顺序执行两种优化并进入 FullConversion。

## 3. 真实代码位置

- `generate_test_models.py::generate()`：三个正向模型和三类拒绝模型。
- `import_onnx.py::BGraphImporter.import_module()`：ONNX → BGraph。
- `FoldBatchNormPattern::matchAndRewrite()`：Conv-BN-ReLU 中消除 BN。
- `FuseElementwisePattern::matchAndRewrite()`：Add-Relu-Mul 链融合。
- `ConvertBGraphToLinalg::runOnOperation()`：FullConversion。
- `tests/E2E/BGraph/onnx_elementwise.py::compile_and_run()`：组合模型执行。

三个真实教学输入：

1. `conv_bn_relu.onnx`：Conv→BN→Relu，展示 BN folding。
2. `tests/Dialect/BGraph/fuse-elementwise.mlir`：Add→Relu→Mul，展示 Region fusion。
3. `elementwise_constant.onnx`：Conv→BN→Add→Relu→Mul→ReduceMean，展示组合闭环。

指导中提到的 Add→Relu→Mul→Relu fixture 并不存在，教程不补造末尾 Relu。

## 4. 调用链

```text
generate()
→ onnx.save(ModelProto)
→ _validate_model()
→ onnx.checker + onnx.shape_inference
→ BGraphImporter.import_module()
→ func.FuncOp + arith.constant + bgraph.*
→ custom graph passes
→ applyFullConversion()
→ One-Shot Bufferize
→ convert-linalg-to-loops / SCF / CF / LLVM conversions
→ mlir-runner 或 mlir-translate
```

不要把它们统称为“模型代码”：ONNX 是交换格式；BGraph 是图语义 IR；Linalg 是
结构化计算 IR；MemRef/loops 是显式存储和控制流；LLVM Dialect 仍是 MLIR；`.ll`
才是 LLVM IR。

## 5. IR 前后变化

### ONNX graph

组合 fixture 的 node 顺序来自生成器：

```text
a,w,bias → Conv → BatchNormalization → Add(b) → Relu → Mul(factor)
         → ReduceMean(axes=[0,1,2,3]) → result
```

### BGraph IR

```mlir
%0 = "bgraph.conv2d"(...) {...} : (...) -> tensor<1x1x2x2xf32>
%1 = "bgraph.batch_norm"(%0, ...) {...} : (...) -> tensor<1x1x2x2xf32>
%2 = "bgraph.add"(%1, %b) : (...) -> tensor<1x1x2x2xf32>
%3 = "bgraph.relu"(%2) : (...) -> tensor<1x1x2x2xf32>
%4 = "bgraph.mul"(%3, %factor) : (...) -> tensor<1x1x2x2xf32>
%5 = "bgraph.reduce_mean"(%4) {axes = array<i64: 0, 1, 2, 3>,
                                keep_dims = false} : (...) -> tensor<f32>
```

图级 op、layout、axes 和 tensor shape 都保留。

### BN folding 后

```mlir
%new_w = arith.constant ...
%new_b = arith.constant ...
%0 = "bgraph.conv2d"(%a, %new_w, %new_b) {...}
```

`bgraph.batch_norm` 消失，BN 数学被吸收到新 weight/bias；Conv 仍保留。

### Fusion 后

```mlir
%fused = "bgraph.fused_elementwise"(%normalized, %b, %factor) ({
^bb0(%x: f32, %y: f32, %s: f32):
  %sum = arith.addf %x, %y : f32
  %zero = arith.constant 0.0 : f32
  %relu = arith.maximumf %sum, %zero : f32
  %scaled = arith.mulf %relu, %s : f32
  "bgraph.yield"(%scaled) : (f32) -> ()
}) : (...) -> tensor<1x1x2x2xf32>
```

Add/Relu/Mul 的中间 tensor op 被一个 Region op 取代。

### Linalg IR

```mlir
%conv = linalg.conv_2d_nchw_fchw ...
%fused = linalg.generic
    {indexing_maps = [...], iterator_types = ["parallel", ...]}
    ins(...) outs(%empty : tensor<1x1x2x2xf32>) { ... }
```

BGraph 消失，图语义具体化为 indexing maps、iterator types 和 Linalg body。
ReduceMean 实际降低为“reduction generic + 除法 generic”，不是单个 `linalg.reduce`。

### Bufferized、LLVM Dialect 与 LLVM IR

bufferization 后 `tensor.empty`/tensor results 变成 `memref.alloc` 和 memref operands；
loops lowering 后是控制流、load/store；LLVM conversion 后是 `llvm.func`、
`llvm.getelementptr`、`llvm.load/store`。本章已生成这些文件，第 11 章再逐层解释其机制。

## 6. 核心机制

每层都有自己的 correctness contract：

- importer 检查 opset/domain/dtype/常量属性并建立 SSA。
- verifier 保证单个 BGraph op 局部自洽。
- graph passes 只在前置条件成立时改变 use-def graph。
- FullConversion 通过 legality 保证没有 BGraph 泄漏。
- bufferization 把值语义具体化为内存语义。
- LLVM lowering 解决低层 ABI 和可执行表示。

## 7. 为什么这样设计

直接 ONNX→Linalg 会让 BN folding/fusion 必须从 indexing maps 反推图关系，也让诊断
失去 ONNX 节点级语义。BGraph 提供可验证的中间契约；FullConversion 又避免把高层
op 留给不理解它的后端。

## 8. 常见错误

- 同时对不同输入文件比较 IR，误把模型差异当作 Pass 效果。
- 只看 op 数就推断 runtime speedup。
- 把 `llvm.func` 文件称作 LLVM IR；它还是 LLVM Dialect MLIR。
- 认为 BN folding 总会触发；参数非常量或 Conv 多用户时必须保留 BN。
- 认为 fusion 只支持同 shape；实际实现支持**静态**右对齐广播，但 dynamic shape
  会在 lowering 拒绝。

## 9. 动手练习

为七个快照建立一张表，每行写：剩余 BGraph op、`linalg.generic` 数、是否出现
`memref.alloc`、是否能直接交给 `mlir-runner`。先预测，再用 `rg` 验证。

## 10. 验收标准

- 能展示 00–06 七个快照并解释关键 diff。
- FullConversion 文件中 `rg 'bgraph\.'` 无输出。
- 能回答“BN、fusion、ReduceMean 分别变成了什么”。
- 组合 E2E 运行后 reference/unoptimized/optimized 均为 7.0。

## 11. 面试追问

**问：哪一步真正消除了中间 tensor？**

答：fusion 在 BGraph 层把 Add/Relu/Mul 的 tensor results 变成一个 fused op 的单一
result；lowering 再把 Region 映射为一个 `linalg.generic`，bufferization 因而少物化
多个 destination buffer。

**问：FullConversion 与“打印结果里没看到 BGraph”有什么区别？**

答：前者是由 `ConversionTarget` 和 `applyFullConversion` 强制的程序性质；漏掉
pattern 会失败，而不是依赖人工观察某个样例。
