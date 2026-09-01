# 02｜从 BuddyGraph 代码学习 MLIR C++ 基础

## 1. 本章目标

你将能逐行解释一个 verifier helper 和一个 Region 构造片段中的 MLIR/LLVM C++
类型、ownership、失败传播与 insertion point，而不是只认出 API 名。

## 2. 先运行

```bash
cd /buddy-mlir/jlq/projects/buddygraph
rg -n 'FailureOr<|LogicalResult|SmallVector|ArrayRef|StringRef|IRMapping|InsertionGuard' \
  lib/BuddyGraph
```

选择输出中的 `getF32Tensor()`、`ElementwiseTree::collect()` 和
`FusedElementwiseLowering::matchAndRewrite()`，先标出输入、输出和对象所有者。

## 3. 真实代码位置

- `BGraphOps.cpp::getF32Tensor(Operation *, Type, StringRef)`。
- `InferShapes.cpp::inferShape(Operation *)`。
- `FoldBatchNorm.cpp::getDenseF32(Value)` 与 `asFloats(...)`。
- `FuseElementwise.cpp::ElementwiseTree`、`emitScalar()`。
- `BGraphToLinalg.cpp::createElementwiseGeneric()`。
- `buddygraph-opt.cpp::main()` 中的 `DialectRegistry`。
- Python 侧 `BGraphImporter.context` 展示显式 `MLIRContext` 生命周期。

## 4. 调用链

以 binary verifier 为例：

```text
Operation::verify()
→ generated AddOp verifier glue
→ AddOp::verify()
→ verifyBinary<AddOp>(op)
→ getF32Tensor() × 3
→ inferBroadcastShape()
→ success() / emitOpError() → LogicalResult
```

以 fusion builder 为例：

```text
FuseElementwisePattern::matchAndRewrite(root, rewriter)
→ OperationState + rewriter.create()
→ rewriter.createBlock(region)
→ block->addArgument()
→ OpBuilder::InsertionGuard
→ setInsertionPointToStart(block)
→ emitScalar()
→ rewriter.create<YieldOp>()
→ replaceOp() / eraseOp()
```

## 5. IR 前后变化

以下 C++：

```cpp
BlockArgument argument = block->addArgument(elementType, location);
mapping.map(leaf, argument);
```

建立的是“外层 tensor leaf → 内层 scalar block argument”的对应关系：

```mlir
"bgraph.fused_elementwise"(%tensor) ({
^bb0(%scalar: f32):
  ...
}) : (tensor<2x3xf32>) -> tensor<2x3xf32>
```

`Value` 不是 owning tensor 对象；它是指向 IR definition 的轻量句柄。`%tensor` 的
definition 由函数 argument 或 producer op 所有，`%scalar` 由 Block 所有。

## 6. 核心机制

### Context、Registry 与唯一化

`DialectRegistry` 保存可加载 Dialect 的构造信息，`MLIRContext` 管理加载的
Dialect，以及 Type/Attribute 的 uniquing。C++ driver 把 registry 交给
`MlirOptMain()`；Python importer 显式创建 `ir.Context()`。`Type`、`Attribute`、
`Value` 通常是可复制的轻量句柄，不应按独立 heap object 理解。

### Operation、Value、OpOperand

- `Operation *` 是通用节点；`AddOp` 是生成的 typed wrapper。
- `OpResult` 是 Operation 定义的 Value；`BlockArgument` 是 Block 定义的 Value。
- operand 存储为 `OpOperand`，它把 user 与被使用的 `Value` 连起来；
  `hasOneUse()` 正是沿这张 use-list 判断。

### view 与 owning container

- `StringRef role`：non-owning 字符 view；这里来自字符串字面量，生命周期安全。
- `ArrayRef<int64_t>`：non-owning 连续数组 view；不能返回指向局部临时容器的 view。
- `SmallVector<int64_t>`：owning container，小尺寸时避免 heap allocation。
- C++ 中没有使用 `DenseMap`；ONNX name mapping 实际是 Python `dict`。不要为满足
  术语表而声称代码用了 `DenseMap`。

### 失败类型

- `LogicalResult`：只有成功/失败，适合 verifier、rewrite、pass driver。
- `FailureOr<T>`：失败或一个值，如 `FailureOr<RankedTensorType>`。
- `dyn_cast<T>`：类型不匹配返回空；`cast<T>` 假定已成立，失败会 assert。
- `isa<T>`：只做类型判断。

### RAII 与 insertion point

`PatternRewriter` 继承 builder 能力。`OpBuilder::InsertionGuard guard(rewriter)` 在作用
域结束时恢复原 insertion point，即使中途失败也不会把后续 op 插到 fused Region
里。这是 RAII，不是手工 save/restore。

### ownership

Region 拥有 Block，Block 拥有其中的 Operations 和 block arguments，Operation
拥有自己的 Regions/results/attributes。`Operation *`、`Value`、`ArrayRef` 都不拥有
这些 IR；删除 op 后不能继续使用旧 wrapper/value。

## 7. 为什么这样设计

MLIR 需要频繁遍历和重写大图，轻量 handle 和 arena/Context 管理能减少对象复制与
分配。LLVM ADT 则把常见“小数组、只读 view、集合保持顺序”等需求表达得比
`std::vector`/裸指针组合更清楚。

## 8. 常见错误

- `cast<RankedTensorType>` 用在未经 verifier 保证的类型上，触发 assertion。
- 返回指向局部 `SmallVector` 的 `ArrayRef`，产生悬空 view。
- 在 `rewriter.eraseOp(op)` 后继续访问 `op`。
- 构建 Region 时忘记切换 insertion point，scalar op 落在父 Block。
- 把 `failure()` 当异常；它只是显式控制流，调用者必须传播。

## 9. 动手练习

阅读 `FoldBatchNormPattern::matchAndRewrite()`，为下列对象标注 owning/non-owning：
`batchNorm`、`conv`、`filterAttr`、`SmallVector<float> filter`、`ArrayRef<float>(filter)`、
`newFilter.getResult()`。说明 `DenseElementsAttr::get()` 为什么不会在函数返回后引用
局部 `filter` 内存。

## 10. 验收标准

- 能画出 Operation-result-OpOperand-user 四者关系。
- 能解释 `FailureOr<T>` 与 `LogicalResult` 的选择。
- 能指出本项目一处安全 `StringRef`、一处 `ArrayRef` 和一处 RAII。
- 能说明 `IRMapping` 映射的是 Value，而不是复制 tensor payload。

## 11. 面试追问

**问：MLIR typed Op 是否继承 Operation？**

答：通常不是传统 owning subclass；ODS 生成的是围绕 `Operation *` 的 typed wrapper，
提供静态名称、typed accessor、builder/verifier glue。

**问：为什么 rewrite 返回 `LogicalResult` 而不是 bool？**

答：它与 MLIR failure propagation、diagnostic 和 pattern driver API 一致；
`notifyMatchFailure` 还能附带未匹配原因。

