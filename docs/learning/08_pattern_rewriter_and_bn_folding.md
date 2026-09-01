# 08｜PatternRewriter 与 Conv-BN folding

## 1. 本章目标

你将按十个步骤解释真实 BN Pattern，手算新 weight/bias，验证正向和至少五个不命中
条件，并知道重写失败前后为什么不能留下半成品 IR。

## 2. 先运行

```bash
cd /buddy-mlir/jlq/projects/buddygraph
export BUDDYGRAPH_TMP=/buddy-mlir/jlq/projects/buddygraph/tmp
mkdir -p "$BUDDYGRAPH_TMP"
build/bin/buddygraph-opt --bgraph-fold-bn-into-conv \
  tests/Dialect/BGraph/fold-bn.mlir | \
  /buddy-mlir/llvm/build/bin/FileCheck tests/Dialect/BGraph/fold-bn.mlir

build/bin/buddygraph-opt --bgraph-fold-bn-into-conv \
  tests/Dialect/BGraph/fold-bn.mlir -o "$BUDDYGRAPH_TMP/fold-bn.after.mlir"
rg 'bgraph.batch_norm|dense<\[-5.000000e-01, -2.500000e\+00\]>|bgraph.conv2d' \
  "$BUDDYGRAPH_TMP/fold-bn.after.mlir"
```

第一个 `@fold` 不再含 BN，新 bias 是 `[-0.5, -2.5]`；`@multiple_users` 中 BN 保留。

再构造两个**合法但不命中**的最小输入。第一例让 scale 经过 Relu，因此不再是 Pattern
可直接读取的 constant；第二例让 `variance+epsilon <= 0`：

```bash
sed -n '/^func.func @fold(/,/^}/p' tests/Dialect/BGraph/fold-bn.mlir | \
  sed '/  %conv = "bgraph.conv2d"/i\  %dynamic_scale = "bgraph.relu"(%scale) : (tensor<2xf32>) -> tensor<2xf32>' | \
  sed 's/(%conv, %scale, %beta/(%conv, %dynamic_scale, %beta/' \
  > "$BUDDYGRAPH_TMP/fold-bn-nonconstant.mlir"

sed -n '/^func.func @fold(/,/^}/p' tests/Dialect/BGraph/fold-bn.mlir | \
  sed 's/dense<\[3.0, 15.0\]>/dense<[-2.0, 15.0]>/' \
  > "$BUDDYGRAPH_TMP/fold-bn-bad-denominator.mlir"

build/bin/buddygraph-opt --bgraph-fold-bn-into-conv \
  "$BUDDYGRAPH_TMP/fold-bn-nonconstant.mlir" \
  -o "$BUDDYGRAPH_TMP/fold-bn-nonconstant.after.mlir"
build/bin/buddygraph-opt --bgraph-fold-bn-into-conv \
  "$BUDDYGRAPH_TMP/fold-bn-bad-denominator.mlir" \
  -o "$BUDDYGRAPH_TMP/fold-bn-bad-denominator.after.mlir"
rg -c 'bgraph.batch_norm' "$BUDDYGRAPH_TMP"/fold-bn-*.after.mlir
```

两个手工输出的计数都应为 1。测试文件还固定了 `@multiple_users`、
`@layout_mismatch` 和 `@epsilon_rounding_boundary`，本章共有五个可复现的不命中条件。

## 3. 真实代码位置

- `FoldBatchNorm.cpp::getDenseF32()`、`asFloats()`。
- `FoldBatchNormPattern::matchAndRewrite()`。
- `BGraphFoldBatchNormIntoConv::runOnOperation()`。
- `createFoldBatchNormIntoConvPass()`。
- `tests/Dialect/BGraph/fold-bn.mlir`。
- `tests/E2E/BGraph/onnx_elementwise.py`：数值级证明。

## 4. 调用链

```text
--bgraph-fold-bn-into-conv
→ generated pass factory registration
→ createFoldBatchNormIntoConvPass()
→ BGraphFoldBatchNormIntoConv::runOnOperation()
→ RewritePatternSet.add<FoldBatchNormPattern>()
→ applyPatternsGreedily(ModuleOp)
→ 对每个 BatchNormOp 调用 matchAndRewrite()
→ success：替换；notifyMatchFailure：保持合法 IR
```

真实 Pattern 的十步：

1. root 是 `BatchNormOp`。
2. `getDefiningOp<Conv2DOp>()` 找直接 producer。
3. `conv.getResult().hasOneUse()` 防止删除仍被其他节点使用的 Conv。
4. 检查 groups=1，且 Conv 与 BN 的 layout 完全一致。
5. `getDenseF32()` 通过 `m_Constant` 读取 filter 与四个 BN 参数。
6. 检查 channel 与 element counts；可选 Conv bias 缺失时用零向量。
7. `asFloats()` 转成 owning `SmallVector<float>`，计算 alpha/new bias/new filter。
8. 所有检查完成后才创建新 filter/bias constants。
9. 用 `OperationState` 创建新 Conv，attributes 复制自旧 Conv，result type 取 BN。
10. `replaceOp(batchNorm, newConv.result)`，再 `eraseOp(conv)`。

## 5. IR 前后变化

数学公式：

```text
alpha[c] = gamma[c] / sqrt(variance[c] + epsilon)
W'[c,...] = W[c,...] * alpha[c]
b'[c] = beta[c] + (b[c] - mean[c]) * alpha[c]
```

测试常量为旧 filter `[1, 2]`、`gamma=[2, 4]`、`beta=[0.5, -0.5]`、
`mean=[1, 2]`、`variance=[3, 15]`、`epsilon=1`，旧 Conv 无 bias：

- channel 0：`alpha=2/sqrt(3+1)=1`，新 filter `1×1=1`，
  新 bias `0.5+(0-1)×1=-0.5`；
- channel 1：`alpha=4/sqrt(15+1)=1`，新 filter `2×1=2`，
  新 bias `-0.5+(0-2)×1=-2.5`。

```mlir
// before
%conv = "bgraph.conv2d"(%input, %filter) ...
%bn = "bgraph.batch_norm"(%conv, %scale, %beta, %mean, %variance) ...

// after
%new_filter = arith.constant ...
%new_bias = arith.constant dense<[-0.5, -2.5]> : tensor<2xf32>
%new_conv = "bgraph.conv2d"(%input, %new_filter, %new_bias) ...
```

新 constants 和 Conv 使用 `rewriter.getFusedLoc({conv.loc, bn.loc})` 保留来源组合。

## 6. 核心机制

PatternRewriter 的修改纪律是“先匹配/验证，后 mutation”。`notifyMatchFailure()` 不
表示 IR 非法，只给 debug trace 一个原因。创建新 ops 后若返回 failure，driver 并不
自动理解任意手工副作用；因此代码把所有可失败数值检查放在创建之前。

DenseElementsAttr 是 immutable Context-owned attribute。Pattern 把 payload 复制为
`SmallVector<float>` 后修改，不会改共享旧常量；再创建新的 DenseElementsAttr。

BN lowering 把 f64 attribute 中的 epsilon 物化为 f32 常量，因此 folding 也先把
epsilon 舍入为 f32，再做 `variance + epsilon` 检查和计算。这样避免 folding 接受
而直接 lowering 在 f32 下分母为零的边界。folding 会把 BN 运算重结合进 Conv
weight/bias，E2E 因而按 `1e-6` 容差验证数值，不承诺逐 bit 等价。

`replaceOp` 重定向 BN 的所有 uses，`eraseOp(conv)` 只在旧 Conv result 已因 single-use
且 BN 被替换后无用户时合法。

## 7. 为什么这样设计

BN 是图级语义，BGraph 层能直接识别 Conv→BN、constants、channel 和 layout；降低到
Linalg 后再做会需要分析 generic body/indexing maps。严格前置条件把优化限定在能
证明等价的 inference case。

## 8. 常见错误

五个应保持原图的反例：

1. Conv 多用户：实际 `@multiple_users`，不能删除/替换共享 producer。
2. filter 或 BN 参数不是 Dense f32 constant：Pattern 无法离线计算新 payload。
3. `variance + epsilon <= 0` 或非有限：sqrt/alpha 无合法实数计算。
4. Conv 与 BN layout 不一致：同一 shape 下 channel 轴也可能不同，不能误折叠。
5. f64 epsilon 舍入到 f32 后使分母非正：必须按实际 lowering 精度拒绝。

其他错误：parameter length 与 output channels 不同；Conv bias 不是合法 constant；
在检查结束前创建 constants；忘记旧 Conv 可能无 bias；用 f64 计算 folding，却让
直接 lowering 使用 f32 epsilon。当前实现只显式证明 f32 `variance+epsilon` 正且有限，没有逐项拒绝
非有限 gamma/beta/mean/filter 或证明计算后 weight/bias 有限；这是一条实现边界，不能
扩写成更强的数值 contract。当前端到端数值 fixture 只覆盖 NCHW。

groups!=1 不是一个“合法但不命中”用例，因为当前 Conv verifier 本身拒绝 groups!=1；
Pattern 中的 groups check 是防御性边界。

## 9. 动手练习

阅读题：为测试 channel 1 手算 alpha、新 filter 的两个值与新 bias `-2.5`。

调试题：

```bash
gdb --args build/bin/buddygraph-opt \
  --bgraph-fold-bn-into-conv tests/Dialect/BGraph/fold-bn.mlir
```

在 `buddy::bgraph::(anonymous namespace)::FoldBatchNormPattern::matchAndRewrite`、
`getDenseF32` 和 `PatternRewriter::replaceOp` 附近下断点。若 optimized build 使局部变量
不可见，先用函数断点观察调用栈，不要为教程修改核心编译选项。

## 10. 验收标准

- 正向 FileCheck 通过，`@multiple_users` 保留 BN。
- 能手算两个 channel 的 alpha/filter/bias。
- 能列出五个不命中条件和各自保护的错误。
- E2E 优化前/后与 NumPy 均为 7.0。

## 11. 面试追问

**问：为什么要求 Conv 单用户？**

答：Pattern 最终删除旧 Conv。若还有其他用户，它们需要原始未折叠输出；直接删除或
全局换成 folded Conv 会改变分支语义。

**问：Pattern failure 后如何保证没有部分修改？**

答：代码在创建任何新 op 前完成 producer/use/constant/shape/数值检查；一旦开始
mutation，后续路径不再返回 match failure。
