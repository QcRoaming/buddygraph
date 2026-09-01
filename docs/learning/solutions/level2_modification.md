# Level 2 答案：小修改

## 1. verifier 与 negative test

`BatchNormOp` 已有 custom verifier。新增检查只应在 variance 可静态证明为 dense constant
时触发，概念骨架如下：

```cpp
Attribute attribute;
if (matchPattern(getVariance(), m_Constant(&attribute)))
  if (auto values = dyn_cast<DenseFPElementsAttr>(attribute))
    for (APFloat value : values.getValues<APFloat>())
      if (value.isNegative() || !value.isFinite())
        return emitOpError("constant variance must be finite and nonnegative");
```

测试使用 `// RUN: buddygraph-opt %s -verify-diagnostics`，在非法 Op 上方写与当前 MLIR
实际输出匹配的 `expected-error`。要同时测试正 constant 和函数参数 variance，避免把
“无法静态证明合法”错误收紧成“非法”。APFloat 的确切 finite API 以本地头文件为准。

## 2. identity canonicalization

`DivOp` 已声明 canonicalizer，但注册函数为空。增加一个 `OpRewritePattern<DivOp>`，依次
检查：

1. 右操作数 defining op 是 `arith::ConstantOp`；
2. value 是 `DenseElementsAttr`；
3. attribute 是 splat 且浮点 splat 为一；
4. 左操作数类型与结果类型完全相同。

满足后 `rewriter.replaceOp(op, op.getLhs())`。除法没有交换律，不能匹配
`1 / x`。不要仅按 SSA 名称判断，也不要在需要 broadcast 时返回 shape 不同的输入。
在 `DivOp::getCanonicalizationPatterns` 中注册；正例 CHECK-NOT Div，反例 CHECK Div。

## 3. Pass option

在 `Passes.td` 的 shape pass 定义中加入当前 TableGen 版本支持的 `Option`，概念形态为：

```tablegen
Option<"failOnDynamic", "fail-on-dynamic", "bool", "false",
       "Fail when inferred BGraph results retain dynamic dimensions">
```

实现中，在完成既有推断后遍历 BGraph 结果的 `RankedTensorType`；`!hasStaticShape()` 时
对对应 Op 发诊断并 `signalPassFailure()`。关键验收是默认 false 不改变原 pipeline。
确切生成字段/API 应查看重建后的 `build/include/BuddyGraph/Transforms/Passes.h.inc`，但
不要编辑该文件。

## 4. IR dump 检查

可增加一条独立 RUN（具体输入 split 依现有测试结构调整）：

```mlir
// RUN: buddygraph-opt %s -bgraph-fold-bn-into-conv \
// RUN:   -mlir-print-ir-after=bgraph-fold-bn-into-conv 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DUMP
// DUMP: IR Dump After
// DUMP-LABEL: func.func @fold
// DUMP: bgraph.conv2d
// DUMP-NOT: bgraph.batch_norm
// DUMP-LABEL: func.func @multiple_users
// DUMP: bgraph.batch_norm
```

先手工执行整条命令，因为 banner 大小写和 pass 显示名由当前 MLIR 版本决定；再固定最小
稳定子串。第二个 LABEL 同时结束第一个 `DUMP-NOT` 的检查范围；不要匹配地址、耗时或
临时路径。
