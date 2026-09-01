# 06｜Type、shape refinement 与 verifier 边界

## 1. 本章目标

你将手算 broadcast、Conv 和 ReduceMean shape，运行 shape pass 收紧动态结果，并能
指出当前实现为什么是自定义 Module pass、而不是 `InferTypeOpInterface`。

## 2. 先运行

```bash
cd /buddy-mlir/jlq/projects/buddygraph
build/bin/buddygraph-opt --bgraph-infer-shapes \
  tests/Dialect/BGraph/infer-shapes.mlir | \
  rg 'func.func @|tensor<2x3xf32>|tensor<6x4xf32>|tensor<4x2x3xf32>|tensor<1x4x3x3xf32>'
```

对比原文件中的 `tensor<?...>` 与输出的静态结果。

## 3. 真实代码位置

- `BGraphOps.cpp::inferBroadcastShape()`、`inferReshapeShape()`、
  `inferReduceShape()`：verifier 侧 helper。
- `InferShapes.cpp::broadcast()`、`inferShape()`：Pass 侧推导。
- `BGraphInferShapes::runOnOperation()`：walk 并 `setType()`。
- `Passes.td::BGraphInferShapes`：Module pass 声明。
- `tests/Dialect/BGraph/infer-shapes.mlir`：五类 shape refinement。

## 4. 调用链

```text
PassManager
→ BGraphInferShapes::runOnOperation()
→ ModuleOp::walk(Operation *)
→ 跳过非 bgraph 和 YieldOp
→ inferShape(operation)
→ dyn_cast<具体 Op>
→ 读取 operands type / attributes
→ FailureOr<SmallVector<int64_t>>
→ result(0).setType(RankedTensorType::get(...))
→ 失败则 emitError + signalPassFailure
```

这不是 Interface 调用链。虽然 headers/`.td` include 了 InferType interface 定义，
没有任何 BGraph Op 声明相应 interface trait，也没有实现 `inferReturnTypes()`。

## 5. IR 前后变化

广播：

```mlir
// before
"bgraph.add" : (tensor<2x1xf32>, tensor<1x3xf32>) -> tensor<?x?xf32>
// after
"bgraph.add" : (tensor<2x1xf32>, tensor<1x3xf32>) -> tensor<2x3xf32>
```

Conv 手算：输入 NCHW `[1,2,5,5]`，filter FCHW `[4,2,3,3]`，pads 都为 1，
stride `[2,2]`，dilation `[1,1]`：

```text
effective_kernel = 1 × (3 - 1) + 1 = 3
Hout = floor((5 + 1 + 1 - 3) / 2) + 1 = 3
Wout = 3
result = [N, F, Hout, Wout] = [1,4,3,3]
```

ReduceMean 输入 `[2,3,4]`、axes `[0,2]`：`keep_dims=false` 得 `[3]`；若为 true，
则得 `[1,3,1]`。

Reshape `[2,3,4]` 到 `[6,-1]`：总元素 24，已知乘积 6，所以 `-1` 推为 4。

## 6. 核心机制

builtin ranked tensor 同时携带 rank、每维大小和 element type。`?` 是 dynamic
dimension，不等于 unranked。verifier 允许“与已知事实不矛盾”的动态结果；shape
pass 利用 inputs/attrs 具体化。两者职责是：

- verifier：拒绝静态矛盾，如 `[2x3] + [4]` 或错误 result shape。
- inference/refinement：把合法但不具体的 `?` 收紧。

binary broadcast 从尾维右对齐；相等或一方为 1 合法。实现还容许 dynamic dim，
但 BGraphToLinalg 的 `createElementwiseGeneric()` 要求所有 shape 静态，因此 dynamic
只能存在于高层分析阶段，不能完成当前 CPU 闭环。

当前 pass 直接 `setType()`，没有 `TypeConverter`；它是在同一 builtin tensor type
体系内做 refinement。

Conv 空间公式在 verifier 与 shape pass 中都使用 checked add/multiply。静态 kernel
大于 padded input、零 kernel 或整数溢出会返回 failure；只有合法的正输出维才用于
构造 `RankedTensorType`，因此非法输入不会再以 assertion/abort 结束。

## 7. 为什么这样设计

Module pass 易于集中展示全部 Op 的 shape 规则，也适配当前学习项目。Interface 的
优势是 builder、通用 inference 框架和其他 passes 能按统一协议调用具体 op；将来
扩展时可把 `inferShape()` 分散到各 op interface implementation，避免 verifier/pass
规则重复。当前文档必须把它标作改进方向，而不是已实现能力。

## 8. 常见错误

- 把 `?` 当任意值，忽略它仍受 rank 和其他静态维约束。
- Conv 公式忘记 dilation 的 effective kernel 或 padding 前后两侧。
- Reduce axes 为负时未归一化，或 axes 重复。
- Reshape 有多个 `-1`，verifier 应先拒绝。
- 直接运行 FullConversion 处理动态 tensor，得到 `requires statically shaped tensors`。
- 声称项目实现了 `InferTypeOpInterface`；代码没有该 trait/方法。

## 9. 动手练习

手算：输入 `[1,3,8,8]`、filter `[6,3,3,3]`、pads `[1,1,1,1]`、stride `[2,2]`、
dilation `[2,2]` 时输出 shape 是什么？写出 effective kernel 和每一步。

调试：复制 `infer-shapes.mlir` 的 Conv 到项目
`/buddy-mlir/jlq/projects/buddygraph/tmp`，把声明结果改成静态错误
`tensor<1x4x4x4xf32>`，先不跑 shape pass只验证，记录 verifier 诊断。

## 10. 验收标准

- 五个 shape tests 得到预期静态类型。
- 能手算本章练习（见[独立答案](solutions/chapter06_shape_calculation.md)）。
- 能用一例解释“verifier 通过但 inference 仍有工作”。
- 能用源代码证明当前不是 Interface 实现。

## 11. 面试追问

**问：为什么 shape inference 不等于 verifier？**

答：verifier 判断当前类型是否可能合法；inference 从 operands/attrs 计算更具体的
result。一个 `tensor<?x?xf32>` 可以合法，但仍可收紧为 `tensor<2x3xf32>`。

**问：为何当前 lowering 拒绝 dynamic shape？**

答：实现用静态 `tensor.empty` shape 和静态 broadcast affine maps，未实现动态
dimension materialization；前端也选择严格 static MVP。
