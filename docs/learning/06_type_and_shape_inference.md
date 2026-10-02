# 06｜Type、shape refinement 与 verifier 边界

> **本章路线：计算与检查 shape。** 先区分前端推导、verifier 兼容检查和 C++ pass 重算，再手算广播/Conv/Reduce。重点回答：导入器已提供静态类型，为什么还保留 shape pass？

## 1. 本章目标

你将手算 broadcast、Conv 和 ReduceMean shape，运行 shape pass 收紧动态结果，并能
指出当前实现为什么是自定义 Module pass、而不是 `InferTypeOpInterface`。

## 2. 先运行

```bash
cd /home/jlq/project/buddygraph
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

本章中的 Type 是 builtin ranked tensor，Interface 是尚未采用的改进方向。要亲手定义
parameterized TypeDef、注册 `addTypes` 并实现 custom OpInterface，请完成
[专题第 03–05 章](dialect_lab/README.md)。专题使用 metadata Type，避免把教学机制
误接进当前 BGraph tensor lowering。

Conv 空间公式在 verifier 与 shape pass 中都使用 checked add/multiply。静态 kernel
大于 padded input、零 kernel 或整数溢出会返回 failure；只有合法的正输出维才用于
构造 `RankedTensorType`，因此非法输入不会再以 assertion/abort 结束。

### 用同一个 Add 串起四处 shape 边界

设输入分别为 `tensor<2x1xf32>`、`tensor<1x3xf32>`。右对齐逐维比较：第一维由 2 与 1
得到 2，第二维由 1 与 3 得到 3，期望结果是 `[2,3]`。

| 阶段 | 它拿到什么 | 它做什么 |
|---|---|---|
| ONNX inference | protobuf graph 与 shapes | 把输出静态 shape 写入推导后的模型信息 |
| Python `_tensor_type` | 结果 value name | 取出 `[2,3]` 并构造 RankedTensorType |
| BGraph verifier | operands type、声明 result type | 计算期望 shape，检查声明是否兼容，不修改 IR |
| `bgraph-infer-shapes` | 合法 BGraph IR | `inferShape` 重算 shape，调用 result `setType` |
| elementwise lowering | 最终 operands/results type | 检查静态 shape，再生成访问 maps |

为什么第三、四步还需要？手写或其他前端生成的 IR 可以声明动态结果，优化也可能产生
待处理类型；标准 ONNX importer 的已静态结果则通常不会因此获得额外信息。

以下是完整的手写输入，可另存于项目 tmp 后使用第 2 节同一 pass：

```mlir
func.func @refine(%a: tensor<2x1xf32>, %b: tensor<1x3xf32>) {
  %0 = "bgraph.add"(%a, %b)
      : (tensor<2x1xf32>, tensor<1x3xf32>) -> tensor<?x?xf32>
  return
}
```

预期验证通过；shape pass 后 `%0` 为 `tensor<2x3xf32>`。这里故意使用 void 函数，
以单独观察 result type。再把声明改成 `tensor<2x4xf32>`，应在 pass 运行前被 verifier
拒绝；“声明未知”与“声明矛盾”是不同情况。

### 本项目 pass 的实现上限

当前是一次 walk 中对已识别 Op 直接重设 result type；不是维护约束 lattice 的固定点
求解器，也不自动转换函数签名。源码没有将旧 result 知识与新推导结果做一般的单调合并。
因此不能概括为“永远只会把 ? 变静态”：动态输入也可能推导出更弱的结果信息。
若函数返回类型与被改写的 Value 不一致，pass 后验证还可能失败。

理解这些限制不要求这轮去重写 pass。面试中应说它解决了当前测试中的局部 shape
refinement；若扩展跨函数/控制流推导，需要设计传播顺序、固定点、边界类型更新与冲突诊断。

### 从手算走到源码检查

广播处理缺失前导维时视为 1；维为 1 只表示索引恒为 0，不是分配更大的输入 tensor。
Conv 先验证 rank/layout，再确定空间维轴、有效 kernel 和输出；ReduceMean 先归一化
负 axes，再决定哪些维归约、是否保留 size-1 维。把这些步骤逐项写出，能避免“公式会背，
但不知道 `NCHW` 改动后哪个数组索引需要变”的问题。

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
`/home/jlq/project/buddygraph/tmp`，把声明结果改成静态错误
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
