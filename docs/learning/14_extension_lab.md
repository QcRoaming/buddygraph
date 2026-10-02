# 14｜完整扩展实录：ONNX `Clip` → `bgraph.clamp`

> 状态：Clamp 已实现。本章沿当前源码复盘一次真实的新 Op 全生命周期扩展，并把它作为后续扩展的模板。

> **本章路线：用增量证明独立实现能力。** 先复盘已实现 Clamp 的生命周期，再在实验副本做一个有界增量。关键产物是自己写的契约、代码 diff、正反测试与调试解释，不是重跑既有 E2E。

## 1. 本章目标

你将从 ONNX `Clip` 入口追到 ODS、verifier、shape refinement、canonicalization、
fusion、Linalg lowering 和数值 E2E，理解为什么“定义一个 Op”只是扩展工作的第一层。

## 2. 先运行

```bash
cd /home/jlq/project/buddygraph
export BUDDYGRAPH_TMP="$PWD/tmp"
mkdir -p "$BUDDYGRAPH_TMP/ch14"
export PYTHONPATH="$PWD/.deps:/buddy-mlir/llvm/build/tools/mlir/python_packages/mlir_core${PYTHONPATH:+:$PYTHONPATH}"

rg -n 'Clamp|clamp|Clip' include lib frontend tests
cmake --build build --target check-buddygraph -j2
python3 tests/E2E/BGraph/onnx_clamp.py \
  "$PWD" "$PWD/build" /buddy-mlir/llvm/build/bin \
  "$BUDDYGRAPH_TMP/ch14/clamp-e2e"
```

回归应为 14/14；E2E 应打印 direct/fused 两条路径的 `5.000000e-01`、零误差、
NaN 传播、`fusion=verified finite=true nan=true` 和 `status=PASS`。E2E 在 lowering
前断言有限值与 NaN 两个 fixture 都真正形成 fused op，而不只依赖最终数值。所有
中间文件都在项目内的 `tmp/ch14/`。

## 3. 真实代码位置

| 生命周期 | 文件 | 当前实现 |
|---|---|---|
| ODS | `include/BuddyGraph/IR/BGraphOps.td` | `BGraph_ClampOp` schema |
| verifier/canonicalization | `lib/BuddyGraph/IR/BGraphOps.cpp` | bounds、shape 和 nested Clamp 规则 |
| shape | `lib/BuddyGraph/Transforms/InferShapes.cpp` | result shape = input shape |
| fusion | `lib/BuddyGraph/Transforms/FuseElementwise.cpp` | fusible 分类与 scalar compare/select |
| importer | `frontend/BGraph/import_onnx.py` | ONNX `Clip` 常量 bounds |
| fixtures | `frontend/BGraph/generate_test_models.py` | 正向、NaN、共享 initializer 和拒绝模型 |
| conversion | `lib/BuddyGraph/Conversion/BGraphToLinalg.cpp` | `ClampLowering` |
| tests | `tests/Dialect/BGraph/`、`tests/Frontend/BGraph/`、`tests/Conversion/BGraphToLinalg/`、`tests/E2E/BGraph/onnx_clamp.py` | 分层结构和数值证据 |

这些都属于独立 BuddyGraph 项目；不需要修改 `/buddy-mlir/tools/buddy-opt` 或原
Buddy-MLIR 源码。

## 4. 调用链

```text
ONNX Clip(data, min_initializer, max_initializer)
→ importer 读取两个 finite scalar f32 常量
→ "bgraph.clamp" {min_value, max_value}
→ ClampOp::verify()
→ bgraph-infer-shapes
→ canonicalize: clamp(clamp(x, min, max), min, max) → inner clamp
→ direct ClampLowering 或 bgraph-fuse-elementwise
→ one linalg.generic，Region 内为 cmpf + select
→ One-Shot Bufferize → LLVM Dialect → mlir-runner
```

direct 路径证明独立 Clamp 能被消除，fused 路径证明 Add→Clamp→Mul 可成为一个
`bgraph.fused_elementwise`；两者最终执行同一标量公式。

## 5. IR 前后变化

实际 ODS schema 使用一输入、一结果和两个必需的 `F32Attr`：

```tablegen
def BGraph_ClampOp : BGraph_Op<"clamp", [Pure]> {
  let arguments = (ins AnyRankedTensor:$input,
                       F32Attr:$min_value,
                       F32Attr:$max_value);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}
```

导入后的核心 IR：

```mlir
%0 = "bgraph.clamp"(%x) {
  min_value = -1.0 : f32,
  max_value = 1.0 : f32
} : (tensor<4xf32>) -> tensor<4xf32>
```

lowering 和 fusion 都使用 ordered compare：

```mlir
%below = arith.cmpf olt, %x, %lo : f32
%lowered = arith.select %below, %lo, %x : f32
%above = arith.cmpf olt, %hi, %lowered : f32
%clamped = arith.select %above, %hi, %lowered : f32
```

输入为 NaN 时两个 ordered compare 都为 false，因此 NaN 沿原输入传播。signed-zero
位模式未纳入 MVP contract；项目只承诺数值相等和 NaN 传播。

## 6. 核心机制

按真实依赖顺序阅读：

1. **ODS**：schema 生成 `input/minValue/maxValue/result` accessors；不加
   `SameOperandsAndResultType`，让兼容的动态 result 可被 shape pass 精化。
2. **Verifier**：输入和结果必须是 ranked f32 tensor、shape 兼容；bounds 必须有限且
   `min_value <= max_value`。
3. **Shape**：`inferShape()` 对 Clamp 返回 input shape。
4. **Canonicalization**：仅在两组 bounds 与 replacement type 完全相同时消除外层
   nested Clamp。
5. **Fusion**：`isFusible()`、root pattern 和 `emitScalar()` 都包含 Clamp；Fused
   Region verifier 同时允许 `arith::CmpFOp` 和 `arith::SelectOp`。
6. **Importer**：`SUPPORTED_OPS` 包含 `Clip`；只把当前节点的 min/max 从 runtime
   operands 排除，因此同一 initializer 仍可被别的节点当普通数据复用。
7. **Conversion**：`ClampLowering` 复用 `createElementwiseGeneric()`，并注册进
   FullConversion pattern set。
8. **Tests**：IR 正反例、shape、canonicalization、fusion、importer、direct/fused
   conversion 和数值 E2E 共同形成证据链。

这里的 `Clip` 支持是有意收窄的 ONNX 子集：恰好三个非空 inputs，min/max 必须是
initializer 中的 finite scalar f32。缺失、运行时、非标量、非有限或逆序 bounds 都会
明确拒绝。

## 7. 为什么这样设计

把 bounds 存成 attributes，可让 verifier、canonicalization 和 lowering 直接看到编译期
语义，也避免在 BGraph Clamp 上保留两个运行时 tensor operands。代价是暂不支持 ONNX
的 optional、Constant-node 或动态 bounds。对 MVP 来说，明确拒绝比静默猜默认值可靠。

`_node_attribute_initializers()` 按节点计算元数据输入，而不是把某个 initializer 在整张
图中永久隐藏。`clamp_shared_initializer.onnx` 专门回归这一点：同一个 `clip_max` 既能
作为 Add 的普通 SSA operand，也能作为 Clip 的属性来源。

## 8. 常见错误

- ODS 修改后没有重建 TableGen target，仍在观察旧 generated class。
- verifier 漏掉 NaN/Inf、`min > max` 或 shape 不兼容。
- fusion 生成 `cmpf/select`，却忘记 Fused Region verifier 白名单。
- 把 Clip bounds 在整张图里全局排除，破坏 initializer 的合法复用。
- bounds 不同或 replacement type 不同仍消除 nested Clamp。
- 只实现 fused scalar path，遗漏独立 `ClampLowering`。
- 只通过 parse test，未用 FullConversion 和 runner 证明闭环。

## 9. 动手练习

若已读过 Clamp 全部实现，可做[Square 独立增量](exercises/resume_capstone.md)：
题目固定 MLIR 层契约与验收，不预先提供整份实现。当前基线没有该 Op。

不要照抄实现，按证据反向重建设计：

1. 从 `.td` 找到四个 generated accessors，再定位手写 verifier。
2. 在 `invalid.mlir` 解释 `min>max`、Inf 和 shape mismatch 三个诊断。
3. 展示 dynamic result 经 shape pass 变成 input type。
4. 对比相同和不同 bounds 的 nested Clamp canonicalization。
5. 展示 Add→Clamp→Mul fusion Region 的 `cmpf/select`。
6. 解释共享 initializer 为什么需要“按节点排除”。
7. 分别运行 direct 与 fused FullConversion，并确认无 `bgraph.` 残留。
8. 修改 E2E fixture 中的输入值，手算 NumPy reference，再运行验证。

完成审计后，可在实验分支尝试一个边界明确的增量，例如新增 `min > max` 的 ONNX
importer rejection fixture，并同时补稳定诊断和 FileCheck。不要把 `min == max` 直接
折成常量：这会破坏当前需要保留输入 NaN 的 contract。

## 10. 验收标准

- `rg Clamp` 能覆盖 ODS、IR、shape、fusion、importer、conversion 和 tests。
- `check-buddygraph` 为 14/14。
- direct/fused FullConversion 后均无 `bgraph.`。
- E2E 的 reference、unoptimized、optimized 都为 `5.000000e-01`，并保留 NaN。
- 能解释 constant scalar bounds 的取舍，以及共享 initializer 的处理边界。

## 11. 面试追问

**问：新增一个 Op 为什么要改这么多层？**

答：schema 只让 IR 可表达；可用编译器还需要局部合法性、shape 精化、图优化参与规则、
前端语义映射、目标 lowering 和分层测试。FullConversion 会把遗漏的 lowering 变成明确
失败，而不是让未知高层 Op 流入后端。

**问：为何不采用 `relu(clamp(...)) → clamp(...)`？**

答：没有 fast-math 时，这条规则会遇到 signed-zero 语义问题。当前实现只折叠 bounds
与 replacement type 完全一致的 nested Clamp，contract 更窄也更容易证明。
