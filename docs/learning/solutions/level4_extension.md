# Level 4 答案：`bgraph.clamp` 实现核对

这是一份当前源码的逐层核对表，不提供可盲贴的完整 patch。

## 1. ODS 与 contract

`BGraph_ClampOp` 是一输入、一结果、两个必需 `F32Attr` 的 `Pure` Op，并启用 custom
verifier/canonicalizer。它没有 `SameOperandsAndResultType`，因此 verifier 可接受 shape
兼容的动态 result，再由 shape pass 精化。`ClampOp::verify()` 检查 ranked f32、shape
兼容、finite bounds 和 `min_value <= max_value`。

## 2. importer

`SUPPORTED_OPS` 与 `_import_node()` 都处理字符串 `"Clip"`。data 从 `self.values` 取 SSA
value；min/max 由 `_constant_scalar_f32()` 从 initializer table 读取。创建
`bgraph.clamp` 后，结果写入 `self.values[output]`。

关键边界是 `_node_attribute_initializers(node)`：bounds 只对当前 Clip 不成为 runtime
operands。float initializer 仍会创建 SSA constant，因此它被其他节点复用时不会丢失；
`clamp_shared_initializer.onnx` 覆盖了这个回归点。缺失、运行时和非标量 bounds 都有
带 node name/op type 的稳定错误。

## 3. shape inference

`inferShape(Operation *)` 对 `ClampOp` 返回 input shape，外层 `ModuleOp::walk` 更新 result
type。项目没有为 BGraph Op 实现 `InferTypeOpInterface`，所以不能把它描述为 interface
method。

## 4. canonicalization

`ElideNestedClamp` 只在 input 来自另一个 Clamp、两组 attributes 完全相同且 inner result
type 等于 outer result type 时返回 inner result。不同 bounds 或不同 replacement type
必须保留。项目没有采用会改变 signed-zero 行为的 `relu(clamp(...))` 规则。

## 5. fusion

Clamp 同时出现在 fusible 分类、root pattern 注册和 `emitScalar()`。两个 f32 attributes
先变成 scalar constants，再用 ordered less-than 和 select 实现 clamp。Fused verifier
白名单包含 `arith::CmpFOp` 与 `arith::SelectOp`。`IRMapping` 只映射 tensor leaf，不会
自动产生这段 scalar 语义。

## 6. lowering

`ClampLowering` 调用 `createElementwiseGeneric()`，Region 内用同一 compare/select 公式并
yield。该 pattern 已注册到 FullConversion；source/result 仍是 builtin tensor，因此不
需要 TypeConverter 或 signature conversion。

## 7. 证明闭环

| 层 | 正证据 | 反证据 |
|---|---|---|
| ODS/verifier | 合法 Clamp 可解析 | bounds/shape 错误被诊断 |
| importer | ONNX Clip 变为 Clamp；initializer 可复用 | runtime/缺失/非标量 bounds 拒绝 |
| shape | result 变为 input shape | 不伪造不兼容类型 |
| canonicalization | 同 bounds nested Clamp 消除 | 不同 bounds 保留 |
| fusion | Add→Clamp→Mul 成为一个 Region | 多用户不非法融合 |
| conversion | 无 `bgraph.`，有 `linalg.generic` | 漏 pattern 会 failed to legalize |
| E2E | direct/fused 都为 0.5 | NaN 明确保留 |

全量回归当前为 14/14，`onnx_clamp.py` 打印零误差和 `status=PASS`。如果只读懂 ODS、
却不能解释 direct/fused 两条消除路径与 E2E 数值，这份全链路审计仍未完成。

## 8. 增量扩展参考

推荐新增 `min > max` 或 Constant-node bound 的 importer rejection fixture，核对错误中
包含 node name/op type。不要把 `min == max` 直接改写为常量 tensor：当前 lowering 在
输入为 NaN 时返回 NaN，常量替换会改变这一 contract。
