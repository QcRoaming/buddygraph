# Design decisions

## Verifier、shape inference 与 canonicalization

三者不互相替代：

- verifier 判断当前 IR 是否自洽，例如 dtype、rank、broadcast、channel、axes、
  permutation、region terminator 和静态输出 shape 冲突；失败时 IR 非法。
- `bgraph-infer-shapes` 在输入足够具体时收紧动态结果维度，不改变数值语义。
- canonicalization 删除局部恒等或冗余结构，如 nested Relu、Mul-one、连续
  Reshape、互逆 Transpose 和相同 bounds 的 nested Clamp；模式不匹配仍是合法 IR。项目不做无 fast-math contract
  的 Add-zero 消除，因为它会改变 `-0.0 + +0.0` 的结果符号。

## DRR 与 C++ Pattern 的边界

ODS 用于声明稳定 schema。当前 rewrite 使用 C++：Mul-one pattern 需要检查
DenseElements splat 数值，reshape/transpose 需要跨 op 和 attribute 计算，BN folding
需要重写常量 payload，fusion 需要克隆链到新 Region。这些条件用 C++ 更清晰。
若未来加入只依赖 op 名和静态 operand 位置的一对一替换，可优先用 DRR；不为了
“使用 DRR”而把数值或 use-count 条件藏进 opaque helper。

## 不手写通用 DCE/CSE

MLIR 已提供经过验证的 canonicalizer、CSE 和 symbol DCE。BuddyGraph patterns 只
描述方言特有等价关系；删除 dead op 和合并公共表达式交给通用 passes。重复实现
通用优化既增加错误面，也会让 pass pipeline 的职责不清。

## Layout 与属性表示

Layout 是 TableGen `EnumAttr`，而非自由字符串，因此 parser 和 verifier 获得封闭
取值集合。为了让当前 MLIR Python `Operation.create` 能设置 ODS attributes，方言
设置 `usePropertiesForAttributes = 0`；本地 Python API 尚无 properties 参数。这不
削弱强类型 LayoutAttr，也避免前端退回到手写整段 MLIR。

## BN folding 合法性

仅在以下条件同时满足时折叠：BN 处于 inference 语义；直接输入来自 groups=1 的
Conv；Conv 结果单 use；filter、可选 Conv bias、scale、BN bias、mean、variance 都
是可读取的 f32 常量；Conv 与 BN layout 一致；channel 和 shape 完全一致；将 f64
epsilon 按 lowering 语义舍入到 f32 后，`variance + epsilon > 0` 且有限。

对输出 channel `c`：

```text
alpha[c] = scale[c] / sqrt(variance[c] + epsilon)
W'[c, ...] = W[c, ...] * alpha[c]
b'[c] = (b[c] - mean[c]) * alpha[c] + beta[c]
```

不满足条件时 pass 保留原图，不猜测训练态统计量，也不复制多 use Conv。该变换会把
BN 算术重结合进 Conv constants，数值验收采用容差，不声明严格逐 bit 等价。

## Elementwise fusion 合法性

融合 Add/Sub/Mul/Div/Relu/Clamp 的单 use chain。外部 tensor 成为
`bgraph.fused_elementwise` inputs；Region block arguments 是 f32 scalar，内部只
容许白名单 arith scalar ops（包括 Clamp 所需的 `cmpf/select`），并由 `bgraph.yield`
终止。单 use 约束防止为融合复制
计算，广播 shape 由 verifier 再检查。

lowering 把整段 Region 映射到一个 `linalg.generic` body，因此链中原本每个 tensor
op 的输出不再物化；只保留融合 kernel 的一个结果 tensor。跨分支、多 use、非法广播
或不在白名单的 op 均不会融合。

## 前端边界

前端采取严格拒绝策略：单一 opset 18、默认 domain、静态 f32、NCHW Conv、常量
shape/axes，以及 Clip 的 finite scalar f32 initializer bounds。Clip bounds 只对当前
节点从 runtime operands 排除，initializer 仍可被其他节点当普通 SSA 数据复用。ONNX checker 和 shape inference 先运行；随后每个 node 的输出类型从
推理信息读取。严格边界使方言和 lowering 的闭环可证明，不把“成功解析”误当作
“已支持 ONNX 语义”。
