# 09｜Region、use-def 与 elementwise fusion

> **本章路线：从 tensor 子图到 scalar Region。** 先画 root、内部节点和 leaves，再逐 Value 建立映射，最后看替换与共享 producer 的保留。第 10 章接着解释 Region 为什么能变成一个 generic。

## 1. 本章目标

你将能画出真实 Add→Relu→Mul 的 use-def 图和 fused Region，解释 tensor operands
如何映射到 scalar block arguments，并判断单用户、分支和静态广播下的融合结果。

## 2. 先运行

```bash
cd /home/jlq/project/buddygraph
export BUDDYGRAPH_TMP=/home/jlq/project/buddygraph/tmp
mkdir -p "$BUDDYGRAPH_TMP"
build/bin/buddygraph-opt --bgraph-fuse-elementwise \
  tests/Dialect/BGraph/fuse-elementwise.mlir \
  -o "$BUDDYGRAPH_TMP/fuse-elementwise.after.mlir"
sed -n '1,140p' "$BUDDYGRAPH_TMP/fuse-elementwise.after.mlir"
```

`@fuse` 中 Add/Relu/Mul 变成一个 FusedElementwise；`@do_not_duplicate` 的共享 Add
仍存在。实现可能合法地融合其下游 Relu→Mul 子链，但不会把多用户 Add 克隆进 Region。

## 3. 真实代码位置

- `FuseElementwise.cpp::isFusible()`。
- `ElementwiseTree::{operations, orderedOperations, leaves, collect()}`。
- `emitScalar()`。
- `FuseElementwisePattern<RootOp>::matchAndRewrite()`。
- `BGraphFuseElementwise::runOnOperation()`。
- `BGraphOps.cpp::FusedElementwiseOp::verify()`、`YieldOp::verify()`。
- `BGraphToLinalg.cpp::FusedElementwiseLowering`。
- `tests/Dialect/BGraph/fuse-elementwise.mlir`。

## 4. 调用链

```text
applyPatternsGreedily(top-down)
→ 对 Add/Sub/Mul/Div/Relu/Clamp root 尝试 FuseElementwisePattern
→ 若 root 还有 fusible user：不是 chain root，failure
→ ElementwiseTree::collect(root)
   → producer fusible 且当前 operand.hasOneUse()：递归收集
   → 否则加入 ordered leaves
→ 至少两个 ops 才继续
→ create FusedElementwiseOp + Region + Block
→ 每个 tensor leaf 建一个 scalar BlockArgument
→ IRMapping: leaf Value ↦ BlockArgument
→ emitScalar(root) 递归创建 arith scalar ops
→ create YieldOp
→ replace root，删除已无 uses 的旧 ops
```

随后 lowering：

```text
FusedElementwiseLowering::matchAndRewrite()
→ createElementwiseGeneric()
→ broadcastMap() 为每个 tensor input 建 indexing map
→ linalg.generic body builder
→ IRMapping: source BlockArgument ↦ linalg body argument
→ builder.clone(source scalar ops, mapping)
→ source Yield value ↦ linalg.yield
```

## 5. IR 前后变化

use-def 图：

```mermaid
flowchart LR
  A[arg0 tensor] --> Add
  B[arg1 tensor] --> Add
  Add --> R[Relu]
  R --> Mul
  C[arg2 tensor] --> Mul
  Mul --> Ret[return]
```

Region 结构：

```mermaid
flowchart TB
  subgraph F["bgraph.fused_elementwise tensor layer"]
    I["inputs: tensor A, B, C"] --> B0
    subgraph B0["single Block: scalar layer"]
      BA["%a: f32"] --> SA[arith.addf]
      BB["%b: f32"] --> SA
      SA --> MR[arith.maximumf]
      Z["0.0"] --> MR
      MR --> SM[arith.mulf]
      BC["%c: f32"] --> SM
      SM --> Y[bgraph.yield]
    end
    Y --> O["result: tensor<2x3xf32>"]
  end
```

Region 表达“对每个输出 index 执行的 scalar formula”；外层 tensor operands/result
表达 shape 和广播。`bgraph.yield` 不是函数 return，只结束 fused body。

## 6. 核心机制

`hasOneUse()` 防止把共享 producer 克隆进 Region。对于多用户 Add，collect 把它作为
leaf，所以可以继续融合 Add 之后的单用户子链，但 Add 自身保留并只计算一次。

`llvm::SetVector<Value> leaves` 同时去重和保持确定顺序；该顺序必须与 block arguments
一一对应。`SmallPtrSet` 防止重复收集同一 Operation。

fusion pass 中的 `IRMapping` 只映射 tensor leaves 到 scalar args；scalar body由
`emitScalar()` 显式重建，不是 clone 原 BGraph tensor ops。真正的 `builder.clone()`
发生在 fused-to-Linalg lowering，把 Region 中已是 scalar 的 arith ops 克隆到
`linalg.generic` body。

当前实现支持静态 NumPy 风格广播。`broadcastMap()` 对输入维为 1、输出对应维不为 1
时使用 affine constant 0；其他维映射到输出 loop dimension。所有 input/result 必须
静态，否则 lowering 返回 failure。

### 跟踪一个完整子图的每个 Value

先在纸上写 `a=Add(x,b); r=Relu(a); y=Mul(r,s)`。假设每个内部结果都只用一次，
且 y 的 user 不是可融合 Op。root 是 Mul，反向收集顺序从 root 开始；外部 leaves
按首次遇到的顺序去重，必须保持 operand 顺序和 Region argument 顺序一致。

| 原 Value | 在收集中的角色 | 融合后的对应物 |
|---|---|---|
| x、b、s | 外部 tensor leaves | fused operands 与一一对应的 f32 block arguments |
| a | 内部 Add result | `arith.addf` 的 scalar result |
| r | 内部 Relu result | scalar `arith.maximumf` result |
| y | root result，供外部使用 | fused Op 的 tensor result；原 uses 被重定向 |

Region 的意思是“给定当前元素的 x、b、s，计算一个输出元素”。它不持有外部 tensor
payload，也不在此阶段定义如何循环整个 tensor。第 10 章的 Linalg maps 负责每个迭代点
应从各 tensor 的哪个坐标取值。

### 两次 IRMapping 不要混淆

图 fusion 阶段的映射是 `外部 tensor Value → fused scalar BlockArgument`，`emitScalar`
沿内部子图递归创建 arith 运算。进入 Linalg lowering 后，已有 scalar Region 被迁移，
映射变为 `旧 scalar BlockArgument/result → 新 Linalg body Value`；`builder.clone`
复制 scalar op 并更新映射，yield 单独处理。前者是语义重建，后者是 scalar IR 克隆。

### 共享 producer 为什么成为 leaf

若 `a` 还被另一条分支使用，`a.hasOneUse()` 不成立，收集沿该边停止；当前消费者仍可
与它之后的节点融合，a 作为外部 tensor 传入。旧 Add 不在可删除内部集合中。
所以“单用户”不是要求全图无分叉，也不是共享节点导致整个 pass 失败。

### 三种失败分别在哪一层

| 情况 | 行为 | 需要证明什么 |
|---|---|---|
| 候选 root 仍有可融合 user | Pattern 不命中 | 应尝试更下游 root，不等于 IR 非法 |
| 收集集合不足两个 Op | Pattern 不命中 | 没有可消除的内部中间值 |
| scalar body 违反白名单或 yield/type 契约 | verifier 失败 | 生成/手写 IR 非法 |
| 高层合法，但仍有动态 tensor shape | 后续 static lowering 可失败 | 融合成功不等于 CPU 支持 |

当前 root 策略不是寻找任意图的全局最优子图，也没有估算寄存器压力或重计算成本。
讲“减少中间值”时，先指出实际消失的内部 tensor result，再讨论可能减少的 materialization。
Clamp 的 scalar body 使用 ordered `cmpf/select` 保留 NaN 路径；不能随手改成另一组
min/max 指令而不核对浮点契约。

## 7. 为什么这样设计

Region 让 fused op 保存任意白名单 scalar expression，而不需要为每一种链新增一个
Op。外层保持 tensor contract，内层可以直接成为 `linalg.generic` body。single-use
边界避免重复计算；static broadcast maps 让支持范围有明确 lowering 证据。

## 8. 常见错误

- 从链中间开始融合，留下重叠 fused candidates；代码先检查 fusible user。
- 多用户 producer 仍递归 collect，导致复制计算或删除有用户 op。
- block argument 数/顺序与 leaves 不一致。
- 忘记 `InsertionGuard`，scalar ops 插入父 Block。
- Region 含 `arith.negf` 等白名单外 op，parent verifier 拒绝。
- yield 类型不是 result element type。
- 宣称支持 dynamic broadcast；lowering 明确要求 static shape。

## 9. 动手练习

先不运行，预测测试 `@do_not_duplicate` 的 leaves 和内部 ops：
`p=add(a,b); r=relu(p); y=mul(p,r); return y`。p 同时被 Relu 与 Mul 使用，注意
两条边都必须保留正确关系。再运行测试文件对照该函数输出。

修改题：在项目 `tmp/` 测试副本中给 `@fuse` 的 `%1` 增加一个额外 return-use（可用
另一个 Add 汇合），观察 collect 边界和 fused body inputs 如何变化。

## 10. 验收标准

- 能从 IR 画出 use-def 图并正确标出 single/multiple use。
- 能解释两个阶段各自如何使用 `IRMapping`。
- `@fuse` 只生成一个 fused op，FullConversion 后只生成一个相关 generic。
- 能解释 static broadcast 的 affine constant-0 map。

## 11. 面试追问

**问：为什么 fused op 需要 Region？**

答：链的 scalar formula 随 graph 而变；Region 能把 operands 映射成 block arguments，
保存实际 arith expression 和 yield，同时外层只暴露稳定的 tensor contract。

**问：多用户为何不能直接融合？**

答：把 shared producer 克隆进 fused Region 会重复计算；删除它又会破坏另一个用户。
当前实现把它作为 fused leaf，最多融合其下游单用户链。
