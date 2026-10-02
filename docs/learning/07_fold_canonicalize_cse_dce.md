# 07｜fold、canonicalization、CSE 与 DCE：不要混成“优化”

> **本章路线：四类清理各负责什么。** 先按第 5–6 节辨认四类变化，再做第 2 节隔离实验。最后用 signed-zero 反例检验数学替换。此章明确项目 pattern 与 upstream 通用 pass 的贡献边界。

## 1. 本章目标

你将用同一个临时 module 分别观察 upstream fold、BGraph canonicalization、CSE 和
dead-op deletion，证明 BuddyGraph 当前没有自定义 `fold()` hook，并用 signed-zero
回归理解为何当前实现刻意不做无 fast-math contract 的 Add-zero 消除。

## 2. 先运行

在项目 `tmp/` 创建 `bgraph-cleanup.mlir`：

```bash
cd /home/jlq/project/buddygraph
export BUDDYGRAPH_TMP=/home/jlq/project/buddygraph/tmp
mkdir -p "$BUDDYGRAPH_TMP"
cat >"$BUDDYGRAPH_TMP/bgraph-cleanup.mlir" <<'MLIR'
module {
  func.func @arith_fold() -> f32 {
    %a = arith.constant 1.0 : f32
    %b = arith.constant 2.0 : f32
    %sum = arith.addf %a, %b : f32
    return %sum : f32
  }
  func.func @canonicalize(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %a = "bgraph.relu"(%x) : (tensor<2x3xf32>) -> tensor<2x3xf32>
    %b = "bgraph.relu"(%a) : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %b : tensor<2x3xf32>
  }
  func.func @cse(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %a = "bgraph.relu"(%x) : (tensor<2x3xf32>) -> tensor<2x3xf32>
    %b = "bgraph.relu"(%x) : (tensor<2x3xf32>) -> tensor<2x3xf32>
    %c = "bgraph.add"(%a, %b) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return %c : tensor<2x3xf32>
  }
  func.func @dead(%x: tensor<2x3xf32>) {
    %unused = "bgraph.relu"(%x) : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return
  }
}
MLIR

build/bin/buddygraph-opt --canonicalize "$BUDDYGRAPH_TMP/bgraph-cleanup.mlir" \
  -o "$BUDDYGRAPH_TMP/bgraph-canonicalized.mlir"
build/bin/buddygraph-opt --cse "$BUDDYGRAPH_TMP/bgraph-cleanup.mlir" \
  -o "$BUDDYGRAPH_TMP/bgraph-cse.mlir"
```

查看每个函数：`arith_fold` 应返回常量 3；`canonicalize` 只剩一个 Relu；
`dead` 的 unused Relu 被 canonicalizer 的通用清理删除；`cse` 输出中重复 Relu 被合并。

再用可执行 rank-0 tensor 验证 Add-zero signed-zero 回归：

```bash
cat >"$BUDDYGRAPH_TMP/bgraph-add-zero.mlir" <<'MLIR'
func.func @main() -> f32 {
  %x = arith.constant dense<-0.0> : tensor<f32>
  %z = arith.constant dense<0.0> : tensor<f32>
  %sum = "bgraph.add"(%x, %z)
      : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %v = tensor.extract %sum[] : tensor<f32>
  return %v : f32
}
MLIR

lowering_args=(
  --convert-bgraph-to-linalg
  '--one-shot-bufferize=bufferize-function-boundaries'
  --convert-linalg-to-loops --lower-affine
  --convert-scf-to-cf --convert-cf-to-llvm
  --convert-math-to-llvm --convert-arith-to-llvm
  --finalize-memref-to-llvm --convert-func-to-llvm
  --reconcile-unrealized-casts
)

build/bin/buddygraph-opt "${lowering_args[@]}" \
  "$BUDDYGRAPH_TMP/bgraph-add-zero.mlir" \
  -o "$BUDDYGRAPH_TMP/bgraph-add-zero.before.mlir"
build/bin/buddygraph-opt --canonicalize "${lowering_args[@]}" \
  "$BUDDYGRAPH_TMP/bgraph-add-zero.mlir" \
  -o "$BUDDYGRAPH_TMP/bgraph-add-zero.after.mlir"

/buddy-mlir/llvm/build/bin/mlir-runner \
  "$BUDDYGRAPH_TMP/bgraph-add-zero.before.mlir" -e main -entry-point-result=f32
/buddy-mlir/llvm/build/bin/mlir-runner \
  "$BUDDYGRAPH_TMP/bgraph-add-zero.after.mlir" -e main -entry-point-result=f32

build/bin/buddygraph-opt --canonicalize "$BUDDYGRAPH_TMP/bgraph-add-zero.mlir" \
  | rg '"bgraph.add"'
```

当前实现的两次 runner 都应输出 `0.000000e+00`，最后一条命令必须仍找到
`"bgraph.add"`。过去的泛型 identity pattern 会把第二次结果改成
`-0.000000e+00`；现在已移除 Add-zero rewrite，并由
`tests/E2E/BGraph/signed_zero.mlir` 固定结构和完整 CPU lowering 结果。

## 3. 真实代码位置

- `BGraphOps.td`：`Pure` 和 `hasCanonicalizer = 1`。
- `BGraphOps.cpp::ElideMulByOne`、`ElideNestedRelu`、`CollapseReshape`、
  `CancelTranspose`。
- `ReluOp/AddOp/MulOp/ReshapeOp/TransposeOp::getCanonicalizationPatterns()`；其中
  `AddOp` 当前不注册 pattern。
- `tests/Dialect/BGraph/canonicalize.mlir`、`tests/E2E/BGraph/signed_zero.mlir`。
- driver 调用 `registerAllPasses()`，因此 `--canonicalize`、`--cse` 来自 upstream。

检索自定义 fold 的缺失：

```bash
rg -n 'hasFolder|hasFold|::fold\(' include/BuddyGraph lib/BuddyGraph || true
```

应无输出。

## 4. 调用链

### BGraph canonicalization

```text
--canonicalize
→ upstream CanonicalizerPass
→ 读取 op 的 canonicalization registry
→ ReluOp::getCanonicalizationPatterns()
→ ElideNestedRelu::matchAndRewrite()
→ replaceOp(outer, inner.result)
→ greedy driver 重试 + 删除 trivially dead Pure ops
```

### CSE

```text
--cse
→ upstream CSE pass
→ 比较 operation name / operands / attributes / result types / regions
→ 查询 side-effect information
→ 把第二个等价 Relu 的 uses 替换成第一个
→ 删除重复 op
```

### fold

Canonicalizer 内部会先尝试 op fold hook；本例 `arith.addf` 由 Arith 提供 fold，BGraph
没有自己的 fold hook。不要把 `ElideMulByOne` 称作 `MulOp::fold()`；它是
canonicalization pattern。

## 5. IR 前后变化

Fold：

```mlir
// before
%a = arith.constant 1.0 : f32
%b = arith.constant 2.0 : f32
%sum = arith.addf %a, %b : f32
// after canonicalize invokes Arith fold
%cst = arith.constant 3.0 : f32
```

Canonicalization：

```mlir
relu(relu(%x))  →  relu(%x)
mul(%x, dense<1.0>) → %x
```

当前代码不会执行 `add(%x, dense<0.0>) → %x`。这是有意的 correctness 选择：当
`%x=-0.0`、常量为 `+0.0` 时，原 `addf` 结果是 `+0.0`，替换后却是 `-0.0`。
BuddyGraph 没有 fast-math/`nsz` contract，所以修复选择是移除 rewrite，而不是悄悄
放宽语义。结构检查保证 Add 仍在，runner 检查保证完整 lowering 后输出正零。

CSE：

```mlir
%a = bgraph.relu(%x)
%b = bgraph.relu(%x)  // uses replaced by %a, then deleted
```

DCE-like cleanup：unused `bgraph.relu` 可被删，因为 ODS 标记 `Pure`。当前没有
BuddyGraph 自己写的 DCE pass；这是通用基础设施行为。

## 6. 核心机制

- fold 通常面向单 op、常量/attribute 或恒等结果，API 可返回 attribute/value。
- canonicalization 是注册 patterns 的 best effort greedy rewrite；不保证某种全局
  最优 normal form。
- CSE 识别 structurally equivalent 且可安全复用的表达式。
- DCE 删除无用户且无副作用的 operation。

副作用信息是 CSE/DCE 合法性的核心。若一个 op 可能写文件、修改内存或执行 I/O，
即使 result 未使用也不能随意删除。BGraph 的 `Pure` trait 使通用 passes 有证据进行
清理；fusion 还显式调用 `isMemoryEffectFree()`。

Pass 顺序会改变可见结果：canonicalization 先应用某个 identity pattern 后，CSE 已无
对应 op 可合并；CSE 先合并 producers，可能让后续 pattern 更容易满足 single-use，
也可能改变 use-count，必须用测试固定预期。测试还必须覆盖语义边界，不能只有 op count。

## 7. 为什么这样设计

### 把“项目提供什么”和“框架执行什么”逐项对应

| 项目提供 | MLIR 框架据此执行 | 不能因此声称什么 |
|---|---|---|
| Relu 等 Op 的 canonicalization pattern | greedy driver 匹配、重试、清理 | 自己实现了完整 canonicalizer |
| Pure/副作用信息 | CSE/dead cleanup 的安全判断 | Pure 证明任意浮点替换都等价 |
| driver 的 `registerAllPasses()` | 让 `--cse` 等 CLI 可调用 | 自己编写了通用 CSE 算法 |

如需深挖框架，先只读 `/buddy-mlir/llvm/mlir/lib/Transforms/CSE.cpp` 与
`Canonicalizer.cpp`，再按实际调用查看 greedy rewrite driver。CSE 不能把一个不支配
使用点的 Value 当替代值；相同 op name 不代表 operands/attrs/types 或副作用相同。
这些控制流和语义条件正是复用 upstream 的原因。

方言只实现其特有等价关系，通用 fold/CSE/DCE 交给 upstream。重写通用算法会重复
处理 regions、side effects、dominance 和 symbol 等复杂规则，也不利于维护。

这条设计原则不自动证明每个局部 pattern 正确。Add-zero 缺陷发现后，工程在三种方案
中选择了最保守的一种：移除 rewrite。另两种是只匹配能证明保持 signed zero 的情形，
或为 BGraph/Pass 明确定义允许忽略 signed zero 的 fast-math contract。当前项目没有
后一种 contract，也没有足够收益证明需要更复杂的条件，因此先保正确性并用回归锁定。

## 8. 常见错误

- 把所有 `--canonicalize` 产生的变化都归功于 BGraph pattern；它还调用 fold 和
  通用 dead cleanup。
- 把普通数值相等当成 strict IEEE 等价；Add-zero 即使 result type 完全一致，也会
  改变一组 signed-zero 输入，因此当前没有这条 pattern。
- 忘记 `Pure`，导致 dead op/CSE 不生效。
- 认为 canonicalize 保证执行某个 pattern；pattern 是 best effort，pipeline 应靠
  legality/测试而不是特定遍历偶然性。

## 9. 动手练习

修改题：在项目 `tmp/` 版本中把 `@cse` 的第二个 Relu location 或 attribute 改成不同
结构，预测 CSE 是否仍合并；运行验证。然后说明 location 是否参与 operation
equivalence。

设计题：如果未来要重新引入 Add-zero，列出 canonicalization/fold 需要检查的
DenseElements splat、broadcast/result type、零的符号和 fast-math 条件；不要直接
假设标量 0 与任意 tensor 可替换。

正确性题：构造包含 `-0.0` 输入和 `+0.0` rhs 的可执行 Add，比较 canonicalize 前后
结果和 IR。预期两者都为正零且 Add 保留。解释为什么 7.0 主 E2E 捕获不到该问题，
以及为什么需要专门的 signed-zero regression。

## 10. 验收标准

- 能展示四类变化各自的 IR evidence。
- 能用 `rg` 证明 BGraph 没有 custom fold hook。
- 能指出 `Pure` 对 CSE/DCE 的作用。
- 能说明 pass 顺序为什么可能改变 use-count 和 Pattern 命中。
- 能说明已修复的 Add-zero signed-zero 缺陷、当前策略和结构/数值双重回归。

## 11. 面试追问

**问：为什么不自己写 DCE/CSE？**

答：它们是跨 Dialect 的通用算法，依赖 side-effect、regions 和 dominance 等规则。
BGraph 应正确声明 `Pure` 并提供局部等价 pattern，让 upstream passes 工作。

**问：canonicalization 是否保证得到唯一 IR？**

答：不保证。它是 greedy best effort；pattern benefit、遍历和其他 fold 会影响结果。
语义终止条件应由 verifier/Conversion legality 和测试保证。
