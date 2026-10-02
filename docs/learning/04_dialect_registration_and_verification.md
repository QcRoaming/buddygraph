# 04｜Dialect 注册、解析与 verifier 调用链

> **本章路线：认识 Op 与判断合法性。** 沿 driver→registry→Dialect→Op 的运行链，分别观察 parser、ODS 和 C++ verifier 的失败。输入 IR 在本章只被检查；下一章才追它如何从 ONNX 创建。

## 1. 本章目标

你将能从 `buddygraph-opt main()` 追到 `Conv2DOp::verify()`，运行一个合法输入和至少
两个非法输入，并判断错误属于 parser、ODS 结构检查还是 custom verifier。

## 2. 先运行

```bash
cd /home/jlq/project/buddygraph
build/bin/buddygraph-opt --help | rg 'bgraph-'
build/bin/buddygraph-opt tests/Dialect/BGraph/roundtrip.mlir -o /dev/null
build/bin/buddygraph-opt --verify-diagnostics --split-input-file \
  tests/Dialect/BGraph/invalid.mlir -o /dev/null
```

三条命令都应成功。第三条“成功”表示实际诊断与 `expected-error` 完全匹配，不表示
非法 IR 被接受。

## 3. 真实代码位置

- `tools/buddygraph-opt/buddygraph-opt.cpp::main()`。
- `BGraphDialect.cpp::BGraphDialect::initialize()`。
- `BGraphOps.td` 中 `hasVerifier = 1`。
- `BGraphOps.cpp` 中 13 个 `verify()` 和 helper。
- `tests/Dialect/BGraph/invalid.mlir` 中 14 个 negative sections。
- `tests/Dialect/BGraph/roundtrip.mlir` 中 parse/print round trip。

## 4. 调用链

```text
main(argc, argv)
→ mlir::registerAllPasses()
→ buddy::bgraph::registerPasses()            [Passes.h.inc 生成]
→ registry.insert<BGraphDialect, ...>()
→ MlirOptMain(..., registry)
→ parser 看到 "bgraph.relu"
→ Context 从 registry 加载 BGraphDialect
→ BGraphDialect::initialize()
→ addOperations<GET_OP_LIST>()
→ parser 创建 Operation / typed ReluOp
→ Operation::verify()
→ ODS 生成的结构检查
→ ReluOp::verify()
→ getF32Tensor()
→ success() 或 emitOpError()
```

Pass pipeline 默认在输入 parse 后和 passes 之间验证 IR；失败会让 driver 返回非零。

## 5. IR 前后变化

合法输入：

```mlir
func.func @ok(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "bgraph.relu"(%x) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
```

verifier 不改 IR；它只决定该 IR 能否进入后续阶段。

非法 dtype：

```mlir
%0 = "bgraph.relu"(%x)
    : (tensor<2x3xi32>) -> tensor<2x3xi32>
// 'bgraph.relu' op input must have f32 elements
```

非法广播：

```mlir
%0 = "bgraph.add"(%a, %b)
    : (tensor<2x3xf32>, tensor<4xf32>) -> tensor<2x3xf32>
// 'bgraph.add' op operands are not broadcast compatible
```

非法 fused body 则由 parent verifier 的白名单拒绝 `arith.negf`。

## 6. 核心机制

Frontend semantic checks 与 verifier 不重复：importer 检查 ONNX domain/opset、
training mode、常量属性等协议；verifier 面对任何来源的 BGraph IR，检查 op 局部
不变量。

`getF32Tensor()` 返回 `FailureOr<RankedTensorType>`。它先检查 Type 存在、是 ranked
tensor、element type 是 f32。调用者必须检查 `failed(...)`，不能继续 `cast`。

binary verifier 用 `inferBroadcastShape()` 计算期望 shape，再用
`areCompatibleShapes()` 允许动态维作为待 refinement 信息。Conv verifier 额外检查
rank、attribute 长度/值域、groups=1、channel、bias 和静态空间输出；静态 kernel
必须能放入 padded input，shape 算术溢出也会得到诊断，而不是进入 shape pass 后触发
RankedTensorType assertion。

`Pure` trait 不是 verifier 的替代品；它描述 side effect。`SameOperandsAndResultType`
可提供结构级约束，但 Relu 的 f32 语义仍由 custom verifier 保证。

### 沿一份错误输入区分三层

假设要处理 Relu，先看下面三个故障。它们都可能表现为“工具退出”，但修复位置不同：

| 输入/状态 | 为什么失败 | 应查什么 |
|---|---|---|
| 未加载 BGraph，遇到未知 `bgraph.*` | 工具没有相应 Op 注册信息 | driver registry 与 `initialize` |
| Relu 没有输入却写一个结果 | 不满足 schema 的 operand 数量 | ODS/generated verifier |
| 输入与输出均 `tensor<2xi32>` | ranked tensor 结构成立，但违反 f32 业务契约 | `ReluOp::verify/getF32Tensor` |

验证有先后关系：框架先确保足以安全访问的结构，再调用相关 invariant/hook；因此不要
为了“尽早报错”在不确定有 operand 时直接读取第一个 operand。上表用于理解错误分类，
不是要求背诵所有 Trait/Interface 的完整内部验证次序。

### verifier 不是修复器，也不是优化选择器

错误静态 result shape 应被拒绝，而不是由 verifier 偷偷 `setType()`；Conv 多用户却仍
是合法图，应交 BN Pattern 判断不命中，而不是由 Conv verifier 拒绝。判断一条规则
放在哪里，可以问：没有开启任何优化，单独存在这个 Op 是否也不合法？若是，适合
verifier；若只是本次替换不安全，属于 pattern 条件。

练习时先预测“非法 IR”还是“合法但不优化”，再运行 `--verify-diagnostics` 或 pattern
测试。成功退出在两种测试中含义不同，不能只截图一个 PASS。

## 7. 为什么这样设计

把错误尽可能放在源 Dialect 层，诊断能使用 `input/filter/channel/layout` 等语义词，
而不是等 lowering 后因 indexing map 或 shape assertion 失败。局部 verifier 不应
依赖全图优化顺序；跨多 op 的合法性则由 Pattern 自己匹配。

## 8. 常见错误

- 没注册 Dialect：parser 把 op 当未知或拒绝 custom attribute。
- 只注册 Dialect、没注册 Pass：IR 能 parse，但 CLI pass 名未知。
- verifier 中先 `cast` 后检查：非法输入触发 assertion，而不是用户诊断。
- negative test 把 `expected-error` 放错行，lit 报 unexpected diagnostic。
- 把 dynamic result 当非法；当前 verifier 容许与静态推导不矛盾的 `?`，shape pass
  后再收紧。

## 9. 动手练习

在纸面上设计一个新 negative case：Conv bias 长度为 3、filter output channel 为 4。
写出预期错误文本和应放置的测试 section。若实际修改，请只编辑 BuddyGraph 的
`tests/Dialect/BGraph/invalid.mlir`，运行单测后再恢复或提交到自己的分支。

完整的“新建另一个 Dialect → 注册 → 新增 Op → generated/Trait verifier”带做见
[自定义 Dialect 基础设施实战线](dialect_lab/README.md)。

## 10. 验收标准

- 能解释 pass 注册与 Dialect 注册是两条不同链。
- `invalid.mlir` 的 14 个 section 全部通过 verify-diagnostics。
- 能把一个错误归类为 parser、generated structural verifier 或 custom verifier。
- 能从错误文本定位到准确 `emitOpError()`。

## 11. 面试追问

**问：verifier 和 Pattern match failure 有什么区别？**

答：verifier failure 表示 IR 非法，pipeline 必须停止；Pattern match failure 表示该
合法 IR 不满足本次优化前置条件，保持不变即可。

**问：为什么 importer 检查过还需要 verifier？**

答：BGraph IR 也可能来自手写、其他前端或前序 rewrite。Dialect 必须独立维护自身
不变量，不能信任单一 importer。
