# Square 综合练习：验收提示与定位路线

先完成[题目](../exercises/resume_capstone.md)的设计。本文件提供评审依据，不提供
整份可复制实现，以保留独立完成的证据。

## 设计是否自洽

Square 复用 builtin tensor/f32；shape 等于 input，scalar body 是 `arith.mulf x,x`。
`Pure` 与 shape 相同不替代 f32 检查。静态 CPU lowering 可以继续拒绝动态输入；
高层 verifier 放行不能证明后端支持。

`(-2)^2=4`，`3^2=9`；IEEE 乘法下负零乘自身为正零，NaN 为 NaN，负无穷乘自身为
正无穷。不要只用 `max-error==0` 验证 NaN；需检查非有限类别与零符号。

## 源码路线

| 层 | 对照位置 | 容易漏什么 |
|---|---|---|
| ODS | `BGraphOps.td` Relu/Clamp | 没更新生成链 |
| verifier | `BGraphOps.cpp` dtype/shape helper | 同类型 i32 仍应拒绝 |
| shape | `InferShapes.cpp::inferShape` | 新 Op 落入不支持分支 |
| direct lowering | ReluLowering 与 pattern 注册 | body 写好却没注册 |
| fusion | isFusible、root pattern list、emitScalar | 只改白名单没有 scalar emission |
| tests | Dialect/Conversion/E2E | 只有融合路径“碰巧”通过 |

generated Op list 随 ODS 扩展，通常无需逐个手写 `addOperations<SquareOp>`。
已有 `arith.mulf` 支持应通过源码确认；没有新增 scalar opcode 就不应放宽整个 Region
白名单。前端 `SUPPORTED_OPS` 不属于本题交付。

## 测试是否有力度

把 body 故意改成加法，有限值测试应失败；关闭 fusion 接入，融合结构检查应失败，
即使 direct 数值仍正确；漏 direct pattern，关闭优化的 conversion 必须失败。
三者分别检验数值、优化命中、基础 lowering 完整性。

共享 result 测试须观察分支 uses 与融合内部节点，不能只检查有一个 fused Op。
不能为让测试通过而放宽 legality。记录哪些证据是新运行，哪些沿用历史报告。

## 如何形成面试回答

选择真实发生的错误讲“预期—观察—定位—修复—回归”。故障注入应注明，不能说成
自己发现的生产缺陷。让另一位读者能从 diff 和测试复核贡献，而非只接受第一人称叙述。
