# 简历能力验收：独立实现一个 MLIR 层 Square 增量

这是一道未预先实现的综合练习。只在[实验副本](../dialect_lab/00_lab_setup.md)中完成，
先写设计和预测，再看[独立验收提示](../solutions/resume_capstone.md)。当前基线没有
`bgraph.square`；完成前不要把它写成现有能力。

## 任务契约

新增 `bgraph.square`，输入/输出为 ranked f32 tensor，shape 相同，语义是每个元素
执行一次 f32 `x*x`。本次先实现 MLIR 侧闭环，不新增 ONNX schema 或前端支持声明。
不引入提升精度、自动 fast-math 或其他额外语义。

完成 ODS、verifier、shape pass、直接 Linalg lowering、fusion scalar emission 与测试。
当前 fused Region 已接受 arith.mulf；先检查实际白名单，决定是否真需要修改。
不要为了凑文件数而改没有必要的层。

## 第一步：提交设计，不写实现

写清 operand/result、dtype/shape、副作用、动态维与静态 lowering 边界。预测
`[-2, -0, 3]` 的输出，并讨论 NaN/Inf。列出每个要改的符号和原因；说明为什么此练习
不需要 custom Type、TypeConverter、ONNX 新算子名字。

## 第二步：独立完成代码

可以查现有 Op 的 API，但不要直接索要完整 Square 实现。每完成一个阶段保存 diff
和最小 IR，先验证直接 lowering，再接 fusion。关闭 fusion 的路径必须能工作，
避免某个优化 pass 遮掩了基础 lowering 的遗漏。

## 第三步：测试必须区分不同错误

| 证据 | 必须抓到什么 |
|---|---|
| round-trip | square 已注册，operand/result schema 可读写 |
| verifier negative | 非 f32、错误 shape 被拒绝，诊断落在 square |
| shape | 合法动态结果得到预期推导；不把动态 lowering 当已支持 |
| direct conversion | square 消失，目标 body 出现乘法，legality 仍严格 |
| fusion structure | square 与另一 elementwise Op 确实进入 fused Region |
| shared use | 共享 result 没有被错误删除或重复内联 |
| runner finite | 参考、direct、fused 结果一致 |
| floating boundary | 单独检查 NaN/Inf/signed-zero，不仅比较普通误差 |

先手推有限值预期，再记录实际输出；不能把两条路径“都等于某个意外值”当成正确。
最终 scalar reduction 可能掩盖逐元素错误，应补多个定点提取或其他针对性检查。

## 第四步：模拟追问

让另一位读者随机删一处实验接线，例如 lowering 注册或 shape dispatch。先预测
失败阶段，再定位并恢复；保留可审阅 diff，不做破坏性回滚。限时解释修改层次、
无需修改的层次、类型与语义的区别、fusion 命中证据和拒绝边界。

## 交付物与门槛

实验副本不包含 `.deps`，首次及每个新 shell 的完整验收按
[setup 的 Python 回归依赖](../dialect_lab/00_lab_setup.md)接入基线包目录；lit 会加入
MLIR bindings。不要把缺依赖误判为新 Op 的回归失败。

提交契约笔记、个人 diff、正例与两类反例、direct/fused 结构和数值记录，以及一次
失败定位复盘。实际修改源码后才需构建实验副本；阅读本题无需重建基线。

独立完成对应[导读](../resume_alignment.md)的 L3。若仅能解释现有 Relu/Clamp，
暂记 L2，不把“会读 ConversionPattern”勾成“会写”。
