# Level 3 答案：调试

## 1. Dialect 未注册

根因链是：`buddygraph-opt` 构造 `DialectRegistry`，注册 BGraph Dialect，再交给
`MlirOptMain`。移除注册后 parser 不认识 `bgraph.*`，会报告未注册/未知方言；这不是 Pass
未注册。若开启允许未注册方言的 parser 选项，解析现象会变化，但 typed custom syntax、
verifier 和 transforms 仍没有因此被正确注册。

## 2. failed to legalize

与第 10 章统一使用 lhs/result `tensor<?x3xf32>`、rhs `tensor<3xf32>` 和一个
`bgraph.add`。当前 conversion 的 elementwise lowering 要求所有相关 tensor shape
静态，以便构造迭代空间/映射；Pattern 返回 failure 后，整个 BGraph Dialect 又被
target 标为 illegal，于是 `applyFullConversion` 无法留下该 Op，最终发出
`failed to legalize`。这是 Pattern 不匹配升级为全转换失败的典型路径。

## 3. 错误 yield type

`FusedElementwiseOp::verify()` 检查 Region 结构、terminator 和 yield/结果对应关系。让
parent 结果继续保持 `tensor<...xf32>`，只把 Region 内被 yield 的 constant 改成 f64，
并让 `bgraph.yield` operand 为 f64。这样命中 parent/yield 的 f32 对应关系诊断；若把
parent 结果改成 f64，会先被“result must have f32 elements”拒绝，反而没有测试目标
contract。

## 4. 多用户 fusion

增加外部用户后，中间 producer 的 `hasOneUse()` 为 false。当前 fusion 必须保留这个值
及其 producer，不能把唯一计算搬进 Region 后删除外部仍需的定义。根据第二个用户加入的
位置，较短的单用户后缀仍可能融合；“未全链融合”不等于“完全没有融合”。应以输出 IR
逐项判断。

## 5. BN channel 不匹配

用普通静态 tensor 把 BN 参数长度改坏时，当前 `BatchNormOp::verify()` 会先发现参数不是
“rank 1 with the channel length”，因此 pass 尚未开始，进程返回非零。这题的正确结论
不是 `notifyMatchFailure`。Pattern 内仍重复检查 element counts，作为其他构造路径或
未来 contract 演化下的 defense in depth；但不能把通常不可达的防御分支伪装成当前
合法 IR 的正向实验。若要观察普通 match failure，应改用 Conv 多用户或非常量参数，
它们是合法 IR，fold pass 返回 0 且 BN 保留。
