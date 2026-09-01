# Level 3：调试与故障注入

每次只注入一个故障。先保存原文件副本，记录失败，再恢复并跑全量回归。
临时输入统一放在项目 `$BUDDYGRAPH_TMP`（即
`/buddy-mlir/jlq/projects/buddygraph/tmp`）；开始前先创建该目录。

```bash
export BUDDYGRAPH_TMP=/buddy-mlir/jlq/projects/buddygraph/tmp
mkdir -p "$BUDDYGRAPH_TMP"
```

## 任务 1：移除 Dialect 注册

- **任务**：临时注释 `buddygraph-opt` 中 BGraph Dialect 注册，尝试解析
  `tests/Dialect/BGraph/roundtrip.mlir`。
- **涉及文件**：`tools/buddygraph-opt/buddygraph-opt.cpp`。
- **禁止直接修改**：MLIR 的 registry/parser；故障观察后必须恢复。
- **预期证据**：一条 unknown/unregistered dialect 类诊断、退出码、注册调用的根因说明。
- **提示**：构建后确认执行的是 `build/bin/buddygraph-opt`，不是 `buddy-opt`。

## 任务 2：制造 failed to legalize

- **任务**：给仅含动态 shape 的 `bgraph.add` 运行 `convert-bgraph-to-linalg`，说明 Pattern
  为何拒绝以及 FullConversion 为什么最终报错。
- **涉及文件**：新建 `$BUDDYGRAPH_TMP/dynamic-add.mlir`；阅读
  `lib/BuddyGraph/Conversion/BGraphToLinalg.cpp`。
- **禁止直接修改**：项目源码；本题通过输入触发。
- **预期证据**：`failed to legalize operation 'bgraph.add'` 或当前版本等价诊断，以及
  Pattern 的静态 shape 前置条件。
- **提示**：与第 10 章统一使用 lhs/result `tensor<?x3xf32>`、rhs
  `tensor<3xf32>`，并保留合法函数签名。

## 任务 3：制造错误 yield type

- **任务**：在一个 `bgraph.fused_elementwise` 中让 `bgraph.yield` 的值类型与 Op 结果
  不一致，观察 verifier 失败。
- **涉及文件**：新建 `$BUDDYGRAPH_TMP/bad-yield.mlir`；阅读
  `lib/BuddyGraph/IR/BGraphOps.cpp`。
- **禁止直接修改**：verifier 实现。
- **预期证据**：精确诊断、触发该诊断的 verifier 分支，以及修正后的可解析 IR。
- **提示**：先从 `tests/Dialect/BGraph/fuse-elementwise.mlir` 的 pass 输出获得一个合法
  fused op，再只改一个类型。

## 任务 4：制造多用户 fusion 边界

- **任务**：给 Add 的结果增加第二个外部用户，再运行 `bgraph-fuse-elementwise`；解释
  哪些 Op 保留、哪些可能仍融合。
- **涉及文件**：复制 `tests/Dialect/BGraph/fuse-elementwise.mlir` 到项目 `tmp/`；阅读
  `lib/BuddyGraph/Transforms/FuseElementwise.cpp`。
- **禁止直接修改**：正式测试和 pass。
- **预期证据**：故障注入前后 IR diff、`hasOneUse()` 的实际判断位置、没有错误删除外部
  用户的证明。
- **提示**：第二个用户可以是另一个 `bgraph.relu`，最终 return 两个结果。

## 任务 5：制造 BN channel 不匹配

- **任务**：让 scale/bias/mean/variance 的元素数与 Conv 输出通道数不一致，运行
  `bgraph-fold-bn-into-conv`。
- **涉及文件**：复制 `tests/Dialect/BGraph/fold-bn.mlir` 到项目 `tmp/`；阅读
  `lib/BuddyGraph/Transforms/FoldBatchNorm.cpp`。
- **禁止直接修改**：正式测试和 Pattern。
- **预期证据**：当前实现应在运行 Pattern 前由 `BatchNormOp::verify()` 拒绝该 IR；记录
  非零退出和 channel-length 诊断，并解释 Pattern 中的 element-count 检查为何仍是
  defense in depth。
- **提示**：不要预设所有“优化不满足条件”都会到达 Pattern。先不加 pass 验证输入，
  再加 pass；两者都在同一 verifier 边界失败。

## 本级验收

- 五次实验都包含：输入 diff、命令、退出码/IR、根因、恢复验证。
- 你能区分 parse/verify failure、Pattern match failure 和 FullConversion failure。
- 恢复后 `check-buddygraph` 全部通过。

完成后再看 [Level 3 答案](../solutions/level3_debugging.md)。
