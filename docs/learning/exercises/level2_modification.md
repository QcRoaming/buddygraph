# Level 2：小修改

本级建议在你自己的 BuddyGraph 工作副本完成。每做完一题先回归，再开始下一题。

## 任务 1：增强 verifier 与 negative test

- **任务**：增强 `bgraph.batch_norm` 的现有 custom verifier：当 variance 来自可见的
  dense constant 时，各元素必须有限且非负；动态/非常量 variance 保持当前可接受性。
  添加一个诊断测试，证明确定为负的 variance 会被拒绝。
- **涉及文件**：`lib/BuddyGraph/IR/BGraphOps.cpp`、
  `tests/Dialect/BGraph/invalid.mlir`（或新建同目录测试）。
- **禁止直接修改**：`build/include/**/*.inc`、LLVM/MLIR/Buddy 源码。
- **预期证据**：negative test 用 `-verify-diagnostics` 通过；合法正值和非常量输入仍
  通过；`check-buddygraph` 全部通过。
- **提示**：`BatchNormOp` 已有 `hasVerifier = 1`，不要重复改 ODS。可复用文件中的
  constant matching 风格；诊断文本要稳定且能被 `expected-error` 精确匹配。

## 任务 2：增加 identity canonicalization

- **任务**：为当前尚未实现的 `bgraph.div(%x, %one) -> %x` 建立一个**受约束**的
  identity canonicalization。只匹配可证明为全 1 的 constant，且类型必须完全相同。
- **涉及文件**：`lib/BuddyGraph/IR/BGraphOps.cpp`、
  `tests/Dialect/BGraph/canonicalize.mlir`。
- **禁止直接修改**：通用 `arith` 实现；不得放宽到 shape/broadcast 不安全的情况。
- **预期证据**：`buddygraph-opt --canonicalize` 前后 IR diff；正例消除 Div，非 1 和
  类型不同的反例保留；lit 回归通过。
- **提示**：`DivOp` 已从 `BGraph_BinaryOp` 获得 `hasCanonicalizer = 1`，但当前注册函数
  为空。这是 canonicalization pattern，不是 ODS `fold()` hook。

## 任务 3：增加 Pass option

- **任务**：给 shape inference pass 增加布尔选项 `fail-on-dynamic`。默认值为 false，
  保持当前行为；开启时，只要 BGraph 结果在推断后仍含动态维，就令 pass failure 并输出
  可定位诊断。
- **涉及文件**：`include/BuddyGraph/Transforms/Passes.td`、
  `lib/BuddyGraph/Transforms/InferShapes.cpp`、
  `tests/Dialect/BGraph/infer-shapes.mlir`。
- **禁止直接修改**：生成的 `Passes.h.inc`；不得改变默认 pipeline 的既有结果。
- **预期证据**：`--help` 能看到 option；默认路径原测试通过；显式开启时 negative 用例
  失败且诊断稳定。
- **提示**：TableGen Pass option 生成的是 pass 字段；先观察生成头中字段名，再使用。

## 任务 4：增加 IR dump 检查

- **任务**：给 BN folding 测试增加“该 pass 前后 IR 可观察”的检查，同时不把临时 dump
  文本提交为 golden file。任选 `-mlir-print-ir-after=bgraph-fold-bn-into-conv` 或
  `-mlir-print-ir-after-all`，把 stderr 合并给 FileCheck。
- **涉及文件**：`tests/Dialect/BGraph/fold-bn.mlir`。
- **禁止直接修改**：pass 实现；不得用仅匹配空字符串的宽泛 CHECK。
- **预期证据**：RUN 行可单独复制执行；至少匹配 dump banner；在 `@fold` 范围内 BN
  消失，在 `@multiple_users` 范围内 BN 仍存在。
- **提示**：IR dump 默认写 stderr；测试命令通常需要 `2>&1`。必须用
  `CHECK-LABEL` 划分两个函数，不能把全文件 `CHECK-NOT` 写到 EOF。

## 本级验收

```bash
cmake --build build
cmake --build build --target check-buddygraph
```

提交你的 patch、每题最小命令和输出摘要。答案提供的是实现路线，不应替代你对当前
MLIR 生成 API 的检查。

完成后再看 [Level 2 答案](../solutions/level2_modification.md)。
