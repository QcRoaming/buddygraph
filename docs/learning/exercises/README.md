# 分级练习

这里的练习把 00–15 章中的零散任务组合成四个可验收关卡。题目与答案刻意分开；
完成并保存证据后，再打开 `../solutions/`。

| 级别 | 重点 | 交付证据 | 建议入口 |
|---|---|---|---|
| [Level 1](level1_reading.md) | 阅读与追踪 | 路径、符号、调用链和 use-def 图 | 第 00–05 章 |
| [Level 2](level2_modification.md) | BuddyGraph 内的小修改 | patch、lit 输出和 IR diff | 第 04、07、12 章 |
| [Level 3](level3_debugging.md) | 可恢复的故障注入 | 诊断、根因和修复后回归 | 第 08–12 章 |
| [Level 4](level4_extension.md) | 审计并扩展 `bgraph.clamp` | ODS→importer→lowering→测试闭环 | 第 14 章 |

## 统一规则

1. 从项目根目录 `/buddy-mlir/jlq/projects/buddygraph` 执行命令。
2. 只修改当前 BuddyGraph 项目；不得修改 `/buddy-mlir` 中既有 Buddy/LLVM 源码。
3. 项目没有 `third_party/buddy-mlir/`，因此不存在可修改的 Buddy 副本。
4. Level 2–4 要求学习者在自己的 BuddyGraph 实验副本修改源码。当前目录可能未被
   Git 跟踪，不能把 branch 或
   `git checkout --` 当作既有恢复方案。开始前应先把实验副本纳入自己的版本控制，
   或逐一把待改文件备份到项目外的临时目录。
5. 每题至少保留一种证据：命令输出、IR diff、通过的测试、调试观察或设计解释。

推荐每级新建自己的记录文件，例如 `notes/level1.md`；`notes/` 只是建议，不是项目测试
的一部分。

## 可写实验目录

只读章节固定使用教程生成时的绝对路径。Level 2–4 开始前，明确设置你自己的可写
BuddyGraph 根目录：

```bash
export BUDDYGRAPH_LAB_ROOT=/absolute/path/to/your/buddygraph-lab
cd "$BUDDYGRAPH_LAB_ROOT"
test -f CMakeLists.txt
test -x build/bin/buddygraph-opt
```

如果你选择当前 `/buddy-mlir/jlq/projects/buddygraph`，应先自行建立可恢复的版本控制或
逐文件备份；如果选择另一个完整副本，应按[项目根 README](../../../README.md) 重新配置
该副本的 `build/`。后续练习里的相对路径和 `build/bin/buddygraph-opt` 都指向这个
`BUDDYGRAPH_LAB_ROOT`，不得一边改副本、一边误用原项目的 build。

## 最终回归

```bash
cmake --build build --target check-buddygraph -j2
/usr/bin/python3.10 /buddy-mlir/llvm/build/bin/llvm-lit -sv \
  build/tests/E2E/BGraph/onnx_elementwise.py
```

当前基线期望 `14/14` 通过；原 E2E 的 NumPy、未优化和全优化结果均为
`7.000000e+00`。若测试数因你新增用例而增长，应以“全部通过”为准。手工 runner 示例
及其 `1.600000e+01` 结果见[第 00 章](../00_environment_and_demo.md)。
