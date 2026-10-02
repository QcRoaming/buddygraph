# BuddyGraph 源码实战课程

这套课程帮助你把 BuddyGraph 学到能够解释、修改和接受技术追问的程度。若准备把项目
放进简历，先读[简历陈述与掌握标准](resume_alignment.md)，再按下表进入对应模块。
正文负责教实现过程，[深入追问题库](interview_bank.md)负责检验理解，不能用背答案
替代动手。学习前只需要基本 C++、Python 和 SSA 概念；具体 MLIR 对象关系从第 02 章补起。

全课程围绕一条编译链展开：

```text
ONNX opset 18
→ Python MLIR API importer
→ BGraph tensor IR
→ shape refinement / BN folding / elementwise fusion
→ Linalg / Tensor / Arith / Math
→ bufferized MemRef / loops
→ LLVM Dialect
→ LLVM IR 或 mlir-runner
```

## 先确认事实边界

- 以下命令以 WSL 目标目录 `/home/jlq/project/buddygraph` 为例；克隆到其他位置时
  替换路径。
- 它位于 Buddy-MLIR 文件树之外，有独立 Git 仓库、CMake、`build/` 和
  `buddygraph-opt`；仍需外部已构建的 LLVM/MLIR 工具链。
- `/buddy-mlir` 的既有 Buddy/LLVM 源码和工具链只读使用；本课程的修改练习只允许
  发生在 BuddyGraph 项目内。
- 项目没有 `third_party/buddy-mlir/`，没有复制 Buddy 源码，也没有
  `buddygraph-import` 可执行程序；前端入口是 Python 脚本。
- 项目有独立 Git 仓库；实际修订和工作区状态以 `git log`、`git status` 为准。
- 实现事实优先于最初规划。所有已知差异见 [project_map.md](project_map.md)。

## 运行前提

本教程优先复用当前已配置的 `build`、`.deps` 链接和外部 LLVM/MLIR build。
前两者指向项目目录外的缓存；如果
`build/bin/buddygraph-opt` 或 Python 依赖不存在，先按
[项目根 README](../../README.md) 的“构建”与 `PYTHONPATH` 步骤配置；教程不包含从源码
构建整个 LLVM/MLIR 的过程。

章节中出现的 `/buddy-mlir/...` 是原容器的工具链路径。在 WSL 中运行时，按
项目根 README 将其替换为本机已有的 LLVM/MLIR build 路径。

直接运行前可核对当前 CMake 选择的解释器：

```bash
rg '^_?Python3_EXECUTABLE:' build/CMakeCache.txt
rg '^config.python_executable' build/tests/lit.site.cfg.py
python3 --version
```

当前快照的 lit 配置使用 `/usr/bin/python3.10`；章节中的 `python3` 表示使用已安装
ONNX/NumPy 与 MLIR bindings 的 Python 3。第 00 章的最小 demo 从手写 MLIR 开始，
验证的是 BGraph lowering→runner；第 01 章才把 ONNX importer 纳入完整闭环。

教程中需要保留查看的中间产物统一放在项目内：

```bash
export BUDDYGRAPH_TMP=/home/jlq/project/buddygraph/tmp
mkdir -p "$BUDDYGRAPH_TMP"
```

该路径就是项目根的 `tmp/`，可在 VS Code 中直接打开；它已进入 `.gitignore`，不会把
生成的 ONNX、MLIR、LLVM IR 或 JSON 混入版本控制。教程不再把可观察产物写入系统
`/tmp`。新开 shell 或单独跳到某章时，重新执行该章“先运行”里的两行即可。
若 VS Code 仍隐藏它，请确认设置 `explorer.excludeGitIgnore` 没有开启；该设置会隐藏
所有被 `.gitignore` 排除的文件，而不只是 BuddyGraph 的 `tmp/`。

## 推荐学习顺序

### 先建立结构，再进入命令

每次读一章，先回答三个问题：**这一章接收什么、负责改变或检查什么、下一章依赖
什么结果？** 章节里的第 2 节是观察实验，第 3–6 节才是源码与原理解释；命令较长时
可以先读第 5–6 节再运行。带 `...` 的 IR 是局部示意，不要作为完整输入复制执行。
完整可运行输入会明确给出文件或完整函数。

| 简历主线 | 先学什么 → 再学什么 | 完成时必须交出的证据 |
|---|---|---|
| 自定义方言 | 02 对象关系 → 03 生成链 → 04 注册/验证 → dialect_lab 带做 | 生成/注册图、三类错误、实验新增 Op/Type/Trait/Interface/Pass |
| ONNX 前端 | 05 名字与 SSA → 06 shape 与兼容性 | 逐节点 value map、广播/Conv 手算、拒绝路径 |
| 图优化 | 07 通用清理 → 08 BN 数学与 Pattern → 09 Region fusion | 公式推导、共享用户反例、融合前后 Value 对照 |
| 后端执行 | 10 maps/legality → 11 DPS/buffer/LLVM | 手写 generic、failed-to-legalize 反例、内存冲突与 ABI 解释 |
| 综合证明 | 12 定位错误 → 13 测量边界 → 14 独立增量 → 15 现场展示 | 个人修改的 diff、正反测试和调试记录 |

第一次接触工程先完成第 00 章最小演示，再读第 01 章全链路；随后按上表逐条推进。
每条至少达到[导读的 L2](resume_alignment.md)，并选一条做到 L3。下面保留按编号的
完整索引，便于查阅。

| 阶段 | 章节 | 完成后应能做什么 |
|---|---|---|
| 建立全局图 | [00](00_environment_and_demo.md)–[01](01_end_to_end_pipeline.md) | 跑通 demo，并说清每层 IR |
| 读懂方言 | [02](02_mlir_cpp_foundations.md)–[04](04_dialect_registration_and_verification.md) | 从 ODS 追到生成类、注册和 verifier |
| 读懂前端与类型 | [05](05_onnx_importer_and_ssa_mapping.md)–[06](06_type_and_shape_inference.md) | 从 ONNX name 追到 SSA Value 和结果 shape |
| 读懂图优化 | [07](07_fold_canonicalize_cse_dce.md)–[09](09_region_and_elementwise_fusion.md) | 判断 Pattern 是否应命中并调试失败 |
| 读懂后端 | [10](10_dialect_conversion_to_linalg.md)–[11](11_bufferization_to_llvm.md) | 证明 FullConversion，并追到 LLVM IR |
| 建立工程能力 | [12](12_testing_and_debugging.md)–[13](13_performance_analysis.md) | 写 lit、定位失败、正确解释测量 |
| 独立扩展与表达 | [14](14_extension_lab.md)–[15](15_interview_walkthrough.md) | 沿真实 Clamp 全链路复盘扩展方法并完成面试演示 |

### 专题实战：亲手搭建 MLIR 基础设施

如果你的目标不是继续阅读业务 Pipeline，而是要亲手完成 Dialect 基础设施，请进入
[自定义 Dialect 基础设施实战线](dialect_lab/README.md)。它在隔离的 `bglab` 教学
Dialect 中连续带做：

```text
Dialect → Op → parameterized Type → reusable Trait
→ custom OpInterface → interface-driven Pass → lit/debugging
```

这条专题线是主课程第 02–04、06、12 章的实践补充，不改变当前 BGraph/ONNX 基线；
学习者只在自己的实验副本中落代码。

配套材料：

- [项目—概念映射](project_map.md)
- [进度验收清单](progress_checklist.md)
- [术语表](glossary.md)
- [四级练习](exercises/README.md)
- [Dialect/Op/Type/Trait/Interface/Pass 专题](dialect_lab/README.md)
- [分离答案](solutions/README.md)

## 每章使用方法

每章保留统一的 11 节，以便查找；不要求机械地从第一条命令一路复制到底。建议按
“结构 → 解释 → 预测 → 实验 → 复盘”的顺序：

1. 读章首路线，确认输入、输出和与简历的关系。
2. 看“IR 前后变化”与“核心机制”，把每个新对象对应到图中。
3. 不看结果，写下最小命令的输出预测与一个失败条件。
4. 运行实验，对照真实代码解释预测为何成立或为何错了。
5. 做练习并保留证据，最后用“面试追问”检查是否能脱离正文解释。

连续学习时一次只推进一章。先把命令输出和练习结论记录下来，再进入下一章。
一次性阅读时也不要跳过验收标准；“读过”不是“掌握”。

教程正文默认以当前 BuddyGraph 源码为只读事实。Level 2–4 中明确要求的源码修改应在
学习者自己的 BuddyGraph 实验副本中执行；其中 Level 4 先审计已经实现的 Clamp，再做
一个有界增量，不应把练习修改直接混入基线工程。

## 已验证基线

以下 LLVM/Buddy 修订来自生成课程时的历史快照；14/14 回归是在
2026-10-02 容器迁移后验证的。WSL 目标路径是使用示例，需在目标环境重新验证：

| 项目 | 已验证值 |
|---|---|
| Buddy-MLIR commit | `d7bb40cbac731175dc507f06f0655d81508f2ca2` |
| `/buddy-mlir/jlq` commit | `0a373a0c8ff3c21fb096af1105076565481b8947` |
| LLVM commit | `09b849a2ac83dbf4be1b2f01c767339e46cc34ae` |
| LLVM/MLIR | `21.0.0git`, Release，assertions enabled |
| BuddyGraph build | `build` 链接；实际目录由项目根 README 中的 `BUDDYGRAPH_CACHE` 决定 |
| 主工具 | `build/bin/buddygraph-opt` |
| 回归 | `14/14` 通过 |
| 数值闭环 | NumPy、未优化、全优化均为 `7.000000e+00` |

这些值是本地快照，不代表远程仓库或未来 checkout。开始学习前先运行第 00 章的
版本检查。
