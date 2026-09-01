# BuddyGraph 源码实战课程

这套课程面向“已经读过 MLIR 基础设施文档，但还没有独立跟踪过完整 C++ 工程”的
学习者。课程不重新讲一遍 Toy Tutorial，而是始终沿 BuddyGraph 的真实闭环推进：

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

- 真实源码根目录是 `/buddy-mlir/jlq/projects/buddygraph`。
- 它采用独立 CMake、独立 `build/` 和独立 `buddygraph-opt`，但物理路径仍位于
  `/buddy-mlir` 目录和 `/buddy-mlir/jlq` Git worktree 内；不能声称它位于
  Buddy-MLIR 文件树之外。
- `/buddy-mlir` 的既有 Buddy/LLVM 源码和工具链只读使用；本课程的修改练习只允许
  发生在 BuddyGraph 项目内。
- 项目没有 `third_party/buddy-mlir/`，没有复制 Buddy 源码，也没有
  `buddygraph-import` 可执行程序；前端入口是 Python 脚本。
- 项目当前在 `/buddy-mlir/jlq` worktree 中仍是未跟踪目录。课程不会把未提交状态
  误写为发布版本。
- 实现事实优先于最初规划。所有已知差异见 [project_map.md](project_map.md)。

## 运行前提

本教程优先复用当前已配置的 `build/`、项目 `.deps/` 和
`/buddy-mlir/llvm/build`。如果 `build/bin/buddygraph-opt` 或 Python 依赖不存在，先按
[项目根 README](../../README.md) 的“构建”与 `PYTHONPATH` 步骤配置；教程不包含从源码
构建整个 LLVM/MLIR 的过程。

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
export BUDDYGRAPH_TMP=/buddy-mlir/jlq/projects/buddygraph/tmp
mkdir -p "$BUDDYGRAPH_TMP"
```

该路径就是项目根的 `tmp/`，可在 VS Code 中直接打开；它已进入 `.gitignore`，不会把
生成的 ONNX、MLIR、LLVM IR 或 JSON 混入版本控制。教程不再把可观察产物写入系统
`/tmp`。新开 shell 或单独跳到某章时，重新执行该章“先运行”里的两行即可。
若 VS Code 仍隐藏它，请确认设置 `explorer.excludeGitIgnore` 没有开启；该设置会隐藏
所有被 `.gitignore` 排除的文件，而不只是 BuddyGraph 的 `tmp/`。

## 推荐学习顺序

| 阶段 | 章节 | 完成后应能做什么 |
|---|---|---|
| 建立全局图 | [00](00_environment_and_demo.md)–[01](01_end_to_end_pipeline.md) | 跑通 demo，并说清每层 IR |
| 读懂方言 | [02](02_mlir_cpp_foundations.md)–[04](04_dialect_registration_and_verification.md) | 从 ODS 追到生成类、注册和 verifier |
| 读懂前端与类型 | [05](05_onnx_importer_and_ssa_mapping.md)–[06](06_type_and_shape_inference.md) | 从 ONNX name 追到 SSA Value 和结果 shape |
| 读懂图优化 | [07](07_fold_canonicalize_cse_dce.md)–[09](09_region_and_elementwise_fusion.md) | 判断 Pattern 是否应命中并调试失败 |
| 读懂后端 | [10](10_dialect_conversion_to_linalg.md)–[11](11_bufferization_to_llvm.md) | 证明 FullConversion，并追到 LLVM IR |
| 建立工程能力 | [12](12_testing_and_debugging.md)–[13](13_performance_analysis.md) | 写 lit、定位失败、正确解释测量 |
| 独立扩展与表达 | [14](14_extension_lab.md)–[15](15_interview_walkthrough.md) | 沿真实 Clamp 全链路复盘扩展方法并完成面试演示 |

配套材料：

- [项目—概念映射](project_map.md)
- [进度验收清单](progress_checklist.md)
- [术语表](glossary.md)
- [四级练习](exercises/README.md)
- [分离答案](solutions/README.md)

## 每章使用方法

每章都按同一闭环组织：

1. 先预测命令会生成什么 IR。
2. 运行“先运行”中的最小命令。
3. 沿“真实代码位置”和“调用链”读源码。
4. 完成一个小修改或故障注入。
5. 用测试输出、IR diff 或自己的设计解释验收。

连续学习时一次只推进一章。先把命令输出和练习结论记录下来，再进入下一章。
一次性阅读时也不要跳过验收标准；“读过”不是“掌握”。

教程正文默认以当前 BuddyGraph 源码为只读事实。Level 2–4 中明确要求的源码修改应在
学习者自己的 BuddyGraph 实验副本中执行；其中 Level 4 先审计已经实现的 Clamp，再做
一个有界增量，不应把练习修改直接混入基线工程。

## 已验证基线

生成课程时的事实快照：

| 项目 | 已验证值 |
|---|---|
| Buddy-MLIR commit | `d7bb40cbac731175dc507f06f0655d81508f2ca2` |
| `/buddy-mlir/jlq` commit | `0a373a0c8ff3c21fb096af1105076565481b8947` |
| LLVM commit | `09b849a2ac83dbf4be1b2f01c767339e46cc34ae` |
| LLVM/MLIR | `21.0.0git`, Release，assertions enabled |
| BuddyGraph build | `/buddy-mlir/jlq/projects/buddygraph/build` |
| 主工具 | `build/bin/buddygraph-opt` |
| 回归 | `14/14` 通过 |
| 数值闭环 | NumPy、未优化、全优化均为 `7.000000e+00` |

这些值是本地快照，不代表远程仓库或未来 checkout。开始学习前先运行第 00 章的
版本检查。
