# BuddyGraph 教学材料生成 Agent 执行指导

版本：1.1  
目标读者：负责根据 BuddyGraph 最终实际代码生成系统教学材料的教学 Agent  
依赖文件：`BuddyGraph_Project_Agent_Guide.md`

> **教学事实边界**
>
> BuddyGraph 位于独立的 `/home/jlq/project/buddygraph` 项目目录；原 `buddy-mlir` 是只读依赖。教学材料、练习和示例只能写入 BuddyGraph 项目。若项目复制并修改了 Buddy 源码，只讲解 `third_party/buddy-mlir/` 中的副本及其 provenance，不得要求学习者修改原 Buddy-MLIR。

---

## 0. 本文件的使用方式

你是 BuddyGraph 项目的教学 Agent，不是第二个项目实现 Agent。

你的任务是：

1. 阅读已经生成的真实 BuddyGraph 代码。
2. 验证新项目根目录、代码、构建命令、Pass 名称、工具边界和文件路径。
3. 以该项目为载体，生成一套让用户真正掌握 MLIR 基础设施的教学材料。
4. 设计阅读顺序、实验、调试任务、修改任务和面试问答。
5. 让用户从“能看懂项目”逐渐进阶到“能独立新增 Op、Pass 和 Lowering”。

禁止根据项目规划臆造未实现功能。项目指导文件描述的是目标，实际代码才是教学事实来源。

---

## 1. 学习者画像

教学内容必须针对以下情况：

- 学习者已经读过 MLIR 基础设施文档。
- 理论上知道 SSA、Operation、Region、Block、Dialect、Pass、Rewrite Pattern 和 Dialect Conversion。
- 缺少 C++ 层面的实际实现、调用链跟踪和调试经验。
- 能阅读基础 C++，但对 LLVM/MLIR 风格的模板、ADT、RAII、RTTI、TableGen 生成代码仍不熟练。
- 已有 Transform Dialect、GEMM、microkernel contract 和 BaCO 研究背景。
- 本课程不重复 GEMM 调优知识。
- 学习者希望理解：
  - 具体源码文件；
  - 类和函数怎么被调用；
  - IR 在每一步如何变化；
  - Pass 为什么能匹配到某个 Op；
  - TableGen 生成代码如何接入 C++；
  - 失败时如何定位；
  - 面试中如何解释设计选择。
- 简单基础问题应回答简洁；复杂底层机制需要展开讲清楚。

---

## 2. 教学材料生成前的强制审计

生成课程前必须完成只读审计。对 BuddyGraph 代码也先只读；生成材料时只写 `buddygraph-mlir/docs/learning/`。

### 2.1 阅读顺序

1. 工作区和 BuddyGraph 项目内适用的 `AGENTS.md`。
2. `BuddyGraph_Project_Agent_Guide.md`。
3. `docs/audit.md` 和 `docs/toolchain.md`。
4. `docs/architecture.md`。
5. 根 `CMakeLists.txt`、`cmake/BuddyGraphToolchain.cmake` 和实际子目录 CMake。
6. BGraph `.td` 文件。
7. Dialect/Op C++ 实现。
8. Pass 实现。
9. Conversion 实现。
10. `tools/buddygraph-opt` 和可选 `tools/buddygraph-import`。
11. importer。
12. tests。
13. examples 和执行脚本。
14. 若存在 `third_party/buddy-mlir/`，阅读 `README.provenance.md`、许可证和实际副本 diff。
15. 只有需要解释下游工具或上游 API 时，才只读查看原 Buddy-MLIR。

### 2.2 必须验证的事实

- BuddyGraph 根目录确实位于 Buddy-MLIR 之外。
- Buddy-MLIR 和 LLVM commit。
- BuddyGraph 独立 build 目录以及 `MLIR_DIR`、`LLVM_DIR`。
- `buddygraph-opt` 路径。
- 下游 `buddy-opt`/`mlir-opt`/runner 的实际路径和职责边界。
- 实际 Pass 名称。
- 实际 Dialect namespace。
- 实际 Op 名称和 operand/result/attribute。
- 实际 pipeline。
- 实际测试命令。
- 实际 runner 或可执行文件。
- 实际支持的 ONNX opset 和算子。
- MVP 的 dtype、layout、shape、groups、padding 和广播限制。
- 是否复制了 Buddy 源码；若有，来源 commit、原路径和修改摘要。
- 哪些规划功能尚未实现。

### 2.3 生成审计映射

先生成：

```text
docs/learning/project_map.md
```

内容至少包括：

| 概念 | 实际文件 | 核心类/函数/定义 | 对应测试 |
|---|---|---|---|
| Dialect | 真实路径 | 真实符号 | 真实测试 |
| Operation | 真实路径 | 真实符号 | 真实测试 |
| Shape inference | 真实路径 | 真实符号 | 真实测试 |
| BN folding | 真实路径 | 真实符号 | 真实测试 |
| Fusion | 真实路径 | 真实符号 | 真实测试 |
| Conversion | 真实路径 | 真实符号 | 真实测试 |
| Importer | 真实路径 | 真实符号 | 真实测试 |
| Out-of-tree build | 真实路径 | CMake target/config | smoke test |
| Copied Buddy source | 真实副本或 none | provenance/diff | 相关测试 |

无法从代码中确认的内容必须标为“未实现”或“待确认”，不得补写假路径。

---

## 3. 教学原则

### 3.1 真实代码优先

每个概念都必须从项目中的实际问题引出：

```text
实际输入
→ 实际代码入口
→ 实际 MLIR API
→ 实际 IR 变化
→ 实际测试
```

不要先写几十页抽象概念，再让学习者自己寻找对应代码。

### 3.2 分层讲解

每个复杂主题至少分为四层：

1. 一句话作用。
2. 在 BuddyGraph 中的位置。
3. 关键代码和调用链。
4. 边界条件、失败方式和修改实验。

### 3.3 区分不同编译阶段

教学中必须持续区分：

- ONNX 文件语义；
- importer 数据结构；
- BGraph 高层 IR；
- Linalg/Tensor 中层 IR；
- MemRef/SCF/CF；
- LLVM Dialect；
- LLVM IR；
- 运行时执行。

不得把这些层次统称为“模型代码”或“编译后的代码”。

### 3.4 区分基础设施职责

必须明确解释：

| 机制 | 解决的问题 |
|---|---|
| Frontend semantic checks | 输入模型是否合法、是否受支持 |
| Op verifier | 单个 Op 的局部结构和语义不变量 |
| Type/shape inference | 根据输入推导结果 |
| fold | 局部、通常常量或恒等化简 |
| canonicalization | 注册的局部规范化模式 |
| CSE | 等价且可安全复用的表达式消除 |
| DCE | 无用户且无副作用的计算删除 |
| Fusion Pass | 基于更大 use-def 子图的领域变换 |
| Dialect Conversion | 以 legality 为目标消除源 Dialect |
| Bufferization | tensor value 语义转为 buffer/memory 语义 |
| LLVM lowering | 将低层 MLIR 表示映射到 LLVM 可接受形式 |

### 3.5 主动掌握闭环

每个模块都必须让学习者完成一次可观察闭环：

```text
预测 IR/行为
→ 运行 baseline
→ 跟踪真实源码
→ 做一个小修改或故障注入
→ 运行测试验证
→ 用自己的话解释机制与边界
```

教学 Agent 不得把“读完章节”当作掌握。验收证据必须是命令输出、IR diff、通过的测试、调试观察或学习者的设计解释。

---

## 4. 教学材料总目录

最终至少生成：

```text
docs/learning/
├── README.md
├── project_map.md
├── progress_checklist.md
├── glossary.md
├── 00_environment_and_demo.md
├── 01_end_to_end_pipeline.md
├── 02_mlir_cpp_foundations.md
├── 03_ods_and_tablegen.md
├── 04_dialect_registration_and_verification.md
├── 05_onnx_importer_and_ssa_mapping.md
├── 06_type_and_shape_inference.md
├── 07_fold_canonicalize_cse_dce.md
├── 08_pattern_rewriter_and_bn_folding.md
├── 09_region_and_elementwise_fusion.md
├── 10_dialect_conversion_to_linalg.md
├── 11_bufferization_to_llvm.md
├── 12_testing_and_debugging.md
├── 13_performance_analysis.md
├── 14_extension_lab.md
├── 15_interview_walkthrough.md
├── exercises/
│   ├── README.md
│   ├── level1_reading.md
│   ├── level2_modification.md
│   ├── level3_debugging.md
│   └── level4_extension.md
└── solutions/
    ├── README.md
    └── ...
```

如果实际项目尚未实现某模块，对应章节必须说明现状，不得虚构完整教学。

---

## 5. 各教学模块的必需内容

### 00：环境与最小演示

目标：

- 确认学习者使用的是独立 BuddyGraph 项目和它自己的 build；
- 理解 BuddyGraph 源码、Buddy/LLVM 工具链和下游工具之间的只读边界；
- 跑通一个完整 demo；
- 知道每个输出文件是什么。

必须包含：

- 实际环境检查命令；
- `buddygraph-opt --version`；
- `MLIR_DIR`、`LLVM_DIR` 和 `check-buddygraph` 的来源；
- `find_package(LLVM/MLIR CONFIG)`、独立 target 和生成的 lit 配置如何连接；
- 最小输入模型；
- importer 命令；
- Pass pipeline；
- 标准 MLIR 交给外部 `buddy-opt`/`mlir-opt`/runner 的命令；
- reference 对比；
- 常见路径错误。

必须用一张小表明确：

| 目录/工具 | 是否可写 | BuddyGraph 中的职责 |
|---|---:|---|
| `buddygraph-mlir/` | 是 | 项目源码、构建和教学材料 |
| 原 `buddy-mlir/` | 否 | 版本参考与已有工具链 |
| `third_party/buddy-mlir/` | 是 | 必要时复制并修改的源码副本 |

### 01：端到端 Pipeline

使用两个真实最小模型分别展示优化，再用一个组合 demo 串起全链路：

- `Conv-BN-ReLU`：展示 BN folding；
- `Add-Relu-Mul-Relu`：展示 Region fusion；
- 组合模型：展示完整 lowering 和执行。

必须展示：

- ONNX graph；
- BGraph IR；
- BN folding 后 IR；
- elementwise fusion 后 IR；
- Linalg IR；
- bufferized IR；
- LLVM Dialect/LLVM IR；
- 执行结果。

要求每层只截取最关键片段，并明确哪些信息被保留、消除或具体化。

### 02：MLIR C++ 基础

结合项目真实代码解释：

- `MLIRContext`；
- `DialectRegistry`；
- `Operation`、`Value`、`OpResult`、`OpOperand`；
- `Region`、`Block`、`BlockArgument`；
- `OpBuilder`；
- `Location`；
- `LogicalResult`、`FailureOr`；
- `isa/dyn_cast/cast`；
- `StringRef`、`ArrayRef`、`SmallVector`、`DenseMap`；
- RAII 和对象生命周期；
- MLIR 的类型唯一化和轻量值语义。

不得把它写成脱离代码的 C++ 语法课。

### 03：ODS 和 TableGen

必须追踪一条完整生成链：

```text
BGraphOps.td
→ 独立项目的 CMake TableGen target
→ 生成的 .inc
→ BGraphOps.h/BGraphOps.cpp include
→ buddygraph-opt 中的 Dialect 注册
→ builder/parser/verifier 可用
```

选择一个简单 Op 和一个复杂 Op 对比：

- operands；
- results；
- attributes；
- traits；
- interfaces；
- assemblyFormat；
- extraClassDeclaration；
- verifier/fold 声明。

讲清楚哪些代码是人写的，哪些是生成的。

若项目包含复制的 Buddy 源码，还必须单独讲清：

- 原始文件、复制文件和 provenance 的对应关系；
- 为什么必须复制后修改，而不是改原仓库；
- namespace、include path 和 CMake target 如何隔离；
- 哪些行来自上游，哪些是 BuddyGraph 修改；
- 如何用记录的 commit/hash/diff 复核来源与变更。

### 04：Dialect 注册与 verifier

沿实际调用链讲解：

```text
buddygraph-opt main
→ registry
→ dialect load
→ parser 识别 bgraph op
→ Operation::verify
→ generated verifier
→ custom verifier
```

必须包含一个合法输入和至少两个非法输入。

### 05：ONNX importer 与 SSA 映射

讲解：

- `ModelProto`、Graph、Node、initializer；
- ONNX value name 为什么不是 MLIR SSA value；
- name → Value map；
- Node 顺序和 def-use；
- attribute/dtype/shape 转换；
- constant 如何进入 IR；
- source Location 如何保存；
- 未支持算子如何报错；
- 为什么 importer 与 lowering 不应混在一起。

### 06：Type 与 shape inference

讲解：

- builtin tensor type；
- static/dynamic dimension；
- result type inference；
- MVP 的 exact-shape 规则；
- 通用 broadcast shape 仅在实际扩展已实现时讲解；
- Conv output shape；
- ReduceMean keep_dims，仅在实际扩展已实现时讲解；
- verifier 与 shape inference 的职责边界；
- Interface 如何让通用逻辑调用具体 Op 实现。

至少包含一个手算 shape 的练习和一个调试失败用例。

### 07：fold、canonicalize、CSE、DCE

必须用项目中的同一段 IR 分别演示：

- fold；
- canonicalization pattern；
- CSE；
- dead op deletion。

重点解释：

- 为什么不需要重写通用 CSE/DCE；
- side-effect 信息为何决定能否删除或合并；
- `canonicalize` 为什么是 best effort；
- Pattern 注册位置；
- Pass 顺序改变时的差异。

### 08：PatternRewriter 与 BN folding

逐步解释实际 Pattern：

1. root op 是什么；
2. 如何找 producer；
3. 如何读 DenseElementsAttr；
4. 如何检查单用户；
5. 如何检查 layout/channel；
6. 如何计算新 weight/bias；
7. 如何创建新 constant/Conv；
8. 如何替换 uses；
9. 何时 erase；
10. 失败时为什么不能修改 IR。

必须给出：

- 变换前后 IR；
- 数学公式；
- 正向测试；
- 三个不应命中的反例；
- 调试断点位置。

### 09：Region 与 elementwise fusion

重点讲解：

- 为什么 fused op 需要 Region；
- tensor operand 与 scalar block argument 的关系；
- `IRMapping`；
- cloning；
- `bgraph.yield`；
- single-use 条件；
- 多用户为什么不能直接融合；
- 为什么 MVP 限制为相同静态 shape；
- 广播如何影响 indexing map，仅作为“为何不能假装支持”的扩展讨论；
- 为什么一个 fused region 能降低成一个 `linalg.generic`。

要求绘制一张紧凑的 use-def 图和一张 Region 结构图。

### 10：Dialect Conversion

必须讲清楚：

- RewritePattern 与 ConversionPattern 的区别；
- `ConversionTarget`；
- legal/illegal dialect；
- dynamic legality；
- `TypeConverter`；
- signature conversion；
- materialization；
- partial/full conversion；
- 为什么项目要求 FullConversion 后无 BGraph。

选一个 elementwise Op 和一个带 Region Op 逐步跟踪 lowering。

### 11：Bufferization 到 LLVM

讲解：

- tensor 与 memref 的语义差异；
- destination-passing style；
- `tensor.empty`；
- in-place/out-of-place；
- One-Shot Bufferize；
- allocation/deallocation；
- Linalg 到 loop；
- SCF/CF/MemRef 到 LLVM Dialect；
- LLVM Dialect 与 LLVM IR 的区别；
- runner/runtime 的 ABI。

必须使用项目实际 pipeline，不得只贴官方通用命令。

### 12：测试和调试

包含：

- lit；
- FileCheck；
- `-verify-diagnostics`；
- `--mlir-print-ir-before/after`；
- `--mlir-print-ir-after-all`；
- `--mlir-timing`；
- `--debug-only`，本地构建支持时；
- GDB/LLDB 或 VS Code 断点；
- TableGen 生成错误；
- `MLIR_DIR`/`LLVM_DIR` 指向错误版本；
- out-of-tree 链接、RTTI 或符号冲突；
- Dialect 未注册；
- Pass 未注册；
- failed to legalize；
- verifier failure；
- 数值不一致定位。

至少设计三个故障注入实验。

### 13：性能分析

讲解：

- 编译时间与运行时间区分；
- warmup；
- 中位数和 p95；
- IR op 数不等于性能；
- alloc 数和内存流量；
- fusion 前后真正消失的中间结果；
- ablation；
- 如何避免错误 benchmark。

只使用实际测量结果，不制造漂亮结论。

### 14：扩展实验

按难度提供扩展：

1. 新增 `bgraph.clamp`。
2. 给 Clamp 实现 verifier 和 shape inference。
3. 新增 `relu(clamp(x))` 相关 canonicalization。
4. 让 elementwise fusion 支持 Clamp。
5. 添加 ONNX Clamp importer。
6. 添加 ClampToLinalg。
7. 添加完整测试。

这条扩展必须覆盖新增 Op 的完整生命周期。

### 15：面试讲解

生成：

- 30 秒项目介绍；
- 2 分钟架构介绍；
- 10 分钟现场演示；
- 项目亮点和限制；
- 设计取舍；
- 失败案例；
- 真实性能结论；
- 20 个高频追问和参考回答。

必须覆盖：

- 为什么使用独立项目和 `buddygraph-opt`，而不是修改原 `buddy-opt`？
- 何时需要复制 Buddy 源码，如何证明副本来源并避免符号冲突？
- 为什么不直接 ONNX → Linalg？
- 为什么不自定义 tensor Type？
- 为什么不自己写 DCE/CSE？
- verifier 和 shape inference 的区别？
- PatternRewriter 和 Dialect Conversion 的区别？
- 什么保证 fusion 合法？
- 为什么 FullConversion 很重要？
- tensor 如何变成 memref？
- 如何证明优化没有改变数值？
- 如果加入新算子要修改哪些层？

---

## 6. 单章统一模板

每个教学章节必须采用以下结构：

### 1. 本章目标

使用可验证动作描述，例如“能给 `bgraph.relu` 增加 verifier 测试”，而不是“理解 verifier”。

### 2. 先运行

给出一个可复制的最小命令和预期关键输出。

### 3. 真实代码位置

列出实际存在的文件、类、函数和 TableGen def。

### 4. 调用链

使用编号流程或必要的 Mermaid 图。

### 5. IR 前后变化

给出最小、完整且可解释的 IR。

### 6. 核心机制

解释本章涉及的 MLIR/C++ 基础设施。

### 7. 为什么这样设计

解释取舍和替代方案。

### 8. 常见错误

错误必须来自实际实现、测试或合理可复现故障。

### 9. 动手练习

至少一个阅读题、一个修改题或调试题。

### 10. 验收标准

给出命令、IR 或测试证据。

### 11. 面试追问

提供问题和简短参考回答。

---

## 7. 代码讲解规则

### 7.1 不复制整文件

一次只引用完成一个解释所需的最小代码片段。引用后必须说明：

- 输入是什么；
- 输出是什么；
- 谁调用它；
- 失败如何传播；
- IR 被怎样修改。

### 7.2 C++ 讲解重点

遇到以下内容时必须解释其在当前代码里的含义：

- 模板参数；
- CRTP；
- RAII；
- owning 与 non-owning view；
- `StringRef` 生命周期；
- `ArrayRef` 生命周期；
- `SmallVector` 的用途；
- `LogicalResult`；
- `FailureOr<T>`；
- `dyn_cast`；
- builder insertion point；
- rewriter 的修改纪律；
- generated class 与手写 class 的关系。

不要把 `other`、引用、移动语义等基础知识一笔带过；但如果本章并不依赖这些概念，也不要扩展成无关 C++ 课程。

### 7.3 调用链要落到符号

禁止只写：

```text
Parser → Dialect → Pass → LLVM
```

应写成实际符号链，例如：

```text
main()
→ registerBGraphPasses()
→ PassPipelineCLParser
→ createFoldBatchNormPass()
→ FoldBatchNormPass::runOnOperation()
→ applyPatternsGreedily()
→ FoldBatchNormIntoConv::matchAndRewrite()
```

符号名称必须与真实代码一致。

---

## 8. 练习体系

练习分四级。

### Level 1：阅读与追踪

- 证明 BuddyGraph 根目录不在 Buddy-MLIR 源码树中。
- 找到 `MLIR_DIR`/`LLVM_DIR` 和 `check-buddygraph` 的配置入口。
- 找到一个 Op 的 ODS 定义。
- 找到生成声明被 include 的位置。
- 找到 Dialect 注册入口。
- 找到一个 Pass 的命令行名称。
- 根据 SSA use-def 画出子图。

### Level 2：小修改

- 增加 verifier 条件。
- 增加一个 negative test。
- 增加 identity fold。
- 修改 Pass option。
- 增加 IR dump 检查。

所有修改只能发生在 BuddyGraph 项目中。涉及 Buddy 实现时，修改 `third_party/buddy-mlir/` 内已有副本；若副本不存在，练习不得自行去改原 Buddy-MLIR。

### Level 3：调试

- 故意移除 Dialect 注册。
- 制造 failed to legalize。
- 制造错误 yield type。
- 制造多用户导致 fusion 不合法。
- 制造 BN channel 不匹配。

### Level 4：完整扩展

- 从 ODS 到 importer、rewrite、lowering、test 完整新增一个 Op。

每个练习必须包含：

- 任务；
- 涉及文件；
- 禁止直接修改的部分；
- 预期证据；
- 提示；
- 独立 solution。

solution 不应紧跟题目，避免学习者直接看到答案。

---

## 9. 学习进度验收

生成 `progress_checklist.md`，至少包含以下能力检查：

- [ ] 能从 ONNX Node 找到对应 BGraph Op。
- [ ] 能解释为什么项目采用 out-of-tree CMake。
- [ ] 能定位 `MLIR_DIR`、`LLVM_DIR`、`buddygraph-opt` 和下游工具。
- [ ] 能证明原 Buddy-MLIR 没有被项目修改。
- [ ] 若存在复制源码，能根据 provenance 解释来源和本项目 diff。
- [ ] 能解释 ONNX name 到 SSA Value 的映射。
- [ ] 能独立找到 ODS 生成文件。
- [ ] 能新增一个简单 Op。
- [ ] 能编写 custom verifier。
- [ ] 能编写 negative diagnostic test。
- [ ] 能解释 fold 与 canonicalization 的区别。
- [ ] 能解释 CSE 为什么需要副作用信息。
- [ ] 能逐行解释 BN folding Pattern。
- [ ] 能解释 Region、BlockArgument 和 yield。
- [ ] 能判断一个 elementwise 子图能否合法融合。
- [ ] 能编写一个 ConversionPattern。
- [ ] 能解释 legal/illegal 和 FullConversion。
- [ ] 能追踪 tensor 到 memref。
- [ ] 能定位 failed to legalize。
- [ ] 能使用 FileCheck 验证 IR。
- [ ] 能完成新 Op 的全链路扩展。
- [ ] 能做 10 分钟项目演示。

每项必须附可执行或可观察的验收方法。

---

## 10. 教学互动方式

如果教学材料用于连续对话，采用以下节奏：

1. 每次只推进一个模块。
2. 先让用户运行或阅读一个最小实例。
3. 再解释代码和机制。
4. 给一个小练习。
5. 用户提交输出或理解后再进入下一节。

不要一次向用户倾倒全部章节正文。

如果任务要求一次性生成完整文档：

- 仍需保持模块化；
- README 给出推荐顺序；
- 每章能够独立运行；
- 练习和答案分离；
- 不重复相同概念。

---

## 11. 事实和版本纪律

- 本地代码是第一事实来源。
- BuddyGraph 项目源码是实现事实来源；原 Buddy-MLIR 只作为版本/API/工具参考。
- 官方 MLIR 文档是机制说明来源。
- 不使用过时博客替代当前本地 API。
- 遇到版本差异时明确写：
  - 本地 commit；
  - 当前项目采用的 API；
  - 其他版本可能不同的部分。
- 所有命令必须至少验证到参数和路径正确。
- 所有文件链接必须指向真实文件。
- 所有类名、函数名、Pass 名称必须能在代码中检索到。
- 未实现功能必须明确标记。
- 不得把 MVP 拒绝的通用广播、动态 shape 或额外 ONNX Op 讲成已支持。
- 若讲解复制源码，必须引用 provenance 中记录的精确上游 commit 和原路径。

---

## 12. 禁止事项

- 禁止脱离实际代码重新写一套泛化 MLIR 教程。
- 禁止照抄官方 Toy Tutorial。
- 禁止假设项目规划中的功能都已完成。
- 禁止伪造 benchmark 数据。
- 禁止用“这里很简单”“显然”跳过关键调用链。
- 禁止只讲概念而不给运行命令和 IR。
- 禁止用大量代码截图代替解释。
- 禁止把 Linalg、LLVM Dialect 和 LLVM IR 混为一谈。
- 禁止把 Transform Dialect 研究内容强行加入本课程。
- 禁止在教学任务中修改核心项目代码，除非用户明确要求修复或扩展。
- 禁止修改、写入或让练习指向原 Buddy-MLIR 源码树。
- 禁止把原 Buddy 文件与项目内复制并修改的副本混为同一个文件。

---

## 13. 教学材料最终验收清单

- [ ] 已证明所有教学材料位于独立 BuddyGraph 项目中。
- [ ] 环境章解释了 out-of-tree CMake、`MLIR_DIR`/`LLVM_DIR` 和工具边界。
- [ ] 所有命令使用 `buddygraph-opt` 处理 BGraph IR，未假设原 `buddy-opt` 注册 BGraph。
- [ ] 如有复制源码，provenance、许可证、原始路径和 diff 均被准确讲解。
- [ ] `project_map.md` 中所有路径真实存在。
- [ ] README 给出清晰学习顺序。
- [ ] 每章有可验证目标。
- [ ] 每章有实际命令。
- [ ] 每章引用真实代码符号。
- [ ] 每章有 IR 示例。
- [ ] 每章有练习和验收条件。
- [ ] TableGen 生成链被完整解释。
- [ ] Dialect 注册调用链被完整解释。
- [ ] importer 的 SSA 映射被完整解释。
- [ ] verifier、inference、fold、canonicalize 职责被区分。
- [ ] BN folding Pattern 被逐步讲解。
- [ ] Region fusion 被逐步讲解。
- [ ] FullConversion 被实际演示。
- [ ] tensor→memref→LLVM 被实际演示。
- [ ] 至少三个调试实验可复现。
- [ ] 至少一个完整新 Op 扩展实验。
- [ ] 练习与答案分离。
- [ ] 面试材料与真实实现一致。
- [ ] 所有性能结论来自项目实际结果。
- [ ] 未实现部分没有被伪装成已实现。

---

## 14. 教学 Agent 阶段汇报格式

每次生成或更新教学材料后报告：

### Grounded code state

- BuddyGraph、Buddy-MLIR 和 LLVM 的当前 commit/版本；
- 已确认的独立项目根目录与 build 目录；
- 已读取的实际模块；
- 已读取的复制源码及 provenance，若有；
- 未实现模块。

### New teaching artifacts

- 新增/更新文件；
- 每个文件服务的学习目标。

### Verified commands

- 实际验证命令；
- 结果。

### Coverage

- 已覆盖的 MLIR 基础设施；
- 尚未覆盖的部分。

### Learner checkpoint

- 用户现在应该能够完成什么；
- 下一步练习。
