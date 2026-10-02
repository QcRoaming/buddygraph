# 从简历陈述到可验证的项目能力

这份导读按简历里的四条陈述组织学习。目标是：面试官从任何一个名词追问时，你能
解释输入输出、画出 IR、找到实现、说出反例，并完成一个有测试的小修改。
下文的问题是根据项目技术内容设计的模拟追问，不代表某家公司必问的题目。

## 先校准简历里的四条陈述

| 简历中的表述 | 当前代码事实 | 建议写法 |
|---|---|---|
| 自定义方言 `type、op、attribute` | BGraph 自定义 Op 和 Layout EnumAttr，复用 builtin ranked tensor；没有自己的 TypeDef | 写“定义 BGraph 方言及算子、布局属性，复用 builtin tensor 类型” |
| ONNX 节点名到 SSA Value | `self.values` 的键是 ONNX input/output **value name**；node name 进入 `Location` | 写“完成 ONNX value name 到 SSA Value 的映射并保留节点位置” |
| 静态 shape 推导 | 前端调用 ONNX shape inference；C++ `--bgraph-infer-shapes` 实现 BGraph result refinement | 分清“集成前端推导”和“实现方言内 refinement” |
| 设计实现 CSE/DCE | 项目注册 upstream `--cse`、`--canonicalize`，通过 Pure 与方言 pattern 配合通用清理 | 写“实现 BN folding、Region fusion，并集成 canonicalization/CSE 与无用计算清理” |
| 支持 ONNX Clamp | ONNX 名为 `Clip`，导入后叫 `bgraph.clamp`；目前限双 finite scalar f32 initializer bounds | 写“Clip（映射为 Clamp）等 11 种 ONNX 算子类型的受限子集” |
| BGraph → Linalg/Tensor/Arith | 目标还包括 Math，直接 BN lowering 会产生 `math.sqrt` | 写“BGraph → Linalg/Tensor/Arith/Math” |
| 完成 CPU 执行 | 当前是标准 lowering + `mlir-runner` JIT；没有生产推理 runtime、通用 tensor ABI 封装和完整释放链 | 保留“CPU 端到端执行”，不要引申为生产部署或 kernel 加速 |

这里区分的是实现事实。能否用“我实现了”描述自己的贡献，还需要你能解释和复现相应
工作；代码存在、教程读过和独立完成不是同一个证据。

### 与当前工程一致的简历版本

> **BuddyGraph — 基于 MLIR 的 ONNX 模型编译器（静态 f32 子集）**
>
> - 使用 TableGen/ODS 定义 BGraph 方言、算子与布局 EnumAttr，复用 builtin tensor
>   类型，实现 verifier 与 canonicalization，完成注册、校验和 IR round-trip。
> - 基于 Python MLIR bindings 实现 ONNX opset 18 导入器，覆盖 Conv、BatchNormalization、
>   Relu、Add/Sub/Mul/Div、Reshape、Transpose、ReduceMean、Clip 的受限子集；完成
>   value name 到 SSA Value 的映射，集成 ONNX 静态 shape 推导并实现 BGraph shape refinement。
> - 实现 Conv-BN 参数折叠与基于 Region 的 elementwise fusion，结合 MLIR
>   canonicalization/CSE 和无用计算清理，使用结构回归、数值对拍与浮点边界用例验证优化。
> - 使用 Dialect Conversion 将 BGraph 完全降低到 Linalg/Tensor/Arith/Math，接入
>   One-Shot Bufferization 与标准 LLVM lowering，通过 mlir-runner 完成 CPU 端到端执行。

若版面需要压缩，可删算子枚举中的一部分；不要删除“受限子集”却保留一个容易被理解为
完整 ONNX 支持的“11 算子”。目前适用范围见[项目事实映射](project_map.md)。

## 第一条：从 ODS 定义到能被工具识别的方言

### 面试官在核查什么

“定义了一个 Op”包含三段工作：声明结构、接入生成与注册、实现语义。只会解释
`let arguments` 还不足以说明它如何进入 `buddygraph-opt`。

```text
人写 .td schema ──TableGen──> 生成类/accessor/校验代码
                                  ↓ C++ include + library link
driver registry ──Context 加载──> Dialect::initialize 注册 Op/Attr
                                  ↓
文本 parser → Operation → verifier → 合法 IR
```

从[第 02 章](02_mlir_cpp_foundations.md)建立 Operation/Value/ownership 概念，接着读
[第 03 章](03_ods_and_tablegen.md)生成关系与[第 04 章](04_dialect_registration_and_verification.md)
运行时注册。然后完成 [dialect_lab 01–06](dialect_lab/README.md)，亲手补齐 Type、Trait、
Interface 和 Pass 的构建链。`bglab` 的实验结果应作为补充能力单独描述。

### 必须能现场完成的解释

拿 `BGraph_ReluOp`，说明 `AnyRankedTensor` 只约束 ranked tensor，f32 检查还在哪里；
拿 `BGraph_Conv2DOp`，分清 tensor operands、整数数组 attributes、Layout EnumAttr；
拿 `FusedElementwiseOp`，说明 Region 在哪里、谁检查 yield、谁阻止非法 body。

**重点：Type 决定 Value 的类别，Attribute 表示编译期数据，Trait 声明可组合性质，
Interface 提供统一行为查询。** 它们不是“创建 Op 时可随意互换的字段”。`Pure` 也不
证明某个数学重写正确，它提供的是无内存副作用和可推测执行等性质。

### 通过标准

不看正文，画出 `.td → .inc → C++ → library → registry → initialize`，并在源码中
指出每条边。能制造“缺注册”“ODS 类型不对”“custom verifier 拒绝”三类不同错误。
在实验副本新增 Op 后能解释测试为什么有效，才达到“能独立扩展”的标准。

## 第二条：前端语义如何变成 SSA 和 shape

先读[第 05 章](05_onnx_importer_and_ssa_mapping.md)中的名字跟踪实验，再读
[第 06 章](06_type_and_shape_inference.md)的广播与 Conv 手算。

### 需要画出的两张表

第一张是 `value name → Value`：graph input 对应 BlockArgument，float initializer
对应 constant result，node output 对应 BGraph OpResult。第二张是 `name → TensorInfo`：
它提供结果类型，不是运行时 tensor 数据。把两表混在一起会说不清 builder 为何既要
operands 又要 results types。

例如 Add 的 inputs 是 `x`、`bias`，output 是 `sum`，node.name 是 `add_0`。
`self.values["sum"]` 保存输出 Value；`loc("add_0")` 帮助定位来源；打印器可以将结果
写成 `%0`，不改变任何 def-use 关系。`self.values["add_0"]` 并不是这里的正确查询。

### 必须掌握的边界

**重点：导入成功、BGraph 验证通过、可以降低到 CPU 是三个不同条件。** Python
Context 没加载自定义 BGraph binding，`allow_unregistered_dialects` 让 generic Op 可以
创建，不能替代 C++ verifier。高层手写 IR 可有动态维；当前静态 lowering 仍会拒绝它。

解释 Reshape/ReduceMean/Clip 时，必须能指出哪些 initializer 变成 attribute，哪些
保留为普通 SSA 常量，以及为什么过滤规则要按当前 node 判断。共享 Clip bound 后又把
该常量用于 Add，是检测全局错误过滤的反例。

### 通过标准

手工跟踪一个含 3 个节点的图；给每条输入边找到已定义的 Value；手算 `[2,1] + [1,3]`
和一个带 dilation 的 Conv；解释 unsupported opset、未知输入名、非法 shape 分别在哪
一层失败。能独立给 importer 增加一个拒绝测试，而不是只添加到 `SUPPORTED_OPS`。

## 第三条：优化的数学条件、图条件与改写纪律

顺序为[第 07 章](07_fold_canonicalize_cse_dce.md) →
[第 08 章](08_pattern_rewriter_and_bn_folding.md) →
[第 09 章](09_region_and_elementwise_fusion.md)。这一条通常最适合体现你自己的设计理解。

### BN folding 要能推导，不只背公式

从 `y = Conv(x,W)+b` 和 `z = gamma*(y-mean)/sqrt(var+epsilon)+beta` 开始，
推到 `W'=alpha*W`、`b'=alpha*(b-mean)+beta`。说明 alpha 按输出 channel 广播，
不是每个权重位置独立一套参数。然后将公式中的每个量对应到 `DenseElementsAttr`。

数学以外，还需说明：为何只处理推理态、为何要求常量、为何检查 layout、Conv 多用户
时为何当前实现拒绝、为什么先计算检查再创建 replacement，最后才替换 uses 与 erase。
实数等价不能推出浮点逐 bit 等价；权重预乘会改变舍入位置，测试证据要限定到实际覆盖。

### Fusion 要能把 tensor 图改画成 scalar Region

从 `a=Add(x,b); r=Relu(a); y=Mul(r,s)` 开始，画出外部 leaves、内部计算、result use。
解释 `ElementwiseTree::collect` 为什么只沿 single-use producer 继续，遇到共享值为何
把它作为 leaf；解释 `IRMapping` 是 tensor leaf 到 scalar block argument 的映射。

**重点：fusion 不是把几个 tensor Op 原样装进 Region。** 它重建逐元素 scalar 运算，
后续一个 Linalg generic 通过 indexing maps 为每个迭代点提供标量输入。广播是这些 map
的职责，不是把 tensor payload 拷贝多份。

### 通过标准

能完成 BN 数值手算和一个多用户不命中反例；画出融合前后图及 Region；解释 Add-zero
为何可能改变 `-0.0`；把“少了几个 Op”与“正确且更快”分开。最后在实验副本完成一个
带负例的有界修改，并记录失败过的输入、定位过程与回归证据。

## 第四条：从图语义到循环、内存和执行

读[第 10 章](10_dialect_conversion_to_linalg.md)的 indexing map 推导，再读
[第 11 章](11_bufferization_to_llvm.md)的 DPS、读写冲突与 ABI。

### 必须说清的四次变化

| 阶段 | 新出现的信息 | 你需要解释的一个例子 |
|---|---|---|
| BGraph → Linalg | 迭代空间、输入/输出访问 map、scalar body | `[2,3] + [3]` 中 rhs 使用 `(i,j)→(j)` |
| tensor → memref | buffer alias、读写、分配与复用 | 为什么写 destination 不能破坏旧 tensor Value 的后续读取 |
| Linalg/SCF → CF/LLVM Dialect | 显式循环、分支、load/store、descriptor | 一个 memref 不仅是裸指针，还包含 offset/size/stride |
| LLVM Dialect → 执行 | LLVM translation、JIT 和调用约定 | 为何 `tensor<f32>` 需 extract 后才能按 f32 入口调用 |

**重点：FullConversion 成功证明 target legality，不证明数值等价。** `addIllegalDialect`
防止漏掉 BGraph Op；数值正确性仍依赖 lowering 的实现和对拍。当前没有 TypeConverter，
因为这次 conversion 保留 builtin tensor 类型；后续 bufferization 才处理内存表示。

### 通过标准

手写一个完整 elementwise generic，解释所有 map、iterator、body 参数和 yield；制造
“IR 合法但 lowering 不支持”的动态 shape 例子；解释何时需要 out-of-place buffer；
指出 LLVM Dialect 与 `.ll` 的边界，以及当前没有完整 deallocation 的限制。

## 到什么程度可以把项目写进简历

下面是本教程的建议验收尺度，不是招聘行业的统一评级，也不代表你现在已经达到某级。

| 等级 | 可观察表现 | 简历使用建议 |
|---|---|---|
| L0：跟跑 | 能复制命令，无法预测输出 | 可记录学习经历，尚不能支撑“独立设计实现” |
| L1：解释 | 能画 pipeline，区分四条陈述的职责，说明限制 | 可以介绍项目概览；技术追问仍需源码证据 |
| L2：定位与推导 | 不看答案推 BN/shape/maps，定位 verifier/pattern/conversion，设计反例 | 四条简历内容至少都应达到这里 |
| L3：独立修改 | 在副本完成一次跨 ODS/IR/Pass/lowering/tests 的有界变更，能解释调试记录 | 至少一个完整主题达到这里，才有较扎实的个人实现证据 |
| L4：扩展设计 | 能讨论动态 shape、成本模型、TypeConverter、runtime 与释放方案及代价 | 进阶准备；不能把设计方案写成已实现能力 |

建议门槛：四条主线都达到 L2，BN/fusion 或新增 Op 全链路至少一项达到 L3；
Type/Trait/Interface 实验至少能完整接入、解释生成与分派。若简历明确写“实现自定义 Type”，
还必须有实际 TypeDef/注册/测试证据，并注明它属于哪个 Dialect。

用[进度验收清单](progress_checklist.md)保存证据；用[深入追问题库](interview_bank.md)
做无答案口述；最后再用[第 15 章](15_interview_walkthrough.md)练习时间受限的展示。
不要先背第 15 章，再把流畅介绍误当成掌握实现。
