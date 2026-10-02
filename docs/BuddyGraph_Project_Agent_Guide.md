# BuddyGraph 项目生成 Agent 执行指导

> 历史指导文件：本文记录 2026-07-28 在 Buddy-MLIR 内启动项目时的要求。
> BuddyGraph 已于 2026-10-02 迁至 `/home/jlq/project/buddygraph`，当前目录、
> 外部缓存和构建命令以项目根 [README](../README.md) 为准。

版本：1.0  
目标读者：负责在用户现有 Buddy-MLIR 仓库中实现项目的编码 Agent  
项目名称：BuddyGraph——基于 Buddy-MLIR 的轻量神经网络图前端与融合编译器

---

## 0. 本文件的使用方式

你是本项目的实现 Agent。你必须把本文件视为项目需求、工程边界和验收规范，而不是可自由发挥的构想。

开始工作前必须：

1. 完整阅读本文件。
2. 阅读仓库内所有适用的 `AGENTS.md`、`README.md`、构建说明和测试说明。
3. 审计本地 Buddy-MLIR 的实际版本、目录结构、已有修改和构建产物。
4. 根据本地版本调整 API 和文件位置，不得假设当前仓库与某个线上版本完全一致。
5. 先提交审计结果和实施计划，再进行大规模代码修改。

本项目以“可学习、可验证、可写入简历”为目标。代码清晰度、基础设施覆盖度、测试证据和设计说明与功能本身同等重要。

---

## 1. 已知背景

### 1.1 用户当前状态

- 用户已经阅读完 MLIR 基础设施文档，理解 SSA、Operation、Region、Block、Dialect、Pass、Rewrite Pattern、Dialect Conversion 等基本概念。
- 用户缺少一次由自己充分阅读、调试和修改的完整 MLIR 工程实践。
- 用户现有研究主线是：
  - MLIR Transform Dialect；
  - kernel-aware GEMM 专家调度空间；
  - microkernel contract；
  - BLIS/OpenBLAS/libxsmm；
  - BaCO 搜索。
- 本项目不得重复上述研究主线，不得继续扩展 GEMM 搜索、microkernel 调用或 Transform Dialect 调优。
- 用户本地已有编译完成的 Buddy-MLIR 和 LLVM/MLIR 构建产物，应尽量复用。
- Buddy-MLIR 还可能包含用户论文相关修改，必须保护所有无关改动。

### 1.2 本项目在用户技能结构中的作用

本项目用于补足以下能力：

- C++ 中的 MLIR 工程开发；
- ODS/TableGen；
- 自定义 Dialect、Operation、Attribute；
- verifier、fold、canonicalization、Interface；
- Pass 注册和 Pass Pipeline；
- PatternRewriter、use-def 分析、Region 构造；
- Dialect Conversion 和渐进 lowering；
- lit/FileCheck、错误测试和端到端数值验证；
- 模型前端到 CPU 可执行代码的完整链路。

---

## 2. 项目目标

实现一个面向有限 ONNX 子集的神经网络图前端：

```text
PyTorch 小模型
  → ONNX
  → ONNX 子集导入器
  → bgraph 高层 Dialect
  → 形状推导与图优化
  → Linalg/Tensor/Arith/Math
  → Buddy-MLIR/MLIR 既有 lowering
  → LLVM IR
  → CPU 执行
```

项目必须至少形成以下闭环：

1. 一个真实模型文件能够被导入。
2. 导入后生成合法的 `bgraph` IR。
3. `bgraph` IR 能够经过自定义优化。
4. 优化后的 IR 能够 Full Conversion 到标准 MLIR Dialect。
5. 标准 IR 能够继续降低并在 CPU 上执行。
6. 执行结果能够和 NumPy 或 ONNX Runtime 参考结果对比。
7. 所有关键转换都有 lit/FileCheck 或端到端测试。

---

## 3. 明确不做的内容

以下内容不属于本项目范围：

- 完整支持 ONNX 规范。
- 支持多个 ONNX opset。
- 训练语义、自动求导和反向传播。
- GPU、NPU 或自研芯片后端。
- 量化、混合精度和量化校准。
- Transform Dialect 调优。
- GEMM blocking、packing、microkernel 或自动调优。
- 重写 MLIR 已有的通用 DCE、CSE、Canonicalizer。
- 重新实现一套 LLVM lowering 或运行时。
- 为了展示自定义 Type 而重复定义 builtin tensor。
- 未经测量就宣称固定的性能或内存收益。
- 对 Buddy-MLIR 现有模块进行与本项目无关的重构。

任何新增功能都必须说明它服务于哪一项项目目标，否则不得加入。

---

## 4. 权限、安全和仓库保护规则

### 4.1 修改前的强制检查

首先运行并记录：

```bash
pwd
git rev-parse --show-toplevel
git rev-parse --short HEAD
git status --short
git submodule status
```

随后定位：

```bash
rg --files -g 'AGENTS.md' -g 'CMakeLists.txt' -g 'lit.cfg.py' \
  -g 'lit.site.cfg.py' -g '*Ops.td' -g '*Passes.td'
```

检查实际工具和构建目录，命令需要根据本地路径调整：

```bash
./build/bin/buddy-opt --version
./build/bin/buddy-opt --help
ninja -C build -t targets
```

### 4.2 禁止事项

- 禁止执行 `git reset --hard`、`git clean -fdx`、`git checkout -- <path>`。
- 禁止删除或覆盖用户已有修改。
- 禁止修改论文相关 Transform/Microkernel 模块，除非编译注册确实需要且修改范围可证明。
- 禁止擅自提交、推送、创建 PR 或修改远程仓库。
- 禁止为了修复本项目而大范围格式化整个仓库。
- 禁止在未检查已有构建产物前重新完整编译 LLVM。
- 禁止新增大型第三方依赖而不说明必要性和替代方案。
- 禁止把无法解释的临时代码、硬编码绝对路径或机器相关参数写入正式实现。

### 4.3 工作隔离

如果当前工作树包含论文实验改动：

- 优先在独立分支或 worktree 中开发；
- 新 worktree 可以复用原有 `llvm/build`；
- Buddy 部分使用新的 build 目录增量构建；
- 不得通过 stash、reset 或 checkout 隐藏用户改动，除非用户明确授权。

---

## 5. Phase 0：必须先完成的本地审计

在编写 Dialect 前，生成：

```text
docs/buddygraph/audit.md
```

审计报告必须包括：

1. Buddy-MLIR commit。
2. LLVM submodule commit。
3. 仓库实际目录结构。
4. `buddy-opt` 的真实注册入口。
5. Dialect、Pass、Conversion、tests 的惯用放置位置。
6. 当前 build 目录和可复用 target。
7. Python bindings 是否启用。
8. Python 环境中是否已经有 `onnx`、`numpy`、`onnxruntime`。
9. 一个已有 Buddy 示例或最小 MLIR 测试是否能够运行。
10. 本项目实际采用的目录和 CMake target 名称。
11. 发现的本地 API 差异或构建风险。
12. 当前工作树中需要保护的用户修改。

审计完成前不得批量创建项目文件。

如果现有构建本身失败：

- 先判断失败是否与本项目无关；
- 保留完整、最短可复现命令和关键错误；
- 不得以大范围修改构建系统的方式掩盖问题；
- 能安全修复时给出最小修复；
- 不能安全修复时停止并报告阻塞点。

---

## 6. 总体工程结构

下列结构是逻辑要求，不是对本地仓库路径的绝对假设。Phase 0 后按仓库惯例调整：

```text
frontend/
└── BGraph/
    ├── import_onnx.py
    ├── generate_test_models.py
    └── README.md

midend/include/Dialect/BGraph/
├── IR/
│   ├── BGraphDialect.td
│   ├── BGraphOps.td
│   ├── BGraphEnums.td
│   └── BGraph.h
└── Transforms/
    ├── Passes.td
    └── Passes.h

midend/lib/Dialect/BGraph/
├── IR/
│   ├── BGraphDialect.cpp
│   └── BGraphOps.cpp
└── Transforms/
    ├── InferShapes.cpp
    ├── Canonicalize.cpp
    ├── FoldBatchNorm.cpp
    └── FuseElementwise.cpp

midend/include/Conversion/BGraphToLinalg/
midend/lib/Conversion/BGraphToLinalg/

tests/Dialect/BGraph/
tests/Conversion/BGraphToLinalg/
tests/Frontend/BGraph/
tests/E2E/BGraph/

examples/BGraph/
docs/buddygraph/
```

命名约束：

- Dialect namespace：`bgraph`。
- C++ namespace：遵循 Buddy-MLIR 本地惯例，例如 `buddy::bgraph`。
- Pass 参数统一使用 `bgraph-` 前缀。
- 不得将新 Op 塞进现有 `Bud`、Transform 或 Microkernel Dialect。

---

## 7. 前端设计

### 7.1 输入范围

第一版固定一个 ONNX opset，建议 opset 18。若本地 PyTorch/ONNX 工具链更适合另一个单一版本，可以调整，但必须：

- 在 README 中写明；
- importer 主动检查；
- 不接受未验证的其他 opset；
- 错误信息必须包含实际版本和受支持版本。

第一版支持：

```text
Conv
BatchNormalization
Relu
Add
Mul
Sub
Div
Reshape
Transpose
ReduceMean
```

可选扩展只有在核心闭环完成后才能增加：

```text
Clamp
MaxPool
Softmax
Gemm
```

其中 `Gemm` 不允许引入调优、microkernel 或论文实验逻辑。

### 7.2 导入器实现原则

优先使用 Python `onnx` 包解析 `ModelProto`，因为 protobuf、opset 兼容和模型文件解析不是本项目的 MLIR 学习重点。

导入器负责：

- 检查 opset；
- 读取 graph input/output；
- 读取 initializer；
- 遍历拓扑有序的 Node；
- 建立 `ONNX value name → MLIR Value` 映射；
- 转换 dtype、shape 和 attribute；
- 创建 `func.func`；
- 使用 builtin `tensor` 类型；
- 使用 `arith.constant` 或当前版本等价标准 Op 表示 initializer；
- 调用 bgraph Op builder；
- 给所有 Op 附加可追踪的 Location；
- 对不支持的算子、dtype、动态属性给出明确错误。

禁止把正式 importer 实现为大量字符串拼接 MLIR 文本。

推荐路线：

1. 如果 Buddy Python bindings 和自定义 Dialect Python bindings可用，直接通过 Python MLIR API 构造。
2. 如果 bindings 未启用，先完成手写 MLIR → Pass → Lowering 闭环。
3. 再选择最小代价方案补 importer，不得为 importer 重建整套工具链而拖延核心项目。

### 7.3 前端约束

MVP：

- f32；
- ranked tensor；
- 静态空间维度；
- 可允许动态 batch，但不能阻塞静态闭环；
- Conv 仅支持二维推理；
- BatchNormalization 仅支持推理模式；
- Reshape 的目标 shape 必须是常量；
- Transpose permutation 必须是常量；
- ReduceMean axes 必须是常量。

任何不支持的情况必须拒绝并报告，禁止静默生成错误语义。

---

## 8. BGraph Dialect 设计

### 8.1 复用原则

必须复用：

- `builtin.module`；
- `func.func`、`func.return`；
- builtin `tensor`；
- `arith.constant`；
- 标准整数、浮点和数组属性。

不要定义：

- `bgraph.func`；
- `bgraph.return`；
- 与 builtin tensor 完全等价的 `!bgraph.tensor`；
- 无语义必要性的包装 Op。

### 8.2 必需 Op

#### `bgraph.conv2d`

Operands：

- input；
- filter；
- 可选 bias。

Attributes：

- strides；
- pads；
- dilations；
- groups；
- layout。

Verifier 至少检查：

- input/filter/result rank；
- element type；
- strides、pads、dilations 长度和值域；
- groups 大于零；
- channel/group 关系；
- 静态情况下的输出 shape；
- layout 与维度解释一致；
- bias channel 数。

#### `bgraph.batch_norm`

Operands：

- input；
- scale；
- bias；
- mean；
- variance。

Attributes：

- epsilon；
- channel axis 或 layout。

Verifier 至少检查：

- 推理模式；
- epsilon 合法；
- 五个参数 element type 一致；
- 参数为一维；
- channel 静态时长度一致。

#### `bgraph.relu`

- 单输入单输出；
- shaped floating-point tensor；
- 无副作用；
- 输入输出 shape/type 一致。

#### `bgraph.add/sub/mul/div`

- 两个输入；
- 支持项目范围内的 NumPy/ONNX 广播；
- 通过现有 Trait/Interface 或自定义 verifier 检查；
- 结果 shape 能推导；
- 所有 Op 标记为纯计算。

#### `bgraph.reshape`

- MVP 只接受常量目标 shape；
- 检查元素数；
- 支持一个 `-1` 推导维；
- 禁止多个 `-1`；
- 对静态矛盾给出 verifier 错误。

#### `bgraph.transpose`

- permutation 必须完整、无重复、范围合法；
- 结果 shape 可由输入和 permutation 推导。

#### `bgraph.reduce_mean`

- axes 唯一且合法；
- 支持 `keep_dims`；
- 推导结果 rank 和 shape。

#### `bgraph.fused_elementwise`

这是项目中用于体现 Region 基础设施的核心 Op。

要求：

- variadic tensor operands；
- 一个 tensor result；
- 一个 single-block region；
- block argument 为每个输入的标量 element type；
- region 内只能包含允许的纯标量计算；
- 使用 `bgraph.yield` 结束；
- yield 类型与结果 element type 一致；
- verifier 检查 region 结构和广播后的结果 shape。

#### `bgraph.yield`

- Terminator；
- 只能位于 `bgraph.fused_elementwise`；
- operand 数量和类型满足 parent 要求。

### 8.3 Attribute 和 Type

必须至少实现一个有实际语义的 EnumAttr，例如：

```text
#bgraph.layout<nchw>
#bgraph.layout<nhwc>
```

如果 builtin Attribute 已完整表达语义，优先复用。

只有遇到 builtin Type 无法表达的真实语义时才新增 Type。自定义 Type 不是验收硬指标。

### 8.4 Interface、Trait 和折叠

根据本地 MLIR 版本使用合适接口：

- `InferTypeOpInterface` 或 `InferShapedTypeOpInterface`；
- `MemoryEffectOpInterface` / NoMemoryEffect；
- broadcast 相关 Trait；
- elementwise 相关 Trait；
- `DestinationStyleOpInterface` 只有在表示确实符合 DPS 时使用。

实现：

- result type/shape inference；
- identity fold；
- constant fold 中合理且低风险的部分；
- op-specific canonicalization。

不要伪造 Interface 语义来满足形式要求。

---

## 9. 自定义优化设计

### 9.1 简单 canonicalization

使用 DRR 或 C++ Pattern：

```text
relu(relu(x))                     → relu(x)
add(x, 0)                         → x
add(0, x)                         → x
mul(x, 1)                         → x
mul(1, x)                         → x
reshape(reshape(x))               → reshape(x)
transpose(transpose(x, p), p⁻¹)   → x
```

规则必须：

- 保持 dtype 和广播语义；
- 不错误消除 NaN/Inf 相关行为；
- 为所有边界条件添加测试；
- 简单模式可使用 DRR；
- 依赖属性计算、use count 或常量读取时使用 C++ Pattern。

### 9.2 BatchNorm 折叠进 Conv

仅处理推理模式、所有 BN 参数均为常量的情况。

数学关系：

```text
alpha = gamma / sqrt(variance + epsilon)
W'    = W * alpha
b'    = beta + (b - mean) * alpha
```

如果原 Conv 没有 bias，视为全零 bias，但必须显式构造合法常量。

重写前置条件：

- Conv result 只有 BatchNorm 这一个用户；
- filter、bias、scale、BN bias、mean、variance 为可读取的常量；
- dtype 为受支持浮点类型；
- channel/layout 对齐；
- groups 情况处理正确，或 MVP 明确限制 groups=1；
- epsilon 合法；
- DenseElementsAttr 元素数量匹配；
- 不修改共享常量；需要时创建新常量。

重写结果：

- 新 filter 常量；
- 新 bias 常量；
- 新 Conv；
- 删除 BatchNorm；
- 保留或合理 fuse Location；
- 数值结果与参考模型在容差内一致。

失败时必须返回 match failure，不得部分修改 IR。

### 9.3 通用逐元素链融合

允许融合：

```text
add/sub/mul/div/relu
```

核心要求：

- 只融合无副作用 Op；
- 中间结果必须单用户；
- 不跨越 Conv、Reduce、Reshape、Transpose 或函数边界；
- 广播关系必须能在 fused op 中正确表达；
- 不重复计算高成本或多用户节点；
- 保持源 Location；
- 生成一个合法的 `bgraph.fused_elementwise` region；
- lowering 后对应一个 `linalg.generic` 或本地版本等价的单一结构化 Op；
- 真实消除至少一个中间 tensor，而不是只换名字。

实现时必须使用和解释：

- use-def chain；
- `hasOneUse()`；
- `PatternRewriter`；
- `IRMapping`；
- Region/Block/BlockArgument；
- scalar op cloning 或显式 scalar body 构造；
- side-effect/interface 检查。

---

## 10. Pass Pipeline

注册至少以下 Pass：

```text
bgraph-infer-shapes
bgraph-fold-bn-into-conv
bgraph-fuse-elementwise
convert-bgraph-to-linalg
```

推荐 pipeline：

```text
bgraph-infer-shapes
canonicalize
sccp
bgraph-fold-bn-into-conv
bgraph-fuse-elementwise
canonicalize
cse
convert-bgraph-to-linalg
one-shot-bufferize
<本地 Buddy/MLIR 的后续 lowering pipeline>
```

要求：

- 使用本地 Pass 注册惯例；
- Pass 明确声明 dependent dialects；
- Pass option 通过 TableGen 或仓库惯用机制定义；
- 能通过 `buddy-opt --help` 发现；
- 支持 `--pass-pipeline`；
- 测试 pipeline 顺序；
- README 解释顺序原因。

不得手写通用 CSE/DCE。应通过正确的 side-effect、fold、canonicalization 和 Trait 让通用基础设施工作。

---

## 11. BGraphToLinalg Conversion

### 11.1 Conversion 约束

必须使用 Dialect Conversion 基础设施：

- `ConversionTarget`；
- `RewritePatternSet`；
- `OpConversionPattern` 或本地等价模板；
- 必要时使用 `TypeConverter`；
- `applyFullConversion`。

转换结束后：

- `bgraph` Dialect 必须整体非法；
- 不得残留任何 `bgraph.*` Op；
- 不得依赖 `allow-unregistered-dialect`；
- 所有 `unrealized_conversion_cast` 最终必须消解或明确证明合法。

### 11.2 Lowering 目标

优先目标：

- elementwise → `linalg.generic` 或 `linalg.map`；
- fused_elementwise → 一个 `linalg.generic`；
- reduce_mean → `linalg.reduce` 加必要除法；
- transpose → `linalg.transpose` 或本地版本等价表示；
- reshape → `tensor` Dialect 合法 reshape/collapse/expand 形式；
- conv2d → 对应 layout 的 Linalg named convolution；
- batch_norm → 一个或少量 `linalg.generic`，即使未折叠也必须可执行。

对于不同 MLIR 版本 API：

- 先检索本地源码和已有 lowering；
- 复用本地 builder、helper 和 conversion pattern；
- 不从线上示例机械复制不兼容 API；
- 在 `docs/buddygraph/design_decisions.md` 记录版本相关选择。

### 11.3 Bufferization 和 LLVM

自定义 Op 应在 One-Shot Bufferize 前全部消失。

优先复用 Buddy-MLIR 已有：

- Linalg lowering；
- bufferization pipeline；
- ownership-based deallocation；
- SCF/CF/MemRef/Arith 到 LLVM；
- runner/runtime support。

除非确有必要，不为 bgraph Op 实现 `BufferizableOpInterface`。

---

## 12. 构建与注册

需要完成：

1. Dialect TableGen target。
2. Op/Enum/Pass 生成 target。
3. IR library。
4. Transform library。
5. Conversion library。
6. `buddy-opt` 中的 DialectRegistry 注册。
7. Pass 注册。
8. Python bindings 注册，仅在现有构建已支持时进行。
9. lit test suite 接入 `check-buddy` 或本地等价 target。

构建原则：

- 优先只编译受影响 target；
- 首次确认时编译 `buddy-opt`；
- 每阶段运行相关测试；
- 核心闭环完成后再运行完整 `check-buddy`；
- 不因一个新目录而重新编译 LLVM。

---

## 13. 测试规范

### 13.1 Dialect 测试

必须覆盖：

- 每个 Op 的 parse/print round-trip；
- generic form；
- declarative/custom assembly form；
- 正确 verifier；
- 错误 rank；
- 错误 dtype；
- 错误 axis/permutation；
- 错误 broadcasting；
- 错误 region/yield；
- 错误 Conv/BN channel。

错误测试使用本地惯用的：

```text
-verify-diagnostics
```

### 13.2 Rewrite 测试

使用 lit/FileCheck 验证：

- pattern 命中；
- pattern 不应命中时保持原 IR；
- BN 常量被正确重写；
- BN 被删除；
- elementwise 链变成单个 fused op；
- 多用户中间值不被错误融合；
- 有副作用边界不被跨越；
- Location 和类型保持合法。

### 13.3 Conversion 测试

验证：

- 每个 bgraph Op 都有 lowering；
- FullConversion 后不存在 `bgraph.`；
- fused elementwise 对应单个结构化计算；
- 生成 IR 可继续 bufferize；
- 不产生未解释的 cast；
- dynamic batch 扩展若存在，具有专门测试。

### 13.4 前端测试

至少生成：

- Conv-BN-ReLU ONNX；
- elementwise chain ONNX；
- 一个包含不支持 Op 的 ONNX；
- 一个错误 opset 模型；
- 一个错误 attribute 模型。

检查：

- value mapping；
- initializer；
- attribute；
- shape；
- 错误消息。

### 13.5 端到端数值测试

要求：

- 固定随机种子；
- 小尺寸、可快速运行；
- reference 使用 NumPy 或 ONNX Runtime；
- f32 初始容差建议 `atol=1e-5, rtol=1e-5`，实际按算子链评估；
- 输出 max absolute error、max relative error；
- 优化前和优化后分别验证；
- 不用随机一次通过代替稳定测试。

---

## 14. 性能与效果评测

不能预先写“减少内存 30%”或“加速 X%”。

至少记录：

### 编译期

- importer 时间；
- 每个自定义 Pass 时间；
- 总 pipeline 时间；
- 优化前后 Operation 数量。

### IR 结构

- `bgraph` Op 数；
- `linalg.generic` 数；
- bufferization 后 `memref.alloc` 数；
- 中间 tensor/alloc 的静态总字节数，能计算时报告；
- fusion 前后 kernel/loop 数。

### 运行时

- warmup；
- 重复次数；
- 中位数；
- p95；
- 优化关闭；
- 仅 canonicalize/cse；
- BN folding；
- elementwise fusion；
- 全部优化。

所有结论必须来自实际数据，并记录硬件、编译选项、输入 shape 和测量方法。

---

## 15. 必需文档

项目结束时至少包含：

```text
docs/buddygraph/audit.md
docs/buddygraph/architecture.md
docs/buddygraph/design_decisions.md
docs/buddygraph/pass_pipeline.md
docs/buddygraph/testing.md
docs/buddygraph/debugging.md
docs/buddygraph/results.md
examples/BGraph/README.md
```

文档必须回答：

- 为什么需要 BGraph Dialect，而不是导入后直接生成 Linalg？
- 为什么使用 builtin tensor 而不定义自定义 tensor Type？
- verifier、shape inference、canonicalization 分别负责什么？
- DRR 和 C++ Pattern 的边界是什么？
- 为什么不能手写通用 DCE/CSE？
- BN folding 的合法性条件是什么？
- elementwise fusion 为什么能够消除中间张量？
- PatternRewriter 和 Dialect Conversion 有什么区别？
- 为什么在 bufferization 前消除 BGraph？
- Pass 顺序改变会造成什么影响？

---

## 16. 分阶段执行计划

### Phase 0：审计

产物：

- `audit.md`；
- 本地目录映射；
- 实施计划；
- baseline 命令。

退出条件：

- 确认已有构建可用；
- 确认修改范围；
- 确认注册和测试入口。

### Phase 1：Dialect 骨架

实现：

- Dialect；
- EnumAttr；
- relu/add/mul；
- verifier/interface；
- parse/print；
- `buddy-opt` 注册；
- 基础测试。

退出条件：

- 可解析、打印和验证；
- `buddy-opt --help` 可见；
- 相关测试通过。

### Phase 2：完整高层 Op 与 Rewrite

实现：

- Conv、BN、reshape、transpose、reduce；
- shape inference；
- canonicalization；
- BN folding；
- elementwise fusion；
- 负面测试。

退出条件：

- 所有 Pattern 正负条件测试通过；
- IR 验证始终成功；
- 不存在部分重写失败。

### Phase 3：Lowering

实现：

- BGraphToLinalg；
- FullConversion；
- bufferization；
- LLVM/runner 最小闭环。

退出条件：

- 手写 BGraph MLIR 可运行；
- lowering 后无 `bgraph.*`；
- 数值结果正确。

### Phase 4：ONNX 前端

实现：

- 单一 opset；
- 受支持算子；
- initializer/value/attribute 映射；
- 错误处理；
- 前端测试。

退出条件：

- 两个测试模型可自动导入；
- 不支持模型被明确拒绝；
- 不使用正式字符串拼接 IR。

### Phase 5：端到端、评测与整理

实现：

- 与 reference 对拍；
- ablation；
- performance/IR 指标；
- README；
- 文档；
- 完整测试。

退出条件：

- 所有验收项有可重复命令；
- 简历表述只包含已实现、已测量结果；
- 无未说明的已知错误。

---

## 17. 每阶段的 Agent 汇报格式

每个阶段结束必须输出：

### Outcome

本阶段实际完成了什么。

### Changed files

列出新增和修改文件，并说明职责。

### Design decisions

记录本地版本导致的 API/结构选择。

### Verification

列出实际运行命令及结果，不得只说“应该能工作”。

### IR evidence

给出关键转换前后最小 IR。

### Remaining risks

列出真实未解决问题。

### Next checkpoint

下一阶段的最小目标。

如果测试失败，必须先解释失败和影响范围，不得继续堆叠更多功能。

---

## 18. 最终验收清单

项目只有同时满足以下条件才算完成：

- [ ] 本地 Buddy-MLIR commit 和环境已记录。
- [ ] 未破坏用户论文相关修改。
- [ ] BGraph Dialect 使用 ODS/TableGen 定义。
- [ ] 至少一个有意义的 EnumAttr。
- [ ] 所有必需 Op 有 verifier。
- [ ] 纯 Op 正确建模副作用。
- [ ] 形状推导可工作。
- [ ] 至少五个 canonicalization/fold 规则。
- [ ] BN folding 有完整合法性检查。
- [ ] elementwise fusion 使用 Region。
- [ ] FullConversion 后无 BGraph Op。
- [ ] 手写 BGraph IR 可以执行。
- [ ] ONNX 子集模型可以导入。
- [ ] 不支持情况有明确诊断。
- [ ] 优化前后均通过数值对拍。
- [ ] lit/FileCheck 正负测试完整。
- [ ] 端到端测试可重复。
- [ ] 性能结论来自真实测量。
- [ ] 架构、Pass、测试、调试文档完整。
- [ ] README 包含从模型生成到执行的完整命令。
- [ ] 项目可由用户在面试中用 10 分钟演示。

---

## 19. 面向学习的代码质量要求

因为用户将通过阅读本项目学习，代码必须：

- 优先清晰，而不是炫技；
- 一个文件承担一个明确职责；
- 核心类和函数有“为什么这样设计”的注释；
- 不在注释中复述显而易见的代码；
- 复杂 Pattern 明确写出前置条件；
- 错误路径返回 `failure`/`notifyMatchFailure`，不使用模糊布尔值；
- 使用 MLIR/LLVM ADT 和本地惯用风格；
- 避免无意义模板抽象；
- 对 TableGen 生成文件的关系写入文档；
- 对每个 Pass 提供最小输入/输出示例；
- 关键 lowering 能够通过调试器逐步跟踪。

教学价值不能以牺牲工程正确性为代价。
