# 15｜面试讲解与 10 分钟现场演示

## 1. 本章目标

你将能完成 30 秒、2 分钟和 10 分钟三种版本的真实项目介绍，正确陈述亮点、限制、
失败案例和性能边界，并回答 20 个常见追问。

## 2. 先运行

面试前用一条命令确认 demo 未漂移：

```bash
cd /buddy-mlir/jlq/projects/buddygraph
export BUDDYGRAPH_TMP=/buddy-mlir/jlq/projects/buddygraph/tmp
mkdir -p "$BUDDYGRAPH_TMP"
cmake --build build --target check-buddygraph -j2
build/bin/buddygraph-opt --help | rg 'bgraph-|convert-bgraph'
```

预先按第 01 章生成 `$BUDDYGRAPH_TMP/buddygraph-learning/` 的 00–06 snapshots，避免
现场等待依赖安装；但现场至少重新运行一次 importer、一个 graph pass、
FullConversion 和 runner。这里引用的是第 01 章已经给出生成命令的目录，不另设一个
没有生成步骤的 snapshot 路径。

## 3. 真实代码位置

演示只打开以下文件：

1. `include/BuddyGraph/IR/BGraphOps.td`。
2. `lib/BuddyGraph/Transforms/FoldBatchNorm.cpp`。
3. `lib/BuddyGraph/Transforms/FuseElementwise.cpp`。
4. `lib/BuddyGraph/Conversion/BGraphToLinalg.cpp`。
5. `frontend/BGraph/import_onnx.py`。
6. `tests/E2E/BGraph/{onnx_elementwise,onnx_clamp}.py`。
7. `docs/buddygraph/results.md`。

不要在 10 分钟内滚动完整 420 行 conversion；跳到符号。

## 4. 调用链

现场叙事顺序：

```text
问题：有限 ONNX 图怎样进入可验证 MLIR 并执行？
→ importer 建 SSA、保留 node Location
→ ODS 定义 BGraph contract，C++ verifier 守住边界
→ BN folding 与 Region fusion 展示 use-def/PatternRewriter
→ FullConversion 以 BGraph illegal 保证完全消除
→ 标准 bufferization/LLVM lowering + runner
→ lit/E2E/benchmark 证明结构和数值，限制性能结论
```

## 5. IR 前后变化

演示只展示三组 diff：

1. Conv→BN 变成新 constants→Conv。
2. Add→Clamp→Mul 变成 FusedElementwise Region，再变成一个 generic。
3. BGraph module 经 FullConversion 后 `rg 'bgraph\.'` 无输出，runner 返回 7.0。

这三组分别证明：常量图优化、Region/use-def 基础设施、legality/可执行闭环。

## 6. 核心机制

### 30 秒介绍

“我实现了一个独立构建的 MLIR 学习项目 BuddyGraph。它用 Python MLIR API 把固定
opset 18 的静态 f32 ONNX 子集导入自定义 BGraph Dialect，提供 verifier、shape
refinement、Conv-BN folding 和基于 Region 的 elementwise fusion，再用 Full Dialect
Conversion 降到 Linalg，经过 One-Shot Bufferize 和 LLVM lowering 在 CPU 上运行。
项目有 14 个 lit/E2E 测试；NumPy、优化关闭和全优化结果一致，Clamp E2E 同时验证
direct/fused 路径与 NaN 传播，signed-zero 回归
固定了 canonicalization 的 IEEE 边界。性能数据只用于证明
IR/alloc 结构变化，不宣称真实模型加速。”

### 2 分钟架构介绍

1. **边界**：opset 18、static ranked f32、NCHW、groups=1、CPU。
2. **前端**：ONNX checker/inference，`name → Value`，constants 和 locations。
3. **IR**：builtin tensor + Layout EnumAttr，不重复自定义 tensor type。
4. **优化**：BN Pattern 需要 constants/single-use；fusion 把 tensor chain 变 scalar
   Region，静态 broadcast 通过 indexing maps 表达。
5. **后端**：所有 BGraph illegal，12 个 conversion pattern specializations，FullConversion 后进入
   标准 bufferization/LLVM pipeline。
6. **证据**：14/14；主 E2E 为 7.0，Clamp E2E 为 0.5 且保留 NaN；signed-zero 输出正零；generic 7→4、alloc 7→4、
   静态 bytes 88→40。
7. **边界**：runner timing 含进程/parse/JIT，无 dynamic batch、GPU、完整 ONNX 或
   ownership deallocation。

### 10 分钟演示

| 时间 | 动作 | 要证明的点 |
|---:|---|---|
| 0:00–1:00 | 版本、目录、`buddygraph-opt --help` | 独立注册与事实边界 |
| 1:00–2:00 | 打开 ODS Relu/Fused/Conv | schema、traits、generated/handwritten 边界 |
| 2:00–3:00 | 生成并导入 ONNX | name→Value、Location、严格 opset |
| 3:00–5:00 | 跑 BN folding，打开 Pattern | use-def、constants、单用户、数学公式 |
| 5:00–6:30 | 跑 fusion，展示 Region | block args、yield、single-use |
| 6:30–8:00 | FullConversion，`CHECK-NOT bgraph` | legality 与 generic/named Conv |
| 8:00–9:00 | 完整 pipeline + runner | tensor→memref→LLVM 和 7.0 |
| 9:00–10:00 | 展示 tests/results/limits | correctness 与性能纪律 |

### 实际失败案例

- 本地 Python `Operation.create` 没有 ODS properties 参数：Dialect 选择
  `usePropertiesForAttributes = 0`，仍保留强类型 attributes。
- One-Shot Bufferize 曾要求 promised interfaces：driver 显式注册多个 standard
  Dialect external models。
- BN lowering 生成 `math.sqrt`：完整 pipeline 必须加 `convert-math-to-llvm`。

这三例适合说明如何从本地 API/错误出发做最小兼容，而不是照搬旧版本教程。

## 7. 为什么这样设计

独立 BGraph 层保留图语义和诊断，builtin tensor 最大化复用标准基础设施，C++
patterns承载数值/use-count条件，FullConversion 提供后端边界。独立 driver 则保护根
Buddy 的研究修改。这些设计都对应真实约束，而不是为了堆术语。

若未来必须复制 Buddy 源码，只复制到项目 `third_party/buddy-mlir/`，记录 license、
upstream commit、原路径、原始 hash 和 project diff，并用 namespace/include/CMake
target 隔离。当前没有副本，所以不能展示虚构 provenance。

## 8. 常见错误

- 说项目位于 Buddy-MLIR 目录之外；实际只是在构建/注册上隔离。
- 说支持完整 ONNX、dynamic batch、groups、NHWC E2E 或 GPU。
- 说实现了 InferTypeOpInterface/custom fold/TypeConverter；均未实现。
- 把 Add-zero 当作无条件 IEEE identity；项目正是因 signed-zero 反例移除了这条 rewrite。
- 把 generic/alloc 减少包装成真实模型 speedup。
- 把 LLVM Dialect 称作 LLVM IR。
- 只展示成功输出，不讲 negative tests 和未命中条件。
- 10 分钟演示现场编译 LLVM 或安装依赖，浪费叙事时间。

## 9. 动手练习

录制一次 10 分钟演示。复盘时逐项检查：是否出现一个无法从文件定位的类名？是否
把规划功能说成实现？是否在性能处给出测量范围？是否展示了至少一个 negative
condition？超时则优先删背景，不删 correctness/limits。

## 10. 验收标准

- 三种时长均能在限制内完成。
- 10 分钟演示实际跑出 7.0 和 FullConversion 无 BGraph 证据。
- 所有数字能在 `results.md` 定位，所有符号能 `rg` 到。
- 主动陈述至少五项限制和一个真实失败案例。

## 11. 面试追问

1. **为什么独立项目和 `buddygraph-opt`，不改原 `buddy-opt`？**
   根注册/CMake 有受保护研究改动；独立 driver 注册 BGraph 和所需 interfaces，复用
   同一 MLIR build，减小冲突面。物理目录仍在 `/buddy-mlir` 内。
2. **何时需要复制 Buddy 源码？如何证明来源？**
   只有标准 link/API 无法复用且必须修改实现时；副本应在 `third_party`，记录
   license、commit、原路径/hash 和 diff，并隔离 namespace/target。当前没有副本。
3. **为什么不直接 ONNX→Linalg？**
   会过早丢失 Conv/BN/elementwise 图语义，使诊断、BN folding 和 fusion 更难。
4. **为什么不自定义 tensor Type？**
   builtin ranked tensor 已表达 rank/shape/f32，并直接接入 Linalg/bufferization；
   自定义 Type 没有新增 MVP 语义。
5. **为什么不自己写 DCE/CSE？**
   它们是通用、依赖 side-effect/regions/dominance 的基础设施；BGraph 用 `Pure` 和
   局部 patterns 提供合法信息。
6. **verifier 与 shape inference 区别？**
   verifier 拒绝局部矛盾；shape pass 把合法但动态的 result 收紧。当前是自定义
   Module pass，不是 InferType interface。
7. **PatternRewriter 与 Dialect Conversion 区别？**
   前者做局部 best-effort 等价改写；后者由 ConversionTarget legality 驱动，必须
   消除非法 source ops。
8. **什么保证 BN folding 合法？**
   Conv direct producer、single-use、groups=1、Conv/BN layout 一致、全部 Dense f32
   constants、channel counts、按 f32 为正的 variance+epsilon，以及先检查后 mutation。
9. **什么保证 fusion 合法？**
   op 白名单、无副作用、producer single-use 收集边界、broadcast verifier、scalar
   Region 白名单/yield 和 static lowering maps。
10. **为什么 FullConversion 重要？**
    整个 BGraph dialect 被标 illegal，漏 pattern 立即失败，后端不会收到未知高层 op。
11. **tensor 如何变成 memref？**
    Linalg destination-style IR 进入 One-Shot Bufferize，根据 alias/read-write 做
    in-place/out-of-place 决策并重写 function boundaries。
12. **如何证明优化没改数值？**
    同一固定 ONNX fixture 分别跑未优化和全优化 LLVM pipeline，与 NumPy reference
    比较 max absolute/relative error；当前三者打印均为 7.0。
13. **加入新算子要改哪些层？**
    ODS、verifier、shape、可能的 canonicalization/fusion、importer、conversion、
    dialect/region whitelist 和分层 tests；Clamp 是当前可逐层核对的实例。
14. **为何属性不用 ODS properties？**
    本地 MLIR Python `Operation.create` 没 properties 参数；Dialect 用 attributes
    storage 保持 API 构造和强类型 ODS schema。
15. **支持广播吗？**
    binary/fused verifier 和 Linalg maps 支持静态右对齐 NumPy broadcast；dynamic
    shape lowering不支持。
16. **ReduceMean 如何 lowering？**
    第一个 generic 按 axes reduction 求和，第二个 generic 除以静态 element count；
    keep_dims 影响 output indexing map。
17. **Location 如何从 ONNX 保留？**
    node name 通过 `Location.name()`；initializers 用 `initializer:<name>`；输出用
    debug-info printer，BN/fusion rewrite 合并 locations。
18. **LLVM Dialect 是 LLVM IR 吗？**
    不是；前者仍是 MLIR，需 `mlir-translate --mlir-to-llvmir` 才得到 `.ll`。
19. **性能结论是什么？**
    已证明 generic/alloc/静态 bytes 减少，并记录独立进程/JIT微基准；不能声称真实
    模型 kernel speedup。
20. **当前最重要的工程限制？**
   opset 18、static f32、NCHW importer、groups=1、CPU、rank-0 runner adapter、无
   ownership deallocation、Clip 仅支持 constant scalar bounds，以及项目尚未纳入 Git 跟踪。
