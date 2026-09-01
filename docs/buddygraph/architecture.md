# BuddyGraph architecture

## 数据流

```text
ONNX opset 18
  → checker + ONNX shape inference
  → MLIR Python API importer
  → BGraph tensor IR
  → graph rewrites and shape refinement
  → FullConversion to Linalg/Tensor/Arith/Math
  → One-Shot Bufferize
  → loops + LLVM dialect
  → mlir-runner on CPU
```

前端负责 ONNX 协议边界和静态类型材料化；BGraph 保存图级语义；Transform
passes 只做已证明合法的图改写；Conversion 把每个残留 BGraph op 全部消除。
`applyFullConversion` 把 BGraph dialect 标为 illegal，因此漏写 lowering pattern
会直接导致编译失败，而不会把未知图算子悄悄交给后端。

## 为什么需要 BGraph Dialect

导入后直接产生 Linalg 会过早丢失 Conv、BatchNorm、Clamp、广播 elementwise chain 等
图级意图。保留 BGraph 有三个直接收益：verifier 可在源语义层报告错误；BN
folding 和 fusion 不必从低层 loop/indexing map 反推图结构；每一层 lowering 的
正确性和 IR 变化可以独立测试。它也是前端协议与后端实现之间的稳定边界。

## 为什么复用 builtin tensor

项目的差异在操作语义，不在存储类型系统。rank、shape 和 f32 element type 已由
builtin ranked tensor 完整表达，并能直接复用 Linalg、Tensor、bufferization 和
Dialect Conversion 基础设施。自定义 tensor type 会增加 parser、printer、type
conversion 和 bufferization 工作，却不提供 MVP 所需的新语义。

## 组件职责

- ODS：方言、Layout EnumAttr、操作数/结果/attribute schema 和 Pass CLI 声明。
- `BuddyGraphIR`：方言注册、verifier 和 canonicalization patterns。
- `BuddyGraphTransforms`：shape refinement、BN folding、elementwise fusion。
- `BGraphToLinalg`：覆盖全部 BGraph ops 的 conversion patterns 与
  FullConversion target。
- `buddygraph-opt`：注册所需 dialect、全部 MLIR passes、自定义 passes 和
  bufferization external interface models。
- Python frontend：固定 ONNX 协议、名称到 SSA Value 的映射、initializer 常量、
  Clip 同时提供的 finite scalar f32 initializer bounds 和 location 传播；不接受
  optional、runtime 或 Constant-node bounds。

## TableGen 生成关系

`BGraphDialect.td`、`BGraphEnums.td` 和 `BGraphOps.td` 生成 dialect、enum、attribute
及 op 的声明/定义 `.inc` 文件，目标为 `MLIRBGraphOpsIncGen`。`Passes.td` 生成 Pass
基类、注册函数和 CLI 选项，目标为 `BuddyGraphPassesIncGen`。源文件只 include build
目录中的生成 `.inc`；生成文件不是源代码，不提交到仓库。IR/Transform targets 通过
`add_dependencies` 确保编译前完成 TableGen。

## 隔离构建

根工程的 dialect CMake 和 `buddy-opt` 已含用户未提交的 Microkernel 修改。为保护
这些内容，本项目采用 out-of-tree CMake 和独立 `buddygraph-opt`，但仍链接同一个
本地 MLIR 21 build。此隔离只改变注册入口，不改变方言和 pass 的实现方式。
