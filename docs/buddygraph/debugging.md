# Debugging

## 构建和注册

- `Could not find MLIRConfig.cmake`：重新传入
  `-DMLIR_DIR=/buddy-mlir/llvm/build/lib/cmake/mlir` 和对应 `LLVM_DIR`。
- `unknown dialect 'bgraph'`：使用 `build/bin/buddygraph-opt`，而非未注册此项目的
  根 `buddy-opt` 或裸 `mlir-opt`。
- 找不到生成的 `.inc`：构建 `MLIRBGraphOpsIncGen`/`BuddyGraphPassesIncGen`，不要手工创建
  或提交生成文件；确认 source include 和 build include 都在 target 路径中。
- pass 名未知：运行 `build/bin/buddygraph-opt --help | rg bgraph` 检查注册。

## Python frontend

- `No module named onnx`：执行 `python3 -m pip install --target .deps -r requirements.txt`。
- `No module named mlir`：把
  `/buddy-mlir/llvm/build/tools/mlir/python_packages/mlir_core` 加入 `PYTHONPATH`。
- importer 拒绝 dynamic/symbolic shape：这是 MVP 边界；先在 exporter 固化 shape，
  不要绕过检查生成可能无法 lower 的 IR。
- importer 报 node 错误：node name 会成为 MLIR location。用
  `buddygraph-opt -mlir-print-debuginfo` 保留并查看映射。

## Verifier 和 conversion

- 先只运行 `buddygraph-opt input.mlir -o /dev/null`，把 verifier 错误与 rewrite/
  lowering 错误分离。
- 用 `--mlir-print-ir-before-all --mlir-print-ir-after-all` 定位首个破坏 IR 的 pass。
- FullConversion 报 illegal BGraph op：说明该 op 的 pattern 未匹配；检查 layout、
  rank、常量 attribute 和 conversion target，而不是删除 illegal 标记。
- One-Shot Bufferize 报 promised interface：driver 必须注册 Func、Arith、CF、Linalg、
  SCF、Tensor 和 Vector 的 external bufferization models。
- runner 报 `math.sqrt` dialect 未找到：完整 LLVM pipeline 必须包含
  `--convert-math-to-llvm`，BN lowering 会产生 sqrt。
- runner ABI 错误：普通 imported function 返回 tensor；runner 示例需使用单一
  rank-0 输出和 importer 的 `--scalar-return`，或自行提供 ABI wrapper。

## Rewrite 没有触发

- BN folding：检查全部参数是否常量、Conv 是否单 use、groups 是否为 1、Conv/BN
  layout 是否一致，以及 channel 长度和按 f32 计算的 `variance + epsilon`。
- elementwise fusion：检查链是否单 use、算子是否在白名单、broadcast result 是否
  自洽。多 use 是刻意的拒绝条件。
- identity canonicalization：常量必须是 DenseElements splat，且替代 value 与结果
  type 完全一致。

调试后应把最小复现加入 lit，而不是只保留临时命令。
