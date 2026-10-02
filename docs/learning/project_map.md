# BuddyGraph 项目事实映射

本文件是教程的事实入口。路径均相对于 BuddyGraph 根目录
`/home/jlq/project/buddygraph`，除非明确写成绝对路径。

## 仓库、构建与工具边界

| 事实 | 实际状态 | 证据/入口 |
|---|---|---|
| BuddyGraph 根目录 | 位于 `/home/jlq/project/buddygraph`，在 Buddy-MLIR 文件树之外 | 根 `CMakeLists.txt`、`realpath` |
| BuddyGraph Git 状态 | 独立 Git 仓库；修订与工作区状态以实际 checkout 为准 | `git rev-parse --show-toplevel`、`git status --short` |
| 原 Buddy-MLIR | 只读依赖；项目未修改其注册/CMake | `docs/buddygraph/audit.md` |
| LLVM/MLIR 依赖 | 外部 LLVM/MLIR build；原容器验证路径为 `/buddy-mlir/llvm/build` | `build/CMakeCache.txt`、工具 `--version` |
| 独立 build | `build/` | 根 `CMakeLists.txt` |
| `MLIR_DIR` | 外部 LLVM build 的 `lib/cmake/mlir` | `build/CMakeCache.txt` |
| `LLVM_DIR` | 外部 LLVM build 的 `lib/cmake/llvm` | `build/CMakeCache.txt` |
| BGraph driver | `build/bin/buddygraph-opt` | `tools/buddygraph-opt/buddygraph-opt.cpp` |
| 原 `buddy-opt` | `/buddy-mlir/build/bin/buddy-opt`；已验证可解析第 11 章 Linalg 文件，其他 IR 仍取决于其注册 | 第 00、11 章 smoke test |
| upstream `mlir-opt` | `/buddy-mlir/llvm/build/bin/mlir-opt`；处理其已注册的 upstream Dialect | 第 00、11 章 |
| runner | `/buddy-mlir/llvm/build/bin/mlir-runner` | `tests/E2E/BGraph/onnx_elementwise.py` |
| LLVM IR translator | `/buddy-mlir/llvm/build/bin/mlir-translate` | 第 11 章 |
| copied Buddy source | **none** | 项目中无 `third_party/` |

教学指导预期的 `docs/audit.md`、`docs/toolchain.md` 和
`cmake/BuddyGraphToolchain.cmake` 均不存在。实际审计文件是
`docs/buddygraph/audit.md`，工具链发现逻辑直接位于根 `CMakeLists.txt`。

## 概念到真实代码

| 概念 | 实际文件 | 核心类/函数/定义 | 对应测试 |
|---|---|---|---|
| Dialect | `include/BuddyGraph/IR/BGraphDialect.td`、`lib/BuddyGraph/IR/BGraphDialect.cpp` | `BGraph_Dialect`、`BGraphDialect::initialize()` | `tests/Dialect/BGraph/roundtrip.mlir` |
| Layout Attribute | `include/BuddyGraph/IR/BGraphEnums.td` | `BGraph_Layout`、`BGraph_LayoutAttr` | Conv/BN 的全部 `.mlir` 测试 |
| Operations | `include/BuddyGraph/IR/BGraphOps.td` | `BGraph_Conv2DOp`、`BGraph_BatchNormOp`、`BGraph_ReluOp`、`BGraph_ClampOp`、binary/reshape/transpose/reduce/fused/yield defs | `tests/Dialect/BGraph/*.mlir` |
| Verifier | `lib/BuddyGraph/IR/BGraphOps.cpp` | 13 个 `*Op::verify()`、`verifyBinary()`、`getF32Tensor()` | `tests/Dialect/BGraph/invalid.mlir` |
| Canonicalization | `lib/BuddyGraph/IR/BGraphOps.cpp` | `ElideMulByOne`、`ElideNestedRelu`、`ElideNestedClamp`、`CollapseReshape`、`CancelTranspose` | `tests/Dialect/BGraph/canonicalize.mlir`、`tests/E2E/BGraph/signed_zero.mlir` |
| Shape inference | `lib/BuddyGraph/Transforms/InferShapes.cpp` | `inferShape()`、`BGraphInferShapes::runOnOperation()` | `tests/Dialect/BGraph/infer-shapes.mlir` |
| BN folding | `lib/BuddyGraph/Transforms/FoldBatchNorm.cpp` | `FoldBatchNormPattern::matchAndRewrite()` | `tests/Dialect/BGraph/fold-bn.mlir`、E2E |
| Fusion | `lib/BuddyGraph/Transforms/FuseElementwise.cpp` | `ElementwiseTree::collect()`、`emitScalar()`、`FuseElementwisePattern` | `tests/Dialect/BGraph/fuse-elementwise.mlir` |
| Conversion | `lib/BuddyGraph/Conversion/BGraphToLinalg.cpp` | 12 个已注册 lowering pattern specialization、`ConvertBGraphToLinalg::runOnOperation()` | `tests/Conversion/BGraphToLinalg/*` |
| Driver/注册 | `tools/buddygraph-opt/buddygraph-opt.cpp` | `main()`、`DialectRegistry`、`registerPasses()`、`MlirOptMain()` | `buddygraph-opt --help`、全部 lit |
| Importer | `frontend/BGraph/import_onnx.py` | `BGraphImporter`、`_validate_model()`、`_node_attribute_initializers()`、`_import_node()`、`import_module()` | `tests/Frontend/BGraph/importer.mlir` |
| Fixtures | `frontend/BGraph/generate_test_models.py` | `generate()` | frontend/E2E tests |
| 数值 E2E | `tests/E2E/BGraph/onnx_elementwise.py`、`onnx_clamp.py`、`signed_zero.mlir` | NumPy 三方对拍；Clamp/NaN；signed-zero IR/runtime 回归 | 三个文件本身都是 lit test |
| 评测 | `scripts/benchmark.py` | `PROFILES`、`LOWERING`、`sample()`、`static_alloc_bytes()` | 手工运行，结果见 `docs/buddygraph/results.md` |
| Out-of-tree CMake | 根及各子目录 `CMakeLists.txt` | `find_package(MLIR CONFIG)`、`add_mlir_dialect*`、`add_mlir_library`、`add_llvm_executable` | `cmake --build build --target check-buddygraph` |

## 自定义基础设施专题边界

| 概念 | 当前 BGraph 基线 | 带做位置 |
|---|---|---|
| 自定义 Dialect/Op | 已实现 `bgraph` 与全部业务 Op | `dialect_lab/01_custom_dialect.md`、`02_custom_op.md` 从零重建接线 |
| 自定义 TypeDef | 未实现；业务 IR 复用 builtin tensor | `dialect_lab/03_custom_type.md` 的 `!bglab.tag<"...">` |
| 项目自有 Trait | 未实现；使用 upstream Traits | `dialect_lab/04_custom_trait.md` |
| 项目自有 OpInterface | 未实现 | `dialect_lab/05_custom_interface.md` |
| 自定义 Pass | 已实现四个 BGraph passes | `dialect_lab/06_custom_pass.md` 从 Passes.td 带做到 CLI |

`bglab` 是学习者在实验副本中实现的教学 Dialect，不属于当前源码、ONNX 支持表或
14/14 基线。专题教程的职责是给出逐文件实现和验收闭环，不把规划功能写成已实现事实。

## 实际 Pass

| CLI 名称 | TableGen def | 实现类/入口 |
|---|---|---|
| `bgraph-infer-shapes` | `BGraphInferShapes` | `BGraphInferShapes::runOnOperation()` / `createInferShapesPass()` |
| `bgraph-fold-bn-into-conv` | `BGraphFoldBatchNormIntoConv` | `BGraphFoldBatchNormIntoConv::runOnOperation()` / `createFoldBatchNormIntoConvPass()` |
| `bgraph-fuse-elementwise` | `BGraphFuseElementwise` | `BGraphFuseElementwise::runOnOperation()` / `createFuseElementwisePass()` |
| `convert-bgraph-to-linalg` | `ConvertBGraphToLinalg` | `ConvertBGraphToLinalg::runOnOperation()` / `createConvertBGraphToLinalgPass()` |

## 实际 Op schema

| Op | operands | result | attributes/region |
|---|---|---|---|
| `bgraph.conv2d` | input、filter、可选 bias | ranked tensor | strides、pads、dilations、groups、layout |
| `bgraph.batch_norm` | input、scale、bias、mean、variance | ranked tensor | epsilon、layout |
| `bgraph.relu` | input | same-type tensor | 无 |
| `bgraph.clamp` | input | shape-compatible tensor | finite f32 `min_value`、`max_value` |
| `bgraph.add/sub/mul/div` | lhs、rhs | broadcast result tensor | 无 |
| `bgraph.reshape` | input | tensor | DenseI64 `shape` |
| `bgraph.transpose` | input | tensor | DenseI64 `permutation` |
| `bgraph.reduce_mean` | input | tensor | DenseI64 `axes`、Bool `keep_dims` |
| `bgraph.fused_elementwise` | variadic tensor inputs | tensor | single-block Region |
| `bgraph.yield` | scalar value | 无 | terminator，只能位于 fused op |

## 实现与规划的差异

| 规划/教学期望 | 真实状态 | 教学处理 |
|---|---|---|
| 根目录位于 Buddy-MLIR 之外 | 当前物理路径为 `/home/jlq/project/buddygraph`；CMake/build/driver 也隔离 | 明确区分源码独立与共享 LLVM/MLIR 工具链 |
| `docs/toolchain.md`、toolchain CMake | 未实现 | 直接讲根 CMake 和 Cache |
| 自定义 `InferTypeOpInterface` | 未接入任何 BGraph Op | 讲当前 Module pass，并把 Interface 作为扩展 |
| 自定义 `fold()` | 未实现 | 第 07 章明确区分“缺失的 fold hook”和已有 canonicalization |
| DRR patterns | 未实现，全部是 C++ patterns | 解释为何当前条件适合 C++ |
| `TypeConverter`、signature conversion、materialization、dynamic legality | 当前 Conversion 不需要，也未实现 | 先讲机制，再指出当前 builtin types 保持不变 |
| `buddygraph-import` tool | 未实现 | 使用 `frontend/BGraph/import_onnx.py` |
| dynamic batch | 前端和 lowering 闭环不支持 | 不作为已实现功能讲解 |
| ownership deallocation | runner pipeline 未加入 deallocation pass | 说明 alloc 存在，短进程结束回收；不得声称已做完整内存生命周期管理 |
| 手写示例的标量返回 | `elementwise_main.mlir` 在 Add→Relu→Mul 后用 `tensor.extract` 返回 f32 | ReduceMean 的导入/lowering 用 E2E fixture 讲解 |
| `Add-Relu-Mul-Relu` fixture | 实际 fusion lit 是 Add-Relu-Mul | 使用真实链，不补造末尾 Relu |
| Add-zero canonicalization | 已移除无 fast-math contract 的 rewrite；`-0.0 + +0.0` 保留 Add 并输出 `+0.0` | 第 07 章讲解缺陷发现、修复选择与 `signed_zero.mlir` 回归 |

## MVP 边界

- importer：默认 domain、单一 opset 18、f32、ranked static shape。
- Conv：2D、NCHW importer、显式 padding、groups=1；静态几何必须产生正输出维；
  Dialect 有 NHWC enum 和 lowering 分支，但端到端只承诺 NCHW。
- BN：inference mode、单输出。
- Reshape/Transpose/ReduceMean：shape、permutation、axes 必须是常量 initializer。
- Clip：min/max 必须同时提供为 finite scalar f32 initializer；只对当前 Clip 从 runtime
  operands 排除，允许 initializer 被其他节点复用。
- elementwise 与 fusion：支持静态 NumPy 风格右对齐广播；lowering 拒绝动态 shape。
- Conversion：Conv 的 input/filter/可选 bias/result 均须静态，动态输入会明确
  failed-to-legalize，不得 assertion/abort。
- 后端：CPU；无 GPU、量化、训练、控制流或完整 ONNX 支持。
