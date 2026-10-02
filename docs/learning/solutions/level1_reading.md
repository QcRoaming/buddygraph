# Level 1 答案：阅读与追踪

## 1. 工程边界

BuddyGraph 的真实路径为 `/home/jlq/project/buddygraph`，在 `/buddy-mlir` 文件树
之外；`git rev-parse --show-toplevel` 返回独立项目路径。准确说法是：

> BuddyGraph 是具有独立 `project()`、CMake 配置、构建目录、工具和测试目标的
> BuddyGraph 是独立源码与 Git 仓库、独立 CMake 配置、构建目录、工具和测试目标的
> out-of-tree MLIR 工程，同时依赖 `/buddy-mlir/llvm/build` 的 LLVM/MLIR 包。

这里的源码独立不等于工具链自给；构建仍需指定 `MLIR_DIR` 和 `LLVM_DIR`。

## 2. 构建与测试入口

- 根 `CMakeLists.txt` 用 `find_package(MLIR REQUIRED CONFIG)` 和
  `find_package(LLVM REQUIRED CONFIG)` 接收 `MLIR_DIR`/`LLVM_DIR`。
- 当前 `build/CMakeCache.txt` 分别记录
  `/buddy-mlir/llvm/build/lib/cmake/mlir` 和 `/buddy-mlir/llvm/build/lib/cmake/llvm`。
- `tests/CMakeLists.txt` 的 `add_lit_testsuite(check-buddygraph ...)` 定义测试目标。
- `tools/buddygraph-opt/CMakeLists.txt` 的 `add_llvm_executable(buddygraph-opt ...)`
  定义工具。

## 3. ODS、生成和注册链

以 Add 为例：

```text
BGraph_AddOp in BGraphOps.td
→ mlir_tablegen(BGraphOps.h.inc -gen-op-decls)
→ MLIRBGraphOpsIncGen
→ BGraphOps.h with GET_OP_CLASSES
→ BGraphOps.cpp with GET_OP_CLASSES
→ BGraphDialect::initialize() / addOperations<...>()
→ buddygraph-opt.cpp registers BGraphDialect
```

生成的 `buddy::bgraph::registerPasses()` 负责 Pass 注册。真实命令行名包括
`bgraph-infer-shapes`、`bgraph-fold-bn-into-conv`、`bgraph-fuse-elementwise` 和
`convert-bgraph-to-linalg`。

## 4. use-def 图

真实测试的核心是：

```text
%arg0 ─┐
       ├─ bgraph.add ─ %add ─ bgraph.relu ─ %relu ─┐
%arg1 ─┘                                            ├─ bgraph.mul ─ %mul ─ return
%arg2 ──────────────────────────────────────────────┘
```

`%add` 和 `%relu` 各只有一个用户，三者都在受支持的 elementwise 集合中，shape 静态且
广播兼容，因此能组成一个融合 Region。最初规划中的第二个 Relu 不存在于当前 lit 测试，
不能写入事实链。
