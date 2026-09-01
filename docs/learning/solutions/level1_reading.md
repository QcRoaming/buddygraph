# Level 1 答案：阅读与追踪

## 1. 工程边界

当前事实不是“BuddyGraph 根目录不在 Buddy-MLIR 源码树中”。真实路径为
`/buddy-mlir/jlq/projects/buddygraph`，所以它物理上位于 `/buddy-mlir` 之下；其 Git 顶层
目录是 `/buddy-mlir/jlq`，而 `/buddy-mlir` 自己又是另一个 Git worktree。准确说法是：

> BuddyGraph 是具有独立 `project()`、CMake 配置、构建目录、工具和测试目标的
> out-of-tree MLIR 工程，但当前被放置在 `/buddy-mlir/jlq` worktree 内。

这里的 out-of-tree 描述构建集成方式，不等价于文件系统前缀一定在 `/buddy-mlir` 外。

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
