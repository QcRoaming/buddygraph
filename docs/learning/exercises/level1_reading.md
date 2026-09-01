# Level 1：阅读与追踪

## 任务 1：审计工程边界

- **任务**：用 `pwd -P`、Git 顶层目录和 CMake 文件证明或反驳“BuddyGraph 根目录不在
  Buddy-MLIR 源码树中”。解释“独立构建工程”与“物理目录/Git 边界”是否为同一概念。
- **涉及文件**：`CMakeLists.txt`、`README.md`。
- **禁止直接修改**：所有文件；本题只读。
- **预期证据**：三个真实绝对路径，以及一句不夸大独立性的结论。
- **提示**：比较 `git -C . rev-parse --show-toplevel` 和
  `git -C /buddy-mlir rev-parse --show-toplevel`；再看项目是否有自己的 `project()`、
  `find_package(MLIR)` 和 `build/`。

## 任务 2：追踪构建与测试入口

- **任务**：找出 `MLIR_DIR`、`LLVM_DIR` 的配置来源，解释 `check-buddygraph` 如何连接
  `lit` 测试目录，并找到 `buddygraph-opt` 的定义位置。
- **涉及文件**：`CMakeLists.txt`、`tests/CMakeLists.txt`、`tools/buddygraph-opt/CMakeLists.txt`、
  `build/CMakeCache.txt`。
- **禁止直接修改**：`build/CMakeCache.txt` 和 LLVM/Buddy 工具链。
- **预期证据**：配置变量的实际值、三个 CMake 调用点、一次
  `cmake --build build --target check-buddygraph` 输出。
- **提示**：搜索 `find_package`、`add_lit_testsuite`、`add_llvm_executable`。

## 任务 3：从 ODS 追到注册入口

- **任务**：任选 `bgraph.add` 或 `bgraph.relu`，从 ODS 定义追到生成声明被 include、
  生成实现被 include、Dialect 初始化和命令行工具注册。
- **涉及文件**：`include/BuddyGraph/IR/BGraphOps.td`、
  `include/BuddyGraph/IR/BGraphOps.h`、`lib/BuddyGraph/IR/BGraphOps.cpp`、
  `lib/BuddyGraph/IR/BGraphDialect.cpp`、`tools/buddygraph-opt/buddygraph-opt.cpp`。
- **禁止直接修改**：`build/include/BuddyGraph/IR/*.inc`；它们是生成物。
- **预期证据**：一条至少含 6 个节点的调用/生成链，并给出一个 Pass 的真实命令行名。
- **提示**：`rg 'GET_OP|addOperations|registerPasses' include lib tools`。

## 任务 4：画 SSA use-def 子图

- **任务**：阅读 `tests/Dialect/BGraph/fuse-elementwise.mlir`，画出融合前
  Add→Relu→Mul 的
  SSA use-def 子图，标明每个结果的用户数；再说明为什么该图满足融合条件。
- **涉及文件**：`tests/Dialect/BGraph/fuse-elementwise.mlir`、
  `lib/BuddyGraph/Transforms/FuseElementwise.cpp`。
- **禁止直接修改**：测试和实现。
- **预期证据**：一张文本图或 Mermaid 图；其中必须出现 SSA value、producer、user 和
  `hasOneUse()` 判断。
- **提示**：不要把最初规划中的第二个 Relu 画进去；真实测试链到 Mul 为止。

## 本级验收

- 你的结论区分了独立 CMake 工程、物理目录和 Git worktree。
- 你能从 `bgraph.add` 的 ODS 走到 `buddygraph-opt` 的 Dialect/Pass 注册。
- 你能指出一个真实 Pass 名，如 `bgraph-fuse-elementwise`。
- 你的 use-def 图可由测试中的 SSA 名称逐项核对。

完成后再看 [Level 1 答案](../solutions/level1_reading.md)。
