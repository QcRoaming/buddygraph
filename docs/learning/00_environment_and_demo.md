# 00｜环境、工具边界与最小演示

> **本章路线：入口与产物。** 本章先确认工具与文件位置，再把手写 BGraph 降到 LLVM Dialect 并运行。输入是 examples 中的 MLIR，输出是项目 tmp 下的文件与标量 16。先看第 5 节 IR，再按第 2 节运行；第 01 章才接 ONNX。

## 1. 本章目标

完成本章后，你应能用命令证明项目的真实位置和版本，解释各 optimizer、translator
和 runner 的职责边界，并把一个 BGraph 示例降低到 LLVM Dialect 后执行。

## 2. 先运行

```bash
cd /home/jlq/project/buddygraph
export BUDDYGRAPH_TMP=/home/jlq/project/buddygraph/tmp
mkdir -p "$BUDDYGRAPH_TMP"

realpath .
git rev-parse --show-toplevel
git -C /buddy-mlir rev-parse --show-toplevel

build/bin/buddygraph-opt --version
/buddy-mlir/build/bin/buddy-opt --version
/buddy-mlir/llvm/build/bin/mlir-opt --version
/buddy-mlir/llvm/build/bin/mlir-translate --version
/buddy-mlir/llvm/build/bin/mlir-runner --version

cmake --build build --target check-buddygraph -j2
```

关键输出是 `LLVM version 21.0.0git` 和 `14/14` tests passed。BuddyGraph 的
Git top-level 是 `/home/jlq/project/buddygraph`，位于 Buddy-MLIR 文件树之外；
其 LLVM/MLIR 工具链仍来自 `/buddy-mlir/llvm/build`。

运行最小例子：

```bash
build/bin/buddygraph-opt \
  --bgraph-fuse-elementwise \
  --convert-bgraph-to-linalg \
  '--one-shot-bufferize=bufferize-function-boundaries' \
  --convert-linalg-to-loops --lower-affine \
  --convert-scf-to-cf --convert-cf-to-llvm \
  --convert-math-to-llvm --convert-arith-to-llvm \
  --finalize-memref-to-llvm --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  examples/BGraph/elementwise_main.mlir \
  -o "$BUDDYGRAPH_TMP/buddygraph-demo.llvm.mlir"

/buddy-mlir/llvm/build/bin/mlir-runner \
  "$BUDDYGRAPH_TMP/buddygraph-demo.llvm.mlir" \
  -e main -entry-point-result=f32
```

预期输出：`1.600000e+01`。

这里的文件实际位于
`/home/jlq/project/buddygraph/tmp/buddygraph-demo.llvm.mlir`。它是
`buddygraph-opt -o` 生成的中间产物，不是源文件；源输入仍是
`examples/BGraph/elementwise_main.mlir`。项目 `tmp/` 比系统临时目录更容易在 VS Code
中查看，并由 `.gitignore` 排除；固定文件名在重复运行时会被覆盖。

## 3. 真实代码位置

- 根 `CMakeLists.txt`：`find_package(MLIR REQUIRED CONFIG)`、include path、子目录。
- `tools/buddygraph-opt/CMakeLists.txt`：定义 `buddygraph-opt` executable。
- `tools/buddygraph-opt/buddygraph-opt.cpp`：driver 的 `main()`。
- `tests/CMakeLists.txt`：`check-buddygraph` target。
- `tests/lit.site.cfg.py.in`：把 source/build/LLVM/Python 路径写入 lit site config。
- `examples/BGraph/elementwise_main.mlir`：本章真实输入。

注意：项目没有 `cmake/BuddyGraphToolchain.cmake`。以下命令展示实际 cache：

```bash
rg '^(MLIR_DIR|LLVM_DIR|CMAKE_BUILD_TYPE):' build/CMakeCache.txt
```

## 4. 调用链

1. CMake 用 `MLIR_DIR` 找到 `MLIRConfig.cmake`，并从中获得 MLIR CMake modules、
   headers 和 libraries。
2. `add_llvm_executable(buddygraph-opt ...)` 构建独立 driver。
3. `main()` 调用 `mlir::registerAllPasses()` 和生成的
   `buddy::bgraph::registerPasses()`。
4. `DialectRegistry` 注册 BGraph 及 lowering 会用到的标准 Dialect。
5. `MlirOptMain()` 解析命令行、读取 IR、构建 PassManager 并运行 pipeline。
6. 自定义 Op 经 FullConversion 后全部消失；One-Shot Bufferize 随后把 tensor 转为
   buffer；后续 passes 生成 LLVM Dialect。
7. `mlir-runner` 解析 LLVM Dialect module、JIT 编译 `main` 并按 `f32` ABI 打印结果。

## 5. IR 前后变化

输入中的核心链是：

```mlir
%sum = "bgraph.add"(%a, %b)
    : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
%relu = "bgraph.relu"(%sum) : (tensor<4xf32>) -> tensor<4xf32>
%scaled = "bgraph.mul"(%relu, %scale)
    : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
```

fusion 后外层是 tensor op，Region 内是 scalar：

```mlir
%0 = "bgraph.fused_elementwise"(%a, %b, %scale) ({
^bb0(%x: f32, %y: f32, %s: f32):
  %sum = arith.addf %x, %y : f32
  %zero = arith.constant 0.0 : f32
  %relu = arith.maximumf %sum, %zero : f32
  %scaled = arith.mulf %relu, %s : f32
  "bgraph.yield"(%scaled) : (f32) -> ()
}) : (...) -> tensor<4xf32>
```

最终文件除 builtin 容器外，计算与函数 op 应已降为 `llvm.*`，可检查：

```bash
rg 'bgraph\.|linalg\.|tensor\.|memref\.|func\.' \
  "$BUDDYGRAPH_TMP/buddygraph-demo.llvm.mlir" || true
rg 'llvm.func @main' "$BUDDYGRAPH_TMP/buddygraph-demo.llvm.mlir"
```

第一条应无输出，第二条应命中。

## 6. 核心机制

| 目录/工具 | 教学中是否可写 | BuddyGraph 中的职责 |
|---|---:|---|
| `/home/jlq/project/buddygraph/` | 生成教程时只写 `docs/learning/`；学习者实验可改自己的项目副本 | 项目源码、独立 build、测试和教学材料 |
| `/buddy-mlir/` 原 Buddy/LLVM 源码 | 否 | API/version 参考和既有工具链 |
| 项目 `third_party/buddy-mlir/` | 不存在 | 若未来创建，必须有 provenance 才可修改副本 |
| `buddygraph-opt` | 执行工具 | 唯一注册 BGraph 的 optimizer |
| 原 `buddy-opt` | 只读执行 | 已验证可解析第 11 章的 Linalg 文件；能否处理其他 IR 取决于自身注册，不能解析 BGraph |
| `mlir-opt` | 只读执行 | upstream 标准 Dialect 工具 |
| `mlir-translate` | 只读执行 | LLVM Dialect MLIR → LLVM IR |
| `mlir-runner` | 只读执行 | JIT 执行已降低 module |

`MLIR_DIR`/`LLVM_DIR` 是 CMake package 路径，不是可执行文件目录。`build/` 是
BuddyGraph 的 object tree；`/buddy-mlir/llvm/build` 是被复用的 LLVM/MLIR build。

## 7. 为什么这样设计

根 Buddy 注册/CMake 已有用户的 Microkernel 研究修改。独立 driver 避免在同一组
注册文件上叠加工作，同时仍复用已经构建好的 MLIR 21 libraries。代价是 BGraph IR
必须交给 `buddygraph-opt`，不能假设原 `buddy-opt` 知道该 Dialect。

## 8. 常见错误

- `unknown dialect 'bgraph'`：使用了 `mlir-opt`/原 `buddy-opt` 读取 BGraph。
- `Could not find MLIRConfig.cmake`：`MLIR_DIR` 指到了源码或错误 build。
- `No module named onnx/mlir`：Python 没有 `.deps` 或 MLIR bindings 路径。
- runner 报 `math.sqrt`：pipeline 漏了 `--convert-math-to-llvm`。
- 误读示例：手写 demo 的 Add→Relu→Mul 后用 `tensor.extract %scaled[3]` 返回标量；
  ReduceMean 在 ONNX E2E fixture 中演示，不要混淆两个输入。

## 9. 动手练习

阅读题：解释为什么 `git -C /buddy-mlir status` 不能单独证明 BuddyGraph 没有触碰
用户其他改动。然后列出能证明“本项目注册隔离”的三个文件级证据。

调试题：把命令中的 `build/bin/buddygraph-opt` 临时换成
`/buddy-mlir/llvm/build/bin/mlir-opt`，记录第一条错误，再恢复正确命令。不要修改
任何源文件。

## 10. 验收标准

- `check-buddygraph` 通过。
- runner 输出 `1.600000e+01`。
- 能指出 `MLIR_DIR`、`LLVM_DIR` 和表中各工具的绝对路径。
- 能准确说出“源码与构建独立，但 LLVM/MLIR 工具链仍由 `/buddy-mlir` 提供”。

## 11. 面试追问

**问：为什么不用原 `buddy-opt`？**

答：BGraph 需要 Dialect、Pass 和 bufferization external models 注册；根 driver 已有
受保护改动。独立 `buddygraph-opt` 提供相同 `MlirOptMain` 能力而不触碰根注册面。

**问：复用 build 会不会破坏隔离？**

答：不会。复用的是只读 headers/libraries 和工具；BuddyGraph 有自己的 CMake
target、object tree 和 executable。API 版本仍被本地 MLIR commit 固定。
