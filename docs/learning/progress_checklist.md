# 学习进度与证据清单

勾选前先填“我的证据”。建议把命令输出或笔记路径写在每项下方。

- [ ] 能从 ONNX Node 找到对应 BGraph Op。
  - 验收：任选 `generate_test_models.py` 中一个 node，指出 `_import_node()` 分支、
    生成的 op name 和 frontend FileCheck。
- [ ] 能解释项目采用独立 CMake/build 的原因。
  - 验收：指出根 CMake 的 `find_package` 与审计中受保护的根注册文件。
- [ ] 能准确描述项目位置，而不误称在 Buddy-MLIR 文件树之外。
  - 验收：提交 `realpath`、两个 `git rev-parse --show-toplevel` 的输出和解释。
- [ ] 能定位 `MLIR_DIR`、`LLVM_DIR`、`buddygraph-opt` 和下游工具。
  - 验收：从 `build/CMakeCache.txt` 和 `--version` 给出路径。
- [ ] 能证明项目没有修改原 Buddy-MLIR 注册文件。
  - 验收：说明项目只在独立目录新增文件，并指出独立 driver；不要用“根 worktree
    干净”作证，因为根仓库已有其他用户改动。
- [ ] 能确认 copied Buddy source 为 none。
  - 验收：项目内无 `third_party/`，并解释未来若复制必须记录 provenance。
- [ ] 能解释 ONNX name 到 SSA Value 的映射。
  - 验收：逐行解释 `self.values` 的 input、initializer、node output 三类写入。
- [ ] 能独立找到 ODS 生成文件。
  - 验收：列出 source `.td`、CMake target 和 `build/include/BuddyGraph/**/*.inc`。
- [ ] 能解释 generated class 与手写实现如何拼合。
  - 验收：从 `BGraphOps.h` 的宏 include 追到 `Conv2DOp::verify()`。
- [ ] 能新增一个简单 Op 的 ODS schema。
  - 验收：从已实现的 Clamp 重建 ODS contract，再在实验分支完成一个有界 schema 修改并让 TableGen target 构建。
- [ ] 能编写 custom verifier。
  - 验收：定位 Clamp 的 `min <= max`/finite/shape 检查并给出合法/非法命令。
- [ ] 能编写 negative diagnostic test。
  - 验收：`--verify-diagnostics` 测试通过且错误落在目标 op 行。
- [ ] 能解释 verifier 与 shape inference 的区别。
  - 验收：用一个动态结果 Add 说明“合法但可进一步收紧”。
- [ ] 能说明当前 shape inference 不是 Interface。
  - 验收：指出 `inferShape(Operation *)` 和 `walk()`，并说明没有 ODS interface trait。
- [ ] 能解释 fold 与 canonicalization 的区别，并知道当前没有 BGraph fold hook。
  - 验收：从 ODS/C++ 检索证据，演示 nested Relu canonicalization，并说明当前
    为何不注册 Add-zero pattern，以及 `signed_zero.mlir` 如何防止问题回归。
- [ ] 能解释 CSE/DCE 为什么依赖副作用信息。
  - 验收：指出 BGraph ops 的 `Pure` trait 和 fusion 的 `isMemoryEffectFree()`。
- [ ] 能逐步解释 BN folding Pattern。
  - 验收：按十个步骤说明匹配、常量读取、公式、创建、替换和 erase。
- [ ] 能解释 Region、BlockArgument 和 yield。
  - 验收：画出 fused op 的 tensor 外层和 scalar Region 内层。
- [ ] 能判断一个 elementwise 子图是否可融合。
  - 验收：对单链和 `do_not_duplicate` 两个函数预测结果并运行验证。
- [ ] 能编写一个 `OpConversionPattern`。
  - 验收：逐行解释已实现的 `ClampLowering` 或 `ReluLowering` 的 adaptor/result/replacement。
- [ ] 能解释 legal/illegal 和 FullConversion。
  - 验收：故意漏掉一个 pattern，得到 failed to legalize，并解释为何不应把 BGraph
    标成 legal。
- [ ] 能说明当前为什么没有 `TypeConverter`。
  - 验收：指出 source/result 都复用 builtin tensor，函数 signature 不变。
- [ ] 能追踪 tensor 到 memref。
  - 验收：保存 conversion 前、bufferization 后 IR，解释 `tensor.empty` 与
    `memref.alloc` 的对应关系。
- [ ] 能区分 LLVM Dialect 与 LLVM IR。
  - 验收：分别给出 `.llvm.mlir` 和 `mlir-translate` 生成 `.ll` 的片段。
- [ ] 能定位 failed to legalize。
  - 验收：使用 `--mlir-print-ir-before/after-all` 找到第一个残留非法 op。
- [ ] 能使用 lit/FileCheck 验证 IR。
  - 验收：新增一条正向和一条 `CHECK-NOT`，单测与全套测试均通过。
- [ ] 能解释实际性能数字的测量范围。
  - 验收：说明 runner 数字包含独立进程、解析和 JIT，不能当作 kernel latency。
- [ ] 能完成新 Op 的全链路扩展。
  - 验收：独立核对 Clamp 的 ODS、verifier、shape、canonicalization、fusion、importer、
    lowering 和测试，完成一个有界增量，FullConversion 后无 BGraph。
- [ ] 能做 10 分钟项目演示。
  - 验收：按第 15 章脚本现场生成模型、展示两项优化、FullConversion 并运行结果。
