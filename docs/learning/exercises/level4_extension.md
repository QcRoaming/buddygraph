# Level 4：审计并扩展 `bgraph.clamp`

## 总任务

当前工程已经实现 ONNX `Clip` → `bgraph.clamp` → Linalg → runner 的完整闭环。本练习不再
假设 Clamp 缺失，而是要求你先独立重建它的证据链，再做一个边界明确的小扩展。

```mlir
%y = "bgraph.clamp"(%x) {min_value = -1.0 : f32, max_value = 1.0 : f32}
  : (tensor<4xf32>) -> tensor<4xf32>
```

## 涉及文件

- `include/BuddyGraph/IR/BGraphOps.td`
- `lib/BuddyGraph/IR/BGraphOps.cpp`
- `frontend/BGraph/import_onnx.py`
- `frontend/BGraph/generate_test_models.py`
- `lib/BuddyGraph/Transforms/{InferShapes,FuseElementwise}.cpp`
- `lib/BuddyGraph/Conversion/BGraphToLinalg.cpp`
- `tests/Dialect/BGraph/`、`tests/Frontend/BGraph/`、
  `tests/Conversion/BGraphToLinalg/`、`tests/E2E/BGraph/`

## 禁止直接修改

- `/buddy-mlir` 中既有 Buddy、MLIR、LLVM 源码及其构建产物。
- `build/include/BuddyGraph/IR/*.inc` 等 TableGen 生成物。
- 为绕过失败而把 BGraph Dialect 标为 legal，或删除 FullConversion。
- 只打印正确答案而不执行生成 IR 的伪 E2E。

## 分阶段任务与预期证据

### A. ODS 与 verifier

从 ODS 写出生成的 accessors，解释为什么没有 `SameOperandsAndResultType`。逐项核对 ranked
f32、shape 兼容、finite bounds 和 `min <= max`。证据：合法 IR 与至少三个 stable
negative diagnostics。

### B. importer 与 SSA 映射

画出 `Clip` 三个输入在 importer 中的去向：data 变 SSA operand，min/max 变 attributes。
解释为什么 bounds 只对当前节点从 runtime operands 排除。证据：正向 Clip、共享
initializer、运行时/缺失/非标量 bounds 四类 fixture。

### C. shape inference

证明 Clamp 不改变 shape。证据：未精化 result 经 `bgraph-infer-shapes` 变成 input 的
ranked tensor type，并能指出当前实现不是 `InferTypeOpInterface`。

### D. canonicalization

解释 nested Clamp 规则的两个必要条件：bounds 完全相同，replacement type 完全相同。
证据：相同 bounds 命中、不同 bounds 保留；说明为什么不采用
`relu(clamp(x)) → clamp(x)`。

### E. fusion

追踪 Clamp 如何进入 fusible 分类、root patterns、scalar emitter 与 Region verifier。
证据：Add→Clamp→Mul 变成一个 `bgraph.fused_elementwise`，Region 内有 ordered
`arith.cmpf` 和 `arith.select`。

### F. lowering

对比独立 `ClampLowering` 与 fused Region lowering。证据：两条 FullConversion 路径都无
`bgraph.` 残留，并存在 `linalg.generic`、`arith.cmpf` 和 `arith.select`。

### G. 数值闭环

运行 `onnx_clamp.py`，解释 `0.5` reference 的手算过程、direct/fused 零误差和 NaN 传播。
signed-zero bit pattern 不属于当前 MVP gate，不能扩大结论。

### H. 你的增量扩展

任选一个小改动，例如新增 `min > max` 或 Constant-node bound 的 importer 拒绝
fixture。提交前先写 contract，并同时给出正例、反例、命令和证据路径。不要把
`min == max` 直接折成常量（它会破坏 NaN 传播），也不要在没有 reference 的情况下
扩展 optional/dynamic Clip 语义。

## 提示

1. 本地源码与本地 MLIR API 是准绳，不从旧博客猜 builder 签名。
2. `IRMapping` 只负责 tensor leaf 到 BlockArgument；Clamp 的 scalar 语义仍需显式重建。
3. importer 是 Python MLIR API；项目没有 `buddygraph-import` 二进制。
4. conversion 的结束条件是所有 BGraph Op 被消除，而非输出“看起来像 Linalg”。

## 本级验收

提交一张矩阵：每一层的代码符号、正例、反例、命令和证据路径。然后执行：

```bash
cd /buddy-mlir/jlq/projects/buddygraph
cmake --build build --target check-buddygraph -j2
export BUDDYGRAPH_TMP="$PWD/tmp"
mkdir -p "$BUDDYGRAPH_TMP/level4"
export PYTHONPATH="$PWD/.deps:/buddy-mlir/llvm/build/tools/mlir/python_packages/mlir_core${PYTHONPATH:+:$PYTHONPATH}"
python3 tests/E2E/BGraph/onnx_clamp.py \
  "$PWD" "$PWD/build" /buddy-mlir/llvm/build/bin \
  "$BUDDYGRAPH_TMP/level4/clamp-e2e"
```

验收基线是 14/14、`status=PASS`，以及你的增量测试。最后用 5 分钟回答：Clamp contract
在哪里验证、shape 在哪里精化、Region 内如何标量化、FullConversion 如何发现遗漏。

完成后再看 [Level 4 答案](../solutions/level4_extension.md)。
