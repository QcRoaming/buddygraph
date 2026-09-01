# 03｜ODS 与 TableGen：从 `.td` 追到可用 C++ 类

## 1. 本章目标

你将能从 `BGraph_ReluOp` 和 `BGraph_FusedElementwiseOp` 的 ODS 定义追到实际生成
`.inc`、手写 include、Dialect 初始化和 parser/verifier，并能区分人写代码与生成
代码。

## 2. 先运行

```bash
cd /buddy-mlir/jlq/projects/buddygraph
cmake --build build --target MLIRBGraphOpsIncGen BuddyGraphPassesIncGen -j2
find build/include/BuddyGraph -name '*.inc' -printf '%P\n' | sort
```

应看到 `IR/BGraphOps.h.inc`、`IR/BGraphOps.cpp.inc`、Dialect/Enum/Attribute files 和
`Transforms/Passes.h.inc`。生成文件在 build tree，不提交到 source tree。

## 3. 真实代码位置

- `BGraphDialect.td`：`BGraph_Dialect` 与 `BGraph_Op` 基类。
- `BGraphEnums.td`：`Layout` enum 和 `LayoutAttr` 的 assembly format。
- `BGraphOps.td`：所有 op schema。
- `include/BuddyGraph/IR/CMakeLists.txt`：`mlir_tablegen` 与
  `add_mlir_dialect(BGraphOps bgraph)`。
- `BGraphOps.h`：以 `GET_ATTRDEF_CLASSES`、`GET_OP_CLASSES` include 生成声明。
- `BGraphDialect.cpp`：include dialect definitions，并在 `initialize()` 中注册生成的
  op/attribute list。
- `BGraphOps.cpp`：手写 verifier/canonicalization，文件末尾 include op definitions。

## 4. 调用链

```text
BGraphOps.td
→ set(LLVM_TARGET_DEFINITIONS BGraphOps.td)
→ add_mlir_dialect(BGraphOps bgraph)
→ target MLIRBGraphOpsIncGen
→ build/include/BuddyGraph/IR/BGraphOps*.inc
→ BGraphOps.h / BGraphDialect.cpp / BGraphOps.cpp include
→ BGraphDialect::initialize()
→ addOperations<generated list>()
→ buddygraph-opt registry loads dialect
→ parser creates generated typed op wrapper
→ generated structural verification + hand-written verify()
```

Pass 生成链独立：`Passes.td` → `BuddyGraphPassesIncGen` → `Passes.h.inc` → 生成
pass base/registration → 手写 `runOnOperation()`。

## 5. IR 前后变化

简单 Op：

```tablegen
def BGraph_ReluOp : BGraph_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}
```

它让 generic IR：

```mlir
%0 = "bgraph.relu"(%arg0)
    : (tensor<2x3xf32>) -> tensor<2x3xf32>
```

获得 `ReluOp::getInput()`、`getResult()`、`verify()` 声明和 canonicalization 注册
入口。

复杂 Op：

```tablegen
def BGraph_FusedElementwiseOp : BGraph_Op<"fused_elementwise", [Pure]> {
  let arguments = (ins Variadic<AnyRankedTensor>:$inputs);
  let results = (outs AnyRankedTensor:$result);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}
```

它额外生成 variadic operand range 和 Region accessor，但 Region 的 single-block、
scalar arguments、op 白名单与 yield 规则仍由手写 verifier 实现。

Layout 是项目唯一自定义 assembly format：

```tablegen
let assemblyFormat = "`<` $value `>`";
```

对应 `#bgraph.layout<nchw>`。BGraph ops 本身没有 custom `assemblyFormat`，测试使用
generic quoted form；不要虚构漂亮的 custom op syntax。

## 6. 核心机制

ODS 声明结构事实：operands、results、attributes、regions、traits，以及是否需要
手写 hook。TableGen 生成 boilerplate，但不会自动实现项目语义。

- `Pure` 组合了无副作用建模，让 canonicalizer/CSE/DCE/fusion 能安全判断。
- `SameOperandsAndResultType` 给 Relu 结构约束。
- `Terminator` 与 `HasParent<"FusedElementwiseOp">` 约束 Yield。
- `hasVerifier = 1` 只生成声明和调用 glue；body 是 `*Op::verify()`。
- `hasCanonicalizer = 1` 同理，pattern 注册函数由人写。

当前 `.td` include 了 InferType interface 定义，但没有任何 op 在 trait/interface list
中实现它；shape inference 是 Module pass。当前也没有 `hasFolder`/`fold` 声明、
`extraClassDeclaration` 或 DRR pattern。

`usePropertiesForAttributes = 0` 是本地 MLIR Python API 兼容选择：当前
`Operation.create` 只有 attributes 参数。属性仍是 ODS 强类型 schema，不是任意
字符串字典。

## 7. 为什么这样设计

简单结构交给 ODS 能减少 accessor/registration 错误；依赖数值、shape、use-count
和 Region 白名单的规则留在 C++，诊断与控制流更清楚。项目选择 C++
canonicalization 而非 DRR，是因为现有 patterns 需要 Dense splat 检查或属性计算。

## 8. 常见错误

- 修改 `.td` 后只编译 `.cpp`，生成 `.inc` 未更新。
- 手工编辑 build 中的 `.inc`；下次 TableGen 会覆盖。
- 在 `hasVerifier = 1` 后忘记提供 `Op::verify()`，链接失败。
- 把 CMake target 写成审计早期计划的 `BGraphOpsIncGen`；实际 target 是
  `MLIRBGraphOpsIncGen`。
- `HasParent` 写错生成类名，TableGen 报类型不存在。

## 9. 动手练习

阅读题：对 `BGraph_Conv2DOp` 的三个 operand 类别和五个 attributes，逐一在生成的
`BGraphOps.h.inc` 中找到 accessor 名。不要修改生成文件。

小修改预演：阅读已经实现的 `bgraph.clamp` ODS，列出它生成的 accessor 和仍由 C++
手写的 verifier/canonicalization；第 14 章再沿真实代码核对全链路。

## 10. 验收标准

- 两个 TableGen targets 构建成功。
- 能指出每个 `.inc` 的 source `.td` 和 include site。
- 能解释 `Pure`、`hasVerifier`、`hasCanonicalizer` 各自做什么。
- 能明确说出当前未实现 custom fold、shape interface、custom op assembly。

## 11. 面试追问

**问：TableGen 是否替你实现了 verifier？**

答：它生成 schema 级检查和调用 glue；channel、broadcast、shape、Region 白名单等
项目语义在手写 `BGraphOps.cpp` 中。

**问：为什么生成文件不入库？**

答：它们由本地 MLIR 21 TableGen 和 `.td` 确定，build target 能可靠重建；提交会
引入版本漂移和重复 source of truth。
