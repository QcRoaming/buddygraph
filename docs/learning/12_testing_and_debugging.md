# 12｜lit、FileCheck 与分层调试

> **本章路线：把症状定位到第一处原因。** 先按构建/解析/验证/改写/执行分层，再选择对应测试与 IR checkpoint。阅读本章后应能解释一个测试排除了什么错误，而不只是得到 PASS。

## 1. 本章目标

你将能单独运行一条 lit 测试，读懂 substitutions 和 FileCheck 顺序，使用 IR dump、
timing、diagnostic verification 和 debugger 定位 verifier、rewrite、conversion 或数值
层的失败，并完成三个可复现故障注入。

## 2. 先运行

```bash
cd /home/jlq/project/buddygraph
export BUDDYGRAPH_TMP=/home/jlq/project/buddygraph/tmp
mkdir -p "$BUDDYGRAPH_TMP"

/usr/bin/python3.10 /buddy-mlir/llvm/build/bin/llvm-lit -sv \
  build/tests/Dialect/BGraph/invalid.mlir

/usr/bin/python3.10 /buddy-mlir/llvm/build/bin/llvm-lit -sv \
  build/tests/E2E/BGraph/onnx_elementwise.py

build/bin/buddygraph-opt --mlir-timing \
  --bgraph-fold-bn-into-conv --bgraph-fuse-elementwise \
  tests/Dialect/BGraph/fold-bn.mlir -o /dev/null
```

第一条验证 expected diagnostics，第二条运行真实 Python/runner E2E，第三条在 stderr
打印 pass timing。

## 3. 真实代码位置

- `tests/lit.cfg.py`：suffixes、environment、tool substitutions。
- `tests/lit.site.cfg.py.in`：CMake 配置到 lit 的真实路径。
- `tests/CMakeLists.txt`：`check-buddygraph` suite 与 dependencies。
- `tests/Dialect/BGraph/invalid.mlir`：`-verify-diagnostics`。
- `tests/Conversion/BGraphToLinalg/full-conversion.mlir`：多 prefix FileCheck。
- `tests/Frontend/BGraph/importer.mlir`：Python、`not`、locations。
- `tests/E2E/BGraph/onnx_elementwise.py`：数值测试。
- `docs/buddygraph/debugging.md`：项目级已知问题。

## 4. 调用链

```text
cmake --build ... --target check-buddygraph
→ tests/CMakeLists.txt 的 add_lit_testsuite
→ llvm-lit 读取 build/tests/lit.site.cfg.py
→ site config 注入 source/build/LLVM/Python paths
→ load tests/lit.cfg.py
→ 查找 .mlir / .py
→ 展开 RUN: 中的 %s, %S, %t, %PYTHON, buddygraph-opt, FileCheck
→ shell 执行 pipeline
→ FileCheck / expected diagnostics 判断 pass/fail
```

调试一个 pass：

```text
parse + verify input
→ PassManager
→ --mlir-print-ir-before-all
→ pass runOnOperation
→ verifier
→ --mlir-print-ir-after-all
→ failure 时 driver 非零退出
```

## 5. IR 前后变化

FileCheck 不是全文比较。比如：

```text
// FUSED-NOT: bgraph.
// FUSED-COUNT-1: linalg.generic
// FUSED: arith.addf
// FUSED: arith.maximumf
// FUSED: arith.mulf
```

它验证关键结构和顺序。`CHECK-NOT` 的作用范围位于相邻正向 checks 之间；放置错误
可能漏检。`CHECK-COUNT-1` 会把扫描位置推进到匹配后，后续检查不能再寻找此前文本。

Negative diagnostic：

```mlir
// expected-error@+1 {{operands are not broadcast compatible}}
%0 = "bgraph.add" ...
```

`buddygraph-opt --verify-diagnostics` 会要求错误位置和内容匹配，并拒绝多余诊断。

## 6. 核心机制

常用调试参数已在本地 optimized-with-assertions build 验证存在：

```bash
--mlir-print-ir-before-all
--mlir-print-ir-after-all
--mlir-print-ir-after=<pass-name>
--mlir-timing
--debug-only=<category>
```

`--debug-only` 只有在相关代码使用 `LLVM_DEBUG` 且 category 正确时才有输出；BGraph
自定义源码当前没有自己的 debug category。可尝试 upstream
`--debug-only=dialect-conversion`，但具体 trace 受本地 MLIR 版本影响。

GDB/LLDB：

```bash
gdb --args build/bin/buddygraph-opt \
  --bgraph-fuse-elementwise tests/Dialect/BGraph/fuse-elementwise.mlir
```

本地有 `/usr/bin/gdb` 和 `/usr/lib/llvm-14/bin/lldb`；工具版本不同，优先用 GDB
调试 GCC 构建。Release 优化会合并/内联局部变量，但函数断点和 backtrace 仍有用。

## 7. 为什么这样设计

### 先说一个测试排除了什么错误

| 测试 | 排除的错误 | 仍不能证明什么 |
|---|---|---|
| parse/round-trip | 注册、schema、文本表示接线 | 数值正确 |
| verify-diagnostics | 指定非法输入被按预期拒绝 | 所有非法输入都被覆盖 |
| FileCheck 优化后结构 | 某条 pattern 命中/某个 Op 残留 | scalar 计算完全正确 |
| FullConversion 正反例 | target legality 与支持边界 | 所有 maps 都访问正确元素 |
| runner 对拍 | 已覆盖输入上的数值一致 | 全输入证明与真实模型加速 |

`CHECK-NOT` 只检查相邻正匹配之间的区间；如果要禁止整个函数出现 BGraph，需用
函数 label/return 等边界合理覆盖，或在整体输出上另做检查。只检查“出现 fused Op”
还不足以证明指定 Clamp/Square 确实进入 body，需检查内部 scalar 运算与 source 残留。

### 数值错误怎样逐步缩小

固定同一个输入，先确认 reference 与优化关闭路径一致，再逐个启用 BN、canonicalize/CSE、
fusion。找到首个差异后保存该配置的 IR，缩到一个 op/一条链，比较属性、dtype、maps
和 scalar body。正常数值、NaN、Inf、signed zero 分开判定；不能用普通误差阈值覆盖
所有特殊值。修复后同时保留结构与数值回归，防止未来因为优化不再命中而“意外通过”。

测试按 Dialect、Conversion、Frontend、E2E 分层，使语法/合法性错误不必等到 runner
才发现。FileCheck 只绑定语义关键结构，避免 SSA 编号和打印格式小变化造成脆弱测试。

## 8. 常见错误

- TableGen：改 `.td` 后 generated method/link symbol 不一致；先构建 IncGen target。
- `MLIR_DIR`/`LLVM_DIR`：来自不同 build/version，出现 undefined symbol 或 API mismatch。
- out-of-tree 链接：漏目标 library、重复静态注册或 RTTI 配置不同。
- Dialect 未注册：unknown `bgraph`/custom attribute。
- Pass 未注册：unknown command-line argument。
- failed to legalize：pattern 不匹配或 target legality 错。
- verifier failure：输入或前序 rewrite 生成局部非法 IR。
- 数值不一致：先定位 importer constants/attributes，再对比优化前后，再看 lowering；
  不要直接归因于浮点误差。

定位顺序可以压缩成一张表：

| 观察 | 先做什么 | 典型归层 |
|---|---|---|
| 不带任何 pass 也不能读入 | 看第一条 parser/unknown-dialect 诊断 | parse/注册 |
| 能 parse，但普通 `buddygraph-opt input` 失败 | 追 `emitOpError` 或 generated constraint | verifier |
| greedy pass 返回 0、目标 op 保留 | 打开 pattern debug/断点，检查前置条件 | 正常 match failure |
| conversion 报 `failed to legalize` | 同时看残留 illegal op 和 `--debug-only=dialect-conversion` | Pattern/target/FullConversion |
| conversion 生成了新的非法目标 op | 检查 target legality 与 rewrite 产物 | Pattern 或 target 配置错误 |

IR dump 说明“什么残留”，conversion debug trace 或 Pattern 断点说明“为什么没匹配”；
两类证据不能互相替代。

## 9. 动手练习

### 故障 1：错误 Yield

不要直接复制 `invalid.mlir` 的 fused section；它含 `arith.negf`，会先命中白名单
诊断。改用无干扰的最小输入：

```bash
cat >"$BUDDYGRAPH_TMP/bad-yield.mlir" <<'MLIR'
func.func @bad_yield(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "bgraph.fused_elementwise"(%arg0) ({
  ^bb0(%value: f32):
    %bad = arith.constant 0.0 : f64
    "bgraph.yield"(%bad) : (f64) -> ()
  }) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
MLIR

build/bin/buddygraph-opt "$BUDDYGRAPH_TMP/bad-yield.mlir" -o /dev/null
```

预期非零退出，parent verifier 先报告 body 必须以 yielding f32 结束。随后给错误行增加
精确 `expected-error`，用 `--verify-diagnostics` 比较“未声明 expected”与匹配后的行为。

### 故障 2：failed to legalize

使用第 10 章的 dynamic Add。加 before/after-all，确认 BGraph verifier 通过但
conversion 因 static lowering contract 失败。

### 故障 3：数值不一致

只复制 E2E Python 文件到临时目录，把 NumPy reference 中的 factor operand替成 3.0，
但继续使用原项目生成器与编译模型：

```bash
tmpdir=$(mktemp -d "$BUDDYGRAPH_TMP/buddygraph-e2e-fault.XXXXXX")
cp tests/E2E/BGraph/onnx_elementwise.py \
  "$tmpdir/onnx_elementwise_bad_reference.py"
sed -i 's/\* initializers\["factor"\]/\* 3.0/' \
  "$tmpdir/onnx_elementwise_bad_reference.py"

export PYTHONPATH="$PWD/.deps:/buddy-mlir/llvm/build/tools/mlir/python_packages/mlir_core${PYTHONPATH:+:$PYTHONPATH}"
python3 "$tmpdir/onnx_elementwise_bad_reference.py" \
  "$PWD" "$PWD/build" /buddy-mlir/llvm/build/bin "$tmpdir/run"
```

预期 reference 变成 10.5、runner 仍是 7.0，并以 tolerance failure 非零退出。这说明
问题是 reference/model mismatch，不是 BN/fusion bug；正式测试未被修改。

高级可选：在自己的临时分支中移除 driver 的 `BGraphDialect` 注册或
`registerPasses()`，分别观察 parse 与 CLI 失败。务必先保存基线版本或复制
文件，不使用 destructive Git 命令恢复。

## 10. 验收标准

- 14/14 全套测试通过。
- 三个故障都能复现并准确归层。
- 能解释 `%s/%S/%t/%PYTHON` 的来源。
- 能用 FileCheck 顺序规则解释一次检查失败，而非盲目放宽 pattern。

## 11. 面试追问

**问：如何定位 failed to legalize？**

答：先确认 source verifier 通过；打开 conversion 前后 IR 和 conversion debug trace，
找到残留 illegal op，再检查对应 pattern 的 shape/attribute match failure 和 target
legality，而不是把 source Dialect 改 legal。

**问：为什么还要 E2E，lit lowering test 不够吗？**

答：FileCheck 证明结构，不能证明 constants、公式、ABI 与整个 lowering 的数值语义；
E2E 用 NumPy 同时验证优化关闭和开启路径。

如果需要从 Passes.td、generated base、factory、registration 一直带做到 CLI/lit，见
[自定义 Dialect 专题第 06–07 章](dialect_lab/README.md)。
