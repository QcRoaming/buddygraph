# 10｜Dialect Conversion：用 legality 保证消除 BGraph

> **本章路线：语义降低与 legality。** 先手算一个广播 generic 的 maps/body，再读 pattern adaptor 与替换，最后检查 ConversionTarget。重点分开目标 IR 合法与计算数值正确。

## 1. 本章目标

你将能解释 RewritePattern 与 ConversionPattern 的差异，逐步跟踪一个 binary op 和
一个带 Region op 的 lowering，并制造一次可解释的 `failed to legalize`。

## 2. 先运行

```bash
cd /home/jlq/project/buddygraph
export BUDDYGRAPH_TMP=/home/jlq/project/buddygraph/tmp
mkdir -p "$BUDDYGRAPH_TMP"
build/bin/buddygraph-opt --convert-bgraph-to-linalg \
  tests/Conversion/BGraphToLinalg/binary-ops.mlir \
  -o "$BUDDYGRAPH_TMP/binary-linalg.mlir"

build/bin/buddygraph-opt \
  --bgraph-fuse-elementwise --convert-bgraph-to-linalg \
  tests/Dialect/BGraph/fuse-elementwise.mlir \
  -o "$BUDDYGRAPH_TMP/fused-linalg.mlir"

if rg -n 'bgraph\.' "$BUDDYGRAPH_TMP/binary-linalg.mlir" \
    "$BUDDYGRAPH_TMP/fused-linalg.mlir"; then
  echo 'ERROR: FullConversion left a BGraph op' >&2
  exit 1
fi
rg 'linalg.generic|arith.subf|arith.divf|arith.maximumf' \
  "$BUDDYGRAPH_TMP/binary-linalg.mlir" \
  "$BUDDYGRAPH_TMP/fused-linalg.mlir"
```

条件检查应无输出且不退出；`@fuse` 对应一个 `linalg.generic`。还可单独运行 conversion
回归：

```bash
/usr/bin/python3.10 /buddy-mlir/llvm/build/bin/llvm-lit -sv \
  build/tests/Conversion/BGraphToLinalg/full-conversion.mlir
```

## 3. 真实代码位置

- `BGraphToLinalg.cpp::broadcastMap()`、`createElementwiseGeneric()`。
- `BinaryLowering<OpTy, ScalarOp>`、`ReluLowering`。
- `FusedElementwiseLowering`。
- `ReshapeLowering`、`TransposeLowering`、`ReduceMeanLowering`。
- `BatchNormLowering`、`Conv2DLowering`、`applyPadding()`。
- `ConvertBGraphToLinalg::runOnOperation()`。
- `tests/Conversion/BGraphToLinalg/full-conversion.mlir`、`binary-ops.mlir`、
  `dynamic-conv.mlir`。

## 4. 调用链

```text
--convert-bgraph-to-linalg
→ ConvertBGraphToLinalg::runOnOperation()
→ ConversionTarget(getContext())
→ addLegalDialect<arith, func, linalg, math, tensor>()
→ addLegalOp<ModuleOp>()
→ addIllegalDialect<BGraphDialect>()
→ RewritePatternSet.add<12 lowering pattern specializations>()
→ applyFullConversion(ModuleOp, target, patterns)
→ Conversion driver 遍历非法 op
→ OpConversionPattern::matchAndRewrite(op, adaptor, rewriter)
→ replaceOp / replaceOpWithNewOp
→ 若仍有非法 op：pass failure
```

## 5. IR 前后变化

### Binary op

```mlir
// source
%0 = "bgraph.sub"(%lhs, %rhs)
    : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
```

`BinaryLowering<SubOp, arith::SubFOp>` 创建 `tensor.empty`、broadcast indexing maps 和
parallel iterators：

```mlir
%empty = tensor.empty() : tensor<2x3xf32>
%0 = linalg.generic
    {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,
                      affine_map<(d0,d1)->(d1)>,
                      affine_map<(d0,d1)->(d0,d1)>],
     iterator_types = ["parallel", "parallel"]}
    ins(%lhs, %rhs : tensor<2x3xf32>, tensor<3xf32>)
    outs(%empty : tensor<2x3xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %s = arith.subf %a, %b : f32
    linalg.yield %s : f32
  } -> tensor<2x3xf32>
```

### Fused Region

`FusedElementwiseLowering` 先把 source block arguments 映射到 Linalg body arguments，
再 `builder.clone()` scalar arith ops；source `bgraph.yield` 不被克隆，映射后的值交给
`linalg.yield`。

### 其他 Op

- Conv → `linalg.conv_2d_nchw_fchw` 或 NHWC 分支，padding 用 `tensor.pad`，bias 用
  generic init。
- BN → 一个 `linalg.generic`，body 有 sub/add/sqrt/div/mul/add。
- ReduceMean → reduction generic + elementwise divide generic。
- Transpose → `linalg.transpose`。
- Reshape → `tensor.reshape`，shape 由 i64 constant tensor 提供。

## 6. 核心机制

普通 `RewritePattern` 以“找到局部等价替换”为目标，不关心整个 module 最后是否合法。
`OpConversionPattern` 参与 conversion driver，使用 `OpAdaptor` 接收转换后的 operands，
并受 `ConversionTarget` legality 约束。

当前实现没有使用：

- `TypeConverter`：BGraph 和目标都复用 builtin tensor/f32，类型无需改变。
- signature conversion：`func.func` signature 保持 tensor types。
- materialization：没有 source/target type 桥接，所以不需要 cast materialization。
- dynamic legality：所有 BGraph 整体 illegal，标准目标 Dialect 整体 legal。

这些机制仍应理解：若 Clamp 改变类型、函数 ABI 或某些 source op 需按属性条件暂时
合法，就需要 TypeConverter、signature conversion、materialization 或 dynamic
legality。不能把“当前不需要”写成“Dialect Conversion 没有这些机制”。

`applyPartialConversion` 只要求被判 illegal 的 op 消失，可容许 unknown/dynamically
legal source；本项目选择 `applyFullConversion` 加整个 BGraph illegal，强制所有
BGraph op 消除。

这里“没有 BGraph”是 FullConversion 成功的必要条件，不代表任意输入都能转换。
target 只显式允许 Arith、Func、Linalg、Math、Tensor 和 Module；其他未声明合法的
Dialect/Op 仍会导致 conversion 失败。14 个回归中的 pipeline RUN 同时检查 pass
顺序可解析和最终无 BGraph。

### 从数组公式推到 indexing maps

不要先背 generic builder 参数。先写 `Y[i,j] = L[i,j] - R[j]`，其中 L 为 `[2,3]`，
R 为 `[3]`。输出坐标 `(i,j)` 是迭代域；每个输入的 map 回答“本次迭代读取哪里”：

| 对象 | shape | map | 理由 |
|---|---|---|---|
| lhs | `[2,3]` | `(i,j)→(i,j)` | 与输出同形 |
| rhs | `[3]` | `(i,j)→(j)` | 缺少前导维，沿 i 广播 |
| output | `[2,3]` | `(i,j)→(i,j)` | 每次写一个输出位置 |

若 rhs 改成 `[1,3]`，map 必须是 `(i,j)→(0,j)`。若误用 `(i,j)`，i=1 时会访问
size-1 维之外；“输出 shape 正确”不能证明索引正确。两维都不累加到相同输出位置，
所以是 parallel。body 的三个参数分别是 lhs 元素、rhs 元素、destination 旧元素，
本例不用第三个参数，直接 yield 减法结果。

### 为什么 ReduceMean 要初始化，而 Add 不读 destination

对 `[2,3,4]` 沿 `[0,2]` 求均值，结果 `[3]`。输入 map 是 `(i,j,k)→(i,j,k)`，
输出 map 是 `(i,j,k)→(j)`；i、k 是 reduction，j 是 parallel。不同 i/k 贡献到同一个
输出元素，body 要读取累计值，必须先用零初始化。项目再用第二个 generic 除以 8。
这解释了 `tensor.empty`、fill、reduction body、除数之间的依赖，而不是只有两个 Op 名。

### legality 与类型转换是两个独立设计

当前 target 只标记标准目标 Dialect 和 Module 合法。`applyFullConversion` 检查这项
策略；没有自动证明每条算式、自动消除每种 custom Type。若今后让 custom Type
穿过合法 `func.func`，需要相应类型与函数签名策略，不能只设置 `addIllegalDialect`。

Partial 与 Full 都必须消除显式 illegal Op。当前 BGraph 已整体 illegal，因此改成
Partial 也不允许残留 BGraph；Full 的额外要求是其余 Op 也都被判为合法。

### 一个“没有 BGraph 也失败”的反例

保存以下完整输入到项目 tmp，使用本章同一个 `--convert-bgraph-to-linalg`：

```mlir
func.func @foreign(%condition: i1) {
  scf.if %condition {
    scf.yield
  }
  return
}
```

driver 认识 SCF，输入结构可验证，但当前 conversion target 没将 SCF 判为合法，
也没有在本 pass 注册 SCF lowering pattern，因此 FullConversion 失败。
这与 dynamic Add 反例一起区分了“支持输入 schema”“支持 lowering”“满足最终 target”。
`rg bgraph` 无命中只是一条辅助结构证据，退出码与 pass 诊断同样要看。

## 7. 为什么这样设计

FullConversion 把“没有 BGraph”从 FileCheck 样例提升为 pass 的正确性条件。后续
bufferization 不需要 BGraph external interface，也不会默默遇到未降低的高层 op。

## 8. 常见错误

- 把 BGraph 标为 legal 让 pass“成功”，实则把问题推给后端。
- pattern 忘记处理 static shape 或 attribute，返回 failure 后出现 failed to legalize。
- broadcast map 误把 size-1 维映射到 loop dim，而不是 affine constant 0。
- fused lowerer clone 了 `bgraph.yield`，目标 Region 残留 illegal op。
- 声称使用 TypeConverter；代码中没有。
- 把 Linalg IR 当 LLVM IR。

可复现 failed-to-legalize：把下面保存到
`$BUDDYGRAPH_TMP/dynamic-add.mlir`，它能通过 BGraph
verifier，但 elementwise lowering 要求 static shape：

```mlir
func.func @dynamic(%a: tensor<?x3xf32>, %b: tensor<3xf32>)
    -> tensor<?x3xf32> {
  %0 = "bgraph.add"(%a, %b)
      : (tensor<?x3xf32>, tensor<3xf32>) -> tensor<?x3xf32>
  return %0 : tensor<?x3xf32>
}
```

```bash
build/bin/buddygraph-opt --convert-bgraph-to-linalg \
  "$BUDDYGRAPH_TMP/dynamic-add.mlir"
```

## 9. 动手练习

阅读题：从 `ReluLowering` 追到 `createElementwiseGeneric()`，列出 `OpAdaptor`、result
type、empty destination、indexing maps、body builder、replaceOp 六个阶段。

调试题：运行 dynamic case，并加
`--mlir-print-ir-before-all --mlir-print-ir-after-all`，说明失败时哪个非法 op 还在。

## 10. 验收标准

- FullConversion tests 通过且输出无 `bgraph.`。
- 能对比 PatternRewriter 和 ConversionPattern 的终止条件。
- 能解释当前四类 conversion mechanism 为什么未用。
- 能复现并定位 dynamic Add 的 failed to legalize。

## 11. 面试追问

**问：为什么 FullConversion 很重要？**

答：它让 BGraph 的完全消除成为 legality contract；漏 pattern 时 pass 失败，后端不会
收到不理解的 source op。

**问：什么时候必须用 TypeConverter？**

答：当 source/target type system 或函数 signature 改变时，例如自定义 tensor type
转 builtin tensor/memref，需要类型规则、signature conversion 和边界 materialization。
