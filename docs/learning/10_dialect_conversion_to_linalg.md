# 10｜Dialect Conversion：用 legality 保证消除 BGraph

## 1. 本章目标

你将能解释 RewritePattern 与 ConversionPattern 的差异，逐步跟踪一个 binary op 和
一个带 Region op 的 lowering，并制造一次可解释的 `failed to legalize`。

## 2. 先运行

```bash
cd /buddy-mlir/jlq/projects/buddygraph
export BUDDYGRAPH_TMP=/buddy-mlir/jlq/projects/buddygraph/tmp
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
