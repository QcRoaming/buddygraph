# 05｜ONNX importer：从字符串名字到 MLIR SSA

## 1. 本章目标

你将能从一个 ONNX Node 追到具体 BGraph Operation，解释 graph input、initializer
和 node output 如何进入同一 `name → Value` 映射，并验证 location 和拒绝诊断。

## 2. 先运行

```bash
cd /buddy-mlir/jlq/projects/buddygraph
export BUDDYGRAPH_TMP=/buddy-mlir/jlq/projects/buddygraph/tmp
mkdir -p "$BUDDYGRAPH_TMP/buddygraph-importer"
export PYTHONPATH="$PWD/.deps:/buddy-mlir/llvm/build/tools/mlir/python_packages/mlir_core${PYTHONPATH:+:$PYTHONPATH}"

python3 frontend/BGraph/generate_test_models.py "$BUDDYGRAPH_TMP/buddygraph-importer/models"
python3 frontend/BGraph/import_onnx.py \
  "$BUDDYGRAPH_TMP/buddygraph-importer/models/conv_bn_relu.onnx" \
  -o "$BUDDYGRAPH_TMP/buddygraph-importer/conv.mlir"
build/bin/buddygraph-opt -mlir-print-debuginfo \
  "$BUDDYGRAPH_TMP/buddygraph-importer/conv.mlir" | \
  rg 'bgraph\.|initializer:w|loc\("conv"\)|loc\("batch_norm"\)'
```

再观察拒绝路径：

```bash
python3 frontend/BGraph/import_onnx.py \
  "$BUDDYGRAPH_TMP/buddygraph-importer/models/wrong_opset.onnx"
```

预期非零退出，并报告 expected opset 18、found 17。

另外两类拒绝路径可直接运行：

```bash
python3 frontend/BGraph/import_onnx.py \
  "$BUDDYGRAPH_TMP/buddygraph-importer/models/unsupported_operator.onnx"
test $? -ne 0

python3 frontend/BGraph/import_onnx.py \
  "$BUDDYGRAPH_TMP/buddygraph-importer/models/unsupported_attribute.onnx"
test $? -ne 0
```

前者应定位 `unsupported_softmax`/`Softmax`，后者应定位
`auto_pad_conv`/`auto_pad='SAME_UPPER'`。若 shell 启用了 `set -e`，请逐条执行并直接
观察非零退出，或用第 12 章的 lit frontend test 一次验证三类诊断。

## 3. 真实代码位置

- `SUPPORTED_OPSET = 18`、`SUPPORTED_OPS`。
- `_validate_model()`：opset/domain、ONNX checker、shape inference。
- `BGraphImporter.__init__()`：initializer table、tensor info、Context、value map。
- `_collect_tensor_info()`：f32/static shape 边界。
- `_import_initializer()`：DenseElementsAttr + `arith.constant`。
- `_import_node()`：ONNX op dispatch 与 attributes。
- `_create_bgraph()`：`ir.Operation.create()`。
- `import_module()`：`func.FuncOp`、entry block、return。
- `tests/Frontend/BGraph/importer.mlir`：正向映射、locations、三类错误。

## 4. 调用链

```text
main(argv)
→ onnx.load()
→ _validate_model()
   → checker.check_model()
   → shape_inference.infer_shapes(strict_mode=True, data_prop=True)
→ BGraphImporter(...)
   → _collect_tensor_info()
→ import_module()
   → func.FuncOp + add_entry_block()
   → graph inputs: name ↦ BlockArgument
   → float initializers: name ↦ arith.constant result Value
   → for node in graph.node: _import_node()
      → input names lookup Value
      → Operation.create("bgraph.*")
      → output name ↦ OpResult
   → func.ReturnOp
→ module.verify()
→ get_asm(enable_debug_info=True)
```

## 5. IR 前后变化

ONNX 里的名字是字符串：

```text
node Conv inputs=["x", "w", "conv_bias"] outputs=["conv"]
```

MLIR 中变成 def-use 连接：

```mlir
%w = arith.constant ... loc("initializer:w")
%bias = arith.constant ... loc("initializer:conv_bias")
%conv = "bgraph.conv2d"(%arg0, %w, %bias) {...}
    : (...) -> tensor<1x4x5x5xf32> loc("conv")
```

`self.values["x"]` 指向 entry `BlockArgument`，`self.values["w"]` 指向 constant
result，`self.values["conv"]` 指向 Conv result。后续 BN 输入名 `conv` 查到的就是同一
SSA Value。

Reshape shape 和 ReduceMean axes 是 integer initializer，但不会变成 runtime tensor
operand；`_shape_initializers()` 把它们转成 BGraph DenseI64ArrayAttr。

## 6. 核心机制

`MLIRContext` 允许 unregistered dialect，是因为 Python package 没有 BGraph 生成
bindings；importer 仍用 MLIR API 创建 generic `bgraph.*` operation，而不是拼整段
文本。Python-side `module.verify()` 能检查已注册标准 op 和通用 IR 结构，但不会
执行 C++ BGraph custom verifier；输出必须再交给 `buddygraph-opt`。

shape 的来源不是 importer 自己手算每个中间值，而是 ONNX shape inference 结果。
`_tensor_type(name)` 将固定 shape 与 f32 转成 `RankedTensorType`。

Location 使用 node name：初始化常量是 `initializer:<name>`，node 是
`_node_label()`。序列化时必须 `enable_debug_info=True`，否则 location 会被打印器
省略。

## 7. 为什么这样设计

ONNX protobuf/opset 兼容由成熟 `onnx` package 负责；BGraph importer 只负责受支持
子集到 MLIR 的语义桥接。frontend 与 lowering 分开后，不支持的 ONNX 属性在源层
报错，高层优化也不需要理解 protobuf。

## 8. 常见错误

- graph 未拓扑排序：input name 尚未在 `self.values` 中，报 unavailable。
- initializer 是 int 但既不是 shape 也不是 axes：MVP 拒绝。
- 把 Reshape shape 当 runtime operand，生成与 ODS schema 不符的 op。
- 假设 `allow_unregistered_dialects` 等于“BGraph 已验证”；它只允许创建/打印未知
  dialect。
- 忘记排除 initializers from graph inputs，函数签名多出参数。
- 用 `str(module)` 序列化，source locations 丢失。

## 9. 动手练习

阅读题：选择 `elementwise_chain.onnx` 的 Reshape，列出 shape initializer 从
`numpy_helper.to_array()` 到 `DenseI64ArrayAttr` 的完整函数链，并解释为什么它没有
对应 SSA operand。

调试题：运行 unsupported Softmax 和 `auto_pad=SAME_UPPER` fixtures，比较两个错误
分别由哪个检查产生。不要修改 importer。

## 10. 验收标准

- 能列出 value map 的三类 definition 来源。
- 能从 `loc("batch_norm")` 回到 ONNX node name。
- 正向模型经过 `buddygraph-opt -o /dev/null` 验证。
- 三种 negative frontend commands 都非零退出且含具体 node/opset/attribute。

## 11. 面试追问

**问：为什么不直接拼 MLIR 字符串？**

答：API 构造能复用 MLIR Type/Attribute/Region/Location 对象和结构验证，避免 escaping、
SSA 名冲突和文本格式漂移；输出文本只是序列化结果。

**问：ONNX name 与 MLIR `%0` 有什么关系？**

答：没有固定文本对应。name map 保存实际 `Value` handle，打印器可把它编号为任意
SSA name；语义来自 def-use identity，不来自打印名称。
