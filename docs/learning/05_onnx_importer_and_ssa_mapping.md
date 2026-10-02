# 05｜ONNX importer：从字符串名字到 MLIR SSA

> **本章路线：从名字到数据边。** 先用不同的 node name/output name 跟踪一张图，再解释 constants、attributes 和 types 的来源，最后检查拒绝边界。第 06 章继续追同一个 Add 的 shape。

## 1. 本章目标

你将能从一个 ONNX Node 追到具体 BGraph Operation，解释 graph input、initializer
和 node output 如何进入同一 `name → Value` 映射，并验证 location 和拒绝诊断。

## 2. 先运行

```bash
cd /home/jlq/project/buddygraph
export BUDDYGRAPH_TMP=/home/jlq/project/buddygraph/tmp
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
node.name="conv_node_17", op_type="Conv"
inputs=["x", "w", "conv_bias"], outputs=["features"]
```

MLIR 中变成 def-use 连接：

```mlir
%w = arith.constant ... loc("initializer:w")
%bias = arith.constant ... loc("initializer:conv_bias")
%0 = "bgraph.conv2d"(%arg0, %w, %bias) {...}
    : (...) -> tensor<1x4x5x5xf32> loc("conv_node_17")
```

`self.values["x"]` 指向 entry `BlockArgument`，`self.values["w"]` 指向 constant
result，`self.values["features"]` 指向 Conv result。后续 BN 输入名 `features` 查到的
就是同一 SSA Value。`conv_node_17` 是节点标签，进入 Location；`%0` 是打印名称，
不作为该 Python map 的键。此处故意用三个不同名字，以免把三种身份混在一起。

Reshape shape 和 ReduceMean axes 是 integer initializer，但不会变成 runtime tensor
operand；`_node_attribute_initializers()` 识别当前 node 的元数据输入，
`_constant_ints()` 读取整数，分支通过 `_i64_array()` 转成 BGraph DenseI64ArrayAttr。
Clip 的 min/max 则由 `_constant_scalar_f32()` 检查并转为 FloatAttr。

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

### 手工执行一次 importer：两个表，不是一张字典

假设图为 `x,bias → Add → sum → Relu → activated`。令 x 的 shape 为 `[2,1]`，
bias 为 `[1,3]`，结果广播为 `[2,3]`；第 06 章继续用这组 shape。
按 `import_module()` 的顺序填写：

| 时刻 | `self.values` 增加什么 | 类型从哪里来 |
|---|---|---|
| 建立 entry block | `x → %arg0` | graph input 的 TensorInfo |
| 导入 initializer | `bias → constant.result` | initializer dtype/dims 与 DenseElementsAttr |
| 创建 Add | `sum → add.result` | ONNX inference 提供 `sum: [2,3]` |
| 创建 Relu | `activated → relu.result` | ONNX inference 提供 `activated: [2,3]` |
| 创建 return | 无新名字，查 `activated` | 函数输出类型约束 |

创建 Add 时 operands 用 `_value("x")` 和 `_value("bias")`，result type 用
`_tensor_type("sum")`。前者连接 IR 中的定义，后者查询编译期 shape；Python 没有在这里
执行一次 Add 去测量输出大小。若把 Relu 放在 Add 前，读取 sum 时失败，这是拓扑顺序
契约，不是 SSA printer 编号出了问题。

### 元数据的过滤必须按节点发生

同一个 scalar initializer `limit` 可以在 Clip 中作为 bound 属性，又在 Add 中作为
广播 operand。`_attribute_initializers()` 汇总分类，帮助检查允许的 initializer 类型；
`_import_node()` 组装 operands 时使用当前 node 的 `_node_attribute_initializers()`。
若把汇总集合用于所有 node 的 operands 过滤，Add 会丢失输入。

float initializer 仍可能先被导入成 `arith.constant`；某个 node 不把它用作 operand，
不等于整个 module 绝不出现那条 constant。无用常量之后可由清理 pass 删除。

### 导入器写完以后，验证链还没结束

依次检查 ONNX 协议/shape、generic MLIR 结构、C++ BGraph 语义、后端支持条件。
`allow_unregistered_dialects` 只影响创建/解析能力，不替你确认 channel、broadcast、
bound 或 Region 语义。第 06 章用同一个 Add 展示这几层 shape 判断为何不互相替代。

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
