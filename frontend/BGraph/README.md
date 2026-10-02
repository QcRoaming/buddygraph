# ONNX frontend

`import_onnx.py` 用 `onnx` 解析并检查模型，调用 ONNX shape inference，然后
用 MLIR Python API 的 `Operation.create`、`func.FuncOp` 和 builtin attributes
逐个构造操作。它不会拼接一整段 MLIR 字符串。ONNX node name 被保存在 MLIR
location 中，便于从 verifier 或 conversion 错误追溯源节点。

## 映射

| ONNX | BGraph | 关键限制 |
|---|---|---|
| Conv | `bgraph.conv2d` | 2D、NCHW、显式 pads、group=1 |
| BatchNormalization | `bgraph.batch_norm` | inference mode、单输出 |
| Relu | `bgraph.relu` | f32 tensor |
| Clip | `bgraph.clamp` | min/max 必须是 finite scalar f32 initializer |
| Add/Sub/Mul/Div | 同名 BGraph op | NumPy/ONNX 右对齐广播 |
| Reshape | `bgraph.reshape` | shape 必须是 initializer，导入时解析 0/-1 |
| Transpose | `bgraph.transpose` | 常量 permutation |
| ReduceMean | `bgraph.reduce_mean` | opset-18 axes input 必须是 initializer |

导入器只接受一个默认-domain opset 18 声明。非 f32、symbolic/dynamic shape、
未知 domain、未知 op、Clip 的缺失/运行时/非标量/非有限/逆序 bounds 和 MVP 外属性会在生成
MLIR 前失败，并包含 node name。Clip bounds 只对当前 Clip 作为 attributes 读取；同一
float initializer 仍可被其他节点当普通 tensor operand 复用。

```bash
cd /home/jlq/project/buddygraph
export BUDDYGRAPH_TMP=/home/jlq/project/buddygraph/tmp
mkdir -p "$BUDDYGRAPH_TMP"
python3 frontend/BGraph/generate_test_models.py "$BUDDYGRAPH_TMP/models"
python3 frontend/BGraph/import_onnx.py \
  "$BUDDYGRAPH_TMP/models/elementwise_chain.onnx" \
  -o "$BUDDYGRAPH_TMP/model.mlir"
```

`generate_test_models.py` 生成正向、错误 opset、未知算子和不支持属性等固定
fixtures；生成文件不进入版本库。
