# Testing

## 一键回归

```bash
cmake --build build --target check-buddygraph -j2
```

lit 配置复用本地 LLVM 的 `llvm-lit`、FileCheck 和工具目录，并把 `.deps` 与 MLIR
Python bindings 加入测试环境。当前覆盖：

- dialect parse/print round trip；
- verifier 对 dtype、broadcast、Conv channel/空间几何、BN channel、groups、Clamp
  bounds/shape、permutation、axes、reshape 和 fused Region 的负面诊断；
- shape refinement 的 broadcast、Clamp、reshape、transpose、reduce、conv；
- canonicalization、BN folding、fusion 的正向和“必须不改写”条件，包括 Conv/BN
  layout 不一致、f32 epsilon 边界、nested Clamp 和 broadcast identity；
- BN folding 与 fusion 的 fused location 保留源 op locations；
- FullConversion 后 `CHECK-NOT: bgraph.`；
- One-Shot Bufferize 后没有 BGraph 和 bufferization bridge op；
- Conv-BN-ReLU、elementwise ONNX 正向导入；
- ONNX Clip constant-bound 正向导入、initializer 被普通数据/属性复用，以及运行时、
  缺失、非标量、非有限和逆序 bounds 的拒绝；
- 未知 op、错误 opset、`auto_pad` 的前端拒绝；
- 同一常量 ONNX 模型的未优化/全优化 CPU 结果与 NumPy 对拍。
- signed-zero 的优化前后结果与 canonicalization 回归。
- Clamp 的 direct/fused lowering 数值与 NumPy 对拍，并在 lowering 前断言有限值和 NaN
  fixtures 都已形成 fused op，再验证 NaN 传播。

## 数值 E2E

`tests/E2E/BGraph/onnx_elementwise.py` 每次生成固定模型，不依赖随机种子。模型为
常量 Conv→BN→Add→Relu→Mul→ReduceMean，输出 rank-0 tensor；参考值由 NumPy
按同一参数计算。测试分别运行未启用自定义优化和启用 BN folding、
canonicalize/CSE、elementwise fusion 的 pipeline，要求绝对或相对误差不超过
`1e-6`。两条路径都继续完成 bufferization、LLVM conversion 和 runner 执行。

单独运行：

```bash
/usr/bin/python3.10 /buddy-mlir/llvm/build/bin/llvm-lit -sv \
  build/tests/E2E/BGraph
```

## 如何添加测试

- 新 verifier 约束：在 `invalid.mlir` 添加 `--split-input-file` section 和
  `expected-error`。
- 新 rewrite：同一文件中同时给出应匹配和不应匹配的最小函数。
- 新 lowering：在 `full-conversion.mlir` 检查目标结构，并始终检查无 BGraph。
- 新 ONNX 能力：生成器必须增加固定 fixture、正向映射检查及至少一个边界拒绝。
- 改动数值语义：更新 NumPy reference，并同时保留优化前/后的对拍。

测试验证正确性，不把一次性能采样作为 pass/fail 条件。性能数据由独立 benchmark
脚本采集。
