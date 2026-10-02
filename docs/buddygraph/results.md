# Results

下面的测量命令记录 2026-07-28 采集时的旧目录。项目已迁至
`/root/projects/buddygraph`；新运行请使用项目根 README 中的路径。

## 完成范围

已完成固定 ONNX opset 18 → BGraph → Linalg/Tensor → bufferized loops → LLVM dialect
→ CPU runner 的静态 f32 闭环。`applyFullConversion` 保证转换后没有 BGraph op；
确定性 Conv-BN-elementwise 模型在优化关闭和全部优化两种配置下都返回
`7.000000e+00`，与 NumPy 的最大绝对/相对误差均为 0（runner 打印精度下）。
独立 Clamp E2E 还覆盖 ONNX Clip 的 direct/fused lowering：两条路径都返回
`5.000000e-01`、打印误差为 0，并保留 NaN；E2E 在 lowering 前还断言两个 optimized
fixtures 均已形成 fused op。该 Clamp fixture 不属于下文 2026-07-28
benchmark 数据集，因此不改变历史性能表。

## 测量方法

数据由以下命令在 2026-07-28 实际采集：

```bash
export BUDDYGRAPH_TMP=/buddy-mlir/jlq/projects/buddygraph/tmp
mkdir -p "$BUDDYGRAPH_TMP"
python3 scripts/benchmark.py \
  --compile-repetitions 7 \
  --runtime-warmup 3 \
  --runtime-repetitions 20 \
  -o "$BUDDYGRAPH_TMP/buddygraph-results.json"
```

环境为 WSL2 Linux 6.18.33.2 x86_64、Python 3.10.12、本地 MLIR
21.0.0git Release build。模型输入常量 shape 为 `[1,1,2,2]`。所有 compile-time
数字是独立进程 wall time；runtime 是每次启动独立 `mlir-runner` 的 wall time，
包含 MLIR 解析和 JIT 启动，因此不是纯 kernel latency。

## 编译期

单位为 ms；7 次采样。

| 项目 | median | p95 |
|---|---:|---:|
| ONNX importer | 549.17 | 618.24 |
| shape inference invocation | 13.39 | 20.92 |
| BN folding invocation | 10.55 | 15.99 |
| elementwise fusion invocation | 13.22 | 19.68 |
| full pipeline, optimizations off | 32.13 | 38.30 |
| full pipeline, canonicalize+CSE | 34.75 | 39.94 |
| full pipeline, BN folding | 31.82 | 49.67 |
| full pipeline, elementwise fusion | 26.27 | 40.26 |
| full pipeline, all | 33.27 | 45.17 |

Importer 时间主要包含 Python/ONNX 进程启动、模型检查和 shape inference；这些数据
不能解释为单个 C++ pass 的内部计时。

## IR 结构

| 指标 | 优化关闭 | 全部优化 |
|---|---:|---:|
| BGraph ops | 6 | 4 |
| `bgraph.fused_elementwise` | 0 | 1 |
| `linalg.generic` | 7 | 4 |
| named Linalg Conv | 1 | 1 |
| `memref.alloc` | 7 | 4 |
| 可静态计算的 alloc 总字节 | 88 | 40 |

全部优化后保留 Conv、FusedElementwise 和 ReduceMean 三个图级计算，以及 Region
中的 BGraph Yield terminator；BN 被折入 Conv，Add-Relu-Mul 被一个 scalar Region
表达。对应的 generic kernel 从 7 降为 4，静态
allocation 数从 7 降为 4。字节数只统计 bufferized IR 中具有全静态
`memref<...xf32>` result type 的 `memref.alloc`；rank-0 `memref<f32>` 按一个 f32
（4 bytes）计入。原统计漏掉两条 rank-0 alloc，已从 80/32 更正为 88/40；不含常量
存储、stack、allocator metadata 或 JIT runtime 内存。

## Runner 消融

单位为 ms；3 次 warmup 后 20 次独立进程采样。

| 配置 | median | p95 |
|---|---:|---:|
| optimizations off | 60.74 | 77.21 |
| canonicalize + CSE | 68.86 | 104.00 |
| BN folding | 67.16 | 103.52 |
| elementwise fusion | 58.12 | 77.91 |
| all | 51.85 | 76.38 |

该微型模型中 JIT/进程开销占主导，p95 也显示明显噪声。数据证明五种 pipeline 都
可重复执行并保持数值正确，也显示结构优化减少了 IR 和 allocation；它不足以声称
生产 workload 的 kernel speedup。可靠性能结论需要长期进程内 benchmark、更大
shape、CPU 固频和硬件计数器。

## 已知边界

- 只支持 opset 18、静态 f32、NCHW importer 和 groups=1 Conv。
- ONNX Clip 只支持同时提供的 finite scalar f32 initializer bounds；optional、运行时和
  非标量 bounds 不在当前闭环内。
- NHWC 可由方言 attribute 表达和部分 verifier 处理，但前端/Conv lowering 闭环只
  承诺 NCHW。
- runner E2E 使用 rank-0 output adapter；一般 tensor ABI 需要调用 wrapper。
- 没有 ONNX Runtime 对拍；NumPy reference 是当前验收依据。
- 未评测真实模型、动态 batch、GPU、量化或多线程 scaling。

因此可陈述“实现并验证有限 ONNX 子集的 MLIR 图方言、图优化、FullConversion 和
CPU 数值闭环”，不可陈述“支持完整 ONNX”或“在真实模型上获得确定加速”。
