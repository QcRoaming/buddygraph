# 13｜性能与 IR 效果：如何避免漂亮但错误的结论

## 1. 本章目标

你将复现实测脚本，区分 compile-time、独立 runner wall time 与 kernel latency，解释
op/alloc/bytes 指标，并对五种 ablation 给出不过度外推的结论。

## 2. 先运行

```bash
cd /buddy-mlir/jlq/projects/buddygraph
export BUDDYGRAPH_TMP=/buddy-mlir/jlq/projects/buddygraph/tmp
mkdir -p "$BUDDYGRAPH_TMP"
python3 scripts/benchmark.py \
  --compile-repetitions 7 \
  --runtime-warmup 3 \
  --runtime-repetitions 20 \
  -o "$BUDDYGRAPH_TMP/buddygraph-results.json"
```

每次机器负载不同会改变时间。应稳定检查的是：五种 profile 都返回 7.0、结构指标
定义一致，而不是要求时间逐位复现历史数字。

## 3. 真实代码位置

- `scripts/benchmark.py::PROFILES`：off、canonicalize/CSE、BN、fusion、all。
- `LOWERING`：与 E2E 相同的完整 pipeline。
- `sample()`、`summarize()`：wall time、median、nearest-rank p95。
- `static_alloc_bytes()`：只统计静态 `memref<...xf32>` alloc。
- `docs/buddygraph/results.md`：2026-07-28 实际结果和边界。
- `tests/E2E/BGraph/onnx_elementwise.py`：数值 correctness，不做 timing gate。

## 4. 调用链

```text
generate deterministic ONNX model
→ repeat importer process 7×
→ repeat each standalone pass invocation 7×
→ emit off/all BGraph/Linalg/buffer snapshots
→ regex count BGraph/generic/conv/alloc + static bytes
→ for each of 5 profiles:
   → compile full LLVM Dialect pipeline 7×
   → emit one textual LLVM Dialect snapshot
   → mlir-runner warmup 3×
   → independent mlir-runner process 20×
   → verify printed result begins 7.0
   → median / p95
→ JSON
```

## 5. IR 前后变化

已记录的结构结果：

| 指标 | off | all |
|---|---:|---:|
| BGraph ops | 6 | 4 |
| fused op | 0 | 1 |
| `linalg.generic` | 7 | 4 |
| named Conv | 1 | 1 |
| `memref.alloc` | 7 | 4 |
| 静态 alloc bytes | 88 | 40 |

all 中的 4 个 BGraph ops 是 Conv、FusedElementwise、Region 内 Yield、ReduceMean。
BN folding 消除 BN；fusion 把 Add/Relu/Mul 变成 fused+yield。`linalg.generic` 从 7
到 4，说明结构化 computations 减少，但不直接等于机器 loop 或 runtime 比例。

## 6. 核心机制

历史实测（ms）：

| profile | compile median/p95 | runner median/p95 |
|---|---:|---:|
| off | 32.13 / 38.30 | 60.74 / 77.21 |
| canonicalize+CSE | 34.75 / 39.94 | 68.86 / 104.00 |
| BN folding | 31.82 / 49.67 | 67.16 / 103.52 |
| fusion | 26.27 / 40.26 | 58.12 / 77.91 |
| all | 33.27 / 45.17 | 51.85 / 76.38 |

Importer median/p95 为 549.17/618.24 ms；三个单 pass invocation median 分别约
13.39、10.55、13.22 ms。它们都是**独立进程** wall time，包括进程启动和解析。

runner 每次也启动新进程、parse MLIR、JIT，再执行 4-element workload。JIT/启动
远大于 kernel，因此 51.85 vs 60.74 不能用作“kernel 加速 14.6%”结论。p95 噪声也
说明样本对系统调度敏感。3 次所谓 warmup 只能预热 OS/文件缓存等外部状态，不能在
后续独立 runner 进程中复用同一个 JIT 实例。

`static_alloc_bytes()` 解析静态 f32 memref alloc result type，rank-0 `memref<f32>`
按一个 f32（4 bytes）计入；不含 constants、
allocator metadata、alignment、JIT code、stack 或 runtime allocations，也不证明
峰值 resident memory。

## 7. 为什么这样设计

把数值 correctness 与 timing 分开，避免抖动导致测试不稳定；用 ablation 隔离各
Pass 的结构影响；同时记录 median/p95，避免只报最好一次。小 fixture 的主要价值是
验证测量方法和 IR 效果，不是代表真实网络性能。

## 8. 常见错误

- 没 warmup，只报第一次 JIT。
- 报最小值或平均值，不给分布/重复次数。
- 把独立进程 wall time 称为 kernel latency。
- op count 减少就宣称等比例加速。
- alloc 数减少就宣称峰值内存按比例下降。
- 在不同模型/shape/commit 上比较结果却不记录环境。
- timing 作为 lit pass/fail，导致 flaky tests。

## 9. 动手练习

运行两次 benchmark，把 JSON 中 environment、IR metrics、runtime median/p95 放入
对照表。回答：哪些事实应完全一致？哪些允许抖动？若第二次 all 比 off 慢，是否
推翻 fusion 的结构效果？

扩展设计：提出一个“进程内 kernel benchmark”方案，至少说明 JIT 缓存、输入 shape、
迭代次数、CPU affinity/frequency、warmup、计时范围和数值校验。

## 10. 验收标准

- benchmark 完成且五种 profile 数值都为 7.0。
- 能解释 BGraph count 4 为什么包含 Yield。
- 能准确复述 static bytes 的统计范围。
- 性能结论不超出小模型、独立进程、WSL2、本地快照。

## 11. 面试追问

**问：你能说项目获得了加速吗？**

答：只能说在该独立进程/JIT 微型 fixture 上 all profile 的历史 median 较低，且 IR
generic/alloc 减少；这不足以证明生产 kernel speedup，需要进程内稳定 benchmark 和
真实 shapes。

**问：为什么还报告 op/alloc？**

答：它们是机制证据：证明 fusion/BN folding 真正改变结构并减少中间 destinations；
它们帮助解释性能假设，但不是运行时间的替代指标。
