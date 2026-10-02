# BuddyGraph

BuddyGraph 是一个独立构建的 MLIR 学习项目：它把固定为 opset 18 的有限
ONNX 子集导入 `bgraph` 方言，执行 shape refinement、BatchNorm folding 和
elementwise fusion，再通过 Full Dialect Conversion 降到 Linalg、bufferize、
LLVM dialect，并用 `mlir-runner` 在 CPU 上执行。

项目有自己的 CMake 构建、工具、测试和 Git 仓库。它复用外部已构建的
LLVM/MLIR，不修改 Buddy-MLIR 根工程中的方言、`buddy-opt` 或 Microkernel
研究改动。原目录选址的历史依据见
[Phase 0 audit](docs/buddygraph/audit.md)。

本仓库的 WSL 工作目录约定为 `/home/jlq/project/buddygraph`。若克隆到其他
位置，请相应调整教程中的绝对路径。

## 支持范围

- ONNX：只接受默认 domain 的 opset 18。
- 数据：ranked `tensor<...xf32>`；前端闭环要求静态 shape。
- 布局：导入器使用 NCHW；方言以 `#bgraph.layout<nchw|nhwc>` 表达布局。
- 算子：Conv、BatchNormalization、Relu、Add/Sub/Mul/Div、Reshape、
  Transpose、ReduceMean，以及同时提供 finite scalar f32 initializer bounds 的
  ONNX Clip/BGraph Clamp；不支持 optional、runtime 或 Constant-node bounds。
- 优化：shape inference、Conv-BN folding、canonicalization、CSE、
  region-based elementwise fusion。
- 后端：Linalg/Tensor/Arith/Math → One-Shot Bufferize → loops/LLVM → CPU。

这不是完整 ONNX 编译器。当前明确不支持训练态 BatchNormalization、动态或
非 f32 前端输入、分组卷积、`auto_pad`、多个 opset、量化、GPU 和任意 ONNX
控制流。

## 构建

以下命令复用已有的 LLVM/MLIR build，只编译 BuddyGraph。先把 `LLVM_BUILD`
设为本机现有 LLVM build 的路径；当前容器中的值是 `/buddy-mlir/llvm/build`。
Python 依赖和 BuddyGraph 构建产物放在源码目录外；`build`、`.deps` 是指向
外部缓存的便捷链接：

```bash
cd /home/jlq/project/buddygraph
LLVM_BUILD=/path/to/existing/llvm/build
BUDDYGRAPH_CACHE="${BUDDYGRAPH_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/buddygraph}"
mkdir -p "$BUDDYGRAPH_CACHE/deps"
test -e .deps || ln -s "$BUDDYGRAPH_CACHE/deps" .deps
python3 -m pip install --target "$BUDDYGRAPH_CACHE/deps" -r requirements.txt
cmake -S . -B "$BUDDYGRAPH_CACHE/build" -G Ninja \
  -DMLIR_DIR="$LLVM_BUILD/lib/cmake/mlir" \
  -DLLVM_DIR="$LLVM_BUILD/lib/cmake/llvm" \
  -DPython3_EXECUTABLE="$(command -v python3)" \
  -DCMAKE_BUILD_TYPE=Release
test -e build || ln -s "$BUDDYGRAPH_CACHE/build" build
cmake --build "$BUDDYGRAPH_CACHE/build" --target check-buddygraph -j2
```

Python 工具运行时需要同时找到项目依赖和 MLIR bindings：

```bash
export PYTHONPATH="$PWD/.deps:$LLVM_BUILD/tools/mlir/python_packages/mlir_core${PYTHONPATH:+:$PYTHONPATH}"
```

## 生成、导入与验证

生成稳定的测试模型并导入：

```bash
export BUDDYGRAPH_TMP="$PWD/tmp"
mkdir -p "$BUDDYGRAPH_TMP"
python3 frontend/BGraph/generate_test_models.py "$BUDDYGRAPH_TMP/buddygraph-models"
python3 frontend/BGraph/import_onnx.py \
  "$BUDDYGRAPH_TMP/buddygraph-models/conv_bn_relu.onnx" \
  -o "$BUDDYGRAPH_TMP/conv_bn_relu.mlir"
build/bin/buddygraph-opt "$BUDDYGRAPH_TMP/conv_bn_relu.mlir" -o /dev/null
```

这里的 `-o /dev/null` 只解析并验证导入结果，不是 CPU 编译。可执行的完整 lowering
见 [最小 runner 示例](examples/BGraph/README.md)。教程产物统一写入项目 `tmp/`，便于
在 VS Code 项目树查看；该目录已被 `.gitignore` 忽略。

`--scalar-return` 是 runner 测试适配器，只允许单个 rank-0 tensor 输出；默认
导入保持 tensor ABI。完整 CPU pipeline 及其顺序见
[pass_pipeline.md](docs/buddygraph/pass_pipeline.md)。

## 验证和评测

```bash
cmake --build build --target check-buddygraph -j2
export BUDDYGRAPH_TMP="$PWD/tmp"
mkdir -p "$BUDDYGRAPH_TMP"
python3 scripts/benchmark.py --llvm-tools "$LLVM_BUILD/bin" \
  -o "$BUDDYGRAPH_TMP/buddygraph-results.json"
```

当前回归基线为 14/14。

测试包括 ODS round trip、verifier 错误、shape refinement、rewrite 正负条件、
FullConversion、bufferization、前端拒绝路径，以及优化前/后相对 NumPy 的
确定性数值对拍；Clamp 专用 E2E 覆盖 direct/fused lowering 与 NaN 传播，另有
signed-zero 回归固定 strict IEEE 边界。测量结果和不应
过度解读的边界见
[results.md](docs/buddygraph/results.md)。

## 学习教程

准备将项目放进简历时，先读[简历对齐与掌握标准](docs/learning/resume_alignment.md)：
按四条项目陈述定位学习主线、校准实现范围，并用[深入追问](docs/learning/interview_bank.md)
与独立增量检验理解。

完整中文源码课程从 [docs/learning/README.md](docs/learning/README.md) 开始，包含
项目事实映射、术语表、00–15 章、四级练习、独立答案和学习证据清单。建议先完成
第 00 章环境与最小演示，再沿 ONNX → BGraph → Linalg → LLVM 的真实调用链学习。
若要亲手搭建 Dialect、Op、Type、Trait、Interface 和 Pass，直接进入
[自定义 Dialect 基础设施实战线](docs/learning/dialect_lab/README.md)；该专题只要求在
实验副本中实现 `bglab`，不会把教学能力冒充为当前 BGraph 基线。

## 目录

```text
include/BuddyGraph/       ODS、公开头文件和 Pass 声明
lib/BuddyGraph/           方言、优化和 BGraph-to-Linalg 实现
tools/buddygraph-opt/     独立 MLIR optimizer driver
frontend/BGraph/          ONNX opset-18 导入器与模型生成器
tests/                    lit/FileCheck 和数值 E2E 测试
examples/BGraph/          可手工运行的最小 MLIR 示例
docs/buddygraph/          审计、架构、设计、测试和结果
docs/learning/             中文源码课程、练习、答案和证据清单
scripts/benchmark.py      可复现的 compile/IR/runner 指标采集
```
