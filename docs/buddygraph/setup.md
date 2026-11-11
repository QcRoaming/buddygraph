# 轻量环境准备

本机验证环境：Ubuntu 26.04 x86_64，LLVM/MLIR 21.1.8，Python 3.11。
使用发行版预编译 SDK，只编译 BuddyGraph 和 upstream 的最小 Python bindings。
全部依赖放在用户缓存中，不需要编译完整 LLVM，也不需要管理员权限。
其他发行版应使用与本机兼容的预编译 SDK，不能直接混用 Ubuntu 的二进制包。

以下命令在项目根目录执行。需要已有 C++ 编译器、CMake、Ninja、curl、dpkg-deb
和 apt-get。本机的编译器为 GCC 15。

## 1. 预编译 LLVM/MLIR SDK

```bash
export BUDDYGRAPH_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/buddygraph"
mkdir -p "$BUDDYGRAPH_CACHE/debs" "$BUDDYGRAPH_CACHE/toolchain"
(
  cd "$BUDDYGRAPH_CACHE/debs"
  apt-get download \
    llvm-21 llvm-21-dev llvm-21-tools llvm-21-linker-tools llvm-21-runtime \
    libllvm21 libclang-cpp21 libmlir-21 libmlir-21-dev mlir-21-tools \
    libffi-dev zlib1g-dev libzstd-dev libxml2-dev libedit-dev libncurses-dev \
    libtinfo6 libxml2-16 libedit2 libzstd1
  for package in ./*.deb; do
    dpkg-deb -x "$package" "$BUDDYGRAPH_CACHE/toolchain"
  done
)
```

这是 Ubuntu 26.04 软件包的组合，本次版本为 `21.1.8-6ubuntu1`。
SDK 解包后约 1.4 GiB，包含开发头文件、静态/动态库和 runner 等工具；项目没有
复制这些库。`llvm-21-linker-tools` 和 `llvm-21-runtime` 也必须准备好，因为
SDK 的 CMake 导出目标会检查这些包中的文件。

## 2. 独立 Python 环境

本机系统 Python 为 3.14，项目固定依赖使用 Python 3.11 的 venv。
如果尚未安装 uv：

```bash
curl -LsSf https://astral.sh/uv/install.sh -o "$BUDDYGRAPH_CACHE/install-uv.sh"
sh "$BUDDYGRAPH_CACHE/install-uv.sh"
```

```bash
export PATH="$HOME/.local/bin:$PATH"
uv venv "$BUDDYGRAPH_CACHE/venv" --python 3.11 --seed
uv pip install --python "$BUDDYGRAPH_CACHE/venv/bin/python" \
  -r requirements.txt 'nanobind>=2.4,<3.0' 'lit==18.1.8'
source scripts/env.sh
```

`lit` 是测试驱动，版本无需与 LLVM 完全相同。本机验证了 18.1.8。
NumPy/ONNX 使用 requirements.txt 的固定版本；不将旧 `.deps` 加入 PYTHONPATH。

## 3. 只构建前端所需的 MLIR Python bindings

发行版 SDK 没有附带 Python bindings。下载与 SDK **相同版本**的 MLIR 源码包，
约 7 MB；使用 [upstream Python bindings](https://mlir.llvm.org/docs/Bindings/Python/)
源码单独构建 core extension、dialect 注册模块和 Func 的 Python wrapper。
不构建 execution engine Python extension 或其他 dialect Python extension。

```bash
curl --retry 3 -LsSf \
  https://github.com/llvm/llvm-project/releases/download/llvmorg-21.1.8/mlir-21.1.8.src.tar.xz \
  -o "$BUDDYGRAPH_CACHE/mlir-21.1.8.src.tar.xz"
tar -xf "$BUDDYGRAPH_CACHE/mlir-21.1.8.src.tar.xz" -C "$BUDDYGRAPH_CACHE"
cmake -S scripts/python-bindings -B "$BUDDYGRAPH_CACHE/python-build" -G Ninja \
  -DMLIR_DIR="$LLVM_BUILD/lib/cmake/mlir" \
  -DLLVM_DIR="$LLVM_BUILD/lib/cmake/llvm" \
  -DCMAKE_PREFIX_PATH="$BUDDYGRAPH_CACHE/toolchain/usr" \
  -DPython_EXECUTABLE="$BUDDYGRAPH_CACHE/venv/bin/python" \
  -DMLIR_SOURCE_DIR="$BUDDYGRAPH_CACHE/mlir-21.1.8.src" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUDDYGRAPH_CACHE/python-build" -j2
python3 -c 'import numpy,onnx; from mlir import ir; from mlir.dialects import func; print(ir.__file__)'
```

这个辅助 CMake 工程只为本项目使用的 MLIR 21.1.8 API 准备 bindings，不是完整
MLIR Python 包的替代品。C++ API 或 Python ABI 变化时，应使用匹配源码重新构建。
本机 bindings 构建目录约 8 MiB。

## 4. 构建、回归与执行

回到 [README 的构建步骤](../../README.md#构建)，执行项目配置和构建命令，然后按 README 生成、导入模型；CPU 执行步骤见
[最小 runner 示例](../../examples/BGraph/README.md)。预期回归为 17/17，
ONNX CPU 输出为 `7.000000e+00`，最小 MLIR 示例输出为 `1.600000e+01`。后续终端只需 `source scripts/env.sh`，不需要重新准备依赖。

如果已有其他 LLVM/MLIR 21.1.8 SDK 或匹配的 Python bindings，可在 source 前
设置 `LLVM_BUILD`、`BUDDYGRAPH_MLIR_PYTHON_DIR`，并调整 CMake 的 prefix 和 Python
解释器参数，跳过对应依赖准备步骤。
