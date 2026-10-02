# 00｜建立可恢复的 Dialect 实验副本

## 1. 本章目标

建立一个不会污染 BuddyGraph 基线、不会误用原 build 的实验副本，并为每个 checkpoint
保留源码 diff、生成文件和测试输出。

## 2. 先运行

```bash
cd /home/jlq/project/buddygraph
export BUDDYGRAPH_BASE="$PWD"
export BUDDYGRAPH_LAB_ROOT="$PWD/tmp/dialect-lab"

test ! -e "$BUDDYGRAPH_LAB_ROOT/CMakeLists.txt" || {
  echo "实验目录已有源码；请改用一个新的 BUDDYGRAPH_LAB_ROOT" >&2
  exit 1
}
mkdir -p "$BUDDYGRAPH_LAB_ROOT"
rsync -a \
  --exclude=/build/ --exclude=/.deps/ --exclude=/tmp/ \
  --exclude='__pycache__/' --exclude='*.pyc' \
  "$BUDDYGRAPH_BASE/" "$BUDDYGRAPH_LAB_ROOT/"

cmake -S "$BUDDYGRAPH_LAB_ROOT" -B "$BUDDYGRAPH_LAB_ROOT/build" -G Ninja \
  -DMLIR_DIR=/buddy-mlir/llvm/build/lib/cmake/mlir \
  -DLLVM_DIR=/buddy-mlir/llvm/build/lib/cmake/llvm \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUDDYGRAPH_LAB_ROOT/build" --target buddygraph-opt -j2
```

保护条件会拒绝向已有源码树合并文件；若要保留旧 checkpoint，请把
`BUDDYGRAPH_LAB_ROOT` 改成新的项目内路径，而不是覆盖旧目录。这一步会为实验副本
建立自己的 build。后续不能从实验源码调用原项目的
`/home/jlq/project/buddygraph/build/bin/buddygraph-opt`。

### Python 回归依赖

本专题的 C++ checkpoint 不需要复制 Python 包；副本的 `.deps` 已排除。若随后执行
Frontend/E2E 或完整回归，新 shell 中先复用基线依赖：

```bash
export PYTHONPATH="/home/jlq/project/buddygraph/.deps${PYTHONPATH:+:$PYTHONPATH}"
```

lit 会另外加入现有 MLIR Python bindings。只读教程无需构建；实际修改实验源码后，
才按各章 checkpoint 构建并验收。

## 3. 真实代码位置

只读对照以下基线文件：

- 根 `CMakeLists.txt`：设置 source/build include 路径并加入 `include/lib/tools/tests`。
- `include/BuddyGraph/IR/CMakeLists.txt`：ODS/TableGen target。
- `lib/BuddyGraph/IR/CMakeLists.txt`：Dialect library 与生成依赖。
- `tools/buddygraph-opt/CMakeLists.txt`：最终 executable 的 link closure。
- `tools/buddygraph-opt/buddygraph-opt.cpp`：Dialect/Pass 注册入口。
- `tests/lit.site.cfg.py.in`：source、build、LLVM 工具路径的固化位置。

实验修改只发生在 `$BUDDYGRAPH_LAB_ROOT`。

## 4. 调用链

```text
实验 source/CMakeLists.txt
→ cmake configure
→ 实验 build/build.ninja
→ TableGen custom commands
→ 实验 build/include/.../*.inc
→ C++ libraries
→ 实验 build/bin/buddygraph-opt
→ 实验 build/tests/lit.site.cfg.py
```

source tree 与 build tree 必须成对。若 CMakeCache 仍指向基线 source，立即停止，不要
继续解释编译错误。

## 5. IR 前后变化

本章不改变 IR。先验证实验工具仍能处理基线输入：

```bash
cd "$BUDDYGRAPH_LAB_ROOT"
build/bin/buddygraph-opt tests/Dialect/BGraph/roundtrip.mlir -o /dev/null
```

预期没有输出且退出码为 0。它只证明实验副本起点可用，不证明后续 `bglab` 已存在。

## 6. 核心机制

### 为什么不能共用 build

CMakeCache、generated `.inc`、object dependency 和 lit site config 都记录 source/build
绝对路径。副本源码配原 build 容易出现“看见旧 generated class”“测试读了另一份源码”
等假象。

### 建议的 checkpoint

```bash
mkdir -p "$BUDDYGRAPH_LAB_ROOT/notes/dialect-lab"
git -C "$BUDDYGRAPH_LAB_ROOT" init --initial-branch=main
git -C "$BUDDYGRAPH_LAB_ROOT" add .
git -C "$BUDDYGRAPH_LAB_ROOT" commit -m 'Baseline for bglab tutorial'
```

只有实验副本被初始化；不要在 `/buddy-mlir` 根目录执行 `git init`。如果不想创建 Git，
每章结束至少保存 `find` 清单和修改文件副本。

## 7. 为什么这样设计

本专题会连续修改 CMake、TableGen、生成依赖、driver 和 tests。可恢复副本能让“故障
注入”成为安全实验，也避免把教学 Op 混入真实 ONNX/BGraph 支持范围。

## 8. 常见错误

- `BUDDYGRAPH_LAB_ROOT` 指回基线目录，实验直接覆盖当前工程。
- 对已有实验目录再次执行 `rsync`，把两个 checkpoint 静默混在一起。
- 复制了 `build/`，其 Cache 仍引用旧 source。
- 在副本中修改文件，却调用基线 `build/bin/buddygraph-opt`。
- 把 `.inc` 从 build 复制到 source，制造两个事实来源。
- 用 `git checkout --` 恢复尚未纳入 Git 的目录。

快速核验：

```bash
rg '^CMAKE_HOME_DIRECTORY:' "$BUDDYGRAPH_LAB_ROOT/build/CMakeCache.txt"
realpath "$BUDDYGRAPH_LAB_ROOT/build/bin/buddygraph-opt"
```

## 9. 动手练习

故意把命令中的 executable 换成基线路径，分别打印两个工具的 `realpath`。解释为什么
二者版本字符串可能相同，却不能用来验证实验源码。

## 10. 验收标准

- `CMAKE_HOME_DIRECTORY` 等于 `$BUDDYGRAPH_LAB_ROOT`。
- 实验 `buddygraph-opt` 能 parse 基线 roundtrip。
- `git status` 或备份清单只覆盖实验副本。
- 能指出 source `.td` 与 build `.inc` 各自的所有者。

## 11. 面试追问

**问：out-of-tree build 与 out-of-tree Dialect 是一回事吗？**

答：不是。前者指构建产物不写进 source；后者通常指 Dialect 项目不作为 LLVM 源码树
的一部分构建。BuddyGraph 位于 `/home/jlq/project/buddygraph`，不在
`/buddy-mlir` 文件树内，且有独立 CMake/build；构建仍复用既有 LLVM/MLIR 包。

下一章：[创建并注册 `bglab` Dialect](01_custom_dialect.md)。
