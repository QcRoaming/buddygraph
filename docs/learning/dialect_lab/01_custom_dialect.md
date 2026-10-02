# 01｜从零创建并注册 `bglab` Dialect

## 1. 本章目标

新增一个暂时没有 Op/Type 的 `bglab` Dialect，追通 `.td` → generated dialect class →
library → driver registry，并能区分“Dialect 已编译”和“Dialect 已加载”。

## 2. 先运行

在实验副本中确认基线没有 `bglab`：

```bash
cd "$BUDDYGRAPH_LAB_ROOT"
rg -n 'bglab|BGLab' include lib tools tests || true
printf 'module { "bglab.unknown"() : () -> () }\n' | \
  build/bin/buddygraph-opt -o /dev/null -
```

第二条应失败并报告未注册 Dialect 或未知 Op。不要添加
`--allow-unregistered-dialect`；它会掩盖本章要验证的接线。

## 3. 新增 TableGen 与头文件

创建 `include/BuddyGraph/Lab/IR/BGLabDialect.td`：

```tablegen
#ifndef BUDDYGRAPH_LAB_IR_BGLABDIALECT_TD
#define BUDDYGRAPH_LAB_IR_BGLABDIALECT_TD

include "mlir/IR/OpBase.td"

def BGLab_Dialect : Dialect {
  let name = "bglab";
  let cppNamespace = "::buddy::bglab";
  let summary = "Teaching dialect for MLIR infrastructure exercises";
  let useDefaultTypePrinterParser = 0;
}

class BGLab_Op<string mnemonic, list<Trait> traits = []>
    : Op<BGLab_Dialect, mnemonic, traits>;

#endif
```

创建暂时只有 include guard 的 `BGLabOps.td`：

此时没有 TypeDef，先关闭默认 Type parser/printer 生成。第 03 章加入 Type definitions
时再开启；提前开启会生成 override 声明却没有对应定义，空 Dialect checkpoint 会链接失败。

```tablegen
#ifndef BUDDYGRAPH_LAB_IR_BGLABOPS_TD
#define BUDDYGRAPH_LAB_IR_BGLABOPS_TD

include "BuddyGraph/Lab/IR/BGLabDialect.td"

#endif
```

创建 `BGLabDialect.h`：

```cpp
#ifndef BUDDYGRAPH_LAB_IR_BGLABDIALECT_H
#define BUDDYGRAPH_LAB_IR_BGLABDIALECT_H

#include "mlir/IR/Dialect.h"
#include "BuddyGraph/Lab/IR/BGLabOpsDialect.h.inc"

#endif
```

## 4. 接入 TableGen CMake

创建 `include/BuddyGraph/Lab/IR/CMakeLists.txt`：

```cmake
set(LLVM_TARGET_DEFINITIONS BGLabOps.td)
add_mlir_dialect(BGLabOps bglab)
```

再创建两级转发文件：

```cmake
# include/BuddyGraph/Lab/CMakeLists.txt
add_subdirectory(IR)
```

并在已有 `include/BuddyGraph/CMakeLists.txt` 末尾加入：

```cmake
add_subdirectory(Lab)
```

`add_mlir_dialect(BGLabOps bglab)` 会建立目标 `MLIRBGLabOpsIncGen`，生成 Dialect、Op、
Type 的声明/定义 `.inc`。此时 Type/Op 列表为空并不等于没有生成链。

## 5. 实现并构建 Dialect library

创建 `lib/BuddyGraph/Lab/IR/BGLabDialect.cpp`：

```cpp
#include "BuddyGraph/Lab/IR/BGLabDialect.h"

using namespace mlir;
using namespace buddy::bglab;

#include "BuddyGraph/Lab/IR/BGLabOpsDialect.cpp.inc"

void BGLabDialect::initialize() {}
```

创建 `lib/BuddyGraph/Lab/IR/CMakeLists.txt`：

```cmake
add_mlir_dialect_library(BuddyGraphLabIR
  BGLabDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/BuddyGraph/Lab

  DEPENDS
  MLIRBGLabOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
```

创建 `lib/BuddyGraph/Lab/CMakeLists.txt`：

```cmake
add_subdirectory(IR)
```

并在 `lib/BuddyGraph/CMakeLists.txt` 加入：

```cmake
add_subdirectory(Lab)
```

## 6. 注册到 `buddygraph-opt`

在 `tools/buddygraph-opt/buddygraph-opt.cpp`：

```cpp
#include "BuddyGraph/Lab/IR/BGLabDialect.h"
```

把 `buddy::bglab::BGLabDialect` 加入现有 `registry.insert<...>()` 模板参数列表。

在 `tools/buddygraph-opt/CMakeLists.txt` 的 `target_link_libraries` 中加入：

```cmake
BuddyGraphLabIR
```

然后只构建相关目标：

```bash
cmake --build build --target MLIRBGLabOpsIncGen BuddyGraphLabIR buddygraph-opt -j2
find build/include/BuddyGraph/Lab -type f -name '*.inc' -printf '%f\n' | sort
build/bin/buddygraph-opt --show-dialects | tr ',' '\n' | rg '^bglab$'
```

## 7. 调用链

```text
BGLab_Dialect record
→ MLIRBGLabOpsIncGen
→ BGLabOpsDialect.h.inc / .cpp.inc
→ BGLabDialect C++ class
→ BuddyGraphLabIR
→ buddygraph-opt link
→ DialectRegistry::insert<BGLabDialect>()
→ MLIRContext 按需加载
→ BGLabDialect::initialize()
```

注册表保存构造信息；Dialect 实例由 `MLIRContext` 管理。链接 library 并不会自动把
Dialect 加入 registry，加入 registry 也不等于它在程序启动时立即实例化。

## 8. 常见错误

- 只添加 `.td`，没有把 `Lab` 子目录接进 `include/BuddyGraph/CMakeLists.txt`。
- C++ include 了 generated header，但 library 没有依赖 `MLIRBGLabOpsIncGen`。
- driver 注册了 class，却忘记 link `BuddyGraphLabIR`，产生 undefined symbol。
- `let name = "bgraph"` 与现有 Dialect 冲突。
- 用 `BGLabOpsIncGen` 猜 target；本地 helper 实际加 `MLIR` 前缀。

## 9. 动手练习

暂时从 driver 的 `registry.insert` 删除 `BGLabDialect`，保留全部编译和链接。观察同一份
输入为何回到“未注册 Dialect”错误，再恢复这一行。该实验区分 build integration 与
runtime registration。

## 10. 验收标准

- `MLIRBGLabOpsIncGen` 与 `BuddyGraphLabIR` 可单独构建。
- generated dialect `.inc` 位于实验 build tree。
- `--show-dialects` 明确列出 `bglab`。
- `nm -C build/lib/libBuddyGraphLabIR.a | rg BGLabDialect` 有定义。
- 能画出 TableGen target、C++ library、driver registry 三条边。

目前仍不能解析任何 `bglab.*` Op；Dialect namespace 存在不意味着任意助记符自动合法。

## 11. 面试追问

**问：为什么 Op 名是 `bglab.foo`，C++ namespace 却是 `buddy::bglab`？**

答：`Dialect.name` 决定文本 IR namespace，`cppNamespace` 决定生成的 C++ 符号位置；二者
服务不同层，可以不同，但必须各自保持唯一且稳定。

下一章：[定义第一个自定义 Op](02_custom_op.md)。
