# 02｜定义第一个自定义 Op：`bglab.identity`

## 1. 本章目标

用 ODS 定义一个单输入、单结果的 tensor identity Op，注册 generated op list，并用
custom assembly format、round trip 和 generated accessor 证明接线完成。

## 2. 先运行

```bash
cd "$BUDDYGRAPH_LAB_ROOT"
rg -n 'GET_OP_LIST|GET_OP_CLASSES' \
  include/BuddyGraph/IR lib/BuddyGraph/IR \
  build/include/BuddyGraph/IR/BGraphOps*.inc
```

先从真实 BGraph 找到“声明 include、定义 include、注册 list”三个位置，再为 `bglab`
复制这种职责分工，而不是复制整份 BGraph 文件。

## 3. 定义 ODS schema

在 `BGLabOps.td` 的 `#endif` 前加入：

```tablegen
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def BGLab_IdentityOp
    : BGLab_Op<"identity", [Pure, SameOperandsAndResultType]> {
  let summary = "Forward a ranked tensor unchanged";
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}
```

`Pure` 组合无内存副作用和可推测执行性质，`SameOperandsAndResultType` 约束所有 operand/result 类型相同。
`assemblyFormat` 省略显式结果类型，是因为该 trait 让 parser 能从 input type 推出 result。

## 4. 声明 generated Op class

创建 `include/BuddyGraph/Lab/IR/BGLabOps.h`：

```cpp
#ifndef BUDDYGRAPH_LAB_IR_BGLABOPS_H
#define BUDDYGRAPH_LAB_IR_BGLABOPS_H

#include "BuddyGraph/Lab/IR/BGLabDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "BuddyGraph/Lab/IR/BGLabOps.h.inc"

#endif
```

创建 `lib/BuddyGraph/Lab/IR/BGLabOps.cpp`：

```cpp
#include "BuddyGraph/Lab/IR/BGLabOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace buddy::bglab;

#define GET_OP_CLASSES
#include "BuddyGraph/Lab/IR/BGLabOps.cpp.inc"
```

并把 `BGLabOps.cpp` 加到 `BuddyGraphLabIR` 的 source list。

在同一 library 的 `LINK_LIBS PUBLIC` 中保留 `MLIRIR` 并加入
`MLIRInferTypeOpInterface`、`MLIRSideEffectInterfaces`。即使 driver 目前间接链接这些库，
IR library 也应声明自己实际使用的依赖。`SameOperandsAndResultType` 的 `.td` 定义来自
InferType interface 文件；仅 include OpBase/SideEffect 文件不够。

## 5. 注册 generated op list

修改 `BGLabDialect.cpp`：

```cpp
#include "BuddyGraph/Lab/IR/BGLabDialect.h"
#include "BuddyGraph/Lab/IR/BGLabOps.h"

using namespace mlir;
using namespace buddy::bglab;

#include "BuddyGraph/Lab/IR/BGLabOpsDialect.cpp.inc"

void BGLabDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "BuddyGraph/Lab/IR/BGLabOps.cpp.inc"
      >();
}
```

`GET_OP_CLASSES` 生成 class 声明/方法定义；`GET_OP_LIST` 生成逗号分隔的 Op class list，
交给 Dialect 注册。三处宏用途不能互换。

## 6. 构建与观察生成代码

```bash
cmake --build build --target MLIRBGLabOpsIncGen BuddyGraphLabIR buddygraph-opt -j2

rg -n 'class IdentityOp|getInput\(|getResult\(' \
  build/include/BuddyGraph/Lab/IR/BGLabOps.h.inc
rg -n 'IdentityOp' \
  build/include/BuddyGraph/Lab/IR/BGLabOps.cpp.inc | head
```

不要记忆 generated 文件的具体行号；学习目标是根据 record 名、mnemonic 和 accessor
字段定位生成物。

## 7. IR round trip

创建 `tests/Tutorial/BGLab/ops.mlir`：

```mlir
// RUN: buddygraph-opt %s | FileCheck %s

// CHECK-LABEL: func.func @identity
// CHECK: %[[R:.*]] = bglab.identity %arg0 : tensor<2x3xf32>
// CHECK: return %[[R]]
func.func @identity(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = bglab.identity %arg0 : tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
```

先手工运行：

```bash
build/bin/buddygraph-opt tests/Tutorial/BGLab/ops.mlir
```

若 lit 尚未发现新目录，先不改 suffix；现有 `tests/lit.cfg.py` 已递归扫描 `.mlir`。

## 8. 结构 verifier 的边界

下面的 IR 应在 generated verification 阶段失败：

```mlir
%0 = "bglab.identity"(%arg0)
    : (tensor<2x3xf32>) -> tensor<2x4xf32>
```

失败来自 `SameOperandsAndResultType`，不是你写的 `IdentityOp::verify()`——本章没有声明
`hasVerifier = 1`，也没有手写该函数。

## 9. 常见错误

- 只生成 class，忘记 `addOperations<GET_OP_LIST>()`：链接通过但 parser 说未注册 Op。
- 自定义格式没有写 `attr-dict`：未来添加 discardable attributes 时无法打印。
- 在 source tree 寻找 `BGLabOps.h.inc`：它只在 build include 目录。
- 同时 include `BGLabOps.cpp.inc` 两次且都定义 `GET_OP_CLASSES`，导致重复定义。
- 把 `Pure` 理解为“结果值不会改变”；它组合内存副作用与可推测执行性质，不是数值恒等性证明。

## 10. 验收标准

- custom syntax 和 quoted generic syntax 都能 parse/print。
- generated header 中存在 `IdentityOp::getInput()`。
- 类型不一致得到 generated verifier 错误。
- 能解释为什么本 Op 暂时不需要 custom verifier、builder 或 parser。

## 11. 面试追问

**问：typed Op wrapper 是否拥有 `Operation`？**

答：不拥有。`IdentityOp` 是围绕 `Operation *` 的轻量 typed wrapper；实际 Operation
由 Block/Region 的 IR 所有权结构管理，generated accessor 只是类型安全地访问其字段。

下一章：[定义 parameterized custom Type](03_custom_type.md)。
