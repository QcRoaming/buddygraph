# 03｜定义 parameterized Type：`!bglab.tag<"...">`

## 1. 本章目标

定义一个由 `MLIRContext` 唯一化的字符串参数 Type，让它成为真实 SSA Value 的类型，
并新增 `bglab.make_tag` 与 `bglab.mark` 观察 Type 的生成、解析、注册和访问。

## 2. 先运行

先观察 BuddyGraph 为什么没有自定义 Type：

```bash
cd "$BUDDYGRAPH_LAB_ROOT"
rg -n 'TypeDef<|GET_TYPEDEF_LIST|addTypes<' include/BuddyGraph lib/BuddyGraph || true
find build/include/BuddyGraph/IR -name '*Types*.inc' -printf '%f\n' | sort
```

你会看到 `add_mlir_dialect` 已生成空的 TypeDef `.inc`，但基线没有 `TypeDef` record，
`BGraphDialect::initialize()` 也没有 `addTypes`。生成能力存在不等于项目已经定义 Type。

## 3. 定义 TypeDef

创建 `include/BuddyGraph/Lab/IR/BGLabTypes.td`：

```tablegen
#ifndef BUDDYGRAPH_LAB_IR_BGLABTYPES_TD
#define BUDDYGRAPH_LAB_IR_BGLABTYPES_TD

include "BuddyGraph/Lab/IR/BGLabDialect.td"
include "mlir/IR/AttrTypeBase.td"

class BGLab_Type<string name, string typeMnemonic,
                 list<Trait> traits = []>
    : TypeDef<BGLab_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def BGLab_TagType : BGLab_Type<"Tag", "tag"> {
  let summary = "Compile-time annotation tag";
  let parameters = (
      ins StringRefParameter<"annotation label">:$label
  );
  let assemblyFormat = "`<` $label `>`";
}

#endif
```

在 `BGLabOps.td` 中 include 它：

```tablegen
include "BuddyGraph/Lab/IR/BGLabTypes.td"
```

这里的 `StringRefParameter` 不是把 view 悬挂在临时字符串上。TypeDef storage 会把参数
复制进 Context 管理的 allocator，并用参数作为 uniquing key。

## 4. 声明和定义 generated Type class

创建 `BGLabTypes.h`：

```cpp
#ifndef BUDDYGRAPH_LAB_IR_BGLABTYPES_H
#define BUDDYGRAPH_LAB_IR_BGLABTYPES_H

#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "BuddyGraph/Lab/IR/BGLabOpsTypes.h.inc"

#endif
```

创建 `BGLabTypes.cpp`：

```cpp
#include "BuddyGraph/Lab/IR/BGLabTypes.h"
#include "BuddyGraph/Lab/IR/BGLabDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace buddy::bglab;

#define GET_TYPEDEF_CLASSES
#include "BuddyGraph/Lab/IR/BGLabOpsTypes.cpp.inc"
```

在 `BGLabOps.h` 中，必须在 `GET_OP_CLASSES` 之前 include：

```cpp
#include "BuddyGraph/Lab/IR/BGLabTypes.h"
```

否则 generated Op 声明引用 `TagType` 时，C++ 尚不知道该类型。

## 5. 注册 Type

把 `BGLabDialect.td` 的 `useDefaultTypePrinterParser` 从 0 改为 1。此时已有 TypeDef，
generated Type definitions 才会提供对应 Dialect parse/print 实现。把 `BGLabTypes.cpp`
加入 `BuddyGraphLabIR`，并修改 Dialect 初始化：

```cpp
#include "BuddyGraph/Lab/IR/BGLabTypes.h"

void BGLabDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "BuddyGraph/Lab/IR/BGLabOpsTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "BuddyGraph/Lab/IR/BGLabOps.cpp.inc"
      >();
}
```

`useDefaultTypePrinterParser = 1` 让 Dialect 根据 TypeDef 的 mnemonic 与
`assemblyFormat` 分派 parse/print；无需手写整个 `Dialect::parseType()`。

## 6. 让 Type 进入 SSA

在 `BGLabOps.td` 加入：

```tablegen
def BGLab_MakeTagOp : BGLab_Op<"make_tag", [Pure]> {
  let summary = "Create an SSA annotation tag";
  let results = (outs BGLab_TagType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def BGLab_MarkOp : BGLab_Op<"mark", [Pure]> {
  let summary = "Attach compile-time metadata to a tensor";
  let arguments = (ins AnyRankedTensor:$input, BGLab_TagType:$tag);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat =
      "$input `,` $tag attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
```

在 `BGLabOps.cpp` 的 generated definitions include **之前**实现：

```cpp
LogicalResult MarkOp::verify() {
  if (getInput().getType() != getResult().getType())
    return emitOpError("input and result types must match");
  return success();
}
```

本章先用 Op verifier 保存不变量；下一章会把它重构为可复用 Trait。

## 7. IR 与 C++ API

创建或追加 `tests/Tutorial/BGLab/types.mlir`：

```mlir
// RUN: buddygraph-opt %s | FileCheck %s

// CHECK: %[[TAG:.*]] = bglab.make_tag : !bglab.tag<"frontend">
// CHECK: bglab.mark %arg0, %[[TAG]]
func.func @tag(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %tag = bglab.make_tag : !bglab.tag<"frontend">
  %0 = bglab.mark %arg0, %tag
      : (tensor<2x3xf32>, !bglab.tag<"frontend">) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
```

generated C++ API 的关键形式是：

```cpp
auto type = TagType::get(context, "frontend");
llvm::StringRef label = type.getLabel();
```

同一 Context 中以相同参数调用 `get`，得到相同 uniqued storage；不要用 `new TagType`。

本章 checkpoint：

```bash
cmake --build build --target MLIRBGLabOpsIncGen BuddyGraphLabIR buddygraph-opt -j2
build/bin/buddygraph-opt tests/Tutorial/BGLab/types.mlir
```

输出应保留 `!bglab.tag<"frontend">` 与 mark。`.cpp.inc` 还实现 Dialect 的 Type
parse/print，因此需要 Dialect class、AsmParser/AsmPrinter 和 TypeSwitch 的声明，
不能仅 include `Types.h`。

## 8. Type、Attribute 与 Value 的边界

- Type 描述 SSA Value 可具有的类别；`%tag` 的类型是 `!bglab.tag<...>`。
- Attribute 是附着在 Operation/Type 上的不可变编译期数据；字符串参数最终存于 Type
  storage，但 `TagType` 本身仍是 Type。
- Value 是某次 definition 的结果；两个 `make_tag` 可以产生两个不同 Value，却共享
 同一个 uniqued `TagType`。
- `!bglab.tag<"frontend">` 没有 runtime layout/ABI。本专题 Pass 必须在进入普通后端
  pipeline 前消除它；否则后续 conversion 并不知道如何 lowering。

## 9. 常见错误

- 定义了 TypeDef，却忘记 `addTypes<GET_TYPEDEF_LIST>()`：mnemonic 可能已识别，但
  `TagType::get()` 无法使用未注册 storage；assertions 构建可能 fatal。它与拼错 `tag`
  导致的 unknown type mnemonic 是两个阶段。
- 忘记 `GET_TYPEDEF_CLASSES`：C++ 找不到 `TagType` 或 accessor。
- 直接把 `StringAttr` 当 SSA Value type；Attribute 和 Type 的位置不同。
- 在 `mark` 的 assembly format 只打印 input type，导致 tag/result type无法解析。
- 认为自定义 Type 必须立即写 TypeConverter；只有跨 conversion 边界保留它时才需要。

## 10. 验收标准

- `!bglab.tag<"frontend">` 能 round trip。
- generated header 中存在 `TagType::getLabel()`。
- `bglab.make_tag` 的 result 确实是自定义 Type。
- `mark` 的 input/result type 不同会得到 custom verifier 诊断。
- 能解释为什么本 Type 是编译期 metadata，而不是 tensor element type。

## 11. 面试追问

**问：为什么 BuddyGraph 主 IR 仍使用 builtin tensor，而教程又增加 custom Type？**

答：业务 IR 不需要为已有 tensor 语义重复造 Type；专题用 metadata tag 教习 TypeDef
机制，并通过清理 Pass 在后端前消除。学习基础设施不等于必须把自定义 Type 留进生产
pipeline。

下一章：[把局部 verifier 重构为可复用 Trait](04_custom_trait.md)。
