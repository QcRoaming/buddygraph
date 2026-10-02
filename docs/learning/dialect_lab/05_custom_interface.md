# 05｜定义 custom OpInterface：统一查询 forwarding value

## 1. 本章目标

定义 `ForwardingOpInterface`，让 `bglab.identity` 与 `bglab.mark` 各自实现
`getForwardedValue()`，再用 interface dispatch 证明算法无需枚举具体 Op class。

## 2. 先运行

先看当前 BGraph 的对照事实：

```bash
cd "$BUDDYGRAPH_LAB_ROOT"
rg -n 'InferTypeOpInterface|DeclareOpInterfaceMethods|dyn_cast<.*Op>' \
  include/BuddyGraph lib/BuddyGraph
```

基线 `BGraphOps.td` include 了 upstream InferType interface 定义，但没有任何 BGraph Op
实现项目自有 Interface；当前 shape pass 用 `dyn_cast<ReluOp/ClampOp/...>` 枚举具体类。

## 3. 定义 Interface record

创建 `include/BuddyGraph/Lab/IR/BGLabInterfaces.td`：

```tablegen
#ifndef BUDDYGRAPH_LAB_IR_BGLABINTERFACES_TD
#define BUDDYGRAPH_LAB_IR_BGLABINTERFACES_TD

include "mlir/IR/OpBase.td"

def BGLab_ForwardingOpInterface
    : OpInterface<"ForwardingOpInterface"> {
  let description = [{
    Implemented by regionless, memory-effect-free operations whose sole result
    is semantically equal to one of their existing operands. The returned
    non-null value must remain valid after this operation is erased.
  }];
  let methods = [
    InterfaceMethod<
      "Return the SSA value forwarded by this operation.",
      "::mlir::Value",
      "getForwardedValue"
    >
  ];
}

#endif
```

`InterfaceMethod` 描述公共调用契约；没有 method body 表示每个 ConcreteOp 必须提供实现。

## 4. 生成 Interface 声明与分派代码

创建 `BGLabInterfaces.h`：

```cpp
#ifndef BUDDYGRAPH_LAB_IR_BGLABINTERFACES_H
#define BUDDYGRAPH_LAB_IR_BGLABINTERFACES_H

#include "mlir/IR/OpDefinition.h"

namespace buddy::bglab {
#include "BuddyGraph/Lab/IR/BGLabInterfaces.h.inc"
} // namespace buddy::bglab

#endif
```

创建 `BGLabInterfaces.cpp`：

```cpp
#include "BuddyGraph/Lab/IR/BGLabInterfaces.h"

namespace buddy::bglab {
#include "BuddyGraph/Lab/IR/BGLabInterfaces.cpp.inc"
} // namespace buddy::bglab
```

在 `include/BuddyGraph/Lab/IR/CMakeLists.txt` 加：

```cmake
add_mlir_interface(BGLabInterfaces)
```

本地 helper 生成 target `MLIRBGLabInterfacesIncGen`。把该 target 加到
`BuddyGraphLabIR` 的 `DEPENDS`，并把 `BGLabInterfaces.cpp` 加入 sources。

## 5. 让 Op 声明实现 Interface

在 `BGLabOps.td` include interface td：

```tablegen
include "BuddyGraph/Lab/IR/BGLabInterfaces.td"
```

为两个 Op 加 `DeclareOpInterfaceMethods`：

```tablegen
def BGLab_IdentityOp : BGLab_Op<"identity", [
    Pure,
    SameOperandsAndResultType,
    DeclareOpInterfaceMethods<BGLab_ForwardingOpInterface>
  ]> {
  // 原 schema 不变
}

def BGLab_MarkOp : BGLab_Op<"mark", [
    Pure,
    BGLab_FirstOperandAndResultSameType,
    DeclareOpInterfaceMethods<BGLab_ForwardingOpInterface>
  ]> {
  // 原 schema 不变
}
```

并在 `BGLabOps.h` 的 generated Op include 前加入：

```cpp
#include "BuddyGraph/Lab/IR/BGLabInterfaces.h"
```

## 6. 实现每个 ConcreteOp 的方法

在 `BGLabOps.cpp`、generated definitions include 之前加入：

```cpp
Value IdentityOp::getForwardedValue() { return getInput(); }

Value MarkOp::getForwardedValue() { return getInput(); }
```

两者恰好都返回 `getInput()`；未来 Op 可以转发第二个 operand，但必须满足完整契约：
非空、同类型、确实语义等价、在替换位置可见，且删除当前 Op 后仍有效。任意内部
Region yield 的 Value 不能直接替换外部 uses，因为它可能不支配使用点，且随 Op 删除。
本实验将实现者限定为无 Region、无内存副作用、转发现有 operand 的 Op。

如果只在 ODS 写 `BGLab_ForwardingOpInterface` 而不用
`DeclareOpInterfaceMethods<...>`，Op 会声明“实现 interface”，但不会自动声明你要在
ConcreteOp 上定义的方法；本教程使用 `DeclareOpInterfaceMethods` 保持定义位置明确。

## 7. Interface dispatch

先在一个临时 C++ helper 或下一章 Pass 中使用：

```cpp
static Value getForwardedValue(Operation *operation) {
  auto forwarding = dyn_cast<ForwardingOpInterface>(operation);
  if (!forwarding)
    return {};
  return forwarding.getForwardedValue();
}
```

调用链是：

```text
dyn_cast<ForwardingOpInterface>(Operation *)
→ 查 Operation interface map
→ 找到 ConcreteOp Model
→ Model 调用 IdentityOp::getForwardedValue()
   或 MarkOp::getForwardedValue()
→ 返回 Value
```

它不是基于 op name 字符串的 `if/else`，也不是要求所有 Op 继承一个 owning C++ 基类。

## 8. 观察 generated 代码

```bash
cmake --build build --target \
  MLIRBGLabInterfacesIncGen MLIRBGLabOpsIncGen BuddyGraphLabIR -j2

rg -n 'class ForwardingOpInterface|struct Concept|struct Model|getForwardedValue' \
  build/include/BuddyGraph/Lab/IR/BGLabInterfaces.h.inc \
  build/include/BuddyGraph/Lab/IR/BGLabInterfaces.cpp.inc \
  build/include/BuddyGraph/Lab/IR/BGLabOps.h.inc
```

重点辨认 Interface wrapper、Concept/Model 分派与 ConcreteOp method declaration，不要
把 generated 文件逐行背诵。

## 9. 常见错误

- 忘记 `add_mlir_interface`，include 文件从未生成。
- 猜成 `BGLabInterfacesIncGen`；实际 target 是 `MLIRBGLabInterfacesIncGen`。
- generated interface include 的 C++ namespace 与 Op namespace 不一致。
- ODS 声明了 method，却忘记 C++ definition，最终在 link 阶段失败。
- Pass 仍用 `isa<IdentityOp, MarkOp>` 枚举，Interface 只是装饰而未形成抽象边界。
- 把 Interface 当 ABI vtable；它是 MLIR Context 中按 Operation 注册的概念/模型分派。

## 10. 验收标准

- 两个 Op 的 generated class 都声明 `getForwardedValue()`。
- `dyn_cast<ForwardingOpInterface>` 对 identity/mark 成功，对 make_tag 失败。
- 删除任一 ConcreteOp method definition 会得到可解释的 link error。
- 能说明 Trait 描述性质，Interface 暴露跨 Op 行为，Pass 消费 Interface。

## 11. 面试追问

**问：什么时候应使用 MLIR Interface，而不是在 Pass 中 `dyn_cast` 每个 Op？**

答：当多个 Op 对同一算法提供稳定但可能不同的行为时。Interface 把扩展点放到 Op
实现者一侧，使算法依赖能力而不是具体类；如果只有一个 Op 或逻辑没有复用价值，直接
typed dispatch 更简单。

下一章：[实现 interface-driven Pass](06_custom_pass.md)。
