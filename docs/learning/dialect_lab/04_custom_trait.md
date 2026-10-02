# 04｜定义可复用 Trait：首 operand 与 result 同类型

## 1. 本章目标

把 `MarkOp::verify()` 中的结构不变量提炼为 native Op Trait，使后续其他 forwarding Op
可以复用同一验证和查询语义，并理解 Trait 与普通 C++ helper 的差别。

## 2. 先运行

```bash
cd "$BUDDYGRAPH_LAB_ROOT"
rg -n 'SameOperandsAndResultType|Pure|Terminator|HasParent' \
  include/BuddyGraph/IR/BGraphOps.td
rg -n 'verifyTrait' /buddy-mlir/llvm/mlir/include/mlir/IR/OpDefinition.h | head
```

先确认项目已经使用 upstream Traits，但还没有 `buddy::*` 自定义 Trait。

## 3. 定义 C++ Trait

创建 `include/BuddyGraph/Lab/IR/BGLabTraits.h`：

```cpp
#ifndef BUDDYGRAPH_LAB_IR_BGLABTRAITS_H
#define BUDDYGRAPH_LAB_IR_BGLABTRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace buddy::bglab {

template <typename ConcreteType>
class FirstOperandAndResultSameType
    : public mlir::OpTrait::TraitBase<
          ConcreteType, FirstOperandAndResultSameType> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    if (op->getNumOperands() < 1 || op->getNumResults() != 1)
      return op->emitOpError(
          "requires at least one operand and exactly one result");
    if (op->getOperand(0).getType() != op->getResult(0).getType())
      return op->emitOpError(
          "requires the first operand and result to have the same type");
    return mlir::success();
  }
};

} // namespace buddy::bglab

#endif
```

Trait 是以具体 Op class 为模板参数的 mixin。`verifyTrait(Operation *)` 被 MLIR 的通用
verification 流程调用，不需要每个使用者再写转发函数。

## 4. 定义 ODS Trait record

创建 `BGLabTraits.td`：

```tablegen
#ifndef BUDDYGRAPH_LAB_IR_BGLABTRAITS_TD
#define BUDDYGRAPH_LAB_IR_BGLABTRAITS_TD

include "mlir/IR/OpBase.td"

def BGLab_FirstOperandAndResultSameType
    : NativeOpTrait<"FirstOperandAndResultSameType"> {
  let cppNamespace = "::buddy::bglab";
}

#endif
```

在 `BGLabOps.td` include 它，并把 `MarkOp` 改为：

```tablegen
def BGLab_MarkOp : BGLab_Op<"mark", [
    Pure,
    BGLab_FirstOperandAndResultSameType
  ]> {
  // arguments/results/assemblyFormat 保持不变
}
```

删除 `let hasVerifier = 1` 和 `MarkOp::verify()`；同一不变量只能保留一个权威实现。

## 5. 让 generated class 看见 Trait

在 `BGLabOps.h` 中，把下面 include 放在 `GET_OP_CLASSES` 之前：

```cpp
#include "BuddyGraph/Lab/IR/BGLabTraits.h"
```

TableGen 只会在生成的继承列表中写入 C++ trait 名；它不会替你 include 定义该模板的
头文件。若顺序错误，会看到 incomplete/unknown template 错误。

## 6. 调用链

```text
MarkOp ODS trait list
→ generated MarkOp base list
→ FirstOperandAndResultSameType<MarkOp>
→ Operation::verify()
→ structural traits
→ native trait verifyTrait(Operation *)
→ op/interface verifier（若存在）
```

Trait verifier 适合不依赖某个 mnemonic 的可复用不变量。只有 `mark` 才有的 label、
bounds 或业务语义，仍应留在 `MarkOp::verify()`。

## 7. 正反测试

在 `tests/Tutorial/BGLab/traits.mlir` 写：

```mlir
// RUN: buddygraph-opt --verify-diagnostics --split-input-file %s -o /dev/null

func.func @ok(%x: tensor<2x3xf32>, %tag: !bglab.tag<"ok">) {
  %0 = "bglab.mark"(%x, %tag)
      : (tensor<2x3xf32>, !bglab.tag<"ok">) -> tensor<2x3xf32>
  return
}

// -----

func.func @bad(%x: tensor<2x3xf32>, %tag: !bglab.tag<"bad">) {
  // expected-error@+1 {{requires the first operand and result to have the same type}}
  %0 = "bglab.mark"(%x, %tag)
      : (tensor<2x3xf32>, !bglab.tag<"bad">) -> tensor<3x2xf32>
  return
}
```

使用 generic syntax 能显式写出不一致 result type，适合 negative test。

本章 checkpoint：

```bash
cmake --build build --target MLIRBGLabOpsIncGen BuddyGraphLabIR buddygraph-opt -j2
build/bin/buddygraph-opt --verify-diagnostics --split-input-file \
  tests/Tutorial/BGLab/traits.mlir -o /dev/null
```

第二条成功表示期望错误匹配。若仍报旧 `MarkOp::verify` 定义冲突，检查是否已同时
删除 ODS hook 和 C++ 定义。

## 8. Trait 不等于 Interface

Trait 主要表达可组合的性质，并能注入验证或工具方法。算法可用 `hasTrait` 判断某种
性质，但 Trait 没有“每个 Op 提供不同 method implementation”的虚函数式分派表。

下一章的 Interface 会定义统一方法 `getForwardedValue()`；`identity` 和 `mark` 可以
分别实现它，而 Pass 只依赖 Interface。两者在 ODS 中都写入 trait list，但生成的
C++ 机制与设计意图不同。

## 9. 常见错误

- ODS 写了 `NativeOpTrait`，但 `BGLabOps.h` 没 include C++ template。
- `cppNamespace` 写成 `::mlir::OpTrait`，生成类找不到项目 Trait。
- Trait verifier 假定 operand/result 数量正确却不声明 dependent structural traits；
  本例主动检查数量，所以不依赖该假定。
- Op verifier 与 Trait verifier 保留重复诊断，未来只更新其中一个。
- 把只会用于一个 Op 的复杂业务规则强行抽成 Trait。

## 10. 验收标准

- 删除 `MarkOp::verify()` 后，negative test 仍给出 Trait 诊断。
- generated `MarkOp` 继承列表包含项目 Trait。
- 能各举一个适合 Trait、Op verifier、Interface 的规则。
- 能解释 `Pure` 为什么是 upstream Trait，而不是一句 optimizer hint 字符串。

## 11. 面试追问

**问：为什么不继续调用一个普通 `verifyForwardingTypes(op)` helper？**

答：helper 只能复用函数体，每个 Op 仍需声明/实现 verifier 并显式调用。Trait 把性质
加入 Op 的静态组成，verification 与通用 `hasTrait` 查询都能识别它；代价是更强的
schema 承诺，所以不应用于偶然相同的规则。

下一章：[定义并实现 custom OpInterface](05_custom_interface.md)。
