# 06｜定义 custom Pass：通过 Interface 消除 forwarding Op

## 1. 本章目标

实现 `--bglab-strip-forwarders`：Pass 只查询 `ForwardingOpInterface`，替换 forwarding
Op 的 result uses，删除 Op，并清理无用户的 `make_tag`。同时追通 Pass TableGen、base
class、factory、registration 和 CLI。

## 2. 先运行

```bash
cd "$BUDDYGRAPH_LAB_ROOT"
build/bin/buddygraph-opt --help | rg 'bgraph-|bglab-' || true
rg -n 'GEN_PASS_DEF_|GEN_PASS_REGISTRATION|registerPasses' \
  include/BuddyGraph/Transforms lib/BuddyGraph/Transforms \
  tools/buddygraph-opt
```

此时应只有既有 `bgraph-*` passes，没有 `bglab-strip-forwarders`。

## 3. 声明 Pass

创建 `include/BuddyGraph/Lab/Transforms/Passes.td`：

```tablegen
#ifndef BUDDYGRAPH_LAB_TRANSFORMS_PASSES_TD
#define BUDDYGRAPH_LAB_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def BGLabStripForwarders
    : Pass<"bglab-strip-forwarders", "::mlir::ModuleOp"> {
  let summary = "Remove bglab forwarding annotations";
  let constructor = "::buddy::bglab::createStripForwardersPass()";
  let dependentDialects = ["::buddy::bglab::BGLabDialect"];
}

#endif
```

创建 `Passes.h`：

```cpp
#ifndef BUDDYGRAPH_LAB_TRANSFORMS_PASSES_H
#define BUDDYGRAPH_LAB_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace buddy::bglab {

std::unique_ptr<mlir::Pass> createStripForwardersPass();

#define GEN_PASS_REGISTRATION
#include "BuddyGraph/Lab/Transforms/Passes.h.inc"

} // namespace buddy::bglab

#endif
```

## 4. 生成 Pass base 和注册函数

创建 `include/BuddyGraph/Lab/Transforms/CMakeLists.txt`：

```cmake
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc --gen-pass-decls)
add_public_tablegen_target(BGLabPassesIncGen)
```

在 `include/BuddyGraph/Lab/CMakeLists.txt` 中同时保留：

```cmake
add_subdirectory(IR)
add_subdirectory(Transforms)
```

`GEN_PASS_DEF_BGLABSTRIPFORWARDERS` 选择单个 base class definition；
`GEN_PASS_REGISTRATION` 选择全部 CLI registration helpers。二者来自同一生成文件，但
include 位置和职责不同。

## 5. 实现 Pass

创建 `lib/BuddyGraph/Lab/Transforms/StripForwarders.cpp`：

```cpp
#include "BuddyGraph/Lab/Transforms/Passes.h"

#include "BuddyGraph/Lab/IR/BGLabInterfaces.h"
#include "BuddyGraph/Lab/IR/BGLabOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace buddy::bglab {
#define GEN_PASS_DEF_BGLABSTRIPFORWARDERS
#include "BuddyGraph/Lab/Transforms/Passes.h.inc"

namespace {

class StripForwardersPass final
    : public impl::BGLabStripForwardersBase<StripForwardersPass> {
public:
  using Base::Base;

  void runOnOperation() final {
    SmallVector<Operation *> forwarders;
    getOperation().walk([&] (Operation *operation) {
      if (isa<ForwardingOpInterface>(operation))
        forwarders.push_back(operation);
    });

    for (Operation *operation : forwarders) {
      auto forwarding = cast<ForwardingOpInterface>(operation);
      if (operation->getNumResults() != 1 ||
          operation->getNumRegions() != 0 || !isMemoryEffectFree(operation)) {
        operation->emitError("expected one result, no regions and no memory effects");
        signalPassFailure();
        return;
      }
      Value replacement = forwarding.getForwardedValue();
      if (!replacement ||
          !llvm::is_contained(operation->getOperands(), replacement) ||
          replacement.getType() != operation->getResult(0).getType()) {
        operation->emitError("expected an existing operand matching the result type");
        signalPassFailure();
        return;
      }
      operation->getResult(0).replaceAllUsesWith(replacement);
      operation->erase();
    }

    SmallVector<MakeTagOp> deadTags;
    getOperation().walk([&] (MakeTagOp op) {
      if (op->use_empty())
        deadTags.push_back(op);
    });
    for (MakeTagOp op : deadTags)
      op.erase();
  }
};

} // namespace

std::unique_ptr<Pass> createStripForwardersPass() {
  return std::make_unique<StripForwardersPass>();
}

} // namespace buddy::bglab
```

先收集、后修改，避免在普通 walk callback 中删除当前 Operation 导致遍历失效。Pass
也检查 Interface 返回值非空、来自现有 operand、类型相同且 Op 无 Region/内存副作用。
已验证 SSA 的 operand 在这里可见，删除 Op 后仍有效；语义等价本身仍是 Interface
实现者的承诺，不能仅由类型检查证明。这些防御不替代 Trait，也不使失败成为自动事务回滚。

## 6. 构建 Transform library

创建 `lib/BuddyGraph/Lab/Transforms/CMakeLists.txt`：

```cmake
add_mlir_library(BuddyGraphLabTransforms
  StripForwarders.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/BuddyGraph/Lab

  DEPENDS
  BGLabPassesIncGen
  MLIRBGLabInterfacesIncGen
  MLIRBGLabOpsIncGen

  LINK_LIBS PUBLIC
  BuddyGraphLabIR
  MLIRIR
  MLIRPass
  MLIRSideEffectInterfaces
)
```

在 `lib/BuddyGraph/Lab/CMakeLists.txt` 加：

```cmake
add_subdirectory(IR)
add_subdirectory(Transforms)
```

## 7. 注册 CLI Pass

在 `buddygraph-opt.cpp` include：

```cpp
#include "BuddyGraph/Lab/Transforms/Passes.h"
```

在 `main()` 中、调用 `MlirOptMain` 前加入：

```cpp
buddy::bglab::registerPasses();
```

并把 `BuddyGraphLabTransforms` 加入 driver link libraries。构建后核对：

```bash
cmake --build build --target \
  BGLabPassesIncGen BuddyGraphLabTransforms buddygraph-opt -j2
build/bin/buddygraph-opt --help | rg 'bglab-strip-forwarders'
```

Dialect 注册解决 parser/type/op；Pass 注册解决 CLI pipeline name。任何一边都不能替代
另一边。

## 8. IR 前后变化

把下面完整输入保存为实验副本的 `tests/Tutorial/BGLab/strip-input.mlir`：

```mlir
// RUN: buddygraph-opt %s -o /dev/null
func.func @strip(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %tag = bglab.make_tag : !bglab.tag<"frontend">
  %0 = bglab.identity %x : tensor<2x3xf32>
  %1 = bglab.mark %0, %tag
      : (tensor<2x3xf32>, !bglab.tag<"frontend">) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}
```

运行：

```bash
build/bin/buddygraph-opt --bglab-strip-forwarders tests/Tutorial/BGLab/strip-input.mlir
```

输出核心应为：

```mlir
func.func @strip(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  return %x : tensor<2x3xf32>
}
```

Pass 先将 `%0` 的 users 改用 `%x`，再将 `%1` 的 users 改用其 forwarding value，最后
删除无用户 `%tag`。它没有读取 op name 字符串。

## 9. 常见错误

- 只定义 factory，没生成/调用 registration：`--help` 看不到 CLI 名。
- Pass 在解析后才声明 dependent Dialect，却没在 driver registry 注册 Dialect；输入
  parse 仍会先失败。
- 边 walk 边 erase，出现 iterator invalidation。
- `erase()` 前没有替换 result uses，触发“operation with uses”断言。
- Interface method 返回错误类型，Pass 盲目替换造成非法 IR。
- 把 `make_tag` 无条件删除，仍有非-forwarding user 时留下悬空 use。

## 10. 验收标准

- `--help` 能找到唯一的 `bglab-strip-forwarders`。
- Pass 后无 `bglab.identity`、`bglab.mark` 和死 `make_tag`。
- Pass 源码没有 `isa<IdentityOp, MarkOp>` 或 op name 比较。
- 删除 driver 中 `registerPasses()` 后，可解释“library 能构建但 CLI 未知”。
- 能说明 `signalPassFailure()` 与 `notifyMatchFailure()` 的语义差别。

## 11. 面试追问

**问：为什么不用 canonicalization 自动删除这些 Op？**

答：identity 本身可做 canonicalization，但本专题刻意实现显式 Pass 来展示 TableGen
声明、CLI 注册和 Interface 消费。生产设计中若规则始终安全且局部，应优先考虑
canonicalization；若需要显式阶段、策略或跨 Op 协调，独立 Pass 更合适。

下一章：[把全部基础设施接入 lit 并做故障注入](07_integration_and_debugging.md)。
