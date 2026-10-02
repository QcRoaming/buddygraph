# 07｜集成测试与分层调试

## 1. 本章目标

为 `bglab` 建立 round trip、Type、Trait negative diagnostic、Interface-driven Pass 四层
证据，并能根据错误发生阶段定位 TableGen、CMake、registry、verification 或 Pass。

## 2. 先运行

```bash
cd "$BUDDYGRAPH_LAB_ROOT"
cmake --build build --target buddygraph-opt -j2
/usr/bin/python3.10 /buddy-mlir/llvm/build/bin/llvm-lit -sv \
  --filter='Tutorial/BGLab/' build/tests
```

lit 从 build/tests 的 site config 找到 source tests，再按名称筛选新目录；无需把 source
`.mlir` 复制到 build/tests，也无需为了发现新 `.mlir` 文件重新 configure。

## 3. 最小测试集合

前面章节已有 ops/types/traits 等 checkpoint 文件；继续保留它们，并新增下面三个文件
汇总集成边界。先分别保存第 4–6 节代码块，再运行第 2 节命令：

```text
tests/Tutorial/BGLab/
├── roundtrip.mlir
├── invalid.mlir
└── strip-forwarders.mlir
```

`roundtrip.mlir` 检查 Dialect、custom Type、custom assembly 与 Op registration；
`invalid.mlir` 检查 Trait diagnostic；`strip-forwarders.mlir` 同时检查 Interface dispatch
和 Pass registration。

## 4. Round-trip 测试

```mlir
// RUN: buddygraph-opt %s | FileCheck %s

// CHECK-LABEL: func.func @roundtrip
// CHECK: %[[TAG:.*]] = bglab.make_tag : !bglab.tag<"frontend">
// CHECK: %[[ID:.*]] = bglab.identity %arg0 : tensor<2x3xf32>
// CHECK: bglab.mark %[[ID]], %[[TAG]]
func.func @roundtrip(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %tag = bglab.make_tag : !bglab.tag<"frontend">
  %0 = bglab.identity %arg0 : tensor<2x3xf32>
  %1 = bglab.mark %0, %tag
      : (tensor<2x3xf32>, !bglab.tag<"frontend">) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}
```

不要使用只有 `CHECK: bglab` 的宽泛断言；它无法区分哪个 Op/Type 没有 round trip。

## 5. Negative diagnostic 测试

```mlir
// RUN: buddygraph-opt --verify-diagnostics %s -o /dev/null

func.func @bad(%x: tensor<2x3xf32>, %tag: !bglab.tag<"bad">) {
  // expected-error@+1 {{requires the first operand and result to have the same type}}
  %0 = "bglab.mark"(%x, %tag)
      : (tensor<2x3xf32>, !bglab.tag<"bad">) -> tensor<3x2xf32>
  return
}
```

这里用 generic syntax 显式展示不同 result type；mark 的 custom format 也会显式打印
functional type，并没有从 input 自动推导出相同 result。测试命令成功表示 expected
diagnostic 匹配，不表示非法 IR 被接受。

## 6. Pass 测试

```mlir
// RUN: buddygraph-opt --bglab-strip-forwarders %s | FileCheck %s

// CHECK-LABEL: func.func @strip
// CHECK-NOT: bglab.
// CHECK: return %arg0 : tensor<2x3xf32>
func.func @strip(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %tag = bglab.make_tag : !bglab.tag<"pass-test">
  %0 = bglab.identity %arg0 : tensor<2x3xf32>
  %1 = bglab.mark %0, %tag
      : (tensor<2x3xf32>, !bglab.tag<"pass-test">) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}
```

`CHECK-NOT` 的范围从上一个正匹配延伸到下一个正匹配或文件结尾；把函数 label 和 return
作为边界，避免检查意外跨函数。

## 7. 错误分层表

| 症状 | 最可能阶段 | 首查位置 |
|---|---|---|
| `Unknown def BGLab_*` | TableGen record/include | `.td` include 顺序和拼写 |
| 缺少 `*.inc` | CMake generation | `MLIRBGLabOpsIncGen`/Interface/Pass target |
| C++ unknown `TagType` | generated include 顺序 | `BGLabTypes.h` 是否早于 Op classes |
| undefined `getForwardedValue` | link | ConcreteOp method definition |
| unknown dialect/type/op | runtime registry | driver `registry.insert`、`addTypes/addOperations` |
| unknown pass argument | pass registration | `registerPasses()` 与 transforms link |
| parse 成功后 verifier error | IR legality | generated/Trait/Op/Interface verifier |
| Pass 运行但 IR 不变 | matching/dispatch | `isa<ForwardingOpInterface>` 与注册的 Model |
| erase/use assertion | IR mutation | replace uses 与 erase 顺序 |

不要从最终链接错误倒推 ODS 语义；先确定错误属于哪一层。

## 8. 三个故障注入

按顺序做，每次只破坏一处并恢复：

1. 将输入的 `!bglab.tag` mnemonic 改成 `!bglab.typo`：Type parser 报未知类型名。
   若另做删除 `addTypes` 的实验，预期是已识别 mnemonic 后构造未注册 Type storage
   失败，assertions 构建可能 fatal；不要把这条错误也解释成未知 mnemonic。
2. 保留 transforms library，删除 driver 的 `bglab::registerPasses()`：IR 可 parse，
   `--bglab-strip-forwarders` 变成 unknown argument。
3. 从 `MarkOp` 移除 Interface trait，同时删除 `MarkOp::getForwardedValue()` 定义，
   重新构建 driver。Op 仍合法，但 Pass 不再把它加入 forwarders；
   `CHECK-NOT: bglab.` 抓到行为回归。

如果第 3 项只删 ODS trait 却保留方法定义，generated 声明消失，会先在 C++ 编译阶段
失败，观察不到运行时 dispatch 的变化。恢复时也要同时恢复这两处。

每次记录“仍然成功的层”和“第一次失败的层”，比只保存错误末行更有价值。

## 9. 生成依赖审计

```bash
ninja -C build -t query MLIRBGLabOpsIncGen
ninja -C build -t query MLIRBGLabInterfacesIncGen
ninja -C build -t query BGLabPassesIncGen
ninja -C build -t query BuddyGraphLabIR
ninja -C build -t query BuddyGraphLabTransforms
```

再检查 generated 文件只在 build tree：

```bash
find include lib -name '*.inc' -print
find build/include/BuddyGraph/Lab -name '*.inc' -print | sort
```

第一条应无输出，第二条应列出 Dialect/Op/Type/Interface/Pass 生成物。

## 10. 最终验收标准

- 三个集成测试及此前各章 checkpoint 测试全部通过；不把总数固定为三个。
- `buddygraph-opt --help` 同时列出既有 `bgraph-*` 和新 `bglab-*` pass。
- Type 与 Op 均能 custom/generic syntax round trip。
- Trait negative diagnostic 稳定且落在目标 Op 行。
- Pass 不枚举 Identity/Mark 具体类，且 after IR 无 `bglab.`。
- 原 BuddyGraph tests 仍全部通过；新增测试会使总数大于基线 14。
- 能独立新增第四个 forwarding Op，完成 schema/注册并实现 Interface 契约后被 Pass
  处理；返回值要非空、同类型、为现有 operand，且 Op 可安全删除。

最终才运行一次全量回归：

```bash
export PYTHONPATH="/home/jlq/project/buddygraph/.deps${PYTHONPATH:+:$PYTHONPATH}"
cmake --build build --target check-buddygraph -j2
```

实验副本在第 00 章排除了 `.deps`；上面显式复用基线 Python 依赖。lit 配置还会加入
已有 MLIR Python bindings 路径。这里只在学习者实际改源码后做全量验收。

## 11. 面试追问

**问：如何证明新 Dialect 不是“只在 TableGen 中存在”？**

答：需要跨层证据：generated class 存在、library 链接、driver registry 加载、文本 IR
round trip、verifier 正反例、Pass CLI 注册和变换结果。任何单层成功都不能替代完整闭环。

返回[专题入口](README.md)，或回到[主课程](../README.md)。
