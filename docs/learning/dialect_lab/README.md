# 自定义 Dialect 基础设施实战线

这条专题线补足主课程中“能读懂 BuddyGraph，但还不能从零接起 MLIR 基础设施”的部分。
它不再以阅读题为主，而是在 BuddyGraph 的**实验副本**中连续实现一个隔离的 `bglab`
教学 Dialect：

```text
Dialect
→ Op
→ parameterized Type
→ reusable Trait
→ custom OpInterface
→ interface-driven Pass
→ lit / FileCheck / negative diagnostics
```

最终可处理下面这段 IR：

```mlir
%tag = bglab.make_tag : !bglab.tag<"frontend">
%0 = bglab.identity %arg0 : tensor<2x3xf32>
%1 = bglab.mark %0, %tag
    : (tensor<2x3xf32>, !bglab.tag<"frontend">) -> tensor<2x3xf32>
```

运行 `--bglab-strip-forwarders` 后，三条教学 Op 都被消除，函数直接返回 `%arg0`。

## 事实边界

当前 BuddyGraph 基线已经有自定义 `bgraph` Dialect、ODS Op、EnumAttr、C++ verifier、
canonicalization、Pass 和 Conversion；它仍复用 builtin tensor type，尚未定义自定义
Type、项目自有 Trait 或项目自有 OpInterface。专题中的 `bglab` 是**学习者要实现的
实验扩展**，不是当前基线能力，也不应写进 BGraph 的 ONNX 支持范围。

本轮教程完善只修改 Markdown 教程与导航，没有把 `bglab` 实现预先写进工程，也没有
为文档改动重新构建项目。学习者执行专题时才会修改自己的实验副本并逐 checkpoint
构建。

## 章节顺序

| 章节 | 产物 | 关键问题 |
|---|---|---|
| [00](00_lab_setup.md) | 可恢复实验副本与证据目录 | 修改谁、构建谁、如何回滚？ |
| [01](01_custom_dialect.md) | 空的 `bglab` Dialect | TableGen、CMake、library、driver 如何连起来？ |
| [02](02_custom_op.md) | `bglab.identity` | ODS 生成了什么，parser 如何找到 Op？ |
| [03](03_custom_type.md) | `!bglab.tag<"...">`、`make_tag`、`mark` | Type 如何唯一化、解析、注册和进入 SSA？ |
| [04](04_custom_trait.md) | `FirstOperandAndResultSameType` | 何时写 Trait，何时只写 Op verifier？ |
| [05](05_custom_interface.md) | `ForwardingOpInterface` | 通用算法如何调用多个 Op 的不同实现？ |
| [06](06_custom_pass.md) | `--bglab-strip-forwarders` | Pass 声明、实现、注册和 IR mutation 如何闭环？ |
| [07](07_integration_and_debugging.md) | lit 正反测试与故障注入 | 如何证明不是“能编译但没接上”？ |

不要跳过 00。后续章节是**累积修改**：第 03 章建立在第 02 章之上，第 06 章需要前面
所有产物。

## 学习方式

每章分成两条路径：

- “只读路径”用于理解当前 BuddyGraph 中对应的真实机制，不修改源码。
- “带做路径”在 `BUDDYGRAPH_LAB_ROOT` 中新增 `bglab`，每次只构建当前 checkpoint。

每完成一章，至少保留四项证据：修改文件清单、TableGen/C++ target 输出、最小 IR、
一个失败案例。只看到 `ninja: no work to do` 或程序退出码为 0，不足以证明注册和语义
都正确。

## 最终目录

专题完成后的新增目录应为：

```text
include/BuddyGraph/Lab/
├── CMakeLists.txt
├── IR/
│   ├── BGLabDialect.h
│   ├── BGLabDialect.td
│   ├── BGLabInterfaces.h
│   ├── BGLabInterfaces.td
│   ├── BGLabOps.h
│   ├── BGLabOps.td
│   ├── BGLabTraits.h
│   ├── BGLabTraits.td
│   ├── BGLabTypes.h
│   ├── BGLabTypes.td
│   └── CMakeLists.txt
│
└── Transforms/
    ├── CMakeLists.txt
    ├── Passes.h
    └── Passes.td

lib/BuddyGraph/Lab/
├── CMakeLists.txt
├── IR/
│   ├── BGLabDialect.cpp
│   ├── BGLabInterfaces.cpp
│   ├── BGLabOps.cpp
│   ├── BGLabTypes.cpp
│   └── CMakeLists.txt
└── Transforms/
    ├── CMakeLists.txt
    └── StripForwarders.cpp
```

测试放入 `tests/Tutorial/BGLab/`。生成的 `.inc` 只存在于 `build/include/BuddyGraph/Lab/`，
永远不要手工编辑或复制回 source tree。

## 完成标准

最终你应能脱离教程回答并演示：

1. Dialect 的 TableGen record、C++ class、Context 实例分别是什么。
2. 一个 `.td` 修改会生成哪些 `.inc`，由哪个 CMake target 保证依赖。
3. Type、Attribute、Value 与 Operation 的关系。
4. Trait 和 Interface 都出现在 ODS trait list 中，但它们解决的问题为何不同。
5. Pass 为何不是 Dialect 的成员函数，CLI 名又如何被注册。
6. parser error、generated verifier、Trait verifier、Op verifier、Pass failure 如何区分。

完成本专题后再回到[主课程第 14 章](../14_extension_lab.md)，新增业务 Op 时会更容易看清
“基础设施接线”和“算子语义闭环”是两组不同工作。
