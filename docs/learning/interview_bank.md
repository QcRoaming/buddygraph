# 面向简历四条陈述的深入追问

先读[简历对齐导读](resume_alignment.md)。本题库用于模拟专业面试中的逐层追问，
并不预测特定公司的题目。每题先脱离答案口述，再打开指定源码验证。

回答顺序建议是：**作用 → 本项目实现 → 一个具体例子 → 失败边界 → 验证方法**。
“用了 MLIR，所以自动正确/自动更快”不能完成这条回答链。

## 一、方言、ODS 与工程接入

### 1. 你为什么要有 BGraph 层，直接 ONNX → Linalg 不可以吗？

可以直接转换，但 BGraph 保留 Conv、BN、广播等图语义，便于在降低到迭代空间之前
做领域优化和给出节点级诊断。代价是维护 schema、verifier、优化和 conversion。
追问时展示 BN Pattern 是如何直接识别 Conv producer 的；不要声称直接转换在原理上不可能。

### 2. 你具体自定义了哪个 Type？

当前业务 BGraph 没有 TypeDef；`tensor<2x3xf32>` 属于 builtin dialect。项目自定义了
Layout EnumAttr。`bglab.tag` 是教程实验，完成后可另列实验实现。
若继续问什么时候需要 custom Type，可举“builtin 类型无法表达的新语义或资源类型”，
然后说明 storage uniquing、参数、parse/print、注册和 lowering 都要负责。

### 3. `.td` 中写完 Relu 后，为什么工具还不一定能解析它？

还需要 TableGen target、生成声明/定义的 include、IR library 链接、driver registry 和
Dialect `addOperations`。能编译 library 只覆盖构建链，不能证明运行时注册完成。
现场画出第 03/04 章的两条链，并指出 `MLIRBGraphOpsIncGen` 与 `GET_OP_LIST`。

### 4. `hasVerifier = 1` 自动生成了哪些语义？

它声明手写 verifier hook；ODS 同时根据 schema 生成结构检查。f32、channel、broadcast
等语义来自 `BGraphOps.cpp`。追问：输入是 ranked i32 tensor，为什么还要被拒绝？
因为 `AnyRankedTensor` 没有约束 element type。用 `invalid.mlir` 定位具体诊断。

### 5. Trait 和 Interface 有什么区别？

Trait 为 Op 组合性质、约束或方法，例如数量与类型关系；Interface 为跨 Op 算法定义
可查询的行为，并分派到具体实现。二者都能出现在 ODS trait list，不能据此视为同一机制。
用实验中的 `FirstOperandAndResultSameType` 和 `ForwardingOpInterface` 各追一遍调用链。
若只会列名词，继续追问 generated Concept/Model 如何调用 ConcreteOp method。

### 6. Pure 是否意味着任何涉及该 Op 的优化都是合法的？

否。Pure 组合内存副作用与可推测执行性质，帮助通用删除/移动分析；它不证明两个浮点
表达式等价。Add-zero 的 signed-zero 反例说明即使 Op 无副作用，某条替换也可能不合法。

### 7. 为什么 round-trip 测试不足以证明 lowering 正确？

round-trip 验证表示能解析、验证、打印；它没有运行 scalar body 或检查访问 map。
还需要 pattern 正负例、conversion legality、bufferization 和 runner 对拍。一个把
`SubFOp` 错写成 `AddFOp` 的 lowering 仍可能全部通过 parse/print。

## 二、ONNX 前端与 shape

### 8. node.name、node.output 和 MLIR `%0` 分别是什么？

node.name 是可选的节点标签，本项目用于 Location；node.output 是数据流 value name，
用于 `self.values`；`%0` 是 printer 给 Value 的文本名字。改 node.name 不应改变数据边。
现场以 `add_0: x,bias → sum` 写出 map 更新，避免把 `self.values["add_0"]` 当输出。

### 9. initializer 为什么有时是 SSA 常量，有时是 attribute？

filter/bias 是数据操作数，float initializer 由 `arith.constant` 定义；Reshape shape、
ReduceMean axes 与受限 Clip bounds 是编译期元数据，由对应 helper 转成 attribute。
追问共享常量：同一 Clip bound 也用于 Add 时，要按当前 node 排除 metadata operand，
不能全局从所有输入中删除该名字。源码是 `_node_attribute_initializers`。

### 10. 为什么用了 MLIR Python API，仍允许 unregistered dialect？

本地 Python package 没有 BGraph 生成 binding，因此通过 generic Operation API 构造。
允许创建未知 Op 不能执行 C++ BGraph verifier；生成物还要交给 buddygraph-opt。
API 的价值是维护 Value/Type/Region/Location 等对象关系，并不等于前端获得所有方言语义。

### 11. 静态 shape 是你推导的还是 ONNX 推导的？

两层职责要分开：`_validate_model` 调 ONNX inference，`_tensor_type` 读取结果；
`InferShapes.cpp` 是项目内 Module pass，从 BGraph operands/attrs refine result type。
没有 BGraph `InferTypeOpInterface` 实现，也没有完整动态数据流求解器。

### 12. `[2,1,4]` 与 `[3,4]` 的 Add 输出是什么？怎样读一个元素？

右对齐为 `[2,1,4]` 与 `[1,3,4]`，结果 `[2,3,4]`。结果 `(i,j,k)` 读取 lhs `(i,0,k)`
与 rhs `(j,k)`。必须同时给 shape 与 indexing maps，才说明把前端 shape 和后端执行接上了。
把第二个输入改成 `[3,5]` 应在最后一维冲突，不是自动截断或 reshape。

### 13. Conv 输出公式里的 dilation 和 pads 分别影响哪里？

有效 kernel 是 `dilation*(kernel-1)+1`；输出是
`floor((input+pad_before+pad_after-effective_kernel)/stride)+1`。
用第 06 章数字逐步手算，再说明 groups/channel/layout 检查不由空间公式替代。
kernel 放不下或算术溢出要诊断失败，不应造负维触发 assertion。

### 14. 支持 opset 18 是否意味着支持其中所有 Clip 形式？

不意味着。当前 Clip 要同时提供两个 finite scalar f32 initializer bounds，且 min≤max。
optional/runtime/Constant-node bounds 不在范围；前端接受的 shape/dtype/layout 同样受限。
专业回答应同时交代“协议版本”和“每个 Op 的子集契约”。

## 三、图优化与浮点正确性

### 15. Conv-BN folding 的公式从哪里来？

把 Conv 输出代入推理态 BN，再按输出 channel 提取
`alpha=gamma/sqrt(var+epsilon)`，得 `W'=alpha*W`、`b'=alpha*(b-mean)+beta`。
没有 Conv bias 时用零。必须解释 W 的哪一维对应输出 channel，以及参数为何需为常量。
手算示例见第 08 章，不能只说“把 BN 融进权重”。

### 16. Conv 有两个用户时，为什么不能直接删掉它？

另一个用户仍需要旧 Conv 结果；原地改 W/b 也会改变那条边的语义。当前 Pattern 要
`hasOneUse()`，不满足便保持不变。可以设计克隆新 Conv 的方案，但会重复计算，需要成本
权衡；不能把该方案说成当前已实现。

### 17. 数学公式相等，为什么优化前后浮点仍可能不同？

浮点每步舍入，先改权重再卷积改变乘加与舍入顺序；实数分配律不等于逐 bit 恒等。
当前固定用例的误差是有限测试证据，不是所有输入的证明。另举 Add-zero 对 `-0.0`
的反例，说明 NaN、signed zero、overflow 和 fast-math contract 都需单独分析。

### 18. `matchAndRewrite` 返回 failure 前能不能先创建几个 Op？

Pattern 必须遵守 driver 的改写纪律，最好先检查再修改。BN 先计算再创建替代节点；
fusion 在构造 scalar body 失败时显式清理临时 fused Op。不能把普通 PatternRewriter
想成自动回滚数据库。成功后通过 rewriter 替换 uses，再删除无用户旧节点。

### 19. 你的 fusion 是怎样选子图的？

可融合 Op 白名单加无副作用检查；root 不能还有可融合 user；从 root 反向收集，只有
single-use 可融合 producer 才进入内部，否则 operand 成为 leaf；内部至少两个 Op。
给一张有共享 producer 的图，指出“停止沿这条边收集”不等于“整个图一律不融合”。

### 20. 为什么 Region 参数是 f32，外层 operand 却是 tensor？

外层 tensor 描述全体元素；Region body 描述一个迭代点的 scalar 计算。后续 Linalg
indexing maps 提供该点的各输入元素。`IRMapping` 连接外层 leaves 和 scalar arguments；
不是在 C++ 中逐元素复制 tensor 数据。

### 21. fusion 阶段与 lowering 阶段各自是否在 clone？

图 fusion 用 `emitScalar` 重建 arith 运算，不原样 clone tensor BGraph Op。
`FusedElementwiseLowering` 才 clone 已有 scalar body，并用新的 argument mapping 更新
uses；`bgraph.yield` 单独转换为 `linalg.yield`。两次映射的源/目标 Value 不一样。

### 22. CSE 合并的依据是什么，为什么还要考虑 dominance？

不能只看 op name；还需 operands、types、attributes 等等价条件和副作用信息。
替代结果必须支配原来的使用，不能拿另一条不可达分支中的结果替换。项目复用 upstream
CSE，而不是实现了一个全图 hash 表就解决所有控制流。Location 通常不作为计算差异。

### 23. fusion 减少了 allocation，是否就证明更快？

减少中间 tensor materialization 是可能收益；真正性能还受访存、缓存、寄存器压力、
向量化、重复计算和 kernel 形状影响。当前 runner timing 含进程/parse/JIT，不能等同
稳态 kernel 时间。给出实际测量范围和 ablation 设计，比承诺固定倍数更可信。

## 四、Conversion、Bufferization 与执行

### 24. FullConversion 比 PartialConversion 多保证什么？

Full 要求目标范围内所有 Op 都合法；Partial 允许某些未显式标非法的 unknown Op 保留。
两者都必须消除显式 illegal Op。本项目整个 BGraph illegal，所以即使用 Partial 也不能
合法保留它；选 Full 还会拒绝未声明合法的其他 Op。不要把区别简化成“Partial 随便留 BGraph”。

### 25. FullConversion 成功后一定算对了吗？

没有。legality 检查表示层边界；错误 indexing map 或错用加减都可能生成合法目标 IR。
通过 verifier/结构测试证明形状与操作关系，再用参考实现对拍证明覆盖输入上的数值行为。

### 26. 这里为什么不用 TypeConverter，OpAdaptor 又有什么用？

BGraph→Linalg 保持 builtin tensor/f32，所以没有新增类型映射规则。OpAdaptor 给 pattern
提供 conversion driver 已映射的 operands；不能把它理解为另一份 tensor 数据。
若引入 custom type 或改变函数签名，则还要处理 TypeConverter、signature conversion、
materialization 以及上下游一致性；当前代码未实现这些扩展。

### 27. 一个 Linalg generic 的四个核心部分是什么？

迭代域、每个 operand/output 的 indexing map、parallel/reduction iterator 类型、scalar
body/yield。以 `[2,3]+[3]` 为例，lhs/output `(i,j)→(i,j)`，rhs `(i,j)→(j)`，两维
parallel，body 执行 addf。输出 destination 的 body 参数存在，但 elementwise 覆写无需读取。

### 28. ReduceMean 为什么不能直接使用全 parallel iterators？

被归约的轴有多个输入贡献到同一输出元素，须标 reduction 并初始化 accumulator。
项目先求和，再按静态归约元素数做除法；keep_dims 决定 output map 保留为 0 的轴还是
省略这些轴。让面试者画 `[2,3,4]` 沿 `[0,2]` 的访问关系，比背 API 名更能验证理解。

### 29. `tensor.empty` 是零初始化吗？

不是，它只提供 shape/type 的未定义初值 destination。elementwise body 不读旧输出，
每点覆写；sum reduction 和 Conv 累加必须先 fill 或初始化 bias。把 empty 当零会产生
合法但数值不确定的 IR。

### 30. One-Shot Bufferize 为什么不能把所有 tensor 都原地变成一个 buffer？

旧 tensor Value 的后续读取仍应看到旧内容。若复用同一 buffer 的写入破坏这些读取，
就有读写冲突，需 out-of-place/copy。DPS 提供候选 destination，分析结合 alias 与
read/write/相应 interface 决定是否安全；它不是零拷贝保证。

### 31. MLIRContext 注册 Dialect 后，为什么 Bufferization 还报 interface 缺失？

认识 Op 的 parser/IR class 与提供 BufferizableOpInterface model 是不同注册需求。
项目 driver 显式注册标准 Dialect 的 external models，BGraph 则先由 FullConversion
消除。定位 `registerBufferizableOpInterfaceExternalModels`，不要靠允许未知 Op 绕过。

### 32. MemRef 为什么不是一个裸指针？

还需要区分 allocated/aligned pointer、offset、各维 size/stride，才能处理布局、子视图
与寻址。静态信息可能在降低/优化后被常量化，但一般 memref ABI 不能直接假设 `float*`。
当前 runner 通过 rank-0 tensor.extract 返回标量，规避了通用 tensor-return wrapper 问题。

### 33. LLVM Dialect、LLVM IR 与 runner 分别做什么？

LLVM Dialect 仍由 MLIR 表示和验证；translator 导出 LLVM IR；runner 内部接续 translation
和 JIT 并按入口 ABI 调用。生成 `.llvm.mlir` 不等于生成了独立可执行程序。
当前 pipeline 也没有完整 ownership deallocation，短进程结束不能证明长期服务无泄漏。

## 五、综合压力测试

### 34. 现场让你新增一个 Op，你怎样拆任务？

先定义 dtype/shape/attributes/数值契约，决定是否真的需要新 Type/Interface；再做 ODS、
verifier、shape、直接 lowering、前端和最小测试；有明确需求才做 canonicalization/fusion。
每层给一个拒绝反例。用已经实现的 Clamp 对照，但实际新增内容要在自己的实验副本完成。

### 35. 优化前后误差不一致，你先看哪里？

同一输入固定 seed 和 pipeline，逐个开关优化找第一个差异，保存对应 before/after IR；
检查参数、dtype、maps、body scalar 语义和浮点边界；缩成一个可复现输入，加入回归。
“第一次触发差异的 pass”是定位线索，仍需检查其输出是否合法、下游是否错误处理。

### 36. 哪部分是你最有把握的个人贡献？

回答一条自己能独立复现的具体链，包含设计选择、一次失败、怎么定位、怎么证明修复。
不要把标准 MLIR pass 的所有内部能力都算作自己实现，也不需要虚构缺陷发现经历。
可以诚实选择“我在实验副本完成的扩展”，并展示 diff 和测试记录。

## 一次模拟面试怎样评分

先做 30 秒概览，再由对方从四组各随机取两题，最后做一道白板推导和一道失败定位。
单题可按 0–3 分记录：0=术语混淆；1=说清概念；2=落到本项目源码和反例；3=能推导或
现场验证。分数是学习反馈，不是招聘录用预测。任一主线长期只有 0–1 分，就回对应章节
补实验，而不是只重复口述答案。

完成一次模拟后记录：题号、自己的回答、漏掉的条件、源码证据、下一次可验证的改进。
面试追问的目标是让你的陈述经得住检查，不是要求背诵全部 MLIR 源码。
