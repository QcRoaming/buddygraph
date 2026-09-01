# BuddyGraph / MLIR 术语表

| 术语 | 在本项目中的准确含义 | 真实入口 |
|---|---|---|
| ONNX `ModelProto` | 文件级 protobuf，包含 graph 和 opset imports | `_validate_model()` |
| ONNX value name | 字符串形式的 def-use 标识，不是 MLIR SSA 对象 | `BGraphImporter.values` |
| SSA `Value` | Op result 或 block argument 的轻量句柄 | `Value`、`BlockArgument` |
| `Operation` | 运行时统一 IR 节点；生成的 `ReluOp` 等是 typed wrapper | `BGraphOps.h.inc` |
| Dialect | 一组 op/attribute 的命名空间和注册单元 | `BGraphDialect` / `bgraph` |
| ODS | 用 TableGen 声明 op schema 的系统 | `BGraphOps.td` |
| TableGen | 从 `.td` 生成 C++ 声明、定义、parser/verify glue | `include/BuddyGraph/IR/CMakeLists.txt` |
| Trait | 生成类携带的结构/语义承诺，如 `Pure`、`Terminator` | `BGraphOps.td` |
| Interface | 可由通用算法调用的行为协议 | 当前 BGraph 未接入 shape interface |
| Verifier | 拒绝局部不自洽的 Op；不是优化 | 各 `*Op::verify()` |
| Shape refinement | 用 operands/attributes 收紧结果 tensor shape | `bgraph-infer-shapes` |
| Fold hook | Op 自己提供的局部折叠入口 | 当前 **未实现** |
| Canonicalization | 注册给通用 canonicalizer 的 best-effort patterns | `getCanonicalizationPatterns()` |
| CSE | 合并可安全复用的相同纯表达式 | upstream `--cse` |
| DCE | 删除无用户、无副作用的计算 | canonicalizer/通用 passes |
| PatternRewriter | 受控创建、替换和删除 IR 的 builder | BN/fusion/canonicalization |
| `LogicalResult` | 只有 success/failure 的控制结果 | verifier/pattern/pass |
| `FailureOr<T>` | 失败或携带一个 `T` | shape helper、constant helper |
| Region | Op 所拥有的嵌套 CFG/计算区域 | `bgraph.fused_elementwise` body |
| Block | Region 内的操作序列和 block arguments | fusion 创建的 single block |
| `IRMapping` | 克隆/迁移 IR 时映射旧 Value 到新 Value | fusion leaves、fused lowering clone |
| Dialect Conversion | 以 legality 为终止条件的结构化重写 | `applyFullConversion()` |
| `ConversionTarget` | 声明哪些 dialect/op 合法或非法 | `ConvertBGraphToLinalg::runOnOperation()` |
| `TypeConverter` | 类型和 signature 转换协议 | 当前不需要、未使用 |
| FullConversion | 所有非法 op 必须被消除，否则整个 pass 失败 | BGraph→Linalg |
| Tensor semantics | 值语义，不直接暴露可变存储 | BGraph/Linalg tensor IR |
| Bufferization | 把 tensor value 转成 memref/buffer 操作 | One-Shot Bufferize |
| Destination-passing style | op 显式接收 output/init destination | Linalg `outs(...)` |
| LLVM Dialect | MLIR 中建模 LLVM 操作的 Dialect，不等于文本 LLVM IR | `llvm.func`、`llvm.load` 等 |
| LLVM IR | `mlir-translate --mlir-to-llvmir` 产生的 LLVM 模块 | 第 11 章 |
| lit | 发现测试并执行 `RUN:` 命令的测试框架 | `tests/lit.cfg.py` |
| FileCheck | 按有序 pattern 检查输出，不是全文字符串比较 | 各 `.mlir` test |
| Location | 诊断和源码追踪信息 | ONNX node name → `Location.name()` |
| `StringRef` / `ArrayRef` | non-owning 字符/连续数据 view，调用者保证生命周期 | verifier 和 shape helpers |
| `SmallVector` | 少量元素优先使用栈内存的 LLVM 容器 | shape、operand、attribute 计算 |
| RAII | 作用域结束自动恢复/释放状态 | `OpBuilder::InsertionGuard` |

