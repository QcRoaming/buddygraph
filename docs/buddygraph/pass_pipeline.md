# Pass pipeline

## 推荐顺序

```text
bgraph-infer-shapes
bgraph-fold-bn-into-conv
canonicalize
cse
bgraph-fuse-elementwise
convert-bgraph-to-linalg
one-shot-bufferize{bufferize-function-boundaries}
convert-linalg-to-loops
lower-affine
convert-scf-to-cf
convert-cf-to-llvm
convert-math-to-llvm
convert-arith-to-llvm
finalize-memref-to-llvm
convert-func-to-llvm
reconcile-unrealized-casts
```

shape refinement 先提供更具体的 legality 信息；BN folding 必须在 Conv/BN 仍保留
图语义时执行；canonicalize/CSE 清理 folding 产生的常量和冗余；fusion 再选择单
use elementwise chains；FullConversion 是图层与结构化后端的分界。bufferization
之前保持 tensor semantics，之后再降 loops 和 LLVM。

可复制命令：

```bash
build/bin/buddygraph-opt \
  --bgraph-infer-shapes \
  --bgraph-fold-bn-into-conv \
  --canonicalize --cse \
  --bgraph-fuse-elementwise \
  --convert-bgraph-to-linalg \
  '--one-shot-bufferize=bufferize-function-boundaries' \
  --convert-linalg-to-loops --lower-affine \
  --convert-scf-to-cf --convert-cf-to-llvm \
  --convert-math-to-llvm --convert-arith-to-llvm \
  --finalize-memref-to-llvm --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  input.mlir -o output.llvm.mlir
```

## 各 Pass 的契约

| Pass | 输入 | 成功后保证 | 非匹配行为 |
|---|---|---|---|
| `bgraph-infer-shapes` | 合法 BGraph tensor IR | 可推导的结果 shape 被收紧 | 保持原类型 |
| `bgraph-fold-bn-into-conv` | Conv→BN 与常量参数 | BN 被新 filter/bias Conv 替代 | 原图不变 |
| `canonicalize` | 注册了 patterns 的 IR | 局部等价冗余被删除 | 原图不变 |
| `cse` | side-effect 信息完整的 IR | 公共纯表达式合并 | 原图不变 |
| `bgraph-fuse-elementwise` | 单 use elementwise chain | chain 变成 scalar Region op | 分支/多 use 不融合 |
| `convert-bgraph-to-linalg` | 全部合法 BGraph ops | 不残留任何 `bgraph.*` | FullConversion 失败 |

Conversion 覆盖 Conv、BN、全部 elementwise（含 Clamp）、Reshape、Transpose、ReduceMean、
FusedElementwise。Conv 变成 named Linalg convolution（含 pad/bias init）；BN、
elementwise、ReduceMean 和 fused Region 变成 `linalg.generic`；reshape/transpose
分别变成 Tensor reshape 和 Linalg transpose。
