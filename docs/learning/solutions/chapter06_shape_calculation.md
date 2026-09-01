# 第 06 章答案：Conv shape 手算

输入为 NCHW `[1,3,8,8]`，filter 为 FCHW `[6,3,3,3]`，因此 batch 保持 1，输出
channel 为 6。

每个空间维的 effective kernel：

```text
dilation × (kernel - 1) + 1
= 2 × (3 - 1) + 1
= 5
```

高度和宽度相同：

```text
floor((input + pad_before + pad_after - effective_kernel) / stride) + 1
= floor((8 + 1 + 1 - 5) / 2) + 1
= floor(5 / 2) + 1
= 3
```

所以输出 shape 是 `[1,6,3,3]`。`dilation=2` 时不能把 kernel 仍按 3 代入；否则会
错误得到 `[1,6,4,4]`。
