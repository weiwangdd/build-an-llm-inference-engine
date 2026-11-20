# 第 3 章 张量与算子

正文：[book/03-ops.md](../../book/03-ops.md)

本章新增的文件：

- `ops.h`、`ops.c`：算子（模型里的基本运算），包括矩阵乘、嵌入、两种 RMSNorm、激活函数、softmax、RoPE、因果一维卷积。

后面章节的扩展：

- 第 4 章加了注意力（`op_attention`）和门控 delta 规则（`op_gated_delta`）。
- 第 6 章把三个算子推广到带缓存的版本：`op_attention_past`、`op_conv1d_hist`、`op_gated_delta_state`，求和的项和次序不变。
- 第 7 章的 GPU 内核（`src/07-gpu/gpu_ops.cu`）与这里的算子一一对应，循环一字不差。

命令行工具：`tools/ops.c`。测试：`tests/03-ops/`。
