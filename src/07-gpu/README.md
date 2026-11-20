# 第 7 章 单卡 GPU 后端

正文：[book/07-gpu.md](../../book/07-gpu.md)

本章新增的文件：

- `gpu_ops.h`、`gpu_ops.cu`：CUDA 内核（在 GPU 上并行执行的函数），`gop_xxx` 与 `src/03-ops/ops.h` 的 `op_xxx` 一一对应。这是唯一的 CUDA 文件。
- `gpu.h`、`gpu_model.c`：把模型搬进显存，在 GPU 上跑前向，与 `src/04-layer/model.c` 逐行对应。
- `gpu_none.c`：没有 CUDA 时链接它，GPU 的函数一律报错。

后面章节的扩展：

- 第 8 章的 `model_step_batch` 在 `gpu_model.c` 里也有一份。

命令行工具：与 CPU 版相同的工具，用 `make gpu` 加 `-DUSE_GPU` 编译到 `build/gpu`。测试：`tests/07-gpu/`（第 3–6 章的测试拿 GPU 版原样再跑一遍）。
