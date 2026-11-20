# 第 1 章 读取权重文件

正文：[book/01-weights.md](../../book/01-weights.md)

本章新增的文件：

- `st.h`、`st.c`：用 mmap（把文件直接映射进内存，按需读入）只读解析 safetensors 文件；`st_set` 把多个分片（一个模型拆成的多个文件）合并成一张按名字排序的表。

命令行工具：`tools/st_dump.c`。测试：`tests/01-weights/`。
