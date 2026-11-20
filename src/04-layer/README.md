# 第 4 章 单层前向（第 5、6、8 章在这里扩展）

正文：[book/04-layer.md](../../book/04-layer.md)

本章新增的文件：

- `model.h`、`model.c`：读 config.json、按配置加载并检查一层的权重，算一层的前向（嵌入之后依次经过注意力和前馈层）。

后面章节的扩展：

- 第 5 章（[正文](../../book/05-model.md)）没有新文件，整模前向写在这里：`model_load`、`model_forward`、`logits_argmax`，工具是 `tools/forward.c`，测试在 `tests/05-model/`。
- 第 6 章加了缓存（`model_cache`，存下前面 token 算过的中间结果，生成下一个 token 时不用从头算）和 `model_step`。
- 第 7 章：`model_to_gpu` 之后 `model_forward`、`model_step` 转到 `src/07-gpu/gpu_model.c`，那份代码与这里逐行对应。
- 第 8 章加了 `model_step_batch`，几个序列一起算；`layer_forward`、`model_forward`、`model_step` 都是它的特例。

命令行工具：`tools/layer.c`。测试：`tests/04-layer/`。
