# 第 6 章 生成循环

正文：[book/06-generate.md](../../book/06-generate.md)

本章新增的文件：

- `sample.h`、`sample.c`：采样（按概率从候选 token 里挑一个），依次做温度、top-k、top-p，随机数是 splitmix64。
- `gen.h`、`gen.c`：读 generation_config.json，把“前向 → 采样 → 判断停止”串成生成循环。

后面章节的扩展：

- 缓存和 `model_step` 在 `src/04-layer/model.c` 里，带缓存的三个算子在 `src/03-ops/ops.c` 里。
- 第 8 章的 `src/08-serve/engine.c` 沿用这里的采样代码和次序。

命令行工具：`tools/generate.c`。测试：`tests/06-generate/`。
