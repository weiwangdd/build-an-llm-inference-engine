# Build an LLM Inference Engine (From Scratch)

从零构建大模型推理引擎。

这是一本边写代码边写成的书。目标是用 C 和 CUDA，从读取权重文件开始，一步步写出一个能跑开源大模型的推理引擎。读者只需要会 C 和 Python。

书和代码在同一个仓库里，写作顺序就是开发顺序。

## 状态

刚开始。正文和代码都是初稿，会随开发改动。

| 章 | 内容 | 状态 |
| --- | --- | --- |
| [前言](book/00-preface.md) | 目标、原则、路线 | 初稿 |
| [第 1 章](book/01-weights.md) | 读取权重文件（safetensors） | 初稿，尚未在真实模型上验证 |

## 构建与测试

需要一个 C17 编译器、make 和 Python 3（只用标准库）。

```
make test
```

## 许可

代码（`book/` 以外的所有文件）采用 [MIT 许可](LICENSE)。`book/` 下的正文采用 [CC BY 4.0](book/LICENSE.md)。
