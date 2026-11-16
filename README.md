# Build an LLM Inference Engine (From Scratch)

从零构建大模型推理引擎。

这是一本边写代码边写成的书。目标是用 C 和 CUDA，从读取权重文件开始，一步步写出一个能跑开源大模型的推理引擎。读者只需要会 C 和 Python。

书和代码在同一个仓库里，写作顺序就是开发顺序。

## 状态

刚开始。正文和代码都是初稿，会随开发改动。

| 章 | 内容 | 状态 |
| --- | --- | --- |
| [前言](book/00-preface.md) | 目标、原则、路线 | 初稿 |
| [第 1 章](book/01-weights.md) | 读取权重文件（safetensors） | 初稿，已在 Qwen3-4B 和 Qwen3.8-27B 的全部权重上对拍 |
| [第 2 章](book/02-tokenizer.md) | 分词器（BPE、NFC、预分词、对话模板） | 初稿，已在小词表、全部 Unicode 码点和 Qwen3.8-27B 真实词表上对拍；对话模板与 transformers 对拍 |

## 构建与测试

需要一个 C17 编译器、make 和 Python 3。第 1 章的测试只用 Python 标准库；第 2 章的测试要和官方分词器对拍，需要 Hugging Face 的 `tokenizers` 库（在 0.21.4 到 0.23.3 上验证过，更早的版本预分词会有差异）。建议装在虚拟环境里，再用 `PYTHON` 指定它：

```
python3 -m venv ~/venvs/tok      # 或者 uv venv ~/venvs/tok
~/venvs/tok/bin/pip install "tokenizers>=0.21.4,<=0.23.3" transformers
```

transformers 只用于对话模板的对拍；没装时这一项会打印“跳过”，其余测试照常运行。

```
make test PYTHON=$HOME/venvs/tok/bin/python            # 快速测试，几秒钟
make test-unicode PYTHON=$HOME/venvs/tok/bin/python    # 第 2 章：全部 Unicode 码点的穷举对拍，几分钟
make test-real TOKENIZER=<模型目录>/tokenizer.json PYTHON=$HOME/venvs/tok/bin/python   # 第 2 章：在真实词表上对拍
```

如果系统的 `python3` 已经装了 `tokenizers`，可以省略 `PYTHON=`。

## 许可

代码（`book/` 以外的所有文件）采用 [MIT 许可](LICENSE)。`book/` 下的正文采用 [CC BY-NC-ND 4.0](book/LICENSE.md)。
