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
| [第 3 章](book/03-ops.md) | 张量与算子（矩阵乘、两种 RMSNorm、激活、softmax、RoPE、因果卷积） | 初稿，已在小输入和 Qwen3.8-27B 的真实权重上逐元素对拍，误差不超过 0.5 ulp |
| [第 4 章](book/04-layer.md) | 单层前向（全注意力、门控 delta 规则的线性注意力、前馈层） | 初稿，已在随机小模型上通过两层对拍：逐步误差不超过 0.5 ulp，与官方 transformers（float64）的相对误差不超过 2.4e-4；在 Qwen3.8-27B 的四层真实权重上同样通过 |
| [第 5 章](book/05-model.md) | 整模前向（64 层、输出层、logits）与最朴素的贪心解码 | 初稿，已在 8 层随机小模型上通过两层对拍：逐步误差不超过 0.5 ulp，贪心解码的 token 与官方 transformers（float64）完全相同；在 Qwen3.8-27B 上 64 层逐层对拍通过，“中国的首都是”贪心生成的 8 个 token 与官方 float64 完全相同 |

## 构建与测试

需要一个 C17 编译器、make 和 Python 3。第 1、3 章和第 4 章第一层对拍的测试只用 Python 标准库；第 4、5 章的第二层对拍要调用官方的 transformers 代码，需要 `torch` 和 `transformers`，没装时打印“跳过”；第 2 章的测试要和官方分词器对拍，需要 Hugging Face 的 `tokenizers` 库（在 0.21.4 到 0.23.3 上验证过，更早的版本预分词会有差异）。建议装在虚拟环境里，再用 `PYTHON` 指定它：

```
python3 -m venv ~/venvs/tok      # 或者 uv venv ~/venvs/tok
~/venvs/tok/bin/pip install "tokenizers>=0.21.4,<=0.23.3" transformers jinja2 torch
```

jinja2 只用于对话模板的对拍（transformers 不会自动装上它），torch 只用于第 4、5 章的第二层对拍；没装时对应的测试会打印“跳过”，其余测试照常运行。

```
make test PYTHON=$HOME/venvs/tok/bin/python            # 快速测试，一两分钟
make test-unicode PYTHON=$HOME/venvs/tok/bin/python    # 第 2 章：全部 Unicode 码点的穷举对拍，几分钟
make test-real TOKENIZER=<模型目录>/tokenizer.json PYTHON=$HOME/venvs/tok/bin/python   # 第 2 章：在真实词表上对拍
make test-ops-real MODEL=<模型目录>                     # 第 3 章：拿真实权重当算子输入，十几秒
make test-layer-real MODEL=<模型目录> PYTHON=$HOME/venvs/tok/bin/python   # 第 4 章：在真实模型的四层上对拍，十几分钟
make test-model-real MODEL=<模型目录> PYTHON=$HOME/venvs/tok/bin/python   # 第 5 章：整个模型贪心生成 8 个 token 并对拍
```

如果系统的 `python3` 已经装了 `tokenizers`，可以省略 `PYTHON=`。

## 许可

代码（`book/` 以外的所有文件）采用 [MIT 许可](LICENSE)。`book/` 下的正文采用 [CC BY-NC-ND 4.0](book/LICENSE.md)。
