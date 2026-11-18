# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目性质

这是一本“边写代码边写成”的书：用 C17（后面加 CUDA）从零写一个大模型推理引擎，不依赖 PyTorch 或任何现成推理库。书（`book/`）和代码在同一个仓库里，写作顺序就是开发顺序——每完成一章的代码，对应的 `book/NN-*.md` 也要同步更新（包括 README 的状态表）。正文、注释、提交信息、测试输出都用中文。

`book/00-preface.md` 里有全书路线（11 个阶段，每阶段一章、一条验收标准）和三条原则，改动前值得一读：

- **先正确，后性能**：第一卷只用最朴素的写法，性能优化留到第二卷。
- **每一步都对拍**：每章的验收标准都是与一份独立的参考实现（Python）逐字节/逐元素比较。
- **如实记录**：弯路、错误和排查过程写进书里。

目标模型是 Qwen3.8-27B（也在 Qwen3-4B 上验证过），目标机器是两张 96 GB 的 RTX PRO 6000。

## 命令

需要 C17 编译器、make、Python 3。第 2 章的测试需要 `tokenizers` 库（验证过 0.21.4–0.23.3），对话模板的对拍还需要 `transformers` 和 `jinja2`（前者不会自动带上后者；没装时打印 SKIP），第 4、5、6 章的第二层对拍需要 `torch` 和 `transformers`（没装时打印 SKIP）。系统 Python 通常没有它，装在虚拟环境里，用 `PYTHON=$HOME/venvs/.../bin/python` 传给所有 `make test*` 目标（`tests/run_tok.sh` 读这个环境变量；第 1 章的 `tests/run.sh` 固定用 `python3`，只需标准库）。在 zsh 里不要写 `PYTHON=~/...`，`~` 不会展开。

```
make                 # 构建 build/st_dump、build/tok、build/ops、build/layer、build/forward 和 build/generate
make test            # 第 1 到 6 章快速测试，两三分钟
make test-unicode    # 第 2 章：全部 Unicode 码点的归一化与预分词穷举对拍，几分钟
make test-real TOKENIZER=<模型目录>/tokenizer.json PYTHON=...   # 在真实词表上验收
make test-ops-real MODEL=<模型目录>   # 第 3 章：拿真实权重当算子输入
make test-layer-real MODEL=<模型目录> PYTHON=...   # 第 4 章：真实模型的四层，两层对拍，十几分钟
make test-model-real MODEL=<模型目录> PYTHON=...   # 第 5 章：整模前向加贪心解码 8 个 token，用 -O2 无 ASan 另建的 build/fast/forward
make test-gen-real MODEL=<模型目录> PYTHON=...     # 第 6 章：带缓存贪心 8 个 + 对话模板采样，与不带缓存逐位对拍，同样用 build/fast
make unicode-tables  # 从 UCD 9.0 和 16.0 重新生成 src/unicode_tables.c（下载地址见 tools/gen_unicode.c）
tools/ask.sh [-n N] [-s S] [-g] [-t] "问题"   # 向真实模型提问（MODEL 默认 /data/models/Qwen3.8-27B），边生成边打印，用 build/fast 的 -O2 程序
```

单独跑某一章：`sh tests/run.sh build`（第 1 章）、`sh tests/run_tok.sh build`（第 2 章）、`sh tests/run_ops.sh build`（第 3 章）、`sh tests/run_layer.sh build`（第 4 章）、`sh tests/run_model.sh build`（第 5 章）或 `sh tests/run_gen.sh build`（第 6 章），参数见脚本头部。单个对拍可以手工跑两边再 diff，例如：

```
python3 tests/tok_ref.py encode build/tok_fixtures/tokenizer.json build/tok_fixtures/texts.jsonl > ref.txt
build/tok encode build/tok_fixtures/tokenizer.json build/tok_fixtures/texts.jsonl > got.txt
```

编译选项很严：`-Wall -Wextra -Wpedantic -Wconversion -Werror`，并且默认带 ASan + UBSan（`SAN` 变量）。隐式窄化转换会直接编译失败，需要显式转换。

## 架构

每章由三部分组成，测试就是把它们串起来：

1. `src/` 下的库（引擎本体，不依赖 Python）。
2. `tools/` 下的命令行工具，把库的结果按固定文本格式打印出来。
3. `tests/` 下的 Python 参考实现，打印**完全相同格式**的输出；`tests/run*.sh` 用 `cmp` 逐字节比较两边，并检查所有故意写坏的输入都被以退出码 1 拒绝（不是崩溃，也不是成功）。

因此改 C 端的输出格式时，必须同时改对应的 Python 参考实现，反之亦然。Python 那一侧要与 C 端**独立实现**（例如 `tok_ref.py` 自己实现 byte-level 映射），否则两边可能错成一样；必要时再加一个不依赖 Python 的已知答案（如 CRC-32 测试向量）。

- **第 1 章 权重（`src/st.[ch]`、`tools/st_dump.c`）**：mmap 只读解析 safetensors，`st_set` 把多个分片合并成一张按名字升序的表，跨分片重名算错误。`st_dump` 打印每个张量的 CRC-32、dtype、形状和名字。safetensors 头用 `st.c` 里的专用小 JSON 解析器。测试数据由 `tests/make_fixtures.py` 生成（含一组 `bad_*.safetensors`）。
- **第 2 章 分词器（`src/tok.[ch]`、`src/json.[ch]`、`src/unicode*.[ch]`、`tools/tok.c`）**：流水线是“切出特殊 token → NFC → 预分词 → 字节级 BPE”，与 Hugging Face tokenizers 相同。`json.c` 是通用 JSON 解析器（与第 1 章的专用解析器分开）。`tok` 工具有 `encode/decode/nfc/pretok/roundtrip/chat` 几种模式，文本一律以十六进制输出以避免逐行比较时的歧义，解码结果先按 Rust `from_utf8_lossy` 规则处理。对话模板（`src/chat.[ch]`、`src/sha256.[ch]`）是把 Qwen3.8-27B 的 `chat_template.jinja` 手写成 C 的 `chat_render`，用 SHA-256 确认模板文件就是那一份（副本在 `tests/data/`），参考实现 `tests/chat_ref.py` 调用 transformers 的 `apply_chat_template`。`tests/make_tok_fixtures.py` 在 `tests/corpus/` 上现训练一个约 3000 token 的小词表，配置与 Qwen3.8 的 tokenizer.json 完全相同；24.8 万的真实词表通过 `make test-real` 验收。
- **第 3 章 算子（`src/ops.[ch]`、`tools/ops.c`）**：矩阵乘、嵌入、两种 RMSNorm（`1+w` 与带门控的 `w`）、L2 归一化、silu/sigmoid/softplus、softmax、部分维度的 RoPE、因果一维卷积，定义以 transformers 的 `qwen3_5` 建模代码为准。激活值是 float，大矩阵直接读 BF16 原始字节（逐字节拼，不要求对齐），内部一律用 double 算，所以结果几乎是精确值的正确舍入，后面各章拿它当标尺。这一章的对拍不是逐字节：`ops` 用 `%a` 打印输出，`tests/ops_ref.py`（只用标准库，双精度、`math.fsum`）逐元素比较，容差是 `ulp(r) + 2^-32·Σ|项|`，嵌入要求完全相等。输入是 `tests/make_ops_fixtures.py` 写的小 safetensors 文件和用例清单（`cases.txt`、`bad_cases.txt`）；`tests/make_ops_real.py` 用真实权重生成同格式的用例。
- **第 4 章 单层前向（`src/model.[ch]`、`tools/layer.c`，`src/ops.c` 里新增注意力和门控 delta 规则）**：`model_cfg_load` 读 config.json 的 `text_config`，只接受 Qwen3.8 那一种配置；`layer_load` 按配置检查一层每个张量的名字、dtype、形状；`layer_forward` 每算完一步调用一次 trace 回调。`layer` 工具算“嵌入 → 一层 → 最终归一化”，按“名字 [行,列]”加 `%a` 打印每一步。对拍分两层（Wei 定的方案）：`tests/layer_ref.py`（标准库）拿 C 每一步的输入算精确值，容差同第 3 章，步骤名和顺序必须一致；`tests/layer_official.py` 调用 transformers 的 `Qwen3_5DecoderLayer` 在 float64 下算（屏蔽 fla、causal_conv1d，用 eager 注意力），每个张量最大误差除以均方根不超过 2⁻¹²，没装 torch 打印 SKIP。改 C 的步骤名或顺序时，两个参考实现都要跟着改。小模型由 `tests/make_layer_fixtures.py` 生成。
- **第 5 章 整模前向（`src/model.[ch]` 里的 `model_load`、`model_forward`、`logits_argmax`，`tools/forward.c`）**：嵌入 → 全部层 → 最终归一化 → 顶层的 `lm_head.weight`，输出 [n, vocab] 的 logits；`tie_word_embeddings` 只接受 false。`forward` 工具做最朴素的贪心解码（每一步都从头前向，没有缓存），只打印最后一次前向：`all` 打印每层每一步（名字加 `layers.<i>.` 前缀），`layers` 只打印每层的 `out`；最后一段 `greedy [1,m]` 是十进制 id。对拍仍是两层：`tests/model_ref.py` 复用 `layer_ref.layer_plan` 逐步检查，并检查生成的 token 是那一行的最大值（同样大取 id 小的）；`tests/model_official.py` 逐层加载（27B 的 float64 放不进内存），每层拿 C 的上一层输出做输入按 2⁻¹² 判定，再让官方 float64 自己串起全部层，要求贪心 token 完全相同（利用因果性，对 C 最后一次前向的输入跑一遍就能逐行检查）；官方 BF16 只报告不判定（Wei 定的）。小模型上还用官方 `Qwen3_5ForCausalLM` 和它的 `generate` 核对。另有不依赖 Python 的已知答案：前缀的 logits 逐位相同、接着生成结果不变。小模型由 `tests/make_model_fixtures.py` 生成（8 层，带 lm_head）。
- **第 6 章 生成循环（`src/model.[ch]` 里的 `model_cache`、`model_step`，`src/sample.[ch]`、`src/gen.[ch]`、`tools/generate.c`）**：缓存存全注意力层 RoPE 之后的 k、v，和线性注意力层的卷积窗口（最近 conv_k-1 个 token 的 qkv）、delta 状态 S（double）。`layer_forward`/`model_forward` 与带缓存的版本走同一个内部函数（缓存为 NULL 即第 4、5 章的行为）；`ops.c` 里的 `op_attention_past`、`op_conv1d_hist`、`op_gated_delta_state` 是原算子的推广，求和的项和次序不变，所以**带缓存与不带缓存逐位相同**，改这些代码时必须保持。采样：温度 → top-k（与第 k 名同分的都留）→ top-p，照 transformers 的 logits 处理器；逆变换采样按 id 顺序累加；随机数是 splitmix64（与 Java `SplittableRandom` 相同）。`generation_config.json` 只接受 7 个已知字段加 transformers_version，采样时 temperature/top_k/top_p 必须写明。`generate` 工具的 `--dump` 输出每步 logits、分布、随机数；`--no-cache`、`--chunk` 用于已知答案（输出必须逐字节相同）。对拍（Wei 定的方案：分布对拍加固定随机数）：`tests/gen_ref.py`（标准库，随机数完全相等、保留集合相同、概率相对误差 ≤1e-9、选中 token 相同、停止对）；`tests/gen_official.py warpers` 用官方 logits 处理器作用在 C 的 logits 上，`full` 另在小模型上跑官方 `generate`，把 `torch.multinomial` 换成“记下官方概率、返回 C 的选择”，检查分布（2⁻¹²）和停止位置。另把“提示词加生成 token”交给 `forward` 一次前向，后几行 logits 与生成时逐位相同，并过 `model_ref.py`。

## 约定

- **严格拒绝，绝不猜测**：任何不符合格式或不在支持范围内的输入都要报错（错误信息写进调用方给的 `err` 缓冲区，函数返回 -1），而不是忽略。例如 `tok.c` 只支持 Qwen 用的那一种 tokenizer 配置，预分词正则是手写在 `match_at` 里的，加载时要求文件里的正则与 `QWEN_PATTERN` 一字不差。新增拒绝分支时，在 fixture 脚本里加一个对应的 `bad_*` 样例。
- `src/unicode_tables.c` 是生成文件，不要手改，改 `tools/gen_unicode.c` 后用 `make unicode-tables` 重新生成。为了与参考库逐字节一致，NFC 用 Unicode 9.0 的数据，预分词的字符类别用 16.0（原因见 `tools/gen_unicode.c` 头部）。
- 参考库的行为会随版本变化（第 2 章“参考实现的版本”一节记录了例子）；遇到对拍差异，先确认是不是参考库版本的问题。
- 许可：`book/` 以外是 MIT，`book/` 下的正文是 CC BY-NC-ND 4.0。
