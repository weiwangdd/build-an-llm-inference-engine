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

需要 C17 编译器、make、Python 3。第 2 章的测试需要 `tokenizers` 库（验证过 0.21.4–0.23.3）。系统 Python 通常没有它，装在虚拟环境里，用 `PYTHON=$HOME/venvs/.../bin/python` 传给所有 `make test*` 目标（`tests/run_tok.sh` 读这个环境变量；第 1 章的 `tests/run.sh` 固定用 `python3`，只需标准库）。在 zsh 里不要写 `PYTHON=~/...`，`~` 不会展开。

```
make                 # 构建 build/st_dump 和 build/tok
make test            # 第 1、2 章快速测试，几秒钟
make test-unicode    # 第 2 章：全部 Unicode 码点的归一化与预分词穷举对拍，几分钟
make test-real TOKENIZER=<模型目录>/tokenizer.json PYTHON=...   # 在真实词表上验收
make unicode-tables  # 从 UCD 9.0 和 16.0 重新生成 src/unicode_tables.c（下载地址见 tools/gen_unicode.c）
```

单独跑某一章：`sh tests/run.sh build`（第 1 章）或 `sh tests/run_tok.sh build`（第 2 章，参数见脚本头部）。单个对拍可以手工跑两边再 diff，例如：

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
- **第 2 章 分词器（`src/tok.[ch]`、`src/json.[ch]`、`src/unicode*.[ch]`、`tools/tok.c`）**：流水线是“切出特殊 token → NFC → 预分词 → 字节级 BPE”，与 Hugging Face tokenizers 相同。`json.c` 是通用 JSON 解析器（与第 1 章的专用解析器分开）。`tok` 工具有 `encode/decode/nfc/pretok/roundtrip` 几种模式，文本一律以十六进制输出以避免逐行比较时的歧义，解码结果先按 Rust `from_utf8_lossy` 规则处理。`tests/make_tok_fixtures.py` 在 `tests/corpus/` 上现训练一个约 3000 token 的小词表，配置与 Qwen3.8 的 tokenizer.json 完全相同；24.8 万的真实词表通过 `make test-real` 验收。

## 约定

- **严格拒绝，绝不猜测**：任何不符合格式或不在支持范围内的输入都要报错（错误信息写进调用方给的 `err` 缓冲区，函数返回 -1），而不是忽略。例如 `tok.c` 只支持 Qwen 用的那一种 tokenizer 配置，预分词正则是手写在 `match_at` 里的，加载时要求文件里的正则与 `QWEN_PATTERN` 一字不差。新增拒绝分支时，在 fixture 脚本里加一个对应的 `bad_*` 样例。
- `src/unicode_tables.c` 是生成文件，不要手改，改 `tools/gen_unicode.c` 后用 `make unicode-tables` 重新生成。为了与参考库逐字节一致，NFC 用 Unicode 9.0 的数据，预分词的字符类别用 16.0（原因见 `tools/gen_unicode.c` 头部）。
- 参考库的行为会随版本变化（第 2 章“参考实现的版本”一节记录了例子）；遇到对拍差异，先确认是不是参考库版本的问题。
- 许可：`book/` 以外是 MIT，`book/` 下的正文是 CC BY-NC-ND 4.0。
