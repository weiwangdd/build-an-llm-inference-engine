# 第 2 章 分词器

正文：[book/02-tokenizer.md](../../book/02-tokenizer.md)

本章新增的文件：

- `tok.h`、`tok.c`：分词器（把文字切成 token，也就是词表里的编号），流程是切出特殊 token → NFC 归一化 → 预分词 → 字节级 BPE。
- `json.h`、`json.c`：通用的 JSON 解析器，后面第 4 章读 config.json、第 6 章读 generation_config.json、第 8 章解析请求都用它。
- `unicode.h`、`unicode.c`：NFC 归一化（把同一个字的不同 Unicode 写法统一成一种）和字符类别查询。
- `unicode_tables.h`、`unicode_tables.c`：Unicode 数据表，由 `tools/gen_unicode.c` 生成，不要手改（`make unicode-tables`）。
- `chat.h`、`chat.c`：对话模板，把一组对话消息排成模型认识的文字。
- `sha256.h`、`sha256.c`：SHA-256，用来确认对话模板文件就是支持的那一份。

命令行工具：`tools/tok.c`、`tools/gen_unicode.c`。测试：`tests/02-tokenizer/`（`corpus/` 是训练小词表用的语料，`data/` 里是 Qwen3.8-27B 对话模板的副本）。
