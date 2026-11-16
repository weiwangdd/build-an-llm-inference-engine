#!/usr/bin/env python3
"""对话模板的参考实现：用 transformers 执行官方的 Jinja 模板，输出格式与 `tok chat` 相同。

用法：chat_ref.py tokenizer.json chat_template.jinja requests.jsonl
每行请求是一个 JSON 对象（字段见 src/chat.h）。输出“渲染结果的十六进制 id 序列”，
模板报错时输出 error。
"""
import json, sys
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast


def main(tok_path, tpl_path, req_path):
    with open(tpl_path, encoding="utf-8") as f:
        template = f.read()
    hf = PreTrainedTokenizerFast(tokenizer_file=tok_path)
    tk = Tokenizer.from_file(tok_path)
    out = sys.stdout
    with open(req_path, encoding="utf-8") as f:
        for line in f:
            req = json.loads(line)
            messages = req.pop("messages")
            tools = req.pop("tools", None)
            try:
                text = hf.apply_chat_template(messages, tools=tools, chat_template=template, tokenize=False, **req)
            except Exception:
                out.write("error\n")
                continue
            ids = tk.encode(text).ids
            out.write(text.encode("utf-8").hex() + "".join(f" {i}" for i in ids) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
