#!/usr/bin/env python3
"""对话模板的参考实现：用 transformers 执行官方的 Jinja 模板，输出格式与 `tok chat` 相同。

用法：chat_ref.py tokenizer.json chat_template.jinja requests.jsonl
每行请求是一个 JSON 对象（字段见 src/02-tokenizer/chat.h）。输出“渲染结果的十六进制 id 序列”，
模板报错时输出 error。需要 transformers 和 jinja2（transformers 不会自动装上 jinja2）。
"""
import json, sys
import jinja2  # 缺了它，每个请求都会抛 ImportError，不能被当成“模板报错”吞掉
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
            # transformers 拒绝空对话，但各版本报的异常不同：5.23.0 是 ValueError，
            # 5.8.0.dev0 还没进模板就在 conversation[0] 上抛 IndexError。这里直接按“应当报错”处理。
            if not messages:
                out.write("error\n")
                continue
            try:
                text = hf.apply_chat_template(messages, tools=tools, chat_template=template, tokenize=False, **req)
            # 只有这三类算“按模板应当报错”：模板里的 raise_exception（TemplateError）、
            # 模板运行时的类型错误（如对字符串用 items），以及 transformers 拒绝的其他输入（ValueError）。
            # 其他异常（缺库、参考实现自身的 bug）直接抛出，让测试失败。
            except (jinja2.TemplateError, TypeError, ValueError):
                out.write("error\n")
                continue
            ids = tk.encode(text).ids
            out.write(text.encode("utf-8").hex() + "".join(f" {i}" for i in ids) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
