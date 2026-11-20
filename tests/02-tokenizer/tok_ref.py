#!/usr/bin/env python3
"""参考实现：用 Hugging Face 官方的 tokenizers 库做同样的事，输出格式与 tools/tok.c 相同。

用法与 tok 相同：
  tok_ref.py encode tokenizer.json texts.jsonl
  tok_ref.py decode tokenizer.json ids.txt
  tok_ref.py nfc    tokenizer.json texts.jsonl
  tok_ref.py pretok tokenizer.json texts.jsonl
  tok_ref.py roundtrip tokenizer.json texts.jsonl
roundtrip 不调用编码和解码，直接输出 NFC 归一化的结果：这是“编码再解码”应当得到的答案。
nfc 和 pretok 也要 tokenizer.json，因为归一化和预分词的配置从那里读。
"""
import json, sys
from tokenizers import Tokenizer


def byte_decoder():
    """GPT-2 byte-level 映射的逆：字符 → 字节。与 tok.c 的 init_byte_map 独立实现。"""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def texts(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main(mode, tok_path, inp):
    tk = Tokenizer.from_file(tok_path)
    out = sys.stdout
    if mode == "encode":
        for s in texts(inp):
            out.write(" ".join(map(str, tk.encode(s, add_special_tokens=True).ids)) + "\n")
    elif mode == "decode":
        for line in open(inp):
            ids = [int(x) for x in line.split()]
            keep = tk.decode(ids, skip_special_tokens=False).encode().hex()
            skip = tk.decode(ids, skip_special_tokens=True).encode().hex()
            out.write(f"{keep} {skip}\n")
    elif mode in ("nfc", "roundtrip"):
        for s in texts(inp):
            out.write(tk.normalizer.normalize_str(s).encode().hex() + "\n")
    elif mode == "pretok":
        bd = byte_decoder()
        for s in texts(inp):
            pieces = tk.pre_tokenizer.pre_tokenize_str(tk.normalizer.normalize_str(s))
            out.write(" ".join(bytes(bd[c] for c in p).hex() for p, _ in pieces) + "\n")
    else:
        sys.exit(f"unknown mode {mode}")


if __name__ == "__main__":
    main(*sys.argv[1:])
