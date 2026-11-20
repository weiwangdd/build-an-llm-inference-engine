#!/usr/bin/env python3
"""生成第 8 章服务端测试的输入：一个能接受“文字”的随机小模型。

    tests/08-serve/make_serve_fixtures.py <输出目录>

前几章的小模型词表只有 40 个 token，没法配分词器。这里用第 2 章的办法现训练一个约 3000 token 的小词表
（tests/02-tokenizer/make_tok_fixtures.py 的 build_tokenizer，配置与 Qwen3.8 的 tokenizer.json 相同，有 <|im_start|>、
<|im_end|> 等特殊 token），再造一个词表大小与它相同的 8 层随机模型（其余结构同第 5 章的小模型）。
    tokenizer.json, chat_template.jinja（tests/02-tokenizer/data 里 Qwen3.8-27B 的那一份）
    config.json, model.safetensors, generation_config.json（字段与 27B 的相同，结束符是 <|im_end|> 和 <|endoftext|>）
输出层里 <|im_end|> 那一行放大到 2 倍，让随机模型每一步有百分之几的机会说完，测得到 finish_reason 为 stop。
"""
import json, math, os, random, shutil, sys

sys.path.insert(0, sys.path[0])
# 前面章节的参考实现在相邻的目录里（tests/NN-名字/）
sys.path[1:1] = [os.path.join(sys.path[0], "..", d) for d in ("02-tokenizer", "03-ops", "04-layer", "05-model")]
import make_model_fixtures  # noqa: E402
from make_layer_fixtures import weights  # noqa: E402
from make_ops_fixtures import write  # noqa: E402
from make_tok_fixtures import build_tokenizer, corpus  # noqa: E402


def main(out):
    os.makedirs(out, exist_ok=True)
    tk = build_tokenizer(corpus())
    tk.save(f"{out}/tokenizer.json")
    here = os.path.dirname(os.path.abspath(__file__))
    shutil.copy(f"{here}/../02-tokenizer/data/qwen38-27b-chat_template.jinja", f"{out}/chat_template.jinja")
    V = tk.get_vocab_size(with_added_tokens=True)
    im_end, eot = tk.token_to_id("<|im_end|>"), tk.token_to_id("<|endoftext|>")

    cfg = make_model_fixtures.config()
    t = cfg["text_config"]
    t["vocab_size"] = V
    H = t["hidden_size"]
    with open(f"{out}/config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    rng = random.Random(81)
    w = weights(cfg, rng)
    head = [rng.gauss(0, 1 / math.sqrt(H)) for _ in range(V * H)]
    for i in range(im_end * H, (im_end + 1) * H):
        head[i] *= 2
    w["lm_head.weight"] = ("BF16", [V, H], head)
    write(f"{out}/model.safetensors", w, pad=1)
    with open(f"{out}/generation_config.json", "w") as f:
        json.dump({"bos_token_id": eot, "do_sample": True, "eos_token_id": [im_end, eot], "pad_token_id": eot,
                   "temperature": 1.0, "top_k": 20, "top_p": 0.95}, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1])
