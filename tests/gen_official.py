#!/usr/bin/env python3
"""第 6 章的第二层对拍：拿官方 transformers 的采样代码检查 C 的分布、选择和停止条件。

    gen_official.py <warpers|full> <config.json> <generation_config.json> <ids> <max_new> [generate 的选项 ...] <分片 ...>
        < build/generate --dump 的输出

参数从第二个起与 build/generate 完全相同。需要 torch 和 transformers（在 5.23.0 上写成）。

warpers：只用官方的 logits 处理器。对 C 打印的每一行 logits，在 float64 下依次套上官方的
    TemperatureLogitsWarper、TopKLogitsWarper、TopPLogitsWarper（顺序和“什么时候加上哪一个”照抄
    transformers 的 GenerationMixin._get_logits_processor），再做 softmax。保留的 token 必须与 C 完全相同，
    概率的相对误差不超过 1e-9。真实模型用这个。
full：在 warpers 的基础上，再构造官方的整个模型 Qwen3_5ForCausalLM（float64），调用它的 generate
    （带官方自己的缓存、自己的停止条件），检查整条流水线。采样的随机数官方用的是 torch 的生成器，
    没法和我们的对上，所以把 torch.multinomial 换成一个“记下官方给的概率，返回 C 选的 token”的函数：
      - 官方算出的概率（来自官方自己的 logits）与 C 的分布：保留的 token 相同，概率差不超过 2^-12；
      - 官方 generate 在同样的地方停下（生成结束符就停、或者生成满 max_new 个）。
    贪心解码时不用替换，官方自己选，生成的 token 必须完全相同。只适合小模型（整个模型要放进内存）。
"""
import json, math, sys

for _m in ("fla", "causal_conv1d"):      # 原因见第 4 章 layer_official.py
    sys.modules[_m] = None

try:
    import torch
    from transformers import GenerationConfig
    from transformers.generation.logits_process import (TemperatureLogitsWarper, TopKLogitsWarper,
                                                        TopPLogitsWarper)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
except ImportError as e:
    print(f"SKIP 需要 torch 和 transformers：{e}")
    sys.exit(3)

sys.path.insert(0, sys.path[0])
from gen_ref import parse_args, settings, read_dump  # noqa: E402
from model_official import Weights, P  # noqa: E402

TOL_FULL = 2.0 ** -12


def warpers(s):
    """照抄 transformers 的 _get_logits_processor：什么条件下加上哪一个、按什么顺序"""
    w = []
    if not s["do_sample"]:
        return w
    if s["temperature"] != 1.0:
        w.append(TemperatureLogitsWarper(s["temperature"]))
    if s["top_k"] != 0:
        w.append(TopKLogitsWarper(top_k=s["top_k"], min_tokens_to_keep=1))
    if s["top_p"] < 1.0:
        w.append(TopPLogitsWarper(top_p=s["top_p"], min_tokens_to_keep=1))
    return w


def official_dist(row, s):
    x = torch.tensor([row], dtype=torch.float64)
    if not s["do_sample"]:
        return [(int(x[0].argmax()), 1.0)]
    for w in warpers(s):
        x = w(None, x)
    p = torch.softmax(x, dim=-1)[0]
    return [(i, float(p[i])) for i in range(len(row)) if p[i] > 0]


def same_dist(k, c, o, rel):
    """c 是 C 的分布，o 是官方的。返回 (出错信息或 None, 最大误差)"""
    if [i for i, _ in c] != [i for i, _ in o]:
        return f"第 {k} 步保留的 token 不同：C {[i for i, _ in c][:10]}，官方 {[i for i, _ in o][:10]}", 0
    worst = 0.0
    for (i, pc), (_, po) in zip(c, o):
        e = abs(pc - po) / po if rel else abs(pc - po)
        if not e <= (1e-9 if rel else TOL_FULL):
            return f"第 {k} 步 token {i} 的概率 C 是 {pc!r}，官方 {po!r}", 0
        worst = max(worst, e)
    return None, worst


def main():
    mode, args = sys.argv[1], sys.argv[2:]
    cfg_path, gcfg, prompt, max_new, o = parse_args(args)
    s, eos = settings(gcfg, o)
    steps, stop = read_dump(sys.stdin)
    tokens = [st["token"] for st in steps]

    worst = 0.0
    for k, st in enumerate(steps):
        msg, e = same_dist(k, st["dist"], official_dist(st["logits"], s), rel=True)
        if msg:
            print(msg)
            return 1
        worst = max(worst, e)
    out = [f"官方 logits 处理器：{len(steps)} 步的分布与 C 相同，最大相对误差 {worst:.1e}"]
    if mode != "full":
        print(out[0])
        return 0

    cfg = Qwen3_5Config.from_dict(json.load(open(cfg_path)))
    tc = cfg.text_config
    tc._attn_implementation = "eager"
    files = [a for a in args[4:] if a.endswith(".safetensors")]
    W = Weights(files)
    model = M.Qwen3_5ForCausalLM(tc)
    sd = {("model." + k[len(P):] if k.startswith(P) else k): W.get(k) for k in W.where}
    model.load_state_dict({k: sd[k] for k in model.state_dict()}, strict=True)
    model.to(torch.float64).eval()

    gc = GenerationConfig(do_sample=s["do_sample"], max_new_tokens=max_new, eos_token_id=sorted(eos),
                          pad_token_id=sorted(eos)[0])
    if s["do_sample"]:
        gc.temperature, gc.top_k, gc.top_p = s["temperature"], s["top_k"], s["top_p"]

    seen = []

    class TooLong(Exception):
        pass

    real = torch.multinomial

    def fake(probs, num_samples, *a, **kw):     # 记下官方的概率，返回 C 在这一步选的 token
        k = len(seen)
        seen.append(probs[0].to(torch.float64).clone())
        if k >= len(tokens):
            raise TooLong()
        return torch.tensor([[tokens[k]]])

    inp = torch.tensor([prompt])
    torch.multinomial = fake
    try:
        with torch.no_grad():
            gen = model.generate(inp, attention_mask=torch.ones_like(inp), generation_config=gc)
    except TooLong:
        print(f"C 在第 {len(tokens)} 个 token 停下（{stop}），官方 generate 还要接着生成")
        return 1
    finally:
        torch.multinomial = real
    gen = [int(v) for v in gen[0, len(prompt):]]
    if gen != tokens:
        what = "贪心解码" if not s["do_sample"] else "照 C 的选择走"
        print(f"官方 generate（{what}）生成的是 {gen}，C 是 {tokens}")
        return 1
    if s["do_sample"]:
        worst = 0.0
        for k, p in enumerate(seen):
            o_dist = [(i, float(p[i])) for i in range(p.shape[0]) if p[i] > 0]
            msg, e = same_dist(k, steps[k]["dist"], o_dist, rel=False)
            if msg:
                print(msg + "（官方的概率来自官方自己算的 logits）")
                return 1
            worst = max(worst, e)
        out.append(f"官方 generate 照 C 的选择走，{len(gen)} 步的分布与 C 相同（概率最多差 {worst:.1e}），"
                   f"停在同一个地方（{stop}）")
    else:
        out.append(f"官方 generate 贪心解码生成的 {len(gen)} 个 token 与 C 相同，停在同一个地方（{stop}）")
    print("；".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
