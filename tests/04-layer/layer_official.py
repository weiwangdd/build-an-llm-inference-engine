#!/usr/bin/env python3
"""第 4 章的第二层对拍：拿官方的 transformers 代码算同一层，与 C 的结果比较。

    layer_official.py <config.json> <层号> <ids> <分片 ...> < build/layer 的输出

需要 torch 和 transformers（在 5.23.0 上写成）。第一层对拍（layer_ref.py）检查的是“每一步都算对了”，
但每一步的公式是我们读官方代码之后自己写的，C 和 layer_ref.py 出自同一个人的理解，可能错成一样。
这一层直接调用 transformers 里的 Qwen3_5DecoderLayer 和 Qwen3_5RMSNorm，检查的是“公式读对了”。

官方代码在 float64 下运行（权重从 BF16 精确地转过去），但它在 RMSNorm、RoPE、softmax、
门控 delta 规则内部会自己转成 float32，所以结果只有 float32 的精度，比不了第一层的 1 ulp。
判定标准是：每个张量的最大误差，除以这个张量的均方根，不超过 TOL。设环境变量 LAYER_VERBOSE=1 打印每个张量的误差。

顺便用 BF16 再跑一遍官方代码（也就是模型平时实际的运行方式），只打印它离我们有多远，不做判定。
"""
import json, math, os, sys

# 不用 flash-linear-attention、causal-conv1d 这些 GPU 加速包，即使装了也不用：
# transformers 找不到它们时会退回到纯 torch 的参考写法，那才是我们要比的“官方公式”。
for _m in ("fla", "causal_conv1d"):
    sys.modules[_m] = None

try:
    import torch
    from safetensors import safe_open
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
except ImportError as e:
    print(f"SKIP 需要 torch 和 transformers：{e}")
    sys.exit(3)

sys.path.insert(0, sys.path[0])
from layer_ref import read_dump  # noqa: E402

P = "model.language_model."
TOL = 2.0 ** -12      # 约 2.4e-4，理由见第 4 章“第二层：和官方代码比”


def load(files, names):
    out = {}
    for f in files:
        with safe_open(f, framework="pt") as st:
            for k in st.keys():
                if k in names:
                    out[k] = st.get_tensor(k)
    missing = set(names) - set(out)
    if missing:
        raise SystemExit(f"找不到张量 {sorted(missing)[0]}")
    return out


def run(tc, layer, ids, W, dtype):
    """用官方模块算：嵌入 → 第 layer 层 → 最终归一化。返回 {名字: [n, 列] 的 float64 张量}。"""
    n = len(ids)
    dec = M.Qwen3_5DecoderLayer(tc, layer)
    pre = P + f"layers.{layer}."
    sd = {k: W[pre + k] for k in dec.state_dict()}
    dec.load_state_dict(sd, strict=True)
    norm = M.Qwen3_5RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)
    norm.load_state_dict({"weight": W[P + "norm.weight"]}, strict=True)
    rot = M.Qwen3_5TextRotaryEmbedding(tc)
    full = tc.layer_types[layer] == "full_attention"   # 层的属性名随版本变（5.23.0 叫 block_type，5.8 叫 layer_type），直接读配置
    dec.to(dtype); norm.to(dtype)

    got = {}

    def hook(name):
        def f(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            got[name] = o.detach().to(torch.float64).reshape(n, -1)
        return f

    hooks = {"input_layernorm": "attn_in", "post_attention_layernorm": "mlp_in", "mlp": "mlp_out",
             "mlp.gate_proj": "gate", "mlp.up_proj": "up"}
    if full:
        hooks.update({"self_attn": "attn_out", "self_attn.q_proj": "q_proj", "self_attn.k_proj": "k_proj",
                      "self_attn.v_proj": "v_proj", "self_attn.q_norm": "q_norm", "self_attn.k_norm": "k_norm"})
    else:
        hooks.update({"linear_attn": "attn_out", "linear_attn.in_proj_qkv": "qkv", "linear_attn.in_proj_z": "z",
                      "linear_attn.in_proj_a": "a", "linear_attn.in_proj_b": "b", "linear_attn.norm": "gnorm"})
    mods = dict(dec.named_modules())
    for m, name in hooks.items():
        mods[m].register_forward_hook(hook(name))

    with torch.no_grad():
        h = W[P + "embed_tokens.weight"][torch.tensor(ids)].to(dtype).unsqueeze(0)     # [1, n, hidden]
        pos = torch.arange(n).unsqueeze(0)
        cos_sin = rot(h, pos)
        mask = None
        if full:
            # eager 注意力用加法掩码：未来位置加 -inf
            mask = torch.full((n, n), -math.inf, dtype=dtype).triu(1)[None, None]
        out = dec(h, position_embeddings=cos_sin, attention_mask=mask, position_ids=pos)
        got["out"] = out.to(torch.float64).reshape(n, -1)
        got["final_norm"] = norm(out).to(torch.float64).reshape(n, -1)
    return got


def main():
    cfg_path, layer, ids, files = sys.argv[1], int(sys.argv[2]), [int(v) for v in sys.argv[3].split(",")], sys.argv[4:]
    cfg = Qwen3_5Config.from_dict(json.load(open(cfg_path)))
    tc = cfg.text_config
    tc._attn_implementation = "eager"
    torch.manual_seed(0)

    steps = read_dump(sys.stdin)
    C = {s[0]: torch.tensor(s[3], dtype=torch.float64).reshape(s[1], s[2]) for s in steps}

    pre = P + f"layers.{layer}."
    names = set()
    for f in files:
        with safe_open(f, framework="pt") as st:
            names |= {k for k in st.keys() if k.startswith(pre)}
    W = load(files, names | {P + "embed_tokens.weight", P + "norm.weight"})

    ref = run(tc, layer, ids, W, torch.float64)
    bf = run(tc, layer, ids, W, torch.bfloat16)

    order = [s[0] for s in steps if s[0] in ref]
    worst, lines, bad = 0.0, [], []
    for name in order:
        g, r = C[name], ref[name]
        if g.shape != r.shape:
            print(f"{name}：形状不一致，C 是 {list(g.shape)}，官方是 {list(r.shape)}")
            return 1
        rms = r.pow(2).mean().sqrt().item()
        e = (g - r).abs().max().item() / rms if rms > 0 else (g - r).abs().max().item()
        eb = (bf[name] - g).abs().max().item() / rms if rms > 0 else 0.0
        lines.append(f"    {name:<12} 与官方 float64 差 {e:.1e}，官方 BF16 差 {eb:.1e}")
        if not e <= TOL:
            bad.append(name)
        worst = max(worst, e)
    kind = "全注意力" if tc.layer_types[layer] == "full_attention" else "线性注意力"
    if bad:
        print(f"第 {layer} 层（{kind}）与官方实现不一致：{'、'.join(bad)} 超过 {TOL:.1e}")
        print("\n".join(lines))
        return 1
    print(f"第 {layer} 层（{kind}），{len(order)} 个张量，与官方 float64 的最大相对误差 {worst:.1e}")
    if os.environ.get("LAYER_VERBOSE"):
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
