#!/usr/bin/env python3
"""用真实模型的权重生成第 3 章算子的测试用例：tests/make_ops_real.py <模型目录> <输出目录>

模型目录里要有 config.json、model.safetensors.index.json 和各个分片。输出目录里写：
    inputs.safetensors  激活值一侧的输入（随机数，或从嵌入表里取出的真实行）
    cases.txt           格式同 make_ops_fixtures.py；权重直接用模型里的张量名，文件列表里带上对应的分片

维度、eps、theta 都从 config.json 和权重的形状读出，不写死。顺便打印各个归一化权重的统计量：
零中心的 RMSNorm 权重应当在 0 附近，带门控的那种应当在 1 附近。
"""
import json, math, os, random, sys

sys.path.insert(0, sys.path[0])
from make_ops_fixtures import write  # noqa: E402
from ops_ref import Tensors, bf16_row  # noqa: E402

P = "model.language_model."


def main(model, out):
    os.makedirs(out, exist_ok=True)
    cfg = json.load(open(f"{model}/config.json"))
    tc = cfg.get("text_config", cfg)
    rope = tc.get("rope_parameters") or tc
    eps = tc["rms_norm_eps"]
    theta = rope["rope_theta"]
    head_dim = tc["head_dim"]
    rot = int(head_dim * rope.get("partial_rotary_factor", 1.0))
    types = tc["layer_types"]
    lin = types.index("linear_attention")
    full = types.index("full_attention")
    print(f"配置：eps={eps} theta={theta} head_dim={head_dim} rot={rot}，第 {lin} 层是线性注意力，第 {full} 层是全注意力")

    wmap = json.load(open(f"{model}/model.safetensors.index.json"))["weight_map"]
    names = {
        "embed": P + "embed_tokens.weight",
        "in_norm": P + f"layers.{lin}.input_layernorm.weight",
        "final_norm": P + "norm.weight",
        "qkv": P + f"layers.{lin}.linear_attn.in_proj_qkv.weight",
        "conv": P + f"layers.{lin}.linear_attn.conv1d.weight",
        "gnorm": P + f"layers.{lin}.linear_attn.norm.weight",
        "down": P + f"layers.{lin}.mlp.down_proj.weight",
        "q": P + f"layers.{full}.self_attn.q_proj.weight",
        "q_norm": P + f"layers.{full}.self_attn.q_norm.weight",
    }
    for n in names.values():
        if n not in wmap:
            raise SystemExit(f"模型里没有张量 {n}")
    W = Tensors(sorted({f"{model}/{wmap[n]}" for n in names.values()}))
    shape = lambda k: W.get(names[k])[1]

    for k in ("in_norm", "final_norm", "q_norm", "gnorm"):
        v = W.floats(names[k])[1]
        mean = math.fsum(v) / len(v)
        print(f"权重 {names[k]}：{len(v)} 个，均值 {mean:+.4f}，范围 [{min(v):+.4f}, {max(v):+.4f}]")

    rng = random.Random(4)
    vocab, hidden = shape("embed")
    ids = [i for i in (0, 1, 100, 248044, 248045) if i < vocab] + [vocab - 1]   # 248044 起是特殊 token，最后一行是填充
    emb = []                                         # 嵌入表只解码要用的几行，当作 RMSNorm 的真实输入
    for i in ids:
        emb += list(bf16_row(W.get(names["embed"])[2], i, hidden))

    heads = shape("q")[0] // (2 * head_dim)          # q_proj 同时产出 query 和门控，所以除以 2
    nv, dv = tc["linear_num_value_heads"], shape("gnorm")[0]
    conv_dim = shape("conv")[0]
    inter = shape("down")[1]
    g = lambda n, s=1.0: [rng.gauss(0, s) for _ in range(n)]
    t = {
        "ids": ("I32", [len(ids)], ids),
        "emb": ("F32", [len(ids), hidden], emb),
        "x_hidden": ("F32", [2, hidden], g(2 * hidden)),
        "x_inter": ("F32", [1, inter], g(inter, 0.1)),
        "x_heads": ("F32", [3, heads, head_dim], g(3 * heads * head_dim)),
        "x_v": ("F32", [3, nv, dv], g(3 * nv * dv)),
        "z_v": ("F32", [3, nv, dv], g(3 * nv * dv, 2)),
        "x_conv": ("F32", [5, conv_dim], g(5 * conv_dim)),
        "pos": ("I32", [6], [0, 1, 2, 1000, 100000, 262143]),
        "x_rope": ("F32", [6, heads, head_dim], g(6 * heads * head_dim)),
    }
    write(f"{out}/inputs.safetensors", t)

    inp = f"{out}/inputs.safetensors"
    files = lambda *ks: " ".join([inp] + sorted({f"{model}/{wmap[names[k]]}" for k in ks}))
    cases = [
        f"嵌入 embed w={names['embed']} {files('embed')}",
        f"输入归一化 rmsnorm x=emb w={names['in_norm']} eps={eps} {files('in_norm')}",
        f"最终归一化 rmsnorm x=emb w={names['final_norm']} eps={eps} {files('final_norm')}",
        f"q_norm rmsnorm x=x_heads w={names['q_norm']} eps={eps} {files('q_norm')}",
        f"门控归一化 rmsnorm_gated x=x_v z=z_v w={names['gnorm']} eps={eps} {files('gnorm')}",
        f"L2归一化 l2norm x=x_v eps=1e-6 {inp}",
        f"线性注意力输入投影 matmul x=x_hidden w={names['qkv']} {files('qkv')}",
        f"全注意力q投影 matmul x=x_hidden w={names['q']} {files('q')}",
        f"前馈下投影 matmul x=x_inter w={names['down']} {files('down')}",
        f"因果卷积 conv1d x=x_conv w={names['conv']} {files('conv')}",
        f"RoPE rope x=x_rope theta={theta} rot={rot} {inp}",
    ]
    with open(f"{out}/cases.txt", "w") as f:
        f.write("\n".join(cases) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
