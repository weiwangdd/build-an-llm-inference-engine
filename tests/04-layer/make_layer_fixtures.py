#!/usr/bin/env python3
"""生成第 4 章单层测试的输入：一个结构与 Qwen3.8 相同、但各维度都很小的随机模型。

    tests/04-layer/make_layer_fixtures.py <输出目录>

输出目录里写：
    config.json, model.safetensors   小模型：4 层（3 个线性注意力 + 1 个全注意力），宽 64
    zero_model.safetensors           同一个模型，但每层的输出投影全是 0，用作已知答案
    cases.txt       每行：<名字> <层号> <token id，逗号分隔>，C 端和参考实现都要跑
    bad_cases.txt   每行：<名字> <config.json> <层号> <ids> <分片>，C 端必须以退出码 1 拒绝
    bad_*.json、bad_*.safetensors  写坏的配置和权重

张量的名字、形状、配置的字段与 Qwen3.8-27B 完全相同，所以 transformers 也能直接加载它（tests/04-layer/layer_official.py）。
权重是随机数，量级大致按“每一层的输出与输入同量级”来选，让各种数值情况都出现。
"""
import copy, json, math, os, random, sys

sys.path.insert(0, sys.path[0])
# 前面章节的参考实现在相邻的目录里（tests/NN-名字/）
sys.path[1:1] = [os.path.join(sys.path[0], "..", d) for d in ("03-ops",)]
from make_ops_fixtures import write  # noqa: E402

CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "tie_word_embeddings": False,
    "text_config": {
        "model_type": "qwen3_5_text",
        "vocab_size": 40,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 4,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "full_attention_interval": 4,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-06,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "attention_bias": False,
        "attn_output_gate": True,
        "attention_dropout": 0.0,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4,
        "output_gate_type": "swish",
        "max_position_embeddings": 4096,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.25,
                            "mrope_section": [1, 1, 0], "mrope_interleaved": True},
        "partial_rotary_factor": 0.25,
        "tie_word_embeddings": False,
        "dtype": "bfloat16",
    },
    "vision_config": {"depth": 1, "hidden_size": 16, "num_heads": 2, "out_hidden_size": 64},
}

P = "model.language_model."


def weights(cfg, rng):
    t = cfg["text_config"]
    H, I, V = t["hidden_size"], t["intermediate_size"], t["vocab_size"]
    hd, nh, nkv = t["head_dim"], t["num_attention_heads"], t["num_key_value_heads"]
    hk, hv, dk, dv = (t["linear_num_key_heads"], t["linear_num_value_heads"],
                      t["linear_key_head_dim"], t["linear_value_head_dim"])
    cd = 2 * hk * dk + hv * dv
    g = lambda n, s: [rng.gauss(0, s) for _ in range(n)]
    mat = lambda r, c: ("BF16", [r, c], g(r * c, 1 / math.sqrt(c)))
    vec = lambda n, mu, s: ("BF16", [n], [mu + v for v in g(n, s)])
    w = {P + "embed_tokens.weight": ("BF16", [V, H], g(V * H, 1.0)),
         P + "norm.weight": vec(H, 0.9, 0.4)}     # 像 27B 那样均值约 0.9，看 1+w 写法
    for i, kind in enumerate(t["layer_types"]):
        L = P + f"layers.{i}."
        w[L + "input_layernorm.weight"] = vec(H, 0.0, 0.3)
        w[L + "post_attention_layernorm.weight"] = vec(H, 0.0, 0.3)
        w[L + "mlp.gate_proj.weight"] = mat(I, H)
        w[L + "mlp.up_proj.weight"] = mat(I, H)
        w[L + "mlp.down_proj.weight"] = mat(H, I)
        if kind == "full_attention":
            w[L + "self_attn.q_proj.weight"] = mat(nh * hd * 2, H)
            w[L + "self_attn.k_proj.weight"] = mat(nkv * hd, H)
            w[L + "self_attn.v_proj.weight"] = mat(nkv * hd, H)
            w[L + "self_attn.o_proj.weight"] = mat(H, nh * hd)
            w[L + "self_attn.q_norm.weight"] = vec(hd, 0.2, 0.3)
            w[L + "self_attn.k_norm.weight"] = vec(hd, 0.2, 0.3)
        else:
            w[L + "linear_attn.in_proj_qkv.weight"] = mat(cd, H)
            w[L + "linear_attn.in_proj_z.weight"] = mat(hv * dv, H)
            w[L + "linear_attn.in_proj_b.weight"] = mat(hv, H)
            w[L + "linear_attn.in_proj_a.weight"] = mat(hv, H)
            w[L + "linear_attn.out_proj.weight"] = mat(H, hv * dv)
            w[L + "linear_attn.conv1d.weight"] = ("BF16", [cd, 1, t["linear_conv_kernel_dim"]], g(cd * 4, 0.5))
            # 与 transformers 的初始化一样：A 在 [1, 16] 里均匀分布；dt_bias 在 0 附近，让衰减有快有慢
            w[L + "linear_attn.A_log"] = ("BF16", [hv], [math.log(rng.uniform(1, 16)) for _ in range(hv)])
            w[L + "linear_attn.dt_bias"] = ("BF16", [hv], g(hv, 1.0))
            w[L + "linear_attn.norm.weight"] = vec(dv, 0.9, 0.1)
    return w


def main(out):
    os.makedirs(out, exist_ok=True)
    rng = random.Random(4)
    cfg = CONFIG
    with open(f"{out}/config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    w = weights(cfg, rng)
    write(f"{out}/model.safetensors", w, pad=1)
    # 已知答案用：把每层最后的输出投影（o_proj、out_proj、down_proj）全部置零，这一层就应当原样输出
    zero = {k: (v[0], v[1], [0.0] * len(v[2])) if k.endswith(("o_proj.weight", "out_proj.weight", "down_proj.weight"))
            else v for k, v in w.items()}
    write(f"{out}/zero_model.safetensors", zero)

    V = cfg["text_config"]["vocab_size"]
    ids8 = "3,17,17,0,39,5,22,8"
    ids70 = ",".join(str(rng.randrange(V)) for _ in range(70))   # 比 transformers 的分块长度 64 还长
    cases = [f"线性注意力_1个token 0 7", f"线性注意力 0 {ids8}", f"线性注意力_第2层 2 {ids8}",
             f"线性注意力_70个token 1 {ids70}",
             f"全注意力_1个token 3 7", f"全注意力 3 {ids8}", f"全注意力_70个token 3 {ids70}"]
    with open(f"{out}/cases.txt", "w") as f:
        f.write("\n".join(cases) + "\n")

    # ---------- 必须被拒绝的输入 ----------
    def badcfg(name, edit):
        c = copy.deepcopy(cfg)
        edit(c["text_config"], c)
        path = f"{out}/bad_{name}.json"
        with open(path, "w") as f:
            json.dump(c, f)
        return path

    def setk(k, v):
        def e(t, c):
            t[k] = v
        return e

    def delk(k):
        def e(t, c):
            del t[k]
        return e

    def rope(k, v):
        def e(t, c):
            t["rope_parameters"][k] = v
        return e

    model = f"{out}/model.safetensors"
    good = f"{out}/config.json"
    bad = []
    for name, edit in [
        ("激活函数不支持", setk("hidden_act", "gelu")),
        ("rope_type不支持", rope("rope_type", "yarn")),
        ("旋转维数为奇数", rope("partial_rotary_factor", 0.1875)),
        ("旋转维数超过head_dim", rope("partial_rotary_factor", 1.5)),
        ("有偏置", setk("attention_bias", True)),
        ("关掉输出门控", setk("attn_output_gate", False)),
        ("门控激活不支持", setk("output_gate_type", "sigmoid")),
        ("layer_types长度不符", setk("layer_types", ["linear_attention"] * 3)),
        ("layer_types不认识", setk("layer_types", ["linear_attention"] * 3 + ["sliding_attention"])),
        ("缺少head_dim", delk("head_dim")),
        ("head_dim不是整数", setk("head_dim", 16.5)),
        ("头数不整除", setk("num_key_value_heads", 3)),
        ("eps为负", setk("rms_norm_eps", -1e-6)),
        ("模型类型不对", lambda t, c: c.update(model_type="llama")),
        ("缺少text_config", lambda t, c: c.pop("text_config")),
    ]:
        bad.append(f"{name} {badcfg(name, edit)} 0 1,2 {model}")
    bad += [
        f"配置文件不存在 {out}/不存在.json 0 1 {model}",
        f"配置不是JSON {out}/model.safetensors 0 1 {model}",
        f"层号越界 {good} 4 1,2 {model}",
        f"层号不是数 {good} x 1,2 {model}",
        f"id越界 {good} 0 1,40 {model}",
        f"id为负 {good} 0 -1 {model}",
        f"id为空 {good} 0 1,,2 {model}",
        f"id有多余字符 {good} 0 1,2x {model}",
    ]
    # 写坏的权重：少一个张量、形状不对、类型不对
    def badw(name, edit):
        ww = dict(w)
        edit(ww)
        path = f"{out}/bad_{name}.safetensors"
        write(path, ww)
        return path

    L0, L3 = P + "layers.0.", P + "layers.3."
    bad += [
        f"缺少A_log {good} 0 1 " + badw("no_alog", lambda ww: ww.pop(L0 + "linear_attn.A_log")),
        f"缺少最终归一化 {good} 0 1 " + badw("no_norm", lambda ww: ww.pop(P + "norm.weight")),
        f"q_norm长度不对 {good} 3 1 " + badw("qnorm", lambda ww: ww.update(
            {L3 + "self_attn.q_norm.weight": ("BF16", [8], [0.0] * 8)})),
        f"q_proj少了门控 {good} 3 1 " + badw("qproj", lambda ww: ww.update(
            {L3 + "self_attn.q_proj.weight": ("BF16", [64, 64], [0.0] * 64 * 64)})),
        f"卷积核形状不对 {good} 0 1 " + badw("conv", lambda ww: ww.update(
            {L0 + "linear_attn.conv1d.weight": ("BF16", [32, 4], [0.0] * 128)})),
        f"矩阵不是BF16 {good} 0 1 " + badw("f32", lambda ww: ww.update(
            {L0 + "mlp.up_proj.weight": ("F32", [96, 64], [0.0] * 96 * 64)})),
        f"嵌入表行数不符 {good} 0 1 " + badw("emb", lambda ww: ww.update(
            {P + "embed_tokens.weight": ("BF16", [39, 64], [0.0] * 39 * 64)})),
    ]
    with open(f"{out}/bad_cases.txt", "w") as f:
        f.write("\n".join(bad) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
