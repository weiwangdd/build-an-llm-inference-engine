#!/usr/bin/env python3
"""生成第 5 章整模测试的输入：一个结构与 Qwen3.8 相同、但各维度都很小的随机模型。

    tests/make_model_fixtures.py <输出目录>

输出目录里写：
    config.json, model.safetensors   小模型：8 层（两组“3 个线性注意力 + 1 个全注意力”），宽 64，词表 40，
                                     带输出层 lm_head.weight
    cases.txt       每行：<名字> <ids> <生成个数>，C 端和参考实现都要跑
    bad_cases.txt   每行：<名字> <config.json> <ids> <生成个数> <输出方式> <分片>，C 端必须以退出码 1 拒绝
    bad_*.json、bad_*.safetensors  写坏的配置和权重

层的配置和权重的生成方式沿用第 4 章的 tests/make_layer_fixtures.py，只是层数从 4 变成 8：
每一层的权重各不相同，如果 C 把第 i 层的权重用在了第 j 层，或者把层的顺序弄反，结果就会不同。
"""
import copy, json, math, os, random, sys

sys.path.insert(0, sys.path[0])
from make_ops_fixtures import write  # noqa: E402
from make_layer_fixtures import CONFIG as LAYER_CONFIG, weights, P  # noqa: E402


def config():
    c = copy.deepcopy(LAYER_CONFIG)
    t = c["text_config"]
    t["num_hidden_layers"] = 8
    t["layer_types"] = (["linear_attention"] * 3 + ["full_attention"]) * 2
    return c


def main(out):
    os.makedirs(out, exist_ok=True)
    rng = random.Random(5)
    cfg = config()
    t = cfg["text_config"]
    V, H = t["vocab_size"], t["hidden_size"]
    with open(f"{out}/config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    w = weights(cfg, rng)
    # 输出层：与嵌入表同形 [vocab, hidden]，量级让 logits 大致是 1 的量级
    w["lm_head.weight"] = ("BF16", [V, H], [rng.gauss(0, 1 / math.sqrt(H)) for _ in range(V * H)])
    write(f"{out}/model.safetensors", w, pad=1)

    ids8 = "3,17,17,0,39,5,22,8"
    ids = [str(rng.randrange(V)) for _ in range(70)]
    ids70 = ",".join(ids)                       # 比 transformers 的分块长度 64 还长
    ids62 = ",".join(ids[:62])                  # 从 62 个生成 6 个，前向的长度从 62 到 67，越过 64
    cases = ["1个token 7 0", f"8个token {ids8} 0", f"70个token {ids70} 0",
             "贪心_从1个token生成8个 7 8", f"贪心_从8个token生成6个 {ids8} 6",
             f"贪心_越过分块边界 {ids62} 6"]
    with open(f"{out}/cases.txt", "w") as f:
        f.write("\n".join(cases) + "\n")

    # ---------- 必须被拒绝的输入 ----------
    model, good = f"{out}/model.safetensors", f"{out}/config.json"

    def badcfg(name, edit):
        c = copy.deepcopy(cfg)
        edit(c)
        path = f"{out}/bad_{name}.json"
        with open(path, "w") as f:
            json.dump(c, f)
        return path

    def badw(name, edit):
        ww = dict(w)
        edit(ww)
        path = f"{out}/bad_{name}.safetensors"
        write(path, ww)
        return path

    def set_text(k, v):
        return lambda c: c["text_config"].__setitem__(k, v)

    nan_head = list(w["lm_head.weight"][2])
    nan_head[5 * H] = float("nan")                  # 第 5 个 token 的分数会是 NaN
    bad = [
        f"顶层tie_word_embeddings为true {badcfg('tie0', lambda c: c.__setitem__('tie_word_embeddings', True))} 1,2 0 all {model}",
        f"text_config的tie_word_embeddings为true {badcfg('tie1', set_text('tie_word_embeddings', True))} 1,2 0 all {model}",
        f"层数比权重多 {badcfg('layers9', lambda c: c['text_config'].update(num_hidden_layers=9, layer_types=c['text_config']['layer_types'] + ['linear_attention']))} 1,2 0 all {model}",
        f"缺少lm_head {good} 1,2 0 all " + badw("no_head", lambda ww: ww.pop("lm_head.weight")),
        f"lm_head形状不对 {good} 1,2 0 all " + badw("head_shape", lambda ww: ww.update(
            {"lm_head.weight": ("BF16", [V - 1, H], [0.0] * (V - 1) * H)})),
        f"缺少第7层的张量 {good} 1,2 0 all " + badw("no_l7", lambda ww: ww.pop(P + "layers.7.mlp.up_proj.weight")),
        f"logits有NaN {good} 1,2 1 all " + badw("nan", lambda ww: ww.update(
            {"lm_head.weight": ("BF16", [V, H], nan_head)})),
        f"id越界 {good} 1,40 0 all {model}",
        f"id为空 {good} 1,,2 0 all {model}",
        f"生成个数不是数 {good} 1,2 x all {model}",
        f"生成个数为负 {good} 1,2 -1 all {model}",
        f"输出方式不认识 {good} 1,2 0 some {model}",
    ]
    with open(f"{out}/bad_cases.txt", "w") as f:
        f.write("\n".join(bad) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
