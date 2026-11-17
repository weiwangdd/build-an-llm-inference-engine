#!/usr/bin/env python3
"""用真实模型生成第 4 章单层测试的用例：tests/make_layer_real.py <模型目录> <输出目录> <token id，逗号分隔>

token id 由 tests/run_layer.sh 用第 2 章的分词器（build/tok）从一句真实的话编码出来。
挑四层来测：第一个线性注意力层、第一个全注意力层、最后一个线性注意力层、最后一个全注意力层，
层的类型从 config.json 的 layer_types 读。每一层的输入都是这句话的嵌入（第 5 章才把层串起来）。

输出目录里写：
    cases.txt   每行：<名字> <层号> <ids>，格式同 make_layer_fixtures.py
    shards.txt  这些层、嵌入表和最终归一化所在的分片，每行一个路径
"""
import json, os, sys

P = "model.language_model."


def main(model, out, ids):
    os.makedirs(out, exist_ok=True)
    tc = json.load(open(f"{model}/config.json"))["text_config"]
    types = tc["layer_types"]
    lin = [i for i, t in enumerate(types) if t == "linear_attention"]
    full = [i for i, t in enumerate(types) if t == "full_attention"]
    layers = [("第一个线性注意力层", lin[0]), ("第一个全注意力层", full[0]),
              ("最后一个线性注意力层", lin[-1]), ("最后一个全注意力层", full[-1])]
    wmap = json.load(open(f"{model}/model.safetensors.index.json"))["weight_map"]
    want = [P + "embed_tokens.weight", P + "norm.weight"]
    shards = set()
    for name, f in wmap.items():
        if name in want or any(name.startswith(P + f"layers.{i}.") for _, i in layers):
            shards.add(f"{model}/{f}")
    for n in want:
        if n not in wmap:
            raise SystemExit(f"模型里没有张量 {n}")
    print(f"token：{ids}（{len(ids.split(','))} 个）；层：" + "、".join(f"{n} {i}" for n, i in layers)
          + f"；用到 {len(shards)} 个分片")
    with open(f"{out}/cases.txt", "w") as f:
        f.write("".join(f"{n}（第{i}层） {i} {ids}\n" for n, i in layers))
    with open(f"{out}/shards.txt", "w") as f:
        f.write("".join(s + "\n" for s in sorted(shards)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
