#!/usr/bin/env python3
"""生成第 6 章生成循环测试的输入。

    tests/06-generate/make_gen_fixtures.py <输出目录>

模型就是第 5 章的 8 层随机小模型（tests/05-model/make_model_fixtures.py，同一个随机种子，所以完全相同），
这里另外写几份 generation_config.json 和用例清单：
    gen_*.json      采样参数各不相同的 generation_config.json
    cases.txt       每行：<名字> <generation_config.json> <ids> <max_new> [generate 的选项 ...]
    tie_model.safetensors, tie_cases.txt
                    同一个模型，但输出层第 2i 行和第 2i+1 行相同，于是每一步的 logits 都两两并列（分数完全相等），
                    专门检查贪心和 top-k 在并列时的取舍；用例格式同 cases.txt
    bad_cases.txt   每行：<名字>，之后是 build/generate 的完整参数，C 端必须以退出码 1 拒绝
"""
import json, os, random, sys

sys.path.insert(0, sys.path[0])
# 前面章节的参考实现在相邻的目录里（tests/NN-名字/）
sys.path[1:1] = [os.path.join(sys.path[0], "..", d) for d in ("03-ops", "04-layer", "05-model")]
import make_model_fixtures  # noqa: E402
from make_layer_fixtures import weights  # noqa: E402
from make_ops_fixtures import write  # noqa: E402

# 与 Qwen3.8-27B 的 generation_config.json 字段完全相同，只是 id 换成小词表里的
QWEN = {"bos_token_id": 39, "do_sample": True, "eos_token_id": [38, 39], "pad_token_id": 39,
        "temperature": 1.0, "top_k": 20, "top_p": 0.95}


def main(out):
    make_model_fixtures.main(out)
    rng = random.Random(6)
    V = 40

    def gen(name, **kw):
        g = dict(QWEN, **kw)
        path = f"{out}/gen_{name}.json"
        with open(path, "w") as f:
            json.dump(g, f, indent=2)
        return path

    # 结束符都是 38、39。随机小模型选中它们的机会不小，所以有的用例会在中途因结束符停下，有的生成满个数；
    # 停止条件另外由 run_gen.sh 里的已知答案专门测
    qwen = gen("qwen")
    cold = gen("cold", temperature=0.5, top_k=0, top_p=0.8)
    hot = gen("hot", temperature=1.5, top_k=5, top_p=1.0)
    plain = gen("plain", temperature=1.0, top_k=0, top_p=1.0)
    gen("greedy", do_sample=False)

    ids8 = "3,17,17,0,39,5,22,8"
    ids70 = ",".join(str(rng.randrange(V)) for _ in range(70))     # 比官方分块算法的块长 64 还长
    cases = [
        f"贪心_从1个token生成8个 {qwen} 7 8 --greedy",
        f"贪心_读generation_config {out}/gen_greedy.json {ids8} 6",
        f"采样_Qwen默认参数 {qwen} {ids8} 16 --seed 1",
        f"采样_低温加top_p {cold} {ids8} 16 --seed 2",
        f"采样_高温加top_k {hot} {ids8} 16 --seed 3",
        f"采样_只有温度 {plain} 7 16 --seed 4",
        f"采样_命令行覆盖参数 {qwen} 7 12 --seed 5 --temperature 0.7 --top-k 3 --top-p 0.6",
        f"采样_70个token的提示词 {qwen} {ids70} 6 --seed 6",
    ]
    with open(f"{out}/cases.txt", "w") as f:
        f.write("\n".join(cases) + "\n")

    # 并列：按 make_model_fixtures 的办法（同一个随机种子）重新生成同一个模型，只把输出层改成两两相同
    r5 = random.Random(5)
    mc = make_model_fixtures.config()
    H = mc["text_config"]["hidden_size"]
    w = weights(mc, r5)
    head = [r5.gauss(0, 1 / H ** 0.5) for _ in range(V * H)]
    for i in range(1, V, 2):
        head[i * H:(i + 1) * H] = head[(i - 1) * H:i * H]
    w["lm_head.weight"] = ("BF16", [V, H], head)
    write(f"{out}/tie_model.safetensors", w, pad=1)
    ties = [
        f"并列_贪心取id小的 {qwen} 7 6 --greedy",
        f"并列_top_k为奇数 {qwen} {ids8} 12 --seed 7 --top-k 3 --top-p 1",
        f"并列_top_k为1 {qwen} {ids8} 8 --seed 8 --top-k 1 --top-p 1",
    ]
    with open(f"{out}/tie_cases.txt", "w") as f:
        f.write("\n".join(ties) + "\n")

    # ---------- 必须被拒绝的输入 ----------
    cfg, model = f"{out}/config.json", f"{out}/model.safetensors"

    def badgen(name, edit):
        g = dict(QWEN)
        edit(g)
        path = f"{out}/bad_gen_{name}.json"
        with open(path, "w") as f:
            json.dump(g, f)
        return path

    def setk(k, v):
        return lambda g: g.__setitem__(k, v)

    bad = [
        f"不支持的字段 {cfg} {badgen('rep', setk('repetition_penalty', 1.1))} 1,2 4 {model}",
        f"缺少eos {cfg} {badgen('noeos', lambda g: g.pop('eos_token_id'))} 1,2 4 {model}",
        f"eos超出词表 {cfg} {badgen('eos40', setk('eos_token_id', [38, 40]))} 1,2 4 {model}",
        f"eos是空数组 {cfg} {badgen('eos0', setk('eos_token_id', []))} 1,2 4 {model}",
        f"eos不是整数 {cfg} {badgen('eosstr', setk('eos_token_id', '39'))} 1,2 4 {model}",
        f"缺少do_sample {cfg} {badgen('nods', lambda g: g.pop('do_sample'))} 1,2 4 {model}",
        f"采样却缺少top_k {cfg} {badgen('notopk', lambda g: g.pop('top_k'))} 1,2 4 {model}",
        f"温度为0 {cfg} {badgen('t0', setk('temperature', 0))} 1,2 4 {model}",
        f"top_p为0 {cfg} {badgen('p0', setk('top_p', 0))} 1,2 4 {model}",
        f"top_p大于1 {cfg} {badgen('p15', setk('top_p', 1.5))} 1,2 4 {model}",
        f"top_k为负 {cfg} {badgen('kneg', setk('top_k', -1))} 1,2 4 {model}",
        f"pad超出词表 {cfg} {badgen('pad', setk('pad_token_id', 40))} 1,2 4 {model}",
        f"max_new为0 {cfg} {qwen} 1,2 0 {model}",
        f"max_new不是数 {cfg} {qwen} 1,2 x {model}",
        f"超过max_position_embeddings {cfg} {qwen} 1 4096 {model}",
        f"id越界 {cfg} {qwen} 1,40 4 {model}",
        f"logits有NaN {cfg} {qwen} 1,2 4 {out}/bad_nan.safetensors",
        f"不认识的选项 {cfg} {qwen} 1,2 4 --beam 2 {model}",
        f"命令行温度为0 {cfg} {qwen} 1,2 4 --temperature 0 {model}",
        f"命令行top_p不是数 {cfg} {qwen} 1,2 4 --top-p abc {model}",
        f"chunk为0 {cfg} {qwen} 1,2 4 --chunk 0 {model}",
        f"seed为负 {cfg} {qwen} 1,2 4 --seed -1 {model}",
        f"缺少分片 {cfg} {qwen} 1,2 4 --seed",
    ]
    with open(f"{out}/bad_cases.txt", "w") as f:
        f.write("\n".join(bad) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
