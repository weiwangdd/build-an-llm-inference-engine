#!/usr/bin/env python3
"""生成第 8 章连续批处理测试的输入。

    tests/make_batch_fixtures.py <输出目录>

模型和 generation_config.json 就是第 6 章的（tests/make_gen_fixtures.py：8 层随机小模型，词表 40，
结束符 38、39）。这里另外写：
    requests.txt    请求清单（格式见 tools/batch.c）：到达时间各不相同、采样参数各不相同、提示词有长有短、
                    有的分块预填充、有的中途取消、有两个完全相同的请求，个数比 --max-batch 多，要排队
    bad_cases.txt   每行：<名字>，之后是 build/batch 的完整参数，C 端必须以退出码 1 拒绝
    bad_*.txt       写坏的请求清单
"""
import os, random, sys

sys.path.insert(0, sys.path[0])
import make_gen_fixtures  # noqa: E402


def main(out):
    make_gen_fixtures.main(out)
    rng = random.Random(8)
    V = 40

    def ids(n):
        return ",".join(str(rng.randrange(V)) for _ in range(n))

    ids8 = "3,17,17,0,39,5,22,8"
    long70 = ids(70)
    reqs = [
        # 到达步 提示词 max_new 选项
        f"0 {ids8} 12 --seed 1",
        f"0 7 10 --greedy",
        f"0 {long70} 8 --seed 2 --chunk 16",                  # 长提示词分 5 块预填充，期间别人在解码
        f"1 {ids(5)} 16 --seed 3 --temperature 0.7 --top-k 5 --top-p 0.9",
        f"2 {ids8} 12 --seed 1",                              # 与第 0 个完全相同，结果也应相同
        f"2 {ids(3)} 20 --seed 4 --cancel-after 3",           # 生成 3 个后取消
        f"3 {ids(12)} 9 --seed 5 --temperature 1.5 --top-k 0 --top-p 1",
        f"3 {ids(30)} 6 --greedy --chunk 7",
        f"5 {ids(2)} 14 --seed 6 --top-p 0.5",
        f"9 {ids(9)} 5 --seed 7 --cancel-after 1",
        f"9 {ids(20)} 11 --seed 8 --chunk 3",
        f"30 {ids(4)} 7 --seed 9",                            # 别的都结束以后才来
    ]
    with open(f"{out}/requests.txt", "w") as f:
        f.write("# 第 8 章：连续批处理的请求清单\n" + "\n".join(reqs) + "\n")

    cfg, gcfg, model = f"{out}/config.json", f"{out}/gen_qwen.json", f"{out}/model.safetensors"

    def badreq(name, lines):
        path = f"{out}/bad_{name}.txt"
        with open(path, "w") as f:
            f.write("".join(l + "\n" for l in lines))
        return path

    bad = [
        f"空清单 {cfg} {gcfg} {badreq('empty', ['# 只有注释'])} {model}",
        f"缺字段 {cfg} {gcfg} {badreq('short', ['0 1,2'])} {model}",
        f"id越界 {cfg} {gcfg} {badreq('id40', ['0 1,40 4'])} {model}",
        f"max_new为0 {cfg} {gcfg} {badreq('new0', ['0 1,2 0'])} {model}",
        f"温度为0 {cfg} {gcfg} {badreq('t0', ['0 1,2 4 --temperature 0'])} {model}",
        f"top_p大于1 {cfg} {gcfg} {badreq('p2', ['0 1,2 4 --top-p 2'])} {model}",
        f"chunk为0 {cfg} {gcfg} {badreq('chunk0', ['0 1,2 4 --chunk 0'])} {model}",
        f"cancel_after为0 {cfg} {gcfg} {badreq('cancel0', ['0 1,2 4 --cancel-after 0'])} {model}",
        f"不认识的选项 {cfg} {gcfg} {badreq('opt', ['0 1,2 4 --beam 2'])} {model}",
        f"到达步为负 {cfg} {gcfg} {badreq('neg', ['-1 1,2 4'])} {model}",
        f"超过max_ctx {cfg} {gcfg} {badreq('ctx', ['0 1,2,3,4,5 4'])} --max-ctx 8 {model}",
        f"max_batch为0 {cfg} {gcfg} {out}/requests.txt --max-batch 0 {model}",
        f"max_ctx为1 {cfg} {gcfg} {out}/requests.txt --max-ctx 1 {model}",
        f"logits有NaN {cfg} {gcfg} {badreq('nan', ['0 1,2 4'])} {out}/bad_nan.safetensors",
        f"缺少分片 {cfg} {gcfg} {out}/requests.txt --max-batch",
    ]
    with open(f"{out}/bad_cases.txt", "w") as f:
        f.write("\n".join(bad) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
