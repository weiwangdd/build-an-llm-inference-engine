#!/usr/bin/env python3
"""第 6 章的第一层对拍：检查 C 的采样和停止条件，只用标准库。

    gen_ref.py <config.json> <generation_config.json> <ids> <max_new> [generate 的选项 ...] <分片 ...> < build/generate --dump 的输出

参数与 build/generate 完全相同（分片不读）。logits 本身对不对不在这里查：它与不带缓存的前向逐位相同，
由 tests/run_gen.sh 用 cmp 检查，不带缓存的前向又由第 5 章的对拍负责。这里拿 C 打印出来的每一行 logits，
独立地做一遍采样，检查：
  1. 随机数：每一步的 u 是不是我们自己写的 splitmix64 从同一个种子给出的那个数（要求完全相等）；
  2. 分布：温度、top-k、top-p 之后保留哪些 token 必须完全相同，概率的相对误差不超过 1e-9。
     概率用 decimal 以 40 位有效数字算，当作精确值；
  3. 选择：按精确的累计概率，u 落在哪个 token 上，必须就是 C 选的那个；
  4. 停止：生成了结束符就停，否则生成满 max_new 个才停，停的原因要对。
2 和 3 只在“恰好压在边界上”时可能因为舍入而不同，所以同时报告离边界最近有多远。
"""
import json, sys
from decimal import Decimal, getcontext

getcontext().prec = 40
M64 = (1 << 64) - 1


class SplitMix64:
    """splitmix64，按它的原始定义写（Steele、Lea、Flood 2014）"""
    def __init__(self, seed):
        self.s = seed & M64

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & M64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)

    def uniform(self):
        return (self.next() >> 11) / 2.0 ** 53


def selfcheck():
    # 已知答案：Java 标准库的 java.util.SplittableRandom 用的是同一个算法（同样的常数和移位），
    # new SplittableRandom(seed).nextLong() 连续调用的结果（按无符号数打印），由 OpenJDK 算出
    for seed, want in ((0, [16294208416658607535, 7960286522194355700, 487617019471545679]),
                       (1234567, [6457827717110365317, 3203168211198807973, 9817491932198370423])):
        r = SplitMix64(seed)
        got = [r.next() for _ in want]
        assert got == want, f"splitmix64 自检失败：种子 {seed} 得到 {got}"


def parse_args(argv):
    cfg, gcfg, ids, max_new = argv[0], argv[1], [int(v) for v in argv[2].split(",")], int(argv[3])
    o = {"greedy": None, "temperature": None, "top_k": None, "top_p": None, "seed": 0}
    a = 4
    while a < len(argv) and argv[a].startswith("--"):
        k = argv[a]
        if k in ("--greedy", "--sample"):
            o["greedy"] = k == "--greedy"
        elif k in ("--no-cache", "--dump"):
            pass
        elif k in ("--temperature", "--top-p"):
            a += 1
            o[k[2:].replace("-", "_")] = float(argv[a])
        elif k in ("--top-k", "--seed", "--chunk"):
            a += 1
            o[k[2:].replace("-", "_")] = int(argv[a])
        a += 1
    return cfg, gcfg, ids, max_new, o


def settings(gcfg, o):
    g = json.load(open(gcfg))
    eos = g["eos_token_id"] if isinstance(g["eos_token_id"], list) else [g["eos_token_id"]]
    s = {"do_sample": g["do_sample"], "temperature": g.get("temperature", 1.0), "top_k": g.get("top_k", 0),
         "top_p": g.get("top_p", 1.0)}
    if o["greedy"] is not None:
        s["do_sample"] = not o["greedy"]
    for k in ("temperature", "top_k", "top_p"):
        if o[k] is not None:
            s[k] = o[k]
    return s, set(eos)


def read_dump(stream):
    """读 build/generate --dump 的输出，返回 ([{logits, dist, draw, token}...], 停止原因)"""
    lines = iter(stream.read().split("\n"))
    steps, stop = [], None
    for line in lines:
        if not line:
            continue
        key, _, val = line.partition(" ")
        if key == "step":
            assert int(val) == len(steps), f"step 编号不对：{line}"
            steps.append({"draw": None})
        elif key == "logits":
            steps[-1]["logits"] = [float.fromhex(next(lines)) for _ in range(int(val))]
        elif key == "dist":
            d = [next(lines).split(" ") for _ in range(int(val))]
            steps[-1]["dist"] = [(int(i), float.fromhex(p)) for i, p in d]
        elif key == "draw":
            steps[-1]["draw"] = float.fromhex(val)
        elif key == "token":
            steps[-1]["token"] = int(val)
        elif key == "stop":
            stop = val
        else:
            raise SystemExit(f"不认识的行：{line[:60]}")
    return steps, stop


def ref_dist(logits, s):
    """返回 (按 id 排好的 [(id, 精确概率 Decimal)], top-p 边界的余量)"""
    V = len(logits)
    if not s["do_sample"]:
        best = max(range(V), key=lambda i: (logits[i], -i))
        return [(best, Decimal(1))], None
    z = [x / s["temperature"] for x in logits]        # 与 C 一样是一次 IEEE 双精度除法，结果完全相同
    order = sorted(range(V), key=lambda i: (-z[i], i))
    K = V
    if 0 < s["top_k"] < V:
        th = z[order[s["top_k"] - 1]]
        K = sum(1 for v in z if v >= th)              # 与第 k 名同分的都保留
    m = Decimal(z[order[0]])
    e = {i: (Decimal(z[i]) - m).exp() for i in order[:K]}
    tot = sum(e.values())
    P, margin = K, None
    if s["top_p"] < 1:
        p, above, P = Decimal(s["top_p"]), Decimal(0), 0
        while P < K and above / tot < p:
            above += e[order[P]]
            P += 1
        # 离边界多远：最后一个保留的“前面累计”和第一个去掉的“前面累计”与 top_p 的距离
        cums, acc = [], Decimal(0)
        for r in range(min(P + 1, K)):
            cums.append(acc / tot)
            acc += e[order[r]]
        margin = min(abs(c - p) for c in cums[max(P - 1, 0):])
    kept = sorted(order[:P])
    sm = sum(e[i] for i in kept)
    return [(i, e[i] / sm) for i in kept], margin


def main():
    selfcheck()
    _, gcfg, prompt, max_new, o = parse_args(sys.argv[1:])
    s, eos = settings(gcfg, o)
    steps, stop = read_dump(sys.stdin)
    rng = SplitMix64(o["seed"])
    worst, pick_margin, p_margin = 0.0, None, None
    for k, st in enumerate(steps):
        if "logits" not in st or "dist" not in st or "token" not in st:
            print(f"第 {k} 步缺少 logits、dist 或 token（要用 --dump）")
            return 1
        ref, margin = ref_dist(st["logits"], s)
        if margin is not None:
            p_margin = margin if p_margin is None else min(p_margin, margin)
        if [i for i, _ in ref] != [i for i, _ in st["dist"]]:
            print(f"第 {k} 步保留的 token 不同：C {[i for i, _ in st['dist']][:10]}，参考 {[i for i, _ in ref][:10]}"
                  f"；top-p 边界余量 {float(margin) if margin is not None else '-'}")
            return 1
        for (i, pr), (_, pc) in zip(ref, st["dist"]):
            e = abs(Decimal(pc) - pr) / pr
            if not e <= Decimal("1e-9"):
                print(f"第 {k} 步 token {i} 的概率 C 是 {pc!r}，参考 {float(pr)!r}，相对误差 {float(e):.1e}")
                return 1
            worst = max(worst, float(e))
        if s["do_sample"]:
            u = rng.uniform()
            if st["draw"] != u:
                print(f"第 {k} 步的随机数 C 是 {st['draw']!r}，应为 {u!r}")
                return 1
            acc, want, gap = Decimal(0), ref[-1][0], None
            du = Decimal(u)
            for i, p in ref:
                acc += p
                d = abs(acc - du)
                gap = d if gap is None else min(gap, d)
                if du < acc:
                    want = i
                    break
            pick_margin = gap if pick_margin is None else min(pick_margin, gap)
        else:
            if st["draw"] is not None:
                print(f"第 {k} 步是贪心解码，不应有随机数")
                return 1
            want = ref[0][0]
        if st["token"] != want:
            print(f"第 {k} 步 C 选了 {st['token']}，应为 {want}")
            return 1
        last = k + 1 == len(steps)
        hit = st["token"] in eos
        if hit and not last:
            print(f"第 {k} 步生成了结束符 {st['token']}，却没有停")
            return 1
    if not steps:
        print("一个 token 也没有生成")
        return 1
    want_stop = "eos" if steps[-1]["token"] in eos else "length"
    if want_stop == "length" and len(steps) != max_new:
        print(f"没有生成结束符，却在第 {len(steps)} 个 token 停了（max_new 是 {max_new}）")
        return 1
    if stop != want_stop:
        print(f"停止原因应为 {want_stop}，C 打印的是 {stop}")
        return 1

    what = "贪心解码" if not s["do_sample"] else f"采样（T={s['temperature']}，top_k={s['top_k']}，top_p={s['top_p']}）"
    msg = f"{what}生成 {len(steps)} 个，停止原因 {stop}；分布最大相对误差 {worst:.1e}"
    if pick_margin is not None:
        msg += f"；随机数离累计概率边界最近 {float(pick_margin):.1e}"
    if p_margin is not None:
        msg += f"，top-p 边界余量 {float(p_margin):.1e}"
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
