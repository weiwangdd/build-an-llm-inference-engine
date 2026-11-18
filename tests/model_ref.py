#!/usr/bin/env python3
"""第 5 章的第一层对拍：逐步检查 C 的整模前向和贪心解码，只用标准库。

    model_ref.py <config.json> <ids> <生成个数 m> <all|layers> <分片 ...> < build/forward 的输出

与第 4 章的 layer_ref.py 是同一个办法：拿 C 打印出来的“这一步的输入”，用双精度独立地算出精确值，
按第 3 章的标准比较。每一层的步骤直接用 layer_ref.layer_plan，第 i 层的输入是第 i-1 层的输出 out。
此外检查：
  - 最后一次前向的输入是“提示词加上前 m-1 个生成的 token”（嵌入要求逐位相等，所以喂错 token 一定能发现）；
  - 每个生成的 token 都是对应那一行 logits 里分数最大的（有几个同样大时取 id 最小的）。

all：每一层的每一步都检查（小模型用）。
layers：C 只打印了每一层的输出，各层内部的步骤没法逐步检查，那由第二层对拍（model_official.py）
        逐层负责；这里只检查嵌入、最终归一化和 logits。真实模型的 logits 有 24.8 万列，纯 Python
        算一行要几分钟，所以只检查最后一行（贪心解码最后一步看的就是这一行）。
"""
import json, sys

sys.path.insert(0, sys.path[0])
from ops_ref import Tensors, bf16_row, _selfcheck  # noqa: E402
from layer_ref import P, read_dump, layer_plan, rmsnorm, matmul, compare  # noqa: E402


def main():
    _selfcheck()
    cfg_path, prompt, m, mode, files = (sys.argv[1], [int(v) for v in sys.argv[2].split(",")],
                                        int(sys.argv[3]), sys.argv[4], sys.argv[5:])
    t = json.load(open(cfg_path))["text_config"]
    W = Tensors(files)
    H, V, L, eps = t["hidden_size"], t["vocab_size"], t["num_hidden_layers"], t["rms_norm_eps"]

    steps = read_dump(sys.stdin)
    greedy = []
    if m > 0:
        if not steps or steps[-1][0] != "greedy" or steps[-1][2] != m:
            print(f"最后一段应为 greedy [1,{m}]")
            return 1
        greedy = [int(v) for v in steps.pop()[3]]
    ids = prompt + greedy[:-1]            # 最后一次前向的输入
    n = len(ids)
    D = {s[0]: s[3] for s in steps}

    plan = []                              # (名字, 行数, 列数, 计算函数, 是否要求完全相等)

    def step(name, cols, fn, exact=False, rows=n):
        plan.append((name, rows, cols, fn, exact))

    def silent(name, cols, fn, exact=False):  # layers 方式：这一步 C 没有打印，跳过
        if name.endswith(".out"):
            plan.append((name, n, cols, None, False))

    emb = W.get(P + "embed_tokens.weight")[2]
    step("embed", H, lambda: [(v, None) for i in ids for v in bf16_row(emb, i, H)], exact=True)
    for i in range(L):
        src = "embed" if i == 0 else f"layers.{i - 1}.out"
        layer_plan(step if mode == "all" else silent, t, W, i, n, D, src, pre=f"layers.{i}.")
    step("final_norm", H, lambda: rmsnorm(D[f"layers.{L - 1}.out"], list(W.floats(P + "norm.weight")[1]), H, eps, True))
    head = W.get("lm_head.weight")[2]
    if mode == "all":
        step("logits", V, lambda: matmul(D["final_norm"], n, head, V, H))
    else:
        last = lambda: matmul(D["final_norm"][(n - 1) * H:], 1, head, V, H)
        plan.append(("logits", n, V, last, False))

    want = [(p[0], p[1], p[2]) for p in plan]
    got = [(s[0], s[1], s[2]) for s in steps]
    if want != got:
        print(f"步骤不一致：应为 {[w[0] for w in want][:8]}…（{len(want)} 步），实际 {[g[0] for g in got][:8]}…（{len(got)} 步）")
        return 1
    worst, loose, total, checked = 0.0, 0, 0, 0
    for name, rows, cols, fn, exact in plan:
        if fn is None:
            continue
        got_v = D[name]
        if name == "logits" and mode != "all":
            got_v = got_v[(n - 1) * V:]   # 只查最后一行
        msg, stat = compare(name, got_v, fn(), exact)
        if msg:
            print(msg)
            return 1
        worst, loose, total, checked = max(worst, stat[0]), loose + stat[1], total + len(got_v), checked + 1

    # 贪心：第 k 个生成的 token 应是第 n_prompt-1+k 行 logits 的最大值（同样大取 id 小的）
    lg = D["logits"]
    for k, tok in enumerate(greedy):
        r = len(prompt) - 1 + k
        row = lg[r * V : (r + 1) * V]
        best = max(range(V), key=lambda j: (row[j], -j))
        if tok != best:
            print(f"第 {k + 1} 个生成的 token 是 {tok}，但那一行分数最大的是 {best}")
            return 1

    msg = f"{L} 层，检查了 {checked} 步、{total} 个元素，最大误差 {worst:.2f} ulp"
    if loose:
        msg += f"，另有 {loose} 个处在大量抵消处，误差在抵消项容差内"
    if greedy:
        msg += f"；贪心生成的 {len(greedy)} 个 token 都是分数最大的"
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
