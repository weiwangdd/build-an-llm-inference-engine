#!/usr/bin/env python3
"""第 4 章的第一层对拍：逐步检查 C 的单层前向，只用标准库。

    layer_ref.py <config.json> <层号> <ids> <分片 ...> < build/layer 的输出

C 端（tools/layer.c）把每一步的结果都打印出来。这里对每一步，拿 C 打印出来的“这一步的输入”，
用双精度独立地算出精确值，再按第 3 章的标准比较：
    |g - r| <= ulp(r) + 2^-32 * s
所以误差不会一步步累积，每一步都要做到正确舍入（只差一点）。

哪一步的输入是哪个张量、张量怎么切成头，是这里自己决定的，不看 C 端怎么做。
例如 q_proj 的输出怎样分成 q 和门控、线性注意力的第 j 个值头用哪个键头，两边各写一遍。
步骤的名字和顺序也要完全一致，C 端多算、少算或换了顺序都算失败。
"""
import json, math, sys

sys.path.insert(0, sys.path[0])
from ops_ref import Tensors, bf16_row, sigmoid, silu, softplus, ulp32, FLT_MAX, _selfcheck  # noqa: E402

P = "model.language_model."


def read_dump(stream):
    """读 build/layer 的输出：[(名字, 行, 列, [值...]), ...]"""
    steps, cur = [], None
    for line in stream:
        line = line.rstrip("\n")
        if " [" in line:                      # 段头“名字 [行,列]”；其余每行一个 %a 格式的数
            name, shape = line.split(" ")
            r, c = (int(v) for v in shape.strip("[]").split(","))
            cur = (name, r, c, [])
            steps.append(cur)
        else:
            cur[3].append(float.fromhex(line))
    return steps


# ---------- 参考计算：每个函数返回 [(精确值, 求和项绝对值之和或 None), ...] ----------

def matmul(x, n, wbuf, out, k):
    res = [None] * (n * out)
    xr = [x[t * k : (t + 1) * k] for t in range(n)]
    for o in range(out):
        w = bf16_row(wbuf, o, k)
        for t in range(n):
            terms = [a * b for a, b in zip(xr[t], w)]
            res[t * out + o] = (math.fsum(terms), math.fsum(map(abs, terms)))
    return res


def rmsnorm(x, w, d, eps, one_plus):
    res = []
    for i in range(0, len(x), d):
        row = x[i : i + d]
        denom = math.sqrt(math.fsum(v * v for v in row) / d + eps)
        res += [(v / denom * ((1.0 + w[j]) if one_plus else w[j]), None) for j, v in enumerate(row)]
    return res


def rmsnorm_gated(x, z, w, d, eps):
    res = []
    for i in range(0, len(x), d):
        row = x[i : i + d]
        denom = math.sqrt(math.fsum(v * v for v in row) / d + eps)
        res += [(v / denom * w[j] * silu(z[i + j]), None) for j, v in enumerate(row)]
    return res


def l2norm(x, d):
    res = []
    for i in range(0, len(x), d):
        row = x[i : i + d]
        denom = math.sqrt(math.fsum(v * v for v in row) + 1e-6)
        res += [(v / denom, None) for v in row]
    return res


def conv1d(x, n, c, w, k):
    res = []
    for t in range(n):
        for j in range(c):
            terms = [w[j * k + (k - 1 - back)] * x[(t - back) * c + j] for back in range(k) if t - back >= 0]
            res.append((math.fsum(terms), math.fsum(map(abs, terms))))
    return res


def rope(x, n, heads, hd, rot, theta):
    half, res = rot // 2, []
    for t in range(n):                    # 位置就是 token 的序号
        ang = [t / theta ** (2 * i / rot) for i in range(half)]
        for h in range(heads):
            v = x[(t * heads + h) * hd : (t * heads + h + 1) * hd]
            out = [(u, None) for u in v]
            for i in range(half):
                c, s, p, q = math.cos(ang[i]), math.sin(ang[i]), v[i], v[i + half]
                out[i] = (p * c - q * s, abs(p * c) + abs(q * s))
                out[i + half] = (q * c + p * s, abs(q * c) + abs(p * s))
            res += out
    return res


def attention(q, k, v, n, heads, kv, d):
    """因果注意力，按定义：先算全部分数，softmax，再加权求和。"""
    rep, scale = heads // kv, 1 / math.sqrt(d)
    res = [None] * (n * heads * d)
    for t in range(n):
        for h in range(heads):
            g = h // rep                          # 每 rep 个相邻的查询头共用一组键值
            qv = q[(t * heads + h) * d : (t * heads + h + 1) * d]
            sc = [math.fsum(a * b for a, b in zip(qv, k[(s * kv + g) * d : (s * kv + g + 1) * d])) * scale
                  for s in range(t + 1)]
            m = max(sc)
            e = [math.exp(x - m) for x in sc]
            tot = math.fsum(e)
            p = [x / tot for x in e]
            for i in range(d):
                terms = [p[s] * v[(s * kv + g) * d + i] for s in range(t + 1)]
                res[(t * heads + h) * d + i] = (math.fsum(terms), math.fsum(map(abs, terms)))
    return res


def gated_delta(q, k, v, g, beta, n, hk, hv, dk, dv):
    """门控 delta 规则。状态 S 是 dk 行 dv 列；同时用全部取绝对值的版本跑一遍，得到求和项大小的上界 s。"""
    rep, scale = hv // hk, 1 / math.sqrt(dk)
    res = [None] * (n * hv * dv)
    for j in range(hv):
        kh = j // rep
        S = [[0.0] * dv for _ in range(dk)]
        A = [[0.0] * dv for _ in range(dk)]       # 绝对值版本
        for t in range(n):
            qt = q[(t * hk + kh) * dk : (t * hk + kh + 1) * dk]
            kt = k[(t * hk + kh) * dk : (t * hk + kh + 1) * dk]
            vt = v[(t * hv + j) * dv : (t * hv + j + 1) * dv]
            dec, b = math.exp(g[t * hv + j]), beta[t * hv + j]
            for r in range(dk):
                S[r] = [x * dec for x in S[r]]
                A[r] = [x * dec for x in A[r]]
            u = [b * (vt[c] - math.fsum(S[r][c] * kt[r] for r in range(dk))) for c in range(dv)]
            ua = [b * (abs(vt[c]) + math.fsum(A[r][c] * abs(kt[r]) for r in range(dk))) for c in range(dv)]
            for r in range(dk):
                S[r] = [S[r][c] + kt[r] * u[c] for c in range(dv)]
                A[r] = [A[r][c] + abs(kt[r]) * ua[c] for c in range(dv)]
            for c in range(dv):
                o = math.fsum(S[r][c] * qt[r] for r in range(dk)) * scale
                oa = math.fsum(A[r][c] * abs(qt[r]) for r in range(dk)) * scale
                res[(t * hv + j) * dv + c] = (o, oa)
    return res


# ---------- 比较（与 ops_ref.check 的判定相同） ----------

def compare(name, got, res, exact=False):
    if len(got) != len(res):
        return f"{name}：元素个数不对，应为 {len(res)}，实际 {len(got)}", None
    worst, bad, loose = 0.0, [], 0
    for i, (gv, (r, s)) in enumerate(zip(got, res)):
        if s is None:
            s = abs(r)
        if math.isnan(r) or math.isinf(r) or abs(r) > FLT_MAX * (1 + 2**-25):
            if not ((math.isnan(r) and math.isnan(gv)) or (not math.isnan(r) and gv == math.copysign(math.inf, r))):
                bad.append((i, gv, r))
            continue
        if exact:
            if not (gv == r and math.copysign(1, gv) == math.copysign(1, r)):
                bad.append((i, gv, r))
            continue
        err = abs(gv - r)
        if not err <= ulp32(r) + math.ldexp(s, -32):
            bad.append((i, gv, r))
        elif err > ulp32(r):
            loose += 1
        else:
            worst = max(worst, err / ulp32(r))
    if bad:
        ex = "；".join(f"[{i}] 得到 {gv!r}，应为 {r!r}" for i, gv, r in bad[:3])
        return f"{name}：{len(bad)} 个元素超出容差，例如 {ex}", None
    return None, (worst, loose)


def main():
    _selfcheck()
    cfg_path, layer, ids, files = sys.argv[1], int(sys.argv[2]), [int(v) for v in sys.argv[3].split(",")], sys.argv[4:]
    t = json.load(open(cfg_path))["text_config"]
    W = Tensors(files)
    H, I, eps = t["hidden_size"], t["intermediate_size"], t["rms_norm_eps"]
    n = len(ids)
    L = P + f"layers.{layer}."
    full = t["layer_types"][layer] == "full_attention"
    vec = lambda name: list(W.floats(name)[1])

    def mat(name):
        dt, (rows, cols), buf = W.get(name)
        assert dt == "BF16"
        return buf, rows, cols

    steps = read_dump(sys.stdin)
    D = {s[0]: s[3] for s in steps}

    # 每一步：(名字, 列数, 计算函数, 是否要求完全相等)。计算函数只读 D 里已经检查过的张量。
    plan = []

    def step(name, cols, fn, exact=False):
        plan.append((name, cols, fn, exact))

    def mm(src, wname):
        buf, rows, cols = mat(wname)
        return lambda: matmul(D[src], n, buf, rows, cols)

    def heads_of(src, width, start, length):
        """从每个 token 的 width 维里取出 [start, start+length) 这一段"""
        x = D[src]
        return [v for tt in range(n) for v in x[tt * width + start : tt * width + start + length]]

    emb_buf = W.get(P + "embed_tokens.weight")[2]
    step("embed", H, lambda: [(v, None) for i in ids for v in bf16_row(emb_buf, i, H)], exact=True)
    step("attn_in", H, lambda: rmsnorm(D["embed"], vec(L + "input_layernorm.weight"), H, eps, True))
    if full:
        hd, nh, nkv = t["head_dim"], t["num_attention_heads"], t["num_key_value_heads"]
        rp = t["rope_parameters"]
        rot, theta = int(hd * rp.get("partial_rotary_factor", 1.0)), float(rp["rope_theta"])
        A = L + "self_attn."
        # q_proj 的输出：每个头 2*hd 维，前 hd 维是 q，后 hd 维是门控
        q_part = lambda: [v for tt in range(n) for h in range(nh)
                          for v in D["q_proj"][(tt * nh + h) * 2 * hd : (tt * nh + h) * 2 * hd + hd]]
        g_part = lambda: [v for tt in range(n) for h in range(nh)
                          for v in D["q_proj"][(tt * nh + h) * 2 * hd + hd : (tt * nh + h + 1) * 2 * hd]]
        step("q_proj", 2 * nh * hd, mm("attn_in", A + "q_proj.weight"))
        step("k_proj", nkv * hd, mm("attn_in", A + "k_proj.weight"))
        step("v_proj", nkv * hd, mm("attn_in", A + "v_proj.weight"))
        step("q_norm", nh * hd, lambda: rmsnorm(q_part(), vec(A + "q_norm.weight"), hd, eps, True))
        step("q_rope", nh * hd, lambda: rope(D["q_norm"], n, nh, hd, rot, theta))
        step("k_norm", nkv * hd, lambda: rmsnorm(D["k_proj"], vec(A + "k_norm.weight"), hd, eps, True))
        step("k_rope", nkv * hd, lambda: rope(D["k_norm"], n, nkv, hd, rot, theta))
        step("attn", nh * hd, lambda: attention(D["q_rope"], D["k_rope"], D["v_proj"], n, nh, nkv, hd))
        step("gate_sigmoid", nh * hd, lambda: [(sigmoid(v), None) for v in g_part()])
        step("gated", nh * hd, lambda: [(a * b, None) for a, b in zip(D["attn"], D["gate_sigmoid"])])
        step("attn_out", H, mm("gated", A + "o_proj.weight"))
    else:
        hk, hv = t["linear_num_key_heads"], t["linear_num_value_heads"]
        dk, dv, ck = t["linear_key_head_dim"], t["linear_value_head_dim"], t["linear_conv_kernel_dim"]
        kd, vd = hk * dk, hv * dv
        cd = 2 * kd + vd
        A = L + "linear_attn."
        step("qkv", cd, mm("attn_in", A + "in_proj_qkv.weight"))
        step("z", vd, mm("attn_in", A + "in_proj_z.weight"))
        step("b", hv, mm("attn_in", A + "in_proj_b.weight"))
        step("a", hv, mm("attn_in", A + "in_proj_a.weight"))
        step("conv", cd, lambda: conv1d(D["qkv"], n, cd, vec(A + "conv1d.weight"), ck))
        step("conv_silu", cd, lambda: [(silu(v), None) for v in D["conv"]])
        # 卷积的输出依次是 q、k、v 三段
        step("q_l2", kd, lambda: l2norm(heads_of("conv_silu", cd, 0, kd), dk))
        step("k_l2", kd, lambda: l2norm(heads_of("conv_silu", cd, kd, kd), dk))
        step("beta", hv, lambda: [(sigmoid(v), None) for v in D["b"]])
        A_log, dt_bias = vec(A + "A_log"), vec(A + "dt_bias")
        step("g", hv, lambda: [(-math.exp(A_log[i % hv]) * softplus(v + dt_bias[i % hv]), None)
                               for i, v in enumerate(D["a"])])
        step("delta", vd, lambda: gated_delta(D["q_l2"], D["k_l2"], heads_of("conv_silu", cd, 2 * kd, vd),
                                              D["g"], D["beta"], n, hk, hv, dk, dv))
        step("gnorm", vd, lambda: rmsnorm_gated(D["delta"], D["z"], vec(A + "norm.weight"), dv, eps))
        step("attn_out", H, mm("gnorm", A + "out_proj.weight"))
    step("resid", H, lambda: [(a + b, None) for a, b in zip(D["embed"], D["attn_out"])])
    step("mlp_in", H, lambda: rmsnorm(D["resid"], vec(L + "post_attention_layernorm.weight"), H, eps, True))
    step("gate", I, mm("mlp_in", L + "mlp.gate_proj.weight"))
    step("up", I, mm("mlp_in", L + "mlp.up_proj.weight"))
    step("gate_silu", I, lambda: [(silu(v), None) for v in D["gate"]])
    step("act", I, lambda: [(a * b, None) for a, b in zip(D["gate_silu"], D["up"])])
    step("mlp_out", H, mm("act", L + "mlp.down_proj.weight"))
    step("out", H, lambda: [(a + b, None) for a, b in zip(D["resid"], D["mlp_out"])])
    step("final_norm", H, lambda: rmsnorm(D["out"], vec(P + "norm.weight"), H, eps, True))

    want = [(s[0], n, s[1]) for s in plan]
    got = [(s[0], s[1], s[2]) for s in steps]
    if want != got:
        print(f"步骤不一致：应为 {want}，实际 {got}")
        return 1
    worst, loose, total = 0.0, 0, 0
    for name, cols, fn, exact in plan:
        msg, stat = compare(name, D[name], fn(), exact)
        if msg:
            print(msg)
            return 1
        worst, loose, total = max(worst, stat[0]), loose + stat[1], total + len(D[name])
    kind = "全注意力" if full else "线性注意力"
    msg = f"第 {layer} 层（{kind}），{len(plan)} 步，{total} 个元素，最大误差 {worst:.2f} ulp"
    if loose:
        msg += f"，另有 {loose} 个处在大量抵消处，误差在抵消项容差内"
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
