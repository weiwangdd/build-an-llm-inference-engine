#!/usr/bin/env python3
"""第 5 章的第二层对拍：拿官方的 transformers 代码算整个模型，与 C 的结果比较。

    model_official.py <config.json> <ids> <生成个数 m> <all|layers> <分片 ...> < build/forward 的输出

需要 torch 和 transformers（在 5.23.0 上写成，也在 5.8.0.dev0 上跑过）。做三件事：

1. 逐层对拍（判定）。对每一层 i，把 C 的第 i-1 层输出（第 0 层用嵌入）交给官方的 Qwen3_5DecoderLayer，
   在 float64 下算一遍，和 C 的第 i 层输出比；最终归一化、lm_head 也一样。每个张量的最大误差除以均方根
   不超过 2^-12，与第 4 章相同。每一层的输入都取自 C，误差不会一层层累积，所以容差不用放宽。
2. 官方从嵌入开始自己串起所有层（float64），得到它自己的 logits。贪心解码选出的每个 token 必须与 C 的
   完全相同（判定）。因为模型是因果的，第 t 行 logits 只取决于前 t+1 个 token，所以只要对 C 最后一次
   前向的输入跑一遍，就能逐行检查 C 的每一步选择：只要每一行都选得一样，官方自己贪心解码出来的
   序列就一定与 C 的相同。同时打印官方与 C 的 logits 差多少、每一步第一名领先第二名多少。
3. 官方 BF16 也串一遍（模型平时就是这样运行的），只打印它选的 token 与 C 有几个相同、logits 差多少，
   不做判定（Wei 定的方案）。

真实模型用 float64 要 216 GB，放不进内存，所以三件事都逐层做：每次只加载一层的权重，三种算法都跑完再释放。
小模型（all 方式）另外直接构造官方的整个模型 Qwen3_5ForCausalLM，检查它与逐层串起来的结果一致，
并用它的 generate（官方的生成函数，带缓存）做一次真正的贪心解码，与 C 生成的 token 比较。

设环境变量 MODEL_VERBOSE=1 打印每一层的误差。
"""
import json, math, os, sys

for _m in ("fla", "causal_conv1d"):      # 原因见第 4 章 layer_official.py
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
# 前面章节的参考实现在相邻的目录里（tests/NN-名字/）
sys.path[1:1] = [os.path.join(sys.path[0], "..", d) for d in ("04-layer",)]
from layer_ref import read_dump  # noqa: E402

P = "model.language_model."
TOL = 2.0 ** -12


class Weights:
    """按名字从多个分片里取张量，用到时才读。"""
    def __init__(self, files):
        self.files, self.where = {}, {}
        for f in files:
            h = self.files[f] = safe_open(f, framework="pt")
            for k in h.keys():
                self.where[k] = f

    def get(self, name):
        if name not in self.where:
            raise SystemExit(f"找不到张量 {name}")
        return self.files[self.where[name]].get_tensor(name)

    def prefixed(self, pre):
        return {k[len(pre):]: self.get(k) for k in self.where if k.startswith(pre)}


def rel(a, ref):
    """最大误差除以 ref 的均方根"""
    rms = ref.pow(2).mean().sqrt().item()
    e = (a - ref).abs().max().item()
    return e / rms if rms > 0 else e


def argmax_rows(lg):
    """每一行分数最大的 id，同样大取 id 小的（torch.argmax 也是取第一个）"""
    return [int(v) for v in lg.argmax(dim=-1)]


def margin(row):
    top = row.topk(2).values
    return (top[0] - top[1]).item()


def lm_head(x, w, dtype):
    """x [n, hidden] @ w[vocab, hidden]ᵀ，按块转换权重，免得一次把 24.8 万行全转成 float64（10 GB）"""
    out = []
    for i in range(0, w.shape[0], 16384):
        out.append(x.to(dtype) @ w[i:i + 16384].to(dtype).T)
    return torch.cat(out, dim=1)


def main():
    cfg_path, prompt, m, mode, files = (sys.argv[1], [int(v) for v in sys.argv[2].split(",")],
                                        int(sys.argv[3]), sys.argv[4], sys.argv[5:])
    verbose = bool(os.environ.get("MODEL_VERBOSE"))
    cfg = Qwen3_5Config.from_dict(json.load(open(cfg_path)))
    tc = cfg.text_config
    tc._attn_implementation = "eager"
    torch.manual_seed(0)

    steps = read_dump(sys.stdin)
    greedy = [int(v) for v in steps.pop()[3]] if m > 0 else []
    ids = prompt + greedy[:-1]
    n, L = len(ids), tc.num_hidden_layers
    C = {s[0]: torch.tensor(s[3], dtype=torch.float64).reshape(s[1], s[2]) for s in steps
         if s[0] in ("embed", "final_norm", "logits") or s[0].endswith(".out")}
    W = Weights(files)

    pos = torch.arange(n).unsqueeze(0)
    rot = M.Qwen3_5TextRotaryEmbedding(tc)
    masks = {dt: torch.full((n, n), -math.inf, dtype=dt).triu(1)[None, None] for dt in (torch.float64, torch.bfloat16)}

    def run_layer(dec, i, h, dtype):
        full = tc.layer_types[i] == "full_attention"     # 层类型从配置读（属性名随版本变，见第 4 章）
        with torch.no_grad():
            out = dec(h.to(dtype).unsqueeze(0), position_embeddings=rot(h.to(dtype).unsqueeze(0), pos),
                      attention_mask=masks[dtype] if full else None, position_ids=pos)
        return (out[0] if isinstance(out, tuple) else out)[0]

    emb = W.get(P + "embed_tokens.weight")[torch.tensor(ids)]          # BF16 [n, hidden]
    errs, bad = [], []
    e = rel(C["embed"], emb.to(torch.float64))
    if e != 0:
        bad.append("embed")
    h64, hbf = emb.to(torch.float64), emb.to(torch.bfloat16)
    for i in range(L):
        dec = M.Qwen3_5DecoderLayer(tc, i)
        sd = W.prefixed(P + f"layers.{i}.")
        dec.load_state_dict({k: sd[k] for k in dec.state_dict()}, strict=True)
        dec.to(torch.bfloat16)
        hbf = run_layer(dec, i, hbf, torch.bfloat16)                        # 3. 官方 BF16 串起来
        dec.to(torch.float64)
        src = C["embed"] if i == 0 else C[f"layers.{i - 1}.out"]
        e = rel(C[f"layers.{i}.out"], run_layer(dec, i, src, torch.float64))   # 1. 逐层，输入取自 C
        h64 = run_layer(dec, i, h64, torch.float64)                         # 2. 官方 float64 串起来
        errs.append((f"layers.{i}.out", e, rel(h64, C[f"layers.{i}.out"]), rel(hbf.to(torch.float64), C[f"layers.{i}.out"])))
        if not e <= TOL:
            bad.append(f"第 {i} 层")
        del dec, sd

    norm = M.Qwen3_5RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)
    norm.load_state_dict({"weight": W.get(P + "norm.weight")}, strict=True)
    head = W.get("lm_head.weight")
    with torch.no_grad():
        norm.to(torch.float64)
        e_norm = rel(C["final_norm"], norm(C[f"layers.{L - 1}.out"]))
        e_head = rel(C["logits"], lm_head(C["final_norm"], head, torch.float64))
        lg64 = lm_head(norm(h64), head, torch.float64)
        norm.to(torch.bfloat16)
        lgbf = lm_head(norm(hbf), head, torch.bfloat16).to(torch.float64)
    errs += [("final_norm", e_norm, None, None), ("logits", e_head, rel(lg64, C["logits"]), rel(lgbf, C["logits"]))]
    if not e_norm <= TOL:
        bad.append("最终归一化")
    if not e_head <= TOL:
        bad.append("lm_head")

    if verbose:
        for name, a, b, c in errs:
            extra = "" if b is None else f"；官方 float64 串起来差 {b:.1e}，官方 BF16 串起来差 {c:.1e}"
            print(f"    {name:<16} 逐层与官方 float64 差 {a:.1e}{extra}")
    if bad:
        print(f"与官方实现不一致：{'、'.join(bad[:5])} 超过 {TOL:.1e}（或嵌入不相等）")
        return 1
    worst = max(a for _, a, _, _ in errs)
    msg = [f"{L} 层逐层对拍，最大相对误差 {worst:.1e}；官方 float64 串起 {L} 层后 logits 与 C 差 {errs[-1][2]:.1e}"]

    # 2. 贪心：C 选出的每个 token 都要是官方 float64 那一行分数最大的
    rows = [len(prompt) - 1 + k for k in range(len(greedy))] if greedy else [n - 1]
    want = greedy if greedy else argmax_rows(C["logits"][n - 1:])
    got64, gotbf = argmax_rows(lg64[rows]), argmax_rows(lgbf[rows])
    margins = [margin(lg64[r]) for r in rows]
    if got64 != want:
        k = next(k for k in range(len(want)) if got64[k] != want[k])
        print(f"第 {k + 1} 步选的 token 不同：C 是 {want[k]}，官方 float64 是 {got64[k]}，"
              f"官方第一名领先第二名 {margins[k]:.2e}")
        return 1
    what = f"贪心解码的 {len(greedy)} 个 token" if greedy else "最后一个位置分数最大的 token"
    msg.append(f"{what} 与官方 float64 完全相同（第一名最少领先 {min(margins):.2e}）")
    same = sum(a == b for a, b in zip(gotbf, want))
    msg.append(f"官方 BF16：logits 差 {errs[-1][3]:.1e}，{len(want)} 步中有 {same} 步选的 token 相同")
    if same != len(want):
        diff = [f"第 {k + 1} 步 C {want[k]}、BF16 {gotbf[k]}" for k in range(len(want)) if gotbf[k] != want[k]]
        msg[-1] += "（" + "，".join(diff[:4]) + "）"

    # 小模型：再用官方的整个模型核对一遍逐层串联，并用官方的 generate 真正做一次贪心解码
    if mode == "all":
        full = M.Qwen3_5ForCausalLM(tc)
        sd = {("model." + k[len(P):] if k.startswith(P) else k): W.get(k) for k in W.where}
        full.load_state_dict({k: sd[k] for k in full.state_dict()}, strict=True)
        full.to(torch.float64).eval()
        with torch.no_grad():
            lg_full = full(torch.tensor([ids]), use_cache=False).logits[0]
        e_full = rel(lg_full, lg64)
        if not e_full <= 1e-12:
            print(f"官方整个模型的 logits 与逐层串起来的不一致：差 {e_full:.1e}")
            return 1
        msg.append(f"官方整个模型与逐层串联差 {e_full:.1e}")
        if greedy:
            with torch.no_grad():
                # 显式给出全 1 的 attention_mask：否则 generate 会把等于 pad_token_id 的 token 当成填充遮掉
                inp = torch.tensor([prompt])
                gen = full.generate(inp, attention_mask=torch.ones_like(inp), max_new_tokens=len(greedy),
                                    min_new_tokens=len(greedy), do_sample=False, eos_token_id=None, pad_token_id=0)
            gen = [int(v) for v in gen[0, len(prompt):]]
            if gen != greedy:
                print(f"官方 generate 生成的是 {gen}，C 是 {greedy}")
                return 1
            msg.append("官方 generate 生成的 token 也相同")
    print("；".join(msg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
