#!/usr/bin/env python3
"""第 3 章算子的参考实现，只用标准库，全部用 Python 的 float（双精度）计算。

    ops_ref.py bf16                                   打印全部 BF16 位模式转成 float 的位模式，格式同 build/ops bf16
    ops_ref.py check <算子> [键=值 ...] 文件 ... < got.txt
                                                      读 build/ops 的输出，与参考值逐元素比较

参数的含义与 tools/ops.c 相同。这里的写法刻意与 C 端不同（例如 BF16 按定义从符号、指数、
尾数算出数值，而不是移位；求和用 math.fsum 精确求和），免得两边错成一样。

判定标准（见第 3 章“误差要多小才算对”）：设 r 是精确值（这里用双精度近似），C 的输出 g 是 float，
    |g - r| <= ulp(r) + 2^-32 * s
ulp(r) 是 float 在 r 附近相邻两个数的间距，s 是这个结果所有求和项的绝对值之和（没有求和时 s = |r|）。
第一项允许一次舍入再多一点余量，第二项只在求和项大量抵消、r 本身很小时起作用。
嵌入是纯拷贝，要求完全相等。
"""
import math, os, struct, sys
from array import array
from operator import mul

sys.path.insert(0, sys.path[0])
# 前面章节的参考实现在相邻的目录里（tests/NN-名字/）
sys.path[1:1] = [os.path.join(sys.path[0], "..", d) for d in ("01-weights",)]
from ref_dump import read_header  # noqa: E402

assert sys.byteorder == "little"


# ---------- BF16 ----------

def bf16_value(h):
    """按定义解码：1 位符号、8 位指数（偏置 127）、7 位尾数。"""
    sign = -1.0 if h >> 15 else 1.0
    e, m = (h >> 7) & 0xFF, h & 0x7F
    if e == 0xFF:
        return sign * math.inf if m == 0 else math.nan
    if e == 0:
        return sign * math.ldexp(m, -133)          # 非规格化数：m/128 * 2^-126
    return sign * math.ldexp(128 + m, e - 134)     # (1 + m/128) * 2^(e-127)


def f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def bf16_table():
    out = []
    for h in range(65536):
        v = bf16_value(h)
        if math.isnan(v):
            # IEEE 754 把窄格式的 NaN 扩宽时保留符号和载荷：载荷左移到高位
            out.append((h >> 15) << 31 | 0xFF << 23 | (h & 0x7F) << 16)
        else:
            out.append(f32_bits(v))
    return out


def bf16_fast(buf):
    """大张量用的快速解码：把每个 BF16 的两个字节放到 float 的高两字节。启动时与 bf16_value 全量核对。"""
    n = len(buf) // 2
    b = bytearray(4 * n)
    b[2::4] = buf[0::2]
    b[3::4] = buf[1::2]
    return array("f", bytes(b))


def _selfcheck():
    allbf = bytes(memoryview(array("H", range(65536))).cast("B"))
    for h, v in enumerate(bf16_fast(allbf)):
        r = bf16_value(h)
        assert (math.isnan(v) and math.isnan(r)) or v == r, h


# ---------- 读输入 ----------

class Tensors:
    def __init__(self, paths):
        self.t = {}
        for p in paths:
            mm, base, header = read_header(p)
            for name, e in header.items():
                b, end = e["data_offsets"]
                self.t[name] = (e["dtype"], e["shape"], memoryview(mm)[base + b : base + end])

    def get(self, name):
        return self.t[name]

    def floats(self, name):
        dt, shape, buf = self.t[name]
        if dt == "F32":
            return shape, array("f", bytes(buf))
        if dt == "BF16":
            return shape, bf16_fast(buf)
        raise SystemExit(f"不支持的 dtype {dt}")

    def ints(self, name):
        dt, shape, buf = self.t[name]
        assert dt == "I32"
        return list(array("i", bytes(buf)))


def bf16_row(buf, row, cols):
    return bf16_fast(buf[2 * row * cols : 2 * (row + 1) * cols])


# ---------- 参考算子：每个返回 (形状, [(精确值, 求和项绝对值之和), ...]) ----------

def sigmoid(v):
    if v >= 0:
        return 1.0 / (1.0 + math.exp(-v))
    e = math.exp(v)
    return e / (1.0 + e)


def silu(v):
    return v * sigmoid(v)


def softplus(v):
    if v > 0:
        return v + math.log1p(math.exp(-v))
    return math.log1p(math.exp(v))


def rows(shape, xs):
    d = shape[-1]
    return d, [xs[i : i + d] for i in range(0, len(xs), d)]


def ref(op, a, T):
    if op == "matmul":
        (n, k), x = T.floats(a["x"])
        dt, (out, k2), wbuf = T.get(a["w"])
        res = [None] * (n * out)
        xr = [x[t * k : (t + 1) * k] for t in range(n)]
        for o in range(out):
            w = bf16_row(wbuf, o, k)
            for t in range(n):
                terms = list(map(mul, xr[t], w))   # float * BF16 在双精度下是精确的
                res[t * out + o] = (math.fsum(terms), math.fsum(map(abs, terms)))
        return [n, out], res
    if op == "embed":
        ids = T.ints(a["ids"])
        dt, (nrows, dim), wbuf = T.get(a["w"])
        res = []
        for i in ids:
            res += [(v, None) for v in bf16_row(wbuf, i, dim)]
        return [len(ids), dim], res
    if op in ("rmsnorm", "rmsnorm_gated", "l2norm"):
        shape, x = T.floats(a["x"])
        eps = float(a["eps"])
        d, xs = rows(shape, x)
        w = T.floats(a["w"])[1] if op != "l2norm" else None
        z = rows(shape, T.floats(a["z"])[1])[1] if op == "rmsnorm_gated" else None
        res = []
        for t, row in enumerate(xs):
            ss = math.fsum(v * v for v in row)
            denom = math.sqrt(ss + eps) if op == "l2norm" else math.sqrt(ss / d + eps)
            for i, v in enumerate(row):
                if op == "rmsnorm":
                    r = v / denom * (1.0 + w[i])
                elif op == "l2norm":
                    r = v / denom
                else:
                    r = v / denom * w[i] * silu(z[t][i])
                res.append((r, None))
        return shape, res
    if op in ("silu", "sigmoid", "softplus"):
        shape, x = T.floats(a["x"])
        f = {"silu": silu, "sigmoid": sigmoid, "softplus": softplus}[op]
        return shape, [(f(v), None) for v in x]
    if op == "softmax":
        shape, x = T.floats(a["x"])
        res = []
        for row in rows(shape, x)[1]:
            m = max(row)
            e = [math.exp(v - m) for v in row]
            s = math.fsum(e)
            res += [(v / s, None) for v in e]
        return shape, res
    if op == "rope":
        shape, x = T.floats(a["x"])
        n, heads, hd = shape
        pos = T.ints(a["pos"])
        theta, rot = float(a["theta"]), int(a["rot"])
        half = rot // 2
        res = []
        for t in range(n):
            ang = [pos[t] / theta ** (2 * i / rot) for i in range(half)]
            for h in range(heads):
                v = x[(t * heads + h) * hd : (t * heads + h + 1) * hd]
                out = [(u, None) for u in v]
                for i in range(half):
                    c, s, p, q = math.cos(ang[i]), math.sin(ang[i]), v[i], v[i + half]
                    out[i] = (p * c - q * s, abs(p * c) + abs(q * s))
                    out[i + half] = (q * c + p * s, abs(q * c) + abs(p * s))
                res += out
        return shape, res
    if op == "conv1d":
        (n, c), x = T.floats(a["x"])
        wshape, w = T.floats(a["w"])
        k = wshape[-1]
        res = []
        for t in range(n):
            for j in range(c):
                # 第 t 个输出看的是 token t-k+1 .. t，w[j][k-1] 对应 token t
                terms = [w[j * k + (k - 1 - back)] * x[(t - back) * c + j] for back in range(k) if t - back >= 0]
                res.append((math.fsum(terms), math.fsum(map(abs, terms))))
        return [n, c], res
    raise SystemExit(f"参考实现不认识算子 {op}")


# ---------- 比较 ----------

FLT_MAX = struct.unpack("<f", b"\xff\xff\x7f\x7f")[0]


def ulp32(r):
    r = abs(r)
    if r < math.ldexp(1, -126):
        return math.ldexp(1, -149)
    return math.ldexp(1, math.frexp(r)[1] - 24)


def check(op, args):
    a = {"x": "x", "w": "w", "z": "z", "ids": "ids", "pos": "pos"}
    files = []
    for s in args:
        if "=" in s:
            k, v = s.split("=", 1)
            a[k] = v
        else:
            files.append(s)
    shape, res = ref(op, a, Tensors(files))
    lines = sys.stdin.read().split("\n")
    want = "[" + ",".join(str(d) for d in shape) + "]"
    if lines[0] != want:
        print(f"形状不对：应为 {want}，实际 {lines[0]}")
        return 1
    got = [float.fromhex(s) for s in lines[1:] if s]
    if len(got) != len(res):
        print(f"元素个数不对：应为 {len(res)}，实际 {len(got)}")
        return 1
    worst, bad, loose = 0.0, [], 0
    for i, (g, (r, s)) in enumerate(zip(got, res)):
        if s is None:
            s = abs(r)
        if math.isnan(r) or math.isinf(r) or abs(r) > FLT_MAX * (1 + 2**-25):
            # 结果超出 float 的范围：只能是同号的无穷大；NaN 只能对 NaN
            ok = (math.isnan(r) and math.isnan(g)) or (not math.isnan(r) and g == math.copysign(math.inf, r))
            if not ok:
                bad.append((i, g, r))
            continue
        if op == "embed":
            if not (g == r and math.copysign(1, g) == math.copysign(1, r)):
                bad.append((i, g, r))
            continue
        err = abs(g - r)
        u = err / ulp32(r)
        if not err <= ulp32(r) + math.ldexp(s, -32):   # 写成 not <=，g 是 NaN 时也算超差
            bad.append((i, g, r))
        elif u > 1:
            loose += 1
        else:
            worst = max(worst, u)
    if bad:
        print(f"{len(bad)} 个元素超出容差，例如：")
        for i, g, r in bad[:5]:
            print(f"  [{i}] 得到 {g!r}，应为 {r!r}")
        return 1
    msg = f"{len(got)} 个元素，最大误差 {worst:.2f} ulp"
    if op == "embed":
        msg = f"{len(got)} 个元素，完全一致"
    if loose:
        msg += f"，另有 {loose} 个处在大量抵消处，误差在抵消项容差内"
    print(msg)
    return 0


def main():
    _selfcheck()
    if sys.argv[1] == "bf16":
        sys.stdout.write("".join(f"{u:08x}\n" for u in bf16_table()))
        return 0
    if sys.argv[1] == "check":
        return check(sys.argv[2], sys.argv[3:])
    raise SystemExit("用法见文件头部")


if __name__ == "__main__":
    sys.exit(main())
