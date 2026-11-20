"""第 7 章：比较同一组输入在 CPU 版和 GPU 版上的输出。只用标准库。

    python3 tests/07-gpu/gpu_cmp.py trace    cpu.txt gpu.txt   # layer、forward 的输出（“名字 [行,列]”加 %a）
    python3 tests/07-gpu/gpu_cmp.py generate cpu.txt gpu.txt   # generate --dump 的输出

GPU 版的每一步已经由第 3 到 6 章的参考实现单独检查过（是不是正确舍入）。这里检查两边是不是在算同一件事：
  - 结构完全相同：步骤名、顺序、形状一样；generate 生成的 token 和停止原因一样；forward 的 greedy 一样。
  - 每个张量的最大误差除以 CPU 那个张量的均方根不超过 2^-12（与第 4、5 章和官方对拍的容差相同）。
另外报告有多少个元素逐位相同、最多差几个 ulp（相邻两个 float 之间的距离），这是本章想知道的事实。
退出码 0 表示通过，1 表示不通过。"""
import math
import struct
import sys

TOL = 2.0 ** -12


def ordered(x):
    """把 float 的位模式变成有序整数：相邻的两个 float 差 1"""
    (u,) = struct.unpack("<I", struct.pack("<f", x))
    return -(u & 0x7FFFFFFF) if u & 0x80000000 else u


def read_trace(path):
    """layer / forward 的输出 → [(名字, 形状, [数])]，greedy 一段的数是整数"""
    out = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.endswith("]") and " [" in line:
                name, shape = line.split(" [")
                out.append((name, shape[:-1], []))
            elif out[-1][0] == "greedy":
                out[-1][2].append(int(line))
            else:
                out[-1][2].append(float.fromhex(line))
    return out


def read_generate(path):
    """generate --dump 的输出 → (每一步的 logits 列表, token 列表, 停止原因)"""
    steps, toks, stop = [], [], None
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f]
    i = 0
    while i < len(lines):
        w = lines[i].split()
        if w[0] == "logits":
            v = int(w[1])
            steps.append([float.fromhex(x) for x in lines[i + 1:i + 1 + v]])
            i += 1 + v
            continue
        if w[0] == "dist":
            i += 1 + int(w[1])
            continue
        if w[0] == "token":
            toks.append(int(w[1]))
        elif w[0] == "stop":
            stop = w[1]
        i += 1
    return steps, toks, stop


class Stats:
    def __init__(self):
        self.n = self.same = self.max_ulp = 0
        self.worst = 0.0
        self.where = self.ulp_where = ""

    def add(self, name, a, b):
        if any(not math.isfinite(x) for x in a + b):
            sys.exit(f"{name} 里有 NaN 或无穷大")
        rms = math.sqrt(math.fsum(x * x for x in a) / len(a)) if a else 0.0
        err = max((abs(x - y) for x, y in zip(a, b)), default=0.0)
        rel = err / rms if rms > 0 else (0.0 if err == 0 else math.inf)
        if rel > self.worst:
            self.worst, self.where = rel, name
        for x, y in zip(a, b):
            d = abs(ordered(x) - ordered(y))
            self.same += d == 0
            if d > self.max_ulp:
                self.max_ulp, self.ulp_where = d, name
        self.n += len(a)

    def report(self, ntensors, extra=""):
        msg = f"{ntensors} 个张量 {self.n} 个元素，逐位相同 {self.same}（{100 * self.same / max(self.n, 1):.4f}%）"
        if self.max_ulp:
            msg += f"，最多差 {self.max_ulp} ulp（{self.ulp_where}），最大误差/均方根 {self.worst:.2e}（{self.where}）"
        if self.worst > TOL:
            print(f"误差超过 2^-12：{msg}")
            sys.exit(1)
        print(msg + extra)


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ("trace", "generate"):
        sys.exit("用法：gpu_cmp.py trace|generate <CPU 的输出> <GPU 的输出>")
    st = Stats()
    if sys.argv[1] == "trace":
        a, b = read_trace(sys.argv[2]), read_trace(sys.argv[3])
        if [(n, s) for n, s, _ in a] != [(n, s) for n, s, _ in b]:
            print("两边的步骤名、顺序或形状不同")
            sys.exit(1)
        for (name, _, x), (_, _, y) in zip(a, b):
            if name == "greedy":
                if x != y:
                    print(f"贪心解码的 token 不同：CPU {x}，GPU {y}")
                    sys.exit(1)
            else:
                st.add(name, x, y)
        extra = "，贪心 token 相同" if any(n == "greedy" for n, _, _ in a) else ""
        st.report(sum(n != "greedy" for n, _, _ in a), extra)
    else:
        (sa, ta, pa), (sb, tb, pb) = read_generate(sys.argv[2]), read_generate(sys.argv[3])
        if ta != tb or pa != pb:
            print(f"生成的 token 或停止原因不同：CPU {ta} {pa}，GPU {tb} {pb}")
            sys.exit(1)
        for k, (x, y) in enumerate(zip(sa, sb)):
            st.add(f"step {k} logits", x, y)
        st.report(len(sa), f"，{len(ta)} 个 token 和停止原因（{pa}）相同")


main()
