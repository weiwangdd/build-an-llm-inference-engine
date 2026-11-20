#!/usr/bin/env python3
"""生成第 3 章算子测试的输入：若干个小 safetensors 文件，以及两份用例清单。

    cases.txt      每行：<名字> <算子> [键=值 ...] <文件>，C 端和参考实现都要跑，结果必须在容差内
    bad_cases.txt  每行同上，C 端必须以退出码 1 拒绝
"""
import json, math, os, random, struct, sys

DT = {"F32": "<f", "BF16": None, "I32": "<i"}


def to_bf16(v):
    """float 舍入到 BF16（就近舍入、平局取偶），返回 16 位整数。只用来造测试数据。"""
    u = struct.unpack("<I", struct.pack("<f", v))[0]
    if (u >> 23) & 0xFF == 0xFF:
        return u >> 16
    return (u + 0x7FFF + ((u >> 16) & 1)) >> 16


def f32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


def write(path, tensors, pad=0):
    """tensors: {名字: (dtype, 形状, 数值列表)}。pad 在头部后面加空格，让数据区不对齐。"""
    header, blob = {}, b""
    for name, (dt, shape, vals) in tensors.items():
        if dt == "BF16":
            data = b"".join(struct.pack("<H", to_bf16(v) if isinstance(v, float) else v) for v in vals)
        else:
            data = b"".join(struct.pack(DT[dt], v) for v in vals)
        header[name] = {"dtype": dt, "shape": shape, "data_offsets": [len(blob), len(blob) + len(data)]}
        blob += data
    h = json.dumps(header).encode() + b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + blob)


def main(out):
    os.makedirs(out, exist_ok=True)
    rng = random.Random(3)
    g = lambda s=1.0: rng.gauss(0, s)
    cases, bad = [], []

    def case(name, op, tensors, params="", pad=1):
        path = f"{out}/{name}.safetensors"
        write(path, tensors, pad)
        cases.append(f"{name} {op} {params} {path}".replace("  ", " "))
        return path

    # 矩阵乘：普通的、维度不整齐的、数值跨度很大的、求和大量抵消的
    case("matmul_small", "matmul", {"x": ("F32", [3, 7], [g() for _ in range(21)]),
                                    "w": ("BF16", [5, 7], [g() for _ in range(35)])})
    case("matmul", "matmul", {"x": ("F32", [4, 300], [g() for _ in range(1200)]),
                              "w": ("BF16", [65, 300], [g(0.05) for _ in range(19500)])}, pad=3)
    wide = lambda: math.copysign(10 ** rng.uniform(-20, 20), g())
    case("matmul_range", "matmul", {"x": ("F32", [2, 50], [wide() for _ in range(100)]),
                                    "w": ("BF16", [9, 50], [wide() for _ in range(450)])})
    # 抵消：第 2 列的权重是第 1 列取反，x 的两列几乎相等，结果比各项小好几个数量级
    xs = []
    for _ in range(8):
        v = g()
        xs += [v, f32(v * (1 + 1e-6)), g(1e-4)]
    case("matmul_cancel", "matmul", {"x": ("F32", [8, 3], xs),
                                     "w": ("BF16", [2, 3], [1.0, -1.0, 1.0, 3.0, -3.0, 0.5])})

    # 嵌入：随机的 16 位模式，包括 NaN、无穷大、非规格化数和负零
    table = [rng.randrange(65536) for _ in range(50 * 16)]
    table[:6] = [0x0000, 0x8000, 0x7F80, 0xFF80, 0x0001, 0x7FC1]
    case("embed", "embed", {"ids": ("I32", [6], [0, 49, 7, 7, 1, 30]), "w": ("BF16", [50, 16], table)})

    # RMSNorm：权重是“与 1 的差”，在 0 附近；包括全零行和非常小的行（靠 eps 兜底）
    x = [g() for _ in range(64)] + [0.0] * 64 + [g(1e-6) for _ in range(64)] + [g(1e3) for _ in range(64)]
    case("rmsnorm", "rmsnorm", {"x": ("F32", [4, 64], x), "w": ("BF16", [64], [g(0.1) for _ in range(64)])},
         "eps=1e-6")
    case("rmsnorm_3d_f32w", "rmsnorm", {"x": ("F32", [2, 3, 8], [g() for _ in range(48)]),
                                        "w": ("F32", [8], [g(0.1) for _ in range(8)])}, "eps=1e-5")
    case("rmsnorm_gated", "rmsnorm_gated", {"x": ("F32", [4, 32], [g() for _ in range(128)]),
                                            "z": ("F32", [4, 32], [g(3) for _ in range(128)]),
                                            "w": ("BF16", [32], [1 + g(0.1) for _ in range(32)])}, "eps=1e-6")
    case("l2norm", "l2norm", {"x": ("F32", [3, 16], [g() for _ in range(32)] + [0.0] * 16)}, "eps=1e-6")

    # 激活函数：正常范围的值，加上会让 e^x 溢出的极端值
    ext = [0.0, -0.0, 1e-30, -1e-30, 1e-45, 20.0, -20.0, 30.0, -30.0, 88.0, -88.0, 89.0, -89.0,
           100.0, -100.0, 745.0, -745.0, 1e4, -1e4, 1e30, -1e30, 3.4e38, -3.4e38]
    act = [f32(v) for v in ext] + [g(5) for _ in range(200)]
    for op in ("silu", "sigmoid", "softplus"):
        case(op, op, {"x": ("F32", [len(act)], act)})

    sm = [g(3) for _ in range(30)] + [1e4 + g(3) for _ in range(10)] + [g(), -math.inf, g(), -math.inf, g()] * 2
    case("softmax", "softmax", {"x": ("F32", [5, 10], sm)})
    case("softmax_uniform", "softmax", {"x": ("F32", [2, 4], [0.0] * 8)})

    # RoPE：位置包括 0 和 Qwen3.8 支持的最大位置；rot 小于 head_dim 时其余维度原样复制
    pos = [0, 1, 2, 100, 262143]
    case("rope_partial", "rope", {"x": ("F32", [5, 3, 16], [g() for _ in range(240)]), "pos": ("I32", [5], pos)},
         "theta=10000000 rot=8")
    case("rope_full", "rope", {"x": ("F32", [5, 2, 8], [g() for _ in range(80)]), "pos": ("I32", [5], pos)},
         "theta=10000 rot=8")

    # 因果卷积：权重 [c,k] 和 [c,1,k] 两种形状；token 数少于 k-1 的情形
    case("conv1d", "conv1d", {"x": ("F32", [6, 5], [g() for _ in range(30)]),
                              "w": ("BF16", [5, 4], [g() for _ in range(20)])})
    case("conv1d_3d", "conv1d", {"x": ("F32", [7, 3], [g() for _ in range(21)]),
                                 "w": ("F32", [3, 1, 4], [g() for _ in range(12)])})
    case("conv1d_short", "conv1d", {"x": ("F32", [2, 3], [g() for _ in range(6)]),
                                    "w": ("BF16", [3, 4], [g() for _ in range(12)])})

    # ---------- 必须被拒绝的输入 ----------
    def badf(name, tensors):
        path = f"{out}/bad_{name}.safetensors"
        write(path, tensors)
        return path

    mm = f"{out}/matmul_small.safetensors"
    rope = f"{out}/rope_partial.safetensors"
    rms = f"{out}/rmsnorm.safetensors"
    bad += [
        f"未知算子 gelu {mm}",
        f"未知参数 matmul y=x {mm}",
        f"参数重复 matmul x=x x=x {mm}",
        f"缺少eps rmsnorm {rms}",
        f"eps为负 rmsnorm eps=-1 {rms}",
        f"eps不是数 rmsnorm eps=1e-6abc {rms}",
        f"eps为NaN rmsnorm eps=nan {rms}",
        f"缺少张量 matmul w=nope {mm}",
        f"缺少文件 matmul {out}/不存在.safetensors",
        f"没有文件 matmul x=x",
        f"rot为奇数 rope theta=1e7 rot=7 {rope}",
        f"rot超过head_dim rope theta=1e7 rot=18 {rope}",
        f"theta为零 rope theta=0 rot=8 {rope}",
        f"矩阵乘列数不符 matmul " + badf("matmul_cols", {"x": ("F32", [2, 3], [1.0] * 6),
                                                   "w": ("BF16", [2, 4], [1.0] * 8)}),
        f"矩阵乘权重不是BF16 matmul " + badf("matmul_f32w", {"x": ("F32", [2, 3], [1.0] * 6),
                                                     "w": ("F32", [2, 3], [1.0] * 6)}),
        f"输入不是F32 silu " + badf("silu_bf16", {"x": ("BF16", [4], [1.0] * 4)}),
        f"空输入 silu " + badf("silu_empty", {"x": ("F32", [0], [])}),
        f"嵌入id越界 embed " + badf("embed_oob", {"ids": ("I32", [2], [0, 50]), "w": ("BF16", [50, 4], [0] * 200)}),
        f"嵌入id为负 embed " + badf("embed_neg", {"ids": ("I32", [1], [-1]), "w": ("BF16", [50, 4], [0] * 200)}),
        f"嵌入id类型不对 embed " + badf("embed_i64", {"ids": ("F32", [1], [1.0]), "w": ("BF16", [50, 4], [0] * 200)}),
        f"归一化权重长度不符 rmsnorm eps=1e-6 " + badf("rms_w", {"x": ("F32", [2, 4], [1.0] * 8),
                                                         "w": ("BF16", [5], [0.0] * 5)}),
        f"门控形状不符 rmsnorm_gated eps=1e-6 " + badf("gated_z", {"x": ("F32", [2, 4], [1.0] * 8),
                                                            "z": ("F32", [4, 2], [1.0] * 8),
                                                            "w": ("BF16", [4], [1.0] * 4)}),
        f"位置个数不符 rope theta=1e7 rot=4 " + badf("rope_pos", {"x": ("F32", [2, 1, 4], [1.0] * 8),
                                                            "pos": ("I32", [3], [0, 1, 2])}),
        f"卷积通道不符 conv1d " + badf("conv_c", {"x": ("F32", [2, 3], [1.0] * 6), "w": ("BF16", [4, 4], [1.0] * 16)}),
        f"卷积核形状不对 conv1d " + badf("conv_w", {"x": ("F32", [2, 3], [1.0] * 6), "w": ("BF16", [3, 2, 2], [1.0] * 12)}),
    ]
    with open(f"{out}/cases.txt", "w") as f:
        f.write("\n".join(cases) + "\n")
    with open(f"{out}/bad_cases.txt", "w") as f:
        f.write("\n".join(bad) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
