#!/usr/bin/env python3
"""生成测试用的 safetensors 文件：一份合法的，若干份故意损坏的。"""
import json, os, random, struct, sys

SIZES = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
         "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}


def write(path, header, data, hlen=None):
    h = json.dumps(header, ensure_ascii=False).encode() if isinstance(header, dict) else header
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h) if hlen is None else hlen) + h + data)


def main(out):
    os.makedirs(out, exist_ok=True)
    rng = random.Random(0)
    specs = [
        ("model.embed.weight", "BF16", [50, 16]),
        ("model.layers.0.attn.q.weight", "F32", [16, 16]),
        ("model.layers.0.norm.weight", "F16", [16]),
        ("scalar", "F64", []),
        ("empty", "I32", [0, 4]),
        ("ids", "I64", [3, 2, 1]),
        ("mask", "BOOL", [7]),
        ("名字/带\"引号\"和\\反斜杠\n换行😀", "U8", [5]),
        ("a", "U16", [2]), ("B", "I8", [3]), ("a.b", "U32", [1]), ("é", "I16", [2]),
    ]
    header, blob = {"__metadata__": {"format": "pt", "note": "x{[\"]}"}}, b""
    for name, dt, shape in specs:
        n = SIZES[dt]
        for d in shape:
            n *= d
        header[name] = {"dtype": dt, "shape": shape, "data_offsets": [len(blob), len(blob) + n]}
        blob += bytes(rng.randrange(256) for _ in range(n))
    write(f"{out}/good.safetensors", header, blob)
    # 同一内容，但 JSON 全部用 \uXXXX 转义并加空白，考验解析器
    write(f"{out}/good_escaped.safetensors",
          json.dumps(header, ensure_ascii=True, indent=2).encode() + b"   ", blob)

    one = {"t": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    bad = {
        "bad_short": lambda p: open(p, "wb").write(b"\x01\x02"),
        "bad_hlen": lambda p: write(p, one, b"\0" * 8, hlen=1 << 40),
        "bad_oob": lambda p: write(p, {"t": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}, b"\0" * 8),
        "bad_size": lambda p: write(p, {"t": {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}}, b"\0" * 8),
        "bad_dtype": lambda p: write(p, {"t": {"dtype": "F128", "shape": [2], "data_offsets": [0, 8]}}, b"\0" * 8),
        "bad_missing": lambda p: write(p, {"t": {"dtype": "F32", "shape": [2]}}, b"\0" * 8),
        "bad_json": lambda p: write(p, b'{"t": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}', b"\0" * 8),
        "bad_trailing": lambda p: write(p, json.dumps(one).encode() + b" x", b"\0" * 8),
        "bad_dup": lambda p: write(p, b'{"t":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},'
                                      b'"t":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}', b"\0" * 8),
        "bad_overflow": lambda p: write(p, {"t": {"dtype": "F32", "shape": [1 << 62, 4], "data_offsets": [0, 0]}}, b""),
        "bad_negative": lambda p: write(p, {"t": {"dtype": "F32", "shape": [-2], "data_offsets": [0, 8]}}, b"\0" * 8),
    }
    for name, fn in bad.items():
        fn(f"{out}/{name}.safetensors")


if __name__ == "__main__":
    main(sys.argv[1])
