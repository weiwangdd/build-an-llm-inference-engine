#!/usr/bin/env python3
"""参考实现：只用标准库读 safetensors，输出格式与 st_dump 相同。

它是 C 读取器的"标准答案"。两份实现独立编写，输出逐字节一致才算通过。
"""
import json, mmap, struct, sys


def fnv1a64(buf):
    h = 0xCBF29CE484222325
    for b in buf:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def main(path):
    with open(path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    (hlen,) = struct.unpack("<Q", mm[:8])
    header = json.loads(mm[8 : 8 + hlen].decode("utf-8"))
    header.pop("__metadata__", None)
    base = 8 + hlen
    # C 端按 UTF-8 字节序排序，这里也按字节排，而不是按 Python 的码点序
    for name in sorted(header, key=lambda s: s.encode("utf-8")):
        e = header[name]
        b, end = e["data_offsets"]
        shape = ",".join(str(d) for d in e["shape"])
        esc = "".join(
            "\\\\" if c == "\\" else f"\\u{ord(c):04x}" if ord(c) < 0x20 else c for c in name
        )
        sys.stdout.buffer.write(
            f"{fnv1a64(mm[base + b : base + end]):016x} {e['dtype']} [{shape}] {esc}\n".encode()
        )


if __name__ == "__main__":
    main(sys.argv[1])
