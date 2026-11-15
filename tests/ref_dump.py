#!/usr/bin/env python3
"""参考实现：只用标准库读 safetensors，输出格式与 st_dump 相同。

它是 C 读取器的"标准答案"。两份实现独立编写，输出逐字节一致才算通过。
校验和用 zlib.crc32：它在 C 里实现，几十 GB 的权重也能在几分钟内算完。
"""
import json, mmap, struct, sys, zlib


def read_header(path):
    with open(path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    (hlen,) = struct.unpack("<Q", mm[:8])
    header = json.loads(mm[8 : 8 + hlen].decode("utf-8"))
    header.pop("__metadata__", None)
    return mm, 8 + hlen, header


def main(paths):
    table = {}
    for path in paths:
        mm, base, header = read_header(path)
        for name, e in header.items():
            if name in table:
                sys.exit(f"duplicate tensor name across shards: {name!r}")
            table[name] = (mm, base, e)
    # C 端按 UTF-8 字节序排序，这里也按字节排，而不是按 Python 的码点序
    for name in sorted(table, key=lambda s: s.encode("utf-8")):
        mm, base, e = table[name]
        b, end = e["data_offsets"]
        crc = zlib.crc32(memoryview(mm)[base + b : base + end])
        shape = ",".join(str(d) for d in e["shape"])
        esc = "".join(
            "\\\\" if c == "\\" else f"\\u{ord(c):04x}" if ord(c) < 0x20 else c for c in name
        )
        sys.stdout.buffer.write(f"{crc:08x} {e['dtype']} [{shape}] {esc}\n".encode())


if __name__ == "__main__":
    main(sys.argv[1:])
