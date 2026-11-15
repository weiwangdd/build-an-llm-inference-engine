#!/usr/bin/env python3
"""生成分词器的测试数据。

1. 一个小词表的 tokenizer.json：归一化、预分词、解码器和附加 token 的写法都与
   Qwen3.8 的完全相同，只是词表是在 tests/corpus 上现训练的（约 3000 个 token）。
   真实的 24.8 万词表的对拍在目标机器上做，见第 2 章。
2. 若干个故意写错的 tokenizer.json，C 实现必须拒绝它们。
3. 测试文本（texts.jsonl）和测试 id 序列（ids.txt）。
用法：make_tok_fixtures.py <输出目录> [真实的 tokenizer.json]
4. 覆盖全部 Unicode 码点的穷举文本，加上 20 万条随机的组合字符序列（unicode.jsonl），
   专门考验归一化和预分词。
"""
import copy, glob, json, os, random, sys
from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers

# 与 Qwen3.8 的 tokenizer.json 完全相同的预分词正则
PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
           r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")

SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|音频|>", "<|ñ|>"]
# 非特殊的附加 token：Qwen 的 <think> 等属于这一类。最后一个故意要求“归一化后匹配”，
# 而且内容写成分解形式，用来检查两边都会先把它归一化。
# "<|ñ|>" 和 "中Ā" 不是 Qwen 里有的，专门用来覆盖解码器对附加 token 的两种处理（见 tok.c）。
NORMAL = [("<think>", False), ("</think>", False), ("<tool_call>", False), ("cafe\u0301!", True)]


def corpus():
    lines = []
    for p in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "corpus", "*.txt"))):
        with open(p, encoding="utf-8", newline="") as f:
            lines += f.read().split("\n")
    return lines


def build_tokenizer(lines):
    tk = Tokenizer(models.BPE())
    tk.normalizer = normalizers.NFC()
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PATTERN), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tk.decoder = decoders.ByteLevel()
    tk.post_processor = processors.ByteLevel(trim_offsets=False)
    trainer = trainers.BpeTrainer(vocab_size=3000, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    tk.train_from_iterator(lines * 3, trainer)
    tk.add_special_tokens([AddedToken(s, special=True, normalized=False) for s in SPECIAL])
    tk.add_tokens([AddedToken(s, special=False, normalized=n) for s, n in NORMAL])
    return tk


def bad_variants(good):
    """每个变体只改一处，C 实现必须报错拒绝。"""
    def mod(fn):
        j = copy.deepcopy(good)
        fn(j)
        return json.dumps(j, ensure_ascii=False)

    def first_vocab(j):
        return next(iter(j["model"]["vocab"]))

    out = {
        "bad_normalizer": mod(lambda j: j.__setitem__("normalizer", {"type": "NFKC"})),
        "bad_pattern": mod(lambda j: j["pre_tokenizer"]["pretokenizers"][0]["pattern"].__setitem__("Regex", PATTERN + "|x")),
        "bad_prefix_space": mod(lambda j: j["pre_tokenizer"]["pretokenizers"][1].__setitem__("add_prefix_space", True)),
        "bad_model": mod(lambda j: j["model"].__setitem__("type", "WordPiece")),
        "bad_byte_fallback": mod(lambda j: j["model"].__setitem__("byte_fallback", True)),
        "bad_merge_unknown": mod(lambda j: j["model"]["merges"].append(["Ġ", "zzzzqqq"])),
        "bad_merge_dup": mod(lambda j: j["model"]["merges"].append(j["model"]["merges"][0])),
        "bad_merge_result": mod(lambda j: j["model"]["merges"].append(["Ā", "ā"])),
        "bad_vocab_char": mod(lambda j: j["model"]["vocab"].__setitem__("中", 99999)),
        "bad_vocab_dup_id": mod(lambda j: j["model"]["vocab"].__setitem__("ĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠ", 0)),
        "bad_missing_byte": mod(lambda j: j["model"]["vocab"].pop(first_vocab(j))),
        "bad_added_lstrip": mod(lambda j: j["added_tokens"][0].__setitem__("lstrip", True)),
        "bad_added_id": mod(lambda j: j["added_tokens"][0].__setitem__("id", 5)),
        "bad_negative_id": mod(lambda j: j["model"]["vocab"].__setitem__(first_vocab(j), -1)),
    }
    s = json.dumps(good, ensure_ascii=False)
    out["bad_truncated"] = s[: len(s) // 2]
    # 重复的键：手工拼出来，json.dumps 写不出这种文件
    out["bad_dup_key"] = s.replace('"vocab": {', '"vocab": {"Ġthe": 1, "Ġthe": 2, ', 1)
    out["bad_trailing"] = s + " {}"
    return out


def edge_texts():
    return [
        "", " ", "  ", "\n", "\r\n", " \n ", "\t\t", "a", "  a", "a  ", " a b  c   d", "x\n\n\ny",
        "it's", "IT'S", "'s'S'ſ", "'ſa", "'re're", "don't", "'ll'LL'Ll", "'", "''", "'''s",
        "123456789", "1.5e10", "٣٤٥", "Ⅻ", "½",
        "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|><|im_start|>", "<|im_start|", "<|endoftext|>x", "x<|endoftext|>",
        "<think>\n思考\n</think>\n\n答案", "<|音频|>中间<|音频|>", "<|vision_start|>",
        "cafe\u0301!", "café!", "xcafe\u0301!y", "cafe\u0301\u0301!",
        "\u00a0nbsp\u3000ideographic\u2028sep", "a\u0300\u0301\u0302", "\U0001F600" * 5,
        "x" * 2000, "中" * 500, "!" * 300 + "\n\n", " " * 1000,
    ]


def random_texts(rng, n):
    """随机字符串：从各个区块里取码点，刻意混入空白、标点、组合字符和撇号。"""
    pools = [
        list(range(0x20, 0x7F)) * 4, [0x09, 0x0A, 0x0D, 0x20, 0xA0, 0x3000, 0x2028, 0x85] * 20,
        list(range(0x300, 0x370)), list(range(0x4E00, 0x4F00)), list(range(0xAC00, 0xAC40)),
        list(range(0x1100, 0x1200)), list(range(0x0900, 0x0980)), list(range(0x1F300, 0x1F700)),
        list(range(0x0E00, 0x0E80)), list(range(0x0600, 0x0700)), [0x27, 0x27, 0x73, 0x53, 0x17F, 0x200D, 0xFE0F],
        list(range(0x80, 0x800)),
    ]
    out = []
    for _ in range(n):
        k = rng.randrange(1, 40)
        s = []
        for _ in range(k):
            if rng.random() < 0.05:
                c = rng.randrange(0x110000)
                if 0xD800 <= c < 0xE000:
                    c = 0x41
            else:
                c = rng.choice(rng.choice(pools))
            s.append(chr(c))
        out.append("".join(s))
    return out


def unicode_texts():
    """每个码点放进几种上下文里，覆盖正则每个分支的边界。"""
    ctx = ["x{}", "!{}!", "{}{}", " {}a", "{}\n", "1{}", "{}\u0301", "{}\u0323\u0301{}", "\u1100{}",
           # 缩写分支不分大小写：检查有哪些字符会被当成 s、t、m、d、r、e、v、l
           "'{}a", "'{}{}a", "'{}e", "'r{}", "'v{}", "'l{}"]
    for c in range(0x110000):
        if 0xD800 <= c < 0xE000:
            continue
        ch = chr(c)
        for t in ctx:
            yield t.replace("{}", ch)


def nfc_texts(rng, n):
    """归一化的压力测试：随机拼接带分解或组合类的字符，以及韩文字母。"""
    import unicodedata
    pool = [c for c in range(0x110000) if not 0xD800 <= c < 0xE000 and
            (unicodedata.decomposition(chr(c)) or unicodedata.combining(chr(c)))]
    pool += list(range(0x1100, 0x1113)) + list(range(0x1161, 0x1176)) + list(range(0x11A8, 0x11C3))
    pool += [0xAC00, 0xAC01, 0xD7A3, 0x41, 0x61, 0x20]
    for _ in range(n):
        yield "".join(chr(rng.choice(pool)) for _ in range(rng.randrange(1, 12)))


def main(out, real=None):
    """real 为真实模型的 tokenizer.json 时，不训练小词表，也不生成坏文件，
    只针对这个分词器生成测试文本和 id。"""
    os.makedirs(out, exist_ok=True)
    rng = random.Random(2)
    lines = corpus()
    path = os.path.join(out, "tokenizer.json")
    if real:
        with open(real, "rb") as src, open(path, "wb") as dst:
            dst.write(src.read())
        tk = Tokenizer.from_file(path)
    else:
        tk = build_tokenizer(lines)
        tk.save(path)
    with open(path, encoding="utf-8") as f:
        good = json.load(f)
    if not real:
        for name, text in bad_variants(good).items():
            with open(os.path.join(out, name + ".json"), "w", encoding="utf-8") as f:
                f.write(text)

    texts = [l for l in lines if l] + ["\n".join(lines)] + edge_texts() + random_texts(rng, 3000)
    with open(os.path.join(out, "texts.jsonl"), "w", encoding="utf-8") as f:
        for s in texts:
            f.write(json.dumps(s) + "\n")

    # 编码再解码应当还原出 NFC 归一化后的原文。例外是“归一化后匹配”的附加 token：
    # 小词表里的 "cafe\u0301!" 归一化后含 é，而 é 恰好在字节映射里，解码时会被
    # 当成单个字节 0xE9。Qwen 的附加 token 都不是这一类。
    norm_added = [tk.normalizer.normalize_str(a["content"]) for a in good["added_tokens"] if a["normalized"]]
    with open(os.path.join(out, "roundtrip.jsonl"), "w", encoding="utf-8") as f:
        for s in texts:
            if not any(a in tk.normalizer.normalize_str(s) for a in norm_added):
                f.write(json.dumps(s) + "\n")

    n = tk.get_vocab_size(with_added_tokens=True)
    with open(os.path.join(out, "ids.txt"), "w") as f:
        for i in range(n):                                   # 每个 id 单独解码
            f.write(f"{i}\n")
        for _ in range(2000):                                # 随机 id 序列
            f.write(" ".join(str(rng.randrange(n)) for _ in range(rng.randrange(1, 30))) + "\n")
        for s in texts[:200]:                                # 编码结果再解码
            f.write(" ".join(map(str, tk.encode(s).ids)) + "\n")

    with open(os.path.join(out, "unicode.jsonl"), "w", encoding="utf-8") as f:
        for s in unicode_texts():
            f.write(json.dumps(s) + "\n")
        for s in nfc_texts(rng, 200000):
            f.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
