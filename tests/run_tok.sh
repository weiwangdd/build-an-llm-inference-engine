#!/bin/sh
# 用法：tests/run_tok.sh <build 目录> [unicode | real <tokenizer.json>]
# 第 2 章的对拍：C 分词器（build/tok）与官方 tokenizers 库（tests/tok_ref.py）
# 在同一个 tokenizer.json 上的输出必须逐字节一致；写错的 tokenizer.json 必须被拒绝。
#   不带参数：在现训练的小词表上做快速测试。
#   unicode：对全部 Unicode 码点做归一化和预分词的穷举对拍，几分钟。
#   real：换成真实模型的 tokenizer.json，把上面两组测试都跑一遍。
# 环境变量 PYTHON 可以指定带 tokenizers 库的 Python。
set -u
B=$1; FX=$B/tok_fixtures; fail=0; PY=${PYTHON:-python3}
$PY -c "import tokenizers" 2>/dev/null || { echo "FAIL 需要 Python 的 tokenizers 库：pip install tokenizers"; exit 1; }
# 官方库在不同版本之间行为不同（例如预分词正则用的 Unicode 版本），见第 2 章“参考实现的版本”。
$PY -c '
import sys, tokenizers
v = tuple(int(x) for x in tokenizers.__version__.split(".")[:3] if x.isdigit())
print("参考实现：tokenizers", tokenizers.__version__)
if v < (0, 21, 4):
    print("注意：本章的测试在 tokenizers 0.21.4 到 0.23.3 上验证过；更早的版本预分词会有差异")
'
if [ "${2:-}" = real ]; then
    FX=$B/tok_real
    $PY tests/make_tok_fixtures.py "$FX" "$3" || exit 1
else
    $PY tests/make_tok_fixtures.py "$FX" || exit 1   # 每次重新生成，免得用到旧版脚本留下的数据
fi

check() {  # check <名字> <模式> <输入> ：分别运行两份实现并比较
    case $2 in encode|decode|roundtrip) ca="$FX/tokenizer.json $3";; *) ca="$3";; esac
    $PY tests/tok_ref.py "$2" "$FX/tokenizer.json" "$3" > "$B/ref.txt" || { echo "FAIL 参考实现出错 $1"; fail=1; return; }
    "$B/tok" "$2" $ca > "$B/got.txt"; rc=$?
    if [ $rc -eq 0 ] && cmp -s "$B/ref.txt" "$B/got.txt"; then echo "ok   $1（$(wc -l < "$B/got.txt") 行）"
    else echo "FAIL $1 (rc=$rc)"; diff "$B/ref.txt" "$B/got.txt" | head -5; fail=1; fi
}

if [ "${2:-}" = unicode ]; then
    check "全部码点的归一化" nfc "$FX/unicode.jsonl"
    check "全部码点的预分词" pretok "$FX/unicode.jsonl"
    exit $fail
fi

check "归一化" nfc "$FX/texts.jsonl"
check "预分词" pretok "$FX/texts.jsonl"
check "编码" encode "$FX/texts.jsonl"
check "解码" decode "$FX/ids.txt"
check "编码再解码" roundtrip "$FX/roundtrip.jsonl"

if [ "${2:-}" = real ]; then
    check "全部码点的归一化" nfc "$FX/unicode.jsonl"
    check "全部码点的预分词" pretok "$FX/unicode.jsonl"
    exit $fail
fi

for f in "$FX"/bad_*.json; do
    msg=$("$B/tok" encode "$f" "$FX/texts.jsonl" 2>&1 >/dev/null); rc=$?
    if [ $rc -eq 1 ]; then echo "ok   $(basename "$f")：$msg"
    else echo "FAIL $(basename "$f") 应被拒绝，实际 rc=$rc $msg"; fail=1; fi
done
exit $fail
