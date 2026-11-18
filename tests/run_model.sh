#!/bin/sh
# 用法：tests/run_model.sh <build 目录> [real <模型目录>]
# 第 5 章的对拍：C 的整模前向和贪心解码（build/forward）分两层检查，同第 4 章。
#   第一层 tests/model_ref.py：标准库，逐步检查每一步都是正确舍入，生成的 token 是分数最大的。
#   第二层 tests/model_official.py：调用官方 transformers 的代码，逐层对拍，并检查贪心解码的 token
#     与官方 float64 完全相同；官方 BF16 只报告。需要 torch 和 transformers，没装就跳过（不算通过）。
#   不带参数：在 tests/make_model_fixtures.py 生成的小模型上测试，并检查写坏的输入都被拒绝。
#   real：在真实模型上测试。环境变量 FORWARD 可以指定另一份 forward（make test-model-real 用 -O2、
#     不带 ASan 构建的那份），PROMPT 换提示词，GEN 换生成个数（默认 8）。
set -u
B=$1; fail=0; PY=${PYTHON:-python3}; FWD=${FORWARD:-$B/forward}

if $PY -c "import torch, transformers" 2>/dev/null; then official=1
else official=0; echo "SKIP 第二层对拍（官方 transformers）：需要 torch 和 transformers（pip install torch transformers）"; fi

check() {  # check <名字> <config.json> <ids> <m> <all|layers> <分片 ...>，读 $B/got.txt
    name=$1; shift
    if msg=$($PY tests/model_ref.py "$@" < "$B/got.txt"); then echo "ok   $name 逐步：$msg"
    else echo "FAIL $name 逐步：$msg"; fail=1; fi
    [ $official = 1 ] || return
    if msg=$($PY tests/model_official.py "$@" < "$B/got.txt" 2>"$B/official_err.txt"); then echo "ok   $name 官方：$msg"
    else echo "FAIL $name 官方：$msg $(tail -1 "$B/official_err.txt")"; fail=1; fi
}

if [ "${2:-}" = real ]; then
    M=$3; FX=$B/model_real; mkdir -p "$FX"
    # 提示词用第 2 章的分词器编码；不套对话模板，让模型直接续写
    printf '"%s"\n' "${PROMPT:-中国的首都是}" > "$FX/text.jsonl"
    ids=$("$B/tok" encode "$M/tokenizer.json" "$FX/text.jsonl" | tr ' ' ',') || exit 1
    gen=${GEN:-8}
    shards=$(ls "$M"/*.safetensors) || exit 1
    echo "提示词 token：$ids；生成 $gen 个"
    start=$(date +%s)
    # shellcheck disable=SC2086
    "$FWD" "$M/config.json" "$ids" "$gen" layers $shards > "$B/got.txt" || { echo "FAIL C 端出错"; exit 1; }
    echo "C 端用时 $(( $(date +%s) - start )) 秒"
    out=$(awk '/^greedy \[/ { on = 1; next } on' "$B/got.txt" | paste -sd' ' -)
    echo "$out" > "$FX/out.txt"
    hex=$("$B/tok" decode "$M/tokenizer.json" "$FX/out.txt" | cut -d' ' -f1) || exit 1   # 第一列是保留特殊 token 的解码
    echo "生成的 token：$out，文字：$($PY -c 'import sys; print(repr(bytes.fromhex(sys.argv[1]).decode()))' "$hex")"
    # shellcheck disable=SC2086
    check "真实模型" "$M/config.json" "$ids" "$gen" layers $shards
    exit $fail
fi

FX=$B/model_fixtures
$PY tests/make_model_fixtures.py "$FX" || exit 1
cfg=$FX/config.json; model=$FX/model.safetensors
while read -r name ids m; do
    "$B/forward" "$cfg" "$ids" "$m" all "$model" > "$B/got.txt" || { echo "FAIL $name (rc=$?)"; fail=1; continue; }
    check "$name" "$cfg" "$ids" "$m" all "$model"
    # layers 方式只打印每层的输出，真实模型用它；这里只做第一层，确认两种方式都能对上
    "$B/forward" "$cfg" "$ids" "$m" layers "$model" > "$B/got.txt" || { echo "FAIL $name layers (rc=$?)"; fail=1; continue; }
    if msg=$($PY tests/model_ref.py "$cfg" "$ids" "$m" layers "$model" < "$B/got.txt"); then echo "ok   $name（layers）逐步：$msg"
    else echo "FAIL $name（layers）逐步：$msg"; fail=1; fi
done < "$FX/cases.txt"

# 已知答案，不依赖 Python。模型是因果的：第 t 行 logits 只取决于前 t+1 个 token。
# C 的每一步都只按 token 的先后顺序算，所以不但数值上相等，而且应当逐位相同。
sect() { awk -v s="$2" '/ \[/ { on = ($1 == s); next } on' "$1"; }
"$B/forward" "$cfg" 3,17,17,0,39,5,22,8 0 layers "$model" > "$B/a.txt"
"$B/forward" "$cfg" 3,17,17,0,39 0 layers "$model" > "$B/b.txt"
V=$(awk '/"vocab_size"/ { gsub(/[^0-9]/, ""); print }' "$cfg")
if [ -n "$V" ] && [ "$(sect "$B/a.txt" logits | head -n $((5 * V)))" = "$(sect "$B/b.txt" logits)" ]; then
    echo "ok   已知答案：8 个 token 的前 5 行 logits 与只给前 5 个 token 时逐位相同"
else echo "FAIL 已知答案：前 5 行 logits 与只给前 5 个 token 时不同"; fail=1; fi
# 贪心解码接着往下生成，结果也不受“从哪里开始”的影响
"$B/forward" "$cfg" 7 8 layers "$model" | sect /dev/stdin greedy > "$B/a.txt"
first3=$(head -n 3 "$B/a.txt" | paste -sd, -)
"$B/forward" "$cfg" "7,$first3" 5 layers "$model" | sect /dev/stdin greedy > "$B/b.txt"
if [ "$(tail -n 5 "$B/a.txt")" = "$(cat "$B/b.txt")" ]; then echo "ok   已知答案：从 7 生成 8 个，与从 7,$first3 接着生成 5 个，结果相同"
else echo "FAIL 已知答案：接着生成的结果不同"; fail=1; fi

while read -r name c ids m mode f; do
    msg=$("$B/forward" "$c" "$ids" "$m" "$mode" "$f" 2>&1 >/dev/null); rc=$?
    if [ $rc -eq 1 ]; then echo "ok   拒绝 $name：$msg"
    else echo "FAIL $name 应被拒绝，实际 rc=$rc $msg"; fail=1; fi
done < "$FX/bad_cases.txt"
exit $fail
