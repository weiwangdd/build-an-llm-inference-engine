#!/bin/sh
# 用法：tests/run_layer.sh <build 目录> [real <模型目录>]
# 第 4 章的对拍：C 的单层前向（build/layer）分两层检查。
#   第一层 tests/layer_ref.py：标准库，逐步检查每一步都是正确舍入（与第 3 章同一个标准）。
#   第二层 tests/layer_official.py：调用官方 transformers 的代码算同一层，检查公式读对了。
#     需要 torch 和 transformers，没装就跳过（不算通过）。
#   不带参数：在 tests/make_layer_fixtures.py 生成的小模型上测试，并检查写坏的输入都被拒绝。
#   real：在真实模型上测试（见 tests/make_layer_real.py），要十几分钟，大部分时间花在纯 Python 的参考实现上。
set -u
B=$1; fail=0; PY=${PYTHON:-python3}

if $PY -c "import torch, transformers" 2>/dev/null; then official=1
else official=0; echo "SKIP 第二层对拍（官方 transformers）：需要 torch 和 transformers（pip install torch transformers）"; fi

run_cases() {  # run_cases <config.json> <用例清单> <分片 ...>
    cfg=$1; list=$2; shift 2
    while read -r name layer ids; do
        "$B/layer" "$cfg" "$layer" "$ids" "$@" > "$B/got.txt"; rc=$?
        if [ $rc -ne 0 ]; then echo "FAIL $name (rc=$rc)"; fail=1; continue; fi
        if msg=$($PY tests/layer_ref.py "$cfg" "$layer" "$ids" "$@" < "$B/got.txt"); then echo "ok   $name 逐步：$msg"
        else echo "FAIL $name 逐步：$msg"; fail=1; fi
        [ $official = 1 ] || continue
        if msg=$($PY tests/layer_official.py "$cfg" "$layer" "$ids" "$@" < "$B/got.txt" 2>"$B/official_err.txt"); then echo "ok   $name 官方：$msg"
        else echo "FAIL $name 官方：$msg $(tail -1 "$B/official_err.txt")"; fail=1; fi
    done < "$list"
}

if [ "${2:-}" = real ]; then
    FX=$B/layer_real; mkdir -p "$FX"
    # 输入是一句真实的话，用第 2 章的分词器编码
    printf '%s\n' '"从零构建大模型推理引擎，每一步都对拍。"' > "$FX/text.jsonl"
    ids=$("$B/tok" encode "$3/tokenizer.json" "$FX/text.jsonl" | tr ' ' ',') || exit 1
    $PY tests/make_layer_real.py "$3" "$FX" "$ids" || exit 1
    # shellcheck disable=SC2046
    run_cases "$3/config.json" "$FX/cases.txt" $(cat "$FX/shards.txt")
    exit $fail
fi

FX=$B/layer_fixtures
$PY tests/make_layer_fixtures.py "$FX" || exit 1
run_cases "$FX/config.json" "$FX/cases.txt" "$FX/model.safetensors"

# 已知答案，不依赖 Python：输出投影全为 0 时，每一层的输出必须与输入逐位相同
for layer in 0 3; do
    "$B/layer" "$FX/config.json" $layer 3,17,17,0,39 "$FX/zero_model.safetensors" > "$B/got.txt"
    sect() { awk -v s="$1" '/ \[/ { on = ($1 == s); next } on' "$B/got.txt"; }
    if [ -n "$(sect embed)" ] && [ "$(sect embed)" = "$(sect out)" ]; then echo "ok   已知答案：输出投影为 0 的第 $layer 层原样输出"
    else echo "FAIL 已知答案：输出投影为 0 的第 $layer 层"; fail=1; fi
done

while read -r name cfg layer ids model; do
    msg=$("$B/layer" "$cfg" "$layer" "$ids" "$model" 2>&1 >/dev/null); rc=$?
    if [ $rc -eq 1 ]; then echo "ok   拒绝 $name：$msg"
    else echo "FAIL $name 应被拒绝，实际 rc=$rc $msg"; fail=1; fi
done < "$FX/bad_cases.txt"
exit $fail
