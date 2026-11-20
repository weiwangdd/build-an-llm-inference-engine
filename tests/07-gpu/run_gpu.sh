#!/bin/sh
# 用法：tests/07-gpu/run_gpu.sh <CPU 版 build 目录> <GPU 版 build 目录> [real <模型目录> <tok 程序>]
# 第 7 章：同样的输入交给 CPU 版和 GPU 版，逐层、逐步比较输出（tests/07-gpu/gpu_cmp.py）。
# GPU 版每一步对不对，已经由第 3 到 6 章的测试脚本（换成 GPU 版的程序）各自检查过；这里检查两边一致：
# 步骤、形状、生成的 token 完全相同，每个张量的误差不超过均方根的 2^-12，并报告逐位相同的比例。
#   不带参数：在第 4、5、6 章的小模型上比较第 4 章的单层、第 5 章的整模前向加贪心、第 6 章的生成（--dump）。
#   real：在真实模型上对同一个提示词各做一次整模前向，比较每一层的输出和 logits。
#     PROMPT 换提示词。两边都应当是 -O2、不带 ASan 的那份（make test-gpu-real 就是这样调用的）。
# 只用 Python 标准库。
set -u
C=$1; G=$2; fail=0; PY=${PYTHON:-python3}
ok()  { echo "ok   $*"; }
bad() { echo "FAIL $*"; fail=1; }

compare() {  # compare <名字> <trace|generate>，比较 $G/cmp_cpu.txt 和 $G/cmp_gpu.txt
    if msg=$($PY tests/07-gpu/gpu_cmp.py "$2" "$G/cmp_cpu.txt" "$G/cmp_gpu.txt"); then ok "$1：$msg"
    else bad "$1：$msg"; fi
}

# both <名字> <trace|generate> <程序名> <参数 ...>：两边各跑一次再比较
both() {
    name=$1; kind=$2; prog=$3; shift 3
    "$C/$prog" "$@" > "$G/cmp_cpu.txt" || { bad "$name CPU 版出错"; return; }
    "$G/$prog" "$@" > "$G/cmp_gpu.txt" || { bad "$name GPU 版出错"; return; }
    compare "$name" "$kind"
}

if [ "${3:-}" = real ]; then
    M=$4; TOK=$5; FX=$G/gpu_real; mkdir -p "$FX"
    printf '"%s"\n' "${PROMPT:-中国的首都是}" > "$FX/text.jsonl"
    ids=$("$TOK" encode "$M/tokenizer.json" "$FX/text.jsonl" | tr ' ' ',') || exit 1
    shards=$(ls "$M"/*.safetensors) || exit 1
    echo "提示词 token：$ids"
    for side in cpu gpu; do
        [ $side = cpu ] && bin=$C/forward || bin=$G/forward
        start=$(date +%s)
        # shellcheck disable=SC2086
        "$bin" "$M/config.json" "$ids" 0 layers $shards > "$G/cmp_$side.txt" || { bad "$side 版出错"; exit 1; }
        echo "$side 版一次前向（含加载权重）用时 $(( $(date +%s) - start )) 秒"
    done
    compare "真实模型：每一层的输出、最终归一化和 logits" trace
    exit $fail
fi

FX=$G/gpu_fixtures; mkdir -p "$FX"
$PY tests/04-layer/make_layer_fixtures.py "$FX/layer" || exit 1
$PY tests/05-model/make_model_fixtures.py "$FX/model" || exit 1
$PY tests/06-generate/make_gen_fixtures.py "$FX/gen" || exit 1

while read -r name layer ids; do
    both "单层 $name" trace layer "$FX/layer/config.json" "$layer" "$ids" "$FX/layer/model.safetensors"
done < "$FX/layer/cases.txt"

while read -r name ids m; do
    both "整模 $name" trace forward "$FX/model/config.json" "$ids" "$m" all "$FX/model/model.safetensors"
done < "$FX/model/cases.txt"

while read -r name gcfg ids m opts; do
    # shellcheck disable=SC2086
    both "生成 $name" generate generate "$FX/gen/config.json" "$gcfg" "$ids" "$m" $opts --dump "$FX/gen/model.safetensors"
done < "$FX/gen/cases.txt"
exit $fail
