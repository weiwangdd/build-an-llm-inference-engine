#!/bin/sh
# 用法：tests/run_ops.sh <build 目录> [real <模型目录>]
# 第 3 章的对拍：C 算子（build/ops）与 Python 参考实现（tests/ops_ref.py）逐元素比较。
#   不带参数：在 tests/make_ops_fixtures.py 生成的小输入上测试，并检查写坏的输入都被拒绝。
#   real：拿真实模型的权重当算子的输入（见 tests/make_ops_real.py），几分钟。
# 只用 Python 标准库。
set -u
B=$1; fail=0; PY=${PYTHON:-python3}

run_cases() {  # run_cases <用例清单>
    while read -r name op rest; do
        "$B/ops" $op $rest > "$B/got.txt"; rc=$?
        if [ $rc -ne 0 ]; then echo "FAIL $name (rc=$rc)"; fail=1; continue; fi
        if msg=$($PY tests/ops_ref.py check $op $rest < "$B/got.txt"); then echo "ok   $name：$msg"
        else echo "FAIL $name：$msg"; fail=1; fi
    done < "$1"
}

if [ "${2:-}" = real ]; then
    FX=$B/ops_real
    $PY tests/make_ops_real.py "$3" "$FX" || exit 1
    run_cases "$FX/cases.txt"
    exit $fail
fi

FX=$B/ops_fixtures
$PY tests/make_ops_fixtures.py "$FX" || exit 1

# BF16 转 float：全部 65536 个位模式逐字节比较
"$B/ops" bf16 > "$B/got.txt"
$PY tests/ops_ref.py bf16 > "$B/ref.txt"
if cmp -s "$B/ref.txt" "$B/got.txt"; then echo "ok   BF16 转 float：全部 65536 个位模式一致"
else echo "FAIL BF16 转 float"; diff "$B/ref.txt" "$B/got.txt" | head -5; fail=1; fi
# 已知答案，不依赖 Python：BF16 的 3f80 是 1.0，0001 是最小的非规格化数 2^-133
if [ "$(sed -n '16257p;2p' "$B/got.txt" | tr '\n' ' ')" = "00010000 3f800000 " ]; then echo "ok   BF16 已知答案"
else echo "FAIL BF16 已知答案"; fail=1; fi

run_cases "$FX/cases.txt"

# 已知答案：四个相等的数做 softmax，每个恰好是 1/4
"$B/ops" softmax "$FX/softmax_uniform.safetensors" > "$B/got.txt"
if [ "$(grep -c '^0x1p-2$' "$B/got.txt")" = 8 ]; then echo "ok   softmax 已知答案"
else echo "FAIL softmax 已知答案"; fail=1; fi

while read -r name op rest; do
    msg=$("$B/ops" $op $rest 2>&1 >/dev/null); rc=$?
    if [ $rc -eq 1 ]; then echo "ok   拒绝 $name：$msg"
    else echo "FAIL $name 应被拒绝，实际 rc=$rc $msg"; fail=1; fi
done < "$FX/bad_cases.txt"
exit $fail
