#!/bin/sh
# 用法：tests/run_gen.sh <build 目录> [real <模型目录>]
# 第 6 章的对拍：带缓存的生成循环（build/generate）。
#   已知答案（不依赖 Python）：同一组参数，带缓存、不带缓存（--no-cache，第 5 章的做法）、提示词分块，
#     输出必须逐字节相同；每一步的 logits 与对“提示词加上生成的 token”做一次不带缓存的前向得到的那几行逐位相同。
#   第一层 tests/gen_ref.py：标准库，检查随机数、分布、选择和停止条件。
#   第二层 tests/gen_official.py：官方 transformers 的 logits 处理器；小模型上还跑官方的 generate。
#     需要 torch 和 transformers，没装就跳过（不算通过）。
#   不带参数：在小模型上测试，并检查写坏的输入都被拒绝。
#   real：在真实模型上测试。环境变量 GENERATE、FORWARD 可以指定另一份可执行文件（make test-gen-real 用 -O2、
#     不带 ASan 构建的那份），PROMPT 换提问，GEN 换最多生成个数（默认 48），SEED 换种子（默认 0）。
set -u
B=$1; fail=0; PY=${PYTHON:-python3}; GEN_BIN=${GENERATE:-$B/generate}; FWD=${FORWARD:-$B/forward}

if $PY -c "import torch, transformers" 2>/dev/null; then official=1
else official=0; echo "SKIP 第二层对拍（官方 transformers）：需要 torch 和 transformers（pip install torch transformers）"; fi

ok()  { echo "ok   $*"; }
bad() { echo "FAIL $*"; fail=1; }
sect() { awk -v s="$2" '/ \[/ { on = ($1 == s); next } on' "$1"; }    # forward 输出里的一段
rows() { awk '/^logits / { c = $2; next } c > 0 { print; c-- }' "$1"; }  # generate --dump 里每一步的 logits

# check <名字> <official 方式> <config.json> <generation_config.json> <ids> <max_new> [选项 ...] <分片 ...>
# 读 $B/gen.txt（generate --dump 的输出）
check() {
    name=$1; mode=$2; shift 2
    if msg=$($PY tests/gen_ref.py "$@" < "$B/gen.txt"); then ok "$name 逐步：$msg"
    else bad "$name 逐步：$msg"; fi
    [ $official = 1 ] || return 0
    if msg=$($PY tests/gen_official.py "$mode" "$@" < "$B/gen.txt" 2>"$B/official_err.txt"); then ok "$name 官方：$msg"
    else bad "$name 官方：$msg $(tail -1 "$B/official_err.txt")"; fi
}

# same_as_forward <名字> <config.json> <ids> <forward 的输出方式> <分片 ...>：
# 对“提示词加上除最后一个之外的生成 token”做一次不带缓存的前向，后面那几行 logits 必须与 generate 每一步的逐位相同。
# 前向的输出留在 $B/fwd.txt，第 5 章的 model_ref.py 可以接着检查它。
same_as_forward() {
    name=$1; cfg=$2; ids=$3; how=$4; shift 4
    toks=$(awk '$1 == "token" { print $2 }' "$B/gen.txt")
    m=$(echo "$toks" | wc -l)
    full=$ids
    [ "$m" -gt 1 ] && full="$ids,$(echo "$toks" | head -n $((m - 1)) | paste -sd, -)"
    echo "$full" > "$B/full_ids.txt"
    "$FWD" "$cfg" "$full" 0 "$how" "$@" > "$B/fwd.txt" || { bad "$name 不带缓存的前向出错"; return 1; }
    n=$(echo "$ids" | tr ',' '\n' | wc -l)
    V=$(awk '/^logits / { print $2; exit }' "$B/gen.txt")
    rows "$B/gen.txt" > "$B/rows_gen.txt"
    sect "$B/fwd.txt" logits | tail -n +$(( (n - 1) * V + 1 )) > "$B/rows_fwd.txt"
    if cmp -s "$B/rows_gen.txt" "$B/rows_fwd.txt"; then
        ok "$name 已知答案：$m 步的 logits 与一次不带缓存的前向（$((n + m - 1)) 个 token）的后 $m 行逐位相同"
    else bad "$name 已知答案：带缓存的 logits 与不带缓存的前向不同"; fi
}

if [ "${2:-}" = real ]; then
    M=$3; FX=$B/gen_real; mkdir -p "$FX"; cfg=$M/config.json; gcfg=$M/generation_config.json
    shards=$(ls "$M"/*.safetensors) || exit 1
    # 一、第 5 章的提示词，带缓存贪心解码 8 个：应与第 5 章不带缓存时的结果相同
    printf '"%s"\n' "中国的首都是" > "$FX/text.jsonl"
    ids=$("$B/tok" encode "$M/tokenizer.json" "$FX/text.jsonl" | tr ' ' ',') || exit 1
    start=$(date +%s)
    # shellcheck disable=SC2086
    "$GEN_BIN" "$cfg" "$gcfg" "$ids" 8 --greedy --dump $shards > "$B/gen.txt" || { bad "C 端出错"; exit 1; }
    echo "贪心：提示词 $ids，带缓存生成用时 $(( $(date +%s) - start )) 秒"
    awk '$1 == "token" { print $2 }' "$B/gen.txt" | paste -sd' ' - > "$FX/out.txt"
    hex=$("$B/tok" decode "$M/tokenizer.json" "$FX/out.txt" | cut -d' ' -f1) || exit 1
    echo "生成的 token：$(cat "$FX/out.txt")，文字：$($PY -c 'import sys; print(repr(bytes.fromhex(sys.argv[1]).decode("utf-8", "replace")))' "$hex")"
    # shellcheck disable=SC2086
    check "真实模型_贪心" warpers "$cfg" "$gcfg" "$ids" 8 --greedy $shards
    start=$(date +%s)
    # shellcheck disable=SC2086
    same_as_forward "真实模型_贪心" "$cfg" "$ids" layers $shards
    echo "不带缓存的一次前向用时 $(( $(date +%s) - start )) 秒"

    # 二、套上对话模板提问，按 generation_config.json 的参数采样，直到 <|im_end|> 或 GEN 个
    q=${PROMPT:-中国的首都是哪里？}
    printf '{"messages":[{"role":"user","content":"%s"}],"add_generation_prompt":true,"enable_thinking":false}\n' "$q" > "$FX/chat.jsonl"
    ids=$("$B/tok" chat "$M/tokenizer.json" "$M/chat_template.jinja" "$FX/chat.jsonl" | cut -d' ' -f2- | tr ' ' ',') || exit 1
    case $ids in *[!0-9,]*|"") bad "对话模板渲染失败：$ids"; exit 1;; esac
    gen=${GEN:-48}; seed=${SEED:-0}
    start=$(date +%s)
    # shellcheck disable=SC2086
    "$GEN_BIN" "$cfg" "$gcfg" "$ids" "$gen" --seed "$seed" --dump $shards > "$B/gen.txt" || { bad "C 端出错"; exit 1; }
    echo "采样：提示词 $(echo "$ids" | tr ',' '\n' | wc -l) 个 token，带缓存生成用时 $(( $(date +%s) - start )) 秒，$(tail -1 "$B/gen.txt")"
    awk '$1 == "token" { print $2 }' "$B/gen.txt" | paste -sd' ' - > "$FX/out.txt"
    hex=$("$B/tok" decode "$M/tokenizer.json" "$FX/out.txt" | cut -d' ' -f1) || exit 1
    echo "生成的 token：$(cat "$FX/out.txt")"
    echo "文字：$($PY -c 'import sys; print(repr(bytes.fromhex(sys.argv[1]).decode("utf-8", "replace")))' "$hex")"
    # shellcheck disable=SC2086
    check "真实模型_对话采样" warpers "$cfg" "$gcfg" "$ids" "$gen" --seed "$seed" $shards
    start=$(date +%s)
    # shellcheck disable=SC2086
    same_as_forward "真实模型_对话采样" "$cfg" "$ids" layers $shards || exit 1
    echo "不带缓存的一次前向用时 $(( $(date +%s) - start )) 秒"
    # 这次前向本身再用第 5 章的两层对拍查一遍（这个提示词第 5 章没有查过）
    full=$(cat "$B/full_ids.txt")
    # shellcheck disable=SC2086
    if msg=$($PY tests/model_ref.py "$cfg" "$full" 0 layers $shards < "$B/fwd.txt"); then ok "真实模型_对话采样 前向逐步：$msg"
    else bad "真实模型_对话采样 前向逐步：$msg"; fi
    if [ $official = 1 ]; then
        # shellcheck disable=SC2086
        if msg=$($PY tests/model_official.py "$cfg" "$full" 0 layers $shards < "$B/fwd.txt" 2>"$B/official_err.txt"); then
            ok "真实模型_对话采样 前向官方：$msg"
        else bad "真实模型_对话采样 前向官方：$msg $(tail -1 "$B/official_err.txt")"; fi
    fi
    exit $fail
fi

FX=$B/gen_fixtures
$PY tests/make_gen_fixtures.py "$FX" || exit 1
cfg=$FX/config.json; model=$FX/model.safetensors
run_cases() {  # run_cases <用例清单> <分片>
    model=$2
    while read -r name gcfg ids m opts; do
        # shellcheck disable=SC2086
        "$GEN_BIN" "$cfg" "$gcfg" "$ids" "$m" $opts --dump "$model" > "$B/gen.txt" || { bad "$name (rc=$?)"; continue; }
        # 带缓存、不带缓存、提示词一个一个给、三个三个给：输出逐字节相同
        for v in --no-cache "--chunk 1" "--chunk 3"; do
            # shellcheck disable=SC2086
            "$GEN_BIN" "$cfg" "$gcfg" "$ids" "$m" $opts --dump $v "$model" > "$B/gen_v.txt" || { bad "$name $v (rc=$?)"; continue; }
            cmp -s "$B/gen.txt" "$B/gen_v.txt" || bad "$name：$v 的输出与带缓存的不同"
        done
        ok "$name 已知答案：带缓存、--no-cache、--chunk 1、--chunk 3 的输出逐字节相同（$(tail -1 "$B/gen.txt")，$(grep -c '^token' "$B/gen.txt") 个 token）"
        # shellcheck disable=SC2086
        check "$name" full "$cfg" "$gcfg" "$ids" "$m" $opts "$model"
        same_as_forward "$name" "$cfg" "$ids" all "$model" || continue
        full=$(cat "$B/full_ids.txt")
        if msg=$($PY tests/model_ref.py "$cfg" "$full" 0 all "$model" < "$B/fwd.txt"); then ok "$name 前向逐步：$msg"
        else bad "$name 前向逐步：$msg"; fi
    done < "$1"
}
run_cases "$FX/cases.txt" "$FX/model.safetensors"
run_cases "$FX/tie_cases.txt" "$FX/tie_model.safetensors"
model=$FX/model.safetensors

# 已知答案：停止条件。先不让它停（结束符设成一个这次没生成过的 id），生成 12 个；
# 再从第 4 个起找一个前面没出现过的 token 设成结束符，同样的种子，应当正好在它那里停下，前面的 token 完全相同。
nostop=$FX/gen_nostop.json; stop=$FX/gen_stop.json
args="3,17,17,0,39,5,22,8 12 --seed 11"
for e in 0 1; do      # 第一次用 0 当结束符试一下；如果 0 被生成了，就换成一个确实没生成过的
    printf '{"do_sample": true, "eos_token_id": %s, "temperature": 1.0, "top_k": 0, "top_p": 1.0}\n' "$e" > "$nostop"
    # shellcheck disable=SC2086
    "$GEN_BIN" "$cfg" "$nostop" $args "$model" | awk '$1 == "token" { print $2 }' > "$B/a.txt"
    [ "$(wc -l < "$B/a.txt")" -eq 12 ] && break
    e=$(awk '{ seen[$1] = 1 } END { for (i = 0; i < 40; i++) if (!(i in seen)) { print i; exit } }' "$B/a.txt")
done
at=$(awk 'NR >= 4 && !($1 in seen) { print NR; exit } { seen[$1] = 1 }' "$B/a.txt")
e=$(sed -n "${at}p" "$B/a.txt")
printf '{"do_sample": true, "eos_token_id": [%s], "temperature": 1.0, "top_k": 0, "top_p": 1.0}\n' "$e" > "$stop"
# shellcheck disable=SC2086
"$GEN_BIN" "$cfg" "$stop" $args --dump "$model" > "$B/gen.txt"
awk '$1 == "token" { print $2 }' "$B/gen.txt" > "$B/b.txt"
if [ "$(wc -l < "$B/a.txt")" -eq 12 ] && [ -n "$at" ] && [ "$(cat "$B/b.txt")" = "$(head -n "$at" "$B/a.txt")" ] &&
   [ "$(tail -1 "$B/gen.txt")" = "stop eos" ]; then
    ok "已知答案：不设结束符时生成 12 个；把其中第 $at 个（$e，此前没出现过）设成结束符，正好在第 $at 个处停下，前面的完全相同"
else bad "已知答案：设了结束符之后没有在它第一次出现的地方停下"; fi
# shellcheck disable=SC2086
check "停在结束符" full "$cfg" "$stop" $args "$model"

# 已知答案：同一个种子两次的结果相同，换一个种子结果不同
# shellcheck disable=SC2086
"$GEN_BIN" "$cfg" "$nostop" 7 16 --seed 5 "$model" > "$B/a.txt"
# shellcheck disable=SC2086
"$GEN_BIN" "$cfg" "$nostop" 7 16 --seed 5 "$model" > "$B/b.txt"
"$GEN_BIN" "$cfg" "$nostop" 7 16 --seed 6 "$model" > "$B/c.txt"
if cmp -s "$B/a.txt" "$B/b.txt" && ! cmp -s "$B/a.txt" "$B/c.txt"; then ok "已知答案：种子相同结果相同，种子不同结果不同"
else bad "已知答案：随机数种子不起作用"; fi

while read -r name rest; do
    # shellcheck disable=SC2086
    msg=$("$GEN_BIN" $rest 2>&1 >/dev/null); rc=$?
    if [ $rc -eq 1 ]; then ok "拒绝 $name：$msg"
    else bad "$name 应被拒绝，实际 rc=$rc $msg"; fi
done < "$FX/bad_cases.txt"
exit $fail
