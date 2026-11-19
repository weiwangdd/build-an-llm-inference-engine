#!/bin/sh
# 用法：tests/run_batch.sh <build 目录>
# 第 8 章的对拍：连续批处理（build/batch）与逐个单独生成（build/generate，第 6 章）比较。
#   请求清单里的每个请求，单独交给 generate（同样的选项，--dump 打印每一步的 logits、分布、随机数），
#   输出必须与 batch 里这个请求的那一段逐字节相同；被取消的请求必须是单独生成的前 K 个 token 再加 “stop cancel”。
#   --max-batch 取 1、3、8 各跑一遍：1 就是一个一个排队算，8 是全部挤在一起，结果都必须相同。
#   另查 batch 的日志：批里确实同时有好几个请求，预填充和解码确实混在同一步里。
#   最后检查写坏的输入都被拒绝。
# 环境变量 BATCH、GENERATE 可以换另一份可执行文件。
set -u
B=$1; FX=$B/batch_fixtures; fail=0; PY=${PYTHON:-python3}
BAT=${BATCH:-$B/batch}; GEN_BIN=${GENERATE:-$B/generate}
ok()  { echo "ok   $*"; }
bad() { echo "FAIL $*"; fail=1; }

$PY tests/make_batch_fixtures.py "$FX" > /dev/null || exit 1
cfg=$FX/config.json; gcfg=$FX/gen_qwen.json; model=$FX/model.safetensors

# 每个请求单独跑一遍 generate，作为标准答案：$FX/single_<i>.txt
i=0
grep -v '^#' "$FX/requests.txt" | grep -v '^$' > "$FX/reqs.txt"
while read -r arrive ids max rest; do
    cancel=$(echo "$rest" | awk '{ for (k = 1; k < NF; k++) if ($k == "--cancel-after") print $(k + 1) }')
    opts=$(echo "$rest" | awk '{ s = ""; for (k = 1; k <= NF; k++) { if ($k == "--cancel-after") { k++; continue } s = s " " $k } print s }')
    # shellcheck disable=SC2086
    "$GEN_BIN" "$cfg" "$gcfg" "$ids" "$max" $opts --dump "$model" > "$FX/single_$i.txt" || { bad "请求 $i 单独生成出错"; exit 1; }
    if [ -n "$cancel" ]; then         # 取消：只保留前 cancel 个 token（除非它自己先停了）
        awk -v k="$cancel" '
            /^token / { n++; print; if (n == k) { cut = 1; exit } ; next }
            { print }
            END { if (cut) print "stop cancel" }' "$FX/single_$i.txt" > "$FX/single_$i.cut"
        if ! grep -q '^stop' "$FX/single_$i.cut"; then grep '^stop' "$FX/single_$i.txt" >> "$FX/single_$i.cut"; fi
        mv "$FX/single_$i.cut" "$FX/single_$i.txt"
    fi
    i=$((i + 1))
done < "$FX/reqs.txt"
nreq=$i
ntok=$(cat "$FX"/single_*.txt | grep -c '^token ')
echo "标准答案：$nreq 个请求逐个单独生成，共 $ntok 个 token"

for mb in 1 3 8; do
    "$BAT" "$cfg" "$gcfg" "$FX/requests.txt" --max-batch $mb --dump --log "$FX/log_$mb.txt" "$model" > "$FX/batch_$mb.txt" \
        || { bad "max_batch=$mb：batch 出错"; continue; }
    same=0
    for i in $(seq 0 $((nreq - 1))); do
        awk -v r="req $i" '/^req / { on = ($0 == r); next } on' "$FX/batch_$mb.txt" > "$FX/got_$i.txt"
        if cmp -s "$FX/got_$i.txt" "$FX/single_$i.txt"; then same=$((same + 1))
        else bad "max_batch=$mb：请求 $i 与单独生成不同（$FX/got_$i.txt 对 $FX/single_$i.txt）"; fi
    done
    steps=$(wc -l < "$FX/log_$mb.txt")
    widest=$(awk '{ if (NF - 2 > m) m = NF - 2 } END { print m }' "$FX/log_$mb.txt")
    # 同一步里既有交多个 token 的（预填充）又有交 1 个的（解码）
    mixed=$(awk '{ p = 0; d = 0; for (k = 3; k <= NF; k++) { split($k, a, ":"); if (a[2] > 1) p = 1; else d = 1 } if (p && d) m++ } END { print m + 0 }' "$FX/log_$mb.txt")
    [ "$same" = "$nreq" ] && ok "max_batch=$mb：$nreq 个请求的 logits、分布、随机数、token、停止都与单独生成逐字节相同（$steps 步，一批最多 $widest 个请求，$mixed 步里预填充与解码同批）"
    if [ "$widest" -gt "$mb" ]; then bad "max_batch=$mb：一批里有 $widest 个请求，超过上限"; fi
    if [ $mb -gt 1 ] && [ "$widest" -lt 2 ]; then bad "max_batch=$mb：从来没有两个请求同批，没测到批处理"; fi
    if [ $mb -gt 1 ] && [ "$mixed" -lt 1 ]; then bad "max_batch=$mb：预填充与解码从来没有同批"; fi
done
# max_batch=1 时一步只算一个请求：它就是“排队一个个来”，总步数应当最多
s1=$(wc -l < "$FX/log_1.txt"); s8=$(wc -l < "$FX/log_8.txt")
if [ "$s8" -lt "$s1" ]; then ok "已知答案：max_batch=8 用了 $s8 步，max_batch=1 用了 $s1 步"
else bad "批处理没有减少步数（$s8 对 $s1）"; fi

while read -r name rest; do
    # shellcheck disable=SC2086
    msg=$("$BAT" $rest 2>&1 >/dev/null); rc=$?
    if [ $rc -eq 1 ]; then ok "拒绝 $name：$msg"
    else bad "$name 应被拒绝，实际 rc=$rc $msg"; fi
done < "$FX/bad_cases.txt"
exit $fail
