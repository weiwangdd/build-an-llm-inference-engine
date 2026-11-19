#!/usr/bin/env bash
# ask.sh — 在命令行里向模型提问：套上对话模板，用第 6 章的生成循环回答，边生成边打印。
#
#   tools/ask.sh [选项] "问题"
#     -n N      最多生成 N 个 token，默认 64
#     -s S      随机数种子，默认 0（种子相同，回答完全相同）
#     -g        贪心解码（默认按 generation_config.json 采样）
#     -t        打开思考模式（模型先在 <think> 里写思考过程，要多得多的 token）
#     -G        在 GPU 上算（第 7 章，需要 CUDA，见 Makefile 的 CUDA 变量）
#   环境变量 MODEL 指定模型目录，默认 /data/models/Qwen3.8-27B。
#
# CPU 版是单线程，27B 每个 token 约 13.5 秒（提示词的 token 也算），一个问题要好几分钟。
# 需要 make、C 编译器、xxd，以及系统的 python3（只用标准库，给问题做 JSON 转义）。
set -euo pipefail

usage() { sed -n '4,10p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

max=64; seed=0; mode=(); think=false; gpu=0
while getopts "n:s:gtGh" o; do
    case $o in
        n) max=$OPTARG;; s) seed=$OPTARG;; g) mode=(--greedy);; t) think=true;; G) gpu=1;; *) usage;;
    esac
done
shift $((OPTIND - 1))
[ $# -eq 1 ] || usage

root=$(cd "$(dirname "$0")/.." && pwd)
M=${MODEL:-/data/models/Qwen3.8-27B}
for f in config.json generation_config.json tokenizer.json chat_template.jinja; do
    [ -f "$M/$f" ] || { echo "错误：模型目录 $M 里没有 $f（用 MODEL=<模型目录> 指定）" >&2; exit 1; }
done
shards=("$M"/*.safetensors)

# 和 make test-gen-real 一样：-O2、不带 ASan 另建一份，代码完全相同。源码没变时 make 什么也不做。
# -G 时建的是 GPU 版（第 7 章），放在 build/gpu/fast。
d=build/fast; [ $gpu = 1 ] && d=build/gpu/fast
B=$root/$d
make -s -C "$root" BUILD=$d GPU=$gpu SAN= \
    CFLAGS="-std=c17 -O2 -g -Wall -Wextra -Wpedantic -Wconversion -Werror -Isrc" \
    $d/tok $d/generate >&2

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

python3 -c 'import json, sys
print(json.dumps({"messages": [{"role": "user", "content": sys.argv[1]}],
                  "add_generation_prompt": True, "enable_thinking": sys.argv[2] == "true"},
                 ensure_ascii=False))' "$1" "$think" > "$tmp/chat.jsonl"
ids=$("$B/tok" chat "$M/tokenizer.json" "$M/chat_template.jinja" "$tmp/chat.jsonl" | cut -d' ' -f2- | tr ' ' ',')
case $ids in *[!0-9,]*|"") echo "错误：对话模板渲染失败：$ids" >&2; exit 1;; esac
if [ $gpu = 1 ]; then echo "（提示词 $(tr ',' '\n' <<< "$ids" | wc -l) 个 token，GPU 上先要把权重拷进显存）" >&2
else echo "（提示词 $(tr ',' '\n' <<< "$ids" | wc -l) 个 token，每个 token 约十几秒）" >&2; fi

# 每来一个 token，把目前为止的全部 token 解码一次（跳过特殊 token），只打印新增的字节。
# tok decode 会把不完整的 UTF-8 换成 U+FFFD（efbfbd）：一个汉字可能被切在两个 token 里，
# 所以末尾的 efbfbd 先不打印，等后面的 token 把这个字补齐。
start=$(date +%s); out=(); shown=0; stop=""
while read -r kind val; do
    case $kind in
        token) out+=("$val") ;;
        stop) stop=$val; break ;;
        *) continue ;;
    esac
    echo "${out[*]}" > "$tmp/ids.txt"
    hex=$("$B/tok" decode "$M/tokenizer.json" "$tmp/ids.txt" | cut -d' ' -f2)
    while [ "${hex%efbfbd}" != "$hex" ]; do hex=${hex%efbfbd}; done
    if [ ${#hex} -gt $shown ]; then printf '%s' "${hex:$shown}" | xxd -r -p; shown=${#hex}; fi
done < <("$B/generate" "$M/config.json" "$M/generation_config.json" "$ids" "$max" \
             "${mode[@]}" --seed "$seed" "${shards[@]}")

# 最后把剩下的（包括真正的 U+FFFD）都打印出来
if [ ${#out[@]} -gt 0 ]; then
    echo "${out[*]}" > "$tmp/ids.txt"
    hex=$("$B/tok" decode "$M/tokenizer.json" "$tmp/ids.txt" | cut -d' ' -f2)
    printf '%s' "${hex:$shown}" | xxd -r -p
fi
echo
case $stop in
    eos) why="模型自己停下" ;; length) why="达到 -n $max 的上限" ;; *) why="generate 异常退出"; ;;
esac
echo "（生成 ${#out[@]} 个 token，用时 $(( $(date +%s) - start )) 秒，$why）" >&2
[ -n "$stop" ]
