#!/bin/sh
# 用法：tests/01-weights/run.sh <build 目录>
# 1) 合法文件：C 读取器与 Python 参考实现的输出必须逐字节一致。
# 2) 多个分片：合并后的结果必须与同样内容的单文件一致。
# 3) 损坏文件：C 读取器必须以退出码 1 拒绝（不是崩溃，也不是成功）。
set -u
B=$1; FX=$B/fixtures; fail=0
python3 tests/01-weights/make_fixtures.py "$FX" || exit 1

for f in "$FX"/good*.safetensors; do
    python3 tests/01-weights/ref_dump.py "$f" > "$B/ref.txt" || { echo "FAIL 参考实现出错 $f"; fail=1; continue; }
    "$B/st_dump" "$f" > "$B/got.txt"; rc=$?
    if [ $rc -eq 0 ] && cmp -s "$B/ref.txt" "$B/got.txt"; then echo "ok   $(basename "$f")"
    else echo "FAIL $(basename "$f") (rc=$rc)"; diff "$B/ref.txt" "$B/got.txt" | head -5; fail=1; fi
done

# CRC-32 的已知答案：不依赖 Python 那一侧，防止两边错成一样
if "$B/st_dump" "$FX/good.safetensors" | grep -qx 'cbf43926 U8 \[9\] crc_check'; then echo "ok   CRC-32 测试向量"
else echo "FAIL CRC-32 测试向量"; fail=1; fi

"$B/st_dump" "$FX/good.safetensors" > "$B/single.txt"
"$B/st_dump" "$FX/shard-2-of-2.safetensors" "$FX/shard-1-of-2.safetensors" > "$B/multi.txt"; rc=$?
python3 tests/01-weights/ref_dump.py "$FX"/shard-*.safetensors > "$B/multi_ref.txt"
if [ $rc -eq 0 ] && cmp -s "$B/single.txt" "$B/multi.txt" && cmp -s "$B/multi_ref.txt" "$B/multi.txt"
then echo "ok   两个分片合并"
else echo "FAIL 两个分片合并 (rc=$rc)"; fail=1; fi

msg=$("$B/st_dump" "$FX/good.safetensors" "$FX/shard-1-of-2.safetensors" 2>&1 >/dev/null); rc=$?
if [ $rc -eq 1 ]; then echo "ok   跨分片重名：$msg"
else echo "FAIL 跨分片重名应被拒绝，实际 rc=$rc $msg"; fail=1; fi

msg=$("$B/st_dump" "$FX/good.safetensors" "$FX/不存在.safetensors" 2>&1 >/dev/null); rc=$?
if [ $rc -eq 1 ]; then echo "ok   分片缺失：$msg"
else echo "FAIL 分片缺失应被拒绝，实际 rc=$rc $msg"; fail=1; fi

for f in "$FX"/bad_*.safetensors; do
    msg=$("$B/st_dump" "$f" 2>&1 >/dev/null); rc=$?
    if [ $rc -eq 1 ]; then echo "ok   $(basename "$f")：$msg"
    else echo "FAIL $(basename "$f") 应被拒绝，实际 rc=$rc $msg"; fail=1; fi
done
exit $fail
