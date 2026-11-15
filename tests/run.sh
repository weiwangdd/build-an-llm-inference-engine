#!/bin/sh
# 用法：tests/run.sh <build 目录>
# 1) 合法文件：C 读取器与 Python 参考实现的输出必须逐字节一致。
# 2) 损坏文件：C 读取器必须以退出码 1 拒绝（不是崩溃，也不是成功）。
set -u
B=$1; FX=$B/fixtures; fail=0
python3 tests/make_fixtures.py "$FX" || exit 1

for f in "$FX"/good*.safetensors; do
    python3 tests/ref_dump.py "$f" > "$B/ref.txt" || { echo "FAIL 参考实现出错 $f"; fail=1; continue; }
    "$B/st_dump" "$f" > "$B/got.txt"; rc=$?
    if [ $rc -eq 0 ] && cmp -s "$B/ref.txt" "$B/got.txt"; then echo "ok   $(basename "$f")"
    else echo "FAIL $(basename "$f") (rc=$rc)"; diff "$B/ref.txt" "$B/got.txt" | head -5; fail=1; fi
done

for f in "$FX"/bad_*.safetensors; do
    msg=$("$B/st_dump" "$f" 2>&1 >/dev/null); rc=$?
    if [ $rc -eq 1 ]; then echo "ok   $(basename "$f")：$msg"
    else echo "FAIL $(basename "$f") 应被拒绝，实际 rc=$rc $msg"; fail=1; fi
done
exit $fail
