#!/bin/sh
# 用法：tests/08-serve/run_serve.sh <build 目录>
# 第 8 章：服务端的端到端测试（tests/08-serve/serve_test.py）。需要 tokenizers 库（第 2 章的测试也要它）。
# 环境变量 SERVE 可以换另一份可执行文件。
set -u
B=$1; FX=$B/serve_fixtures; PY=${PYTHON:-python3}
$PY -c "import tokenizers" 2>/dev/null || { echo "FAIL 需要 Python 的 tokenizers 库：pip install tokenizers"; exit 1; }
$PY tests/08-serve/make_serve_fixtures.py "$FX" || exit 1
$PY tests/08-serve/serve_test.py "$B" "$FX"
