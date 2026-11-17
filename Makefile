CC      ?= cc
CFLAGS  ?= -std=c17 -O1 -g -Wall -Wextra -Wpedantic -Wconversion -Werror -Isrc
SAN     ?= -fsanitize=address,undefined -fno-sanitize-recover=all
BUILD   ?= build
PYTHON  ?= python3

TOK_SRC = src/tok.c src/json.c src/unicode.c src/unicode_tables.c src/chat.c src/sha256.c

all: $(BUILD)/st_dump $(BUILD)/tok $(BUILD)/ops $(BUILD)/layer

$(BUILD)/st_dump: tools/st_dump.c src/st.c src/st.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(SAN) -o $@ tools/st_dump.c src/st.c

$(BUILD)/tok: tools/tok.c $(TOK_SRC) src/tok.h src/json.h src/unicode.h src/unicode_tables.h src/chat.h src/sha256.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(SAN) -o $@ tools/tok.c $(TOK_SRC)

$(BUILD)/ops: tools/ops.c src/ops.c src/ops.h src/st.c src/st.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(SAN) -o $@ tools/ops.c src/ops.c src/st.c -lm

$(BUILD)/layer: tools/layer.c src/model.c src/model.h src/ops.c src/ops.h src/st.c src/st.h src/json.c src/json.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(SAN) -o $@ tools/layer.c src/model.c src/ops.c src/st.c src/json.c -lm

# 第 1 到 4 章的快速测试
test: $(BUILD)/st_dump $(BUILD)/tok $(BUILD)/ops $(BUILD)/layer
	sh tests/run.sh $(BUILD)
	sh tests/run_tok.sh $(BUILD)
	PYTHON=$(PYTHON) sh tests/run_ops.sh $(BUILD)
	PYTHON=$(PYTHON) sh tests/run_layer.sh $(BUILD)

# 第 3 章：拿真实模型的权重当算子输入，例如
#   make test-ops-real MODEL=/data/models/Qwen3.8-27B
test-ops-real: $(BUILD)/ops
	PYTHON=$(PYTHON) sh tests/run_ops.sh $(BUILD) real $(MODEL)

# 第 4 章：在真实模型上做单层对拍，例如
#   make test-layer-real MODEL=/data/models/Qwen3.8-27B PYTHON=$HOME/venvs/x/bin/python
# 第二层对拍需要 PYTHON 里装了 torch 和 transformers，没装就只做第一层。
test-layer-real: $(BUILD)/layer $(BUILD)/tok
	PYTHON=$(PYTHON) sh tests/run_layer.sh $(BUILD) real $(MODEL)

# 第 2 章：全部 Unicode 码点的穷举对拍，几分钟
test-unicode: $(BUILD)/tok
	sh tests/run_tok.sh $(BUILD) unicode

# 第 2 章：在真实模型的分词器上验收，例如
#   make test-real TOKENIZER=/data/models/Qwen3.8-27B/tokenizer.json PYTHON=$HOME/venvs/x/bin/python
test-real: $(BUILD)/tok
	PYTHON=$(PYTHON) sh tests/run_tok.sh $(BUILD) real $(TOKENIZER)

# 重新生成 Unicode 数据表。需要 Unicode 9.0 和 16.0 的 UnicodeData.txt 与
# DerivedNormalizationProps.txt，下载地址见 tools/gen_unicode.c。
UCD9  ?= ucd/9.0.0
UCD16 ?= ucd/16.0.0
unicode-tables: tools/gen_unicode.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -o $(BUILD)/gen_unicode tools/gen_unicode.c
	$(BUILD)/gen_unicode $(UCD9) $(UCD16) > src/unicode_tables.c

clean:
	rm -rf $(BUILD)

.PHONY: all test test-unicode test-real test-ops-real test-layer-real unicode-tables clean
