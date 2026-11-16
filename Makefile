CC      ?= cc
CFLAGS  ?= -std=c17 -O1 -g -Wall -Wextra -Wpedantic -Wconversion -Werror -Isrc
SAN     ?= -fsanitize=address,undefined -fno-sanitize-recover=all
BUILD   ?= build
PYTHON  ?= python3

TOK_SRC = src/tok.c src/json.c src/unicode.c src/unicode_tables.c

all: $(BUILD)/st_dump $(BUILD)/tok

$(BUILD)/st_dump: tools/st_dump.c src/st.c src/st.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(SAN) -o $@ tools/st_dump.c src/st.c

$(BUILD)/tok: tools/tok.c $(TOK_SRC) src/tok.h src/json.h src/unicode.h src/unicode_tables.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(SAN) -o $@ tools/tok.c $(TOK_SRC)

# 第 1 章和第 2 章的快速测试
test: $(BUILD)/st_dump $(BUILD)/tok
	sh tests/run.sh $(BUILD)
	sh tests/run_tok.sh $(BUILD)

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

.PHONY: all test test-unicode test-real unicode-tables clean
