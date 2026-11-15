CC      ?= cc
CFLAGS  ?= -std=c17 -O1 -g -Wall -Wextra -Wpedantic -Wconversion -Werror -Isrc
SAN     ?= -fsanitize=address,undefined -fno-sanitize-recover=all
BUILD   ?= build

all: $(BUILD)/st_dump

$(BUILD)/st_dump: tools/st_dump.c src/st.c src/st.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(SAN) -o $@ tools/st_dump.c src/st.c

test: $(BUILD)/st_dump
	sh tests/run.sh $(BUILD)

clean:
	rm -rf $(BUILD)

.PHONY: all test clean
