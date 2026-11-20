CC      ?= cc
CFLAGS  ?= -std=c17 -O1 -g -Wall -Wextra -Wpedantic -Wconversion -Werror
SAN     ?= -fsanitize=address,undefined -fno-sanitize-recover=all
BUILD   ?= build
PYTHON  ?= python3

# 第 7 章：GPU=1 时（make gpu 会这样调用自己）构建 GPU 版，模型交给 gpu_model.c 和 gpu_ops.cu；
# 否则链接 gpu_none.c，GPU 的函数一律报错。
CUDA      ?= /usr/local/cuda
NVCC      ?= $(CUDA)/bin/nvcc
CUDA_ARCH ?= sm_120
# --fmad=false：不把 a * b + c 合并成一条 FMA 指令（只舍入一次），保持与 CPU 相同的舍入
NVFLAGS   ?= -std=c++17 -O2 -g -arch=$(CUDA_ARCH) --fmad=false -Werror all-warnings -Xcompiler -Wall,-Wextra
GPU       ?= 0

# 引擎的源代码按章节分目录，每个文件放在它第一次出现的那一章，后面的章节在原文件上扩展。
# 第 5 章没有新文件：整模前向写在第 4 章的 model.c 里。
CH1 = src/01-weights
CH2 = src/02-tokenizer
CH3 = src/03-ops
CH4 = src/04-layer
CH6 = src/06-generate
CH7 = src/07-gpu
CH8 = src/08-serve
# 头文件用 #include "st.h" 这样的短名字，所以每章的目录都加进 -I。与 CFLAGS 分开，覆盖 CFLAGS 时不用管它。
INC = $(addprefix -I,$(CH1) $(CH2) $(CH3) $(CH4) $(CH6) $(CH7) $(CH8))

ifeq ($(GPU),1)
BACKEND  = $(CH7)/gpu_model.c
GPU_OBJ  = $(BUILD)/gpu_ops.o
DEFS     = -DUSE_GPU
LIBS     = -L$(CUDA)/lib64 -Wl,-rpath,$(CUDA)/lib64 -lcudart -lstdc++
else
BACKEND  = $(CH7)/gpu_none.c
GPU_OBJ  =
DEFS     =
LIBS     =
endif

TOK_SRC = $(CH2)/tok.c $(CH2)/json.c $(CH2)/unicode.c $(CH2)/unicode_tables.c $(CH2)/chat.c $(CH2)/sha256.c

all: $(BUILD)/st_dump $(BUILD)/tok $(BUILD)/ops $(BUILD)/layer $(BUILD)/forward $(BUILD)/generate $(BUILD)/batch $(BUILD)/serve

$(BUILD)/st_dump: tools/st_dump.c $(CH1)/st.c $(CH1)/st.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INC) $(SAN) -o $@ tools/st_dump.c $(CH1)/st.c

$(BUILD)/tok: tools/tok.c $(TOK_SRC) $(CH2)/tok.h $(CH2)/json.h $(CH2)/unicode.h $(CH2)/unicode_tables.h $(CH2)/chat.h $(CH2)/sha256.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INC) $(SAN) -o $@ tools/tok.c $(TOK_SRC)

$(BUILD)/ops: tools/ops.c $(CH3)/ops.c $(CH3)/ops.h $(CH1)/st.c $(CH1)/st.h $(CH7)/gpu_ops.h $(GPU_OBJ)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INC) $(DEFS) $(SAN) -o $@ tools/ops.c $(CH3)/ops.c $(CH1)/st.c $(GPU_OBJ) -lm $(LIBS)

MODEL_SRC = $(CH4)/model.c $(CH3)/ops.c $(CH1)/st.c $(CH2)/json.c $(BACKEND)
MODEL_HDR = $(CH4)/model.h $(CH3)/ops.h $(CH1)/st.h $(CH2)/json.h $(CH7)/gpu.h $(CH7)/gpu_ops.h

$(BUILD)/layer: tools/layer.c $(MODEL_SRC) $(MODEL_HDR) $(GPU_OBJ)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INC) $(DEFS) $(SAN) -o $@ tools/layer.c $(MODEL_SRC) $(GPU_OBJ) -lm $(LIBS)

$(BUILD)/forward: tools/forward.c $(MODEL_SRC) $(MODEL_HDR) $(GPU_OBJ)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INC) $(DEFS) $(SAN) -o $@ tools/forward.c $(MODEL_SRC) $(GPU_OBJ) -lm $(LIBS)

GEN_SRC = $(MODEL_SRC) $(CH6)/gen.c $(CH6)/sample.c
GEN_HDR = $(MODEL_HDR) $(CH6)/gen.h $(CH6)/sample.h

$(BUILD)/generate: tools/generate.c $(GEN_SRC) $(GEN_HDR) $(GPU_OBJ)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INC) $(DEFS) $(SAN) -o $@ tools/generate.c $(GEN_SRC) $(GPU_OBJ) -lm $(LIBS)

# 第 8 章：连续批处理
$(BUILD)/batch: tools/batch.c $(GEN_SRC) $(CH8)/engine.c $(GEN_HDR) $(CH8)/engine.h $(GPU_OBJ)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INC) $(DEFS) $(SAN) -o $@ tools/batch.c $(GEN_SRC) $(CH8)/engine.c $(GPU_OBJ) -lm $(LIBS)

# json.c 在 GEN_SRC 和 TOK_SRC 里都有，sort 顺便去掉重复
SERVE_SRC = $(GEN_SRC) $(CH8)/engine.c $(CH8)/http.c $(CH8)/api.c $(CH8)/server.c $(TOK_SRC)
SERVE_HDR = $(GEN_HDR) $(CH8)/engine.h $(CH8)/http.h $(CH8)/api.h $(CH8)/server.h $(CH2)/tok.h $(CH2)/chat.h

$(BUILD)/serve: tools/serve.c $(SERVE_SRC) $(SERVE_HDR) $(GPU_OBJ)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INC) $(DEFS) $(SAN) -o $@ tools/serve.c $(sort $(SERVE_SRC)) $(GPU_OBJ) -lm $(LIBS)

# 第 7 章：GPU 内核由 nvcc 编译成目标文件，再和其余的 C 代码链接在一起
$(BUILD)/gpu_ops.o: $(CH7)/gpu_ops.cu $(CH7)/gpu_ops.h
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) -c -o $@ $(CH7)/gpu_ops.cu

# 第 1 到 8 章的快速测试（第 7 章的 GPU 版另见 test-gpu）
test: $(BUILD)/st_dump $(BUILD)/tok $(BUILD)/ops $(BUILD)/layer $(BUILD)/forward $(BUILD)/generate $(BUILD)/batch $(BUILD)/serve
	sh tests/01-weights/run.sh $(BUILD)
	sh tests/02-tokenizer/run_tok.sh $(BUILD)
	PYTHON=$(PYTHON) sh tests/03-ops/run_ops.sh $(BUILD)
	PYTHON=$(PYTHON) sh tests/04-layer/run_layer.sh $(BUILD)
	PYTHON=$(PYTHON) sh tests/05-model/run_model.sh $(BUILD)
	PYTHON=$(PYTHON) sh tests/06-generate/run_gen.sh $(BUILD)
	PYTHON=$(PYTHON) sh tests/08-serve/run_batch.sh $(BUILD)
	PYTHON=$(PYTHON) sh tests/08-serve/run_serve.sh $(BUILD)

# 第 3 章：拿真实模型的权重当算子输入，例如
#   make test-ops-real MODEL=/data/models/Qwen3.8-27B
test-ops-real: $(BUILD)/ops
	PYTHON=$(PYTHON) sh tests/03-ops/run_ops.sh $(BUILD) real $(MODEL)

# 第 4 章：在真实模型上做单层对拍，例如
#   make test-layer-real MODEL=/data/models/Qwen3.8-27B PYTHON=$HOME/venvs/x/bin/python
# 第二层对拍需要 PYTHON 里装了 torch 和 transformers，没装就只做第一层。
test-layer-real: $(BUILD)/layer $(BUILD)/tok
	PYTHON=$(PYTHON) sh tests/04-layer/run_layer.sh $(BUILD) real $(MODEL)

# 第 5 章：在真实模型上做整模前向和贪心解码的对拍，例如
#   make test-model-real MODEL=/data/models/Qwen3.8-27B PYTHON=$HOME/venvs/x/bin/python
# 27B 每个 token 要算约 260 亿次乘加，开着 ASan 太慢，所以这里另外用 -O2、不带 ASan 构建一份到
# $(BUILD)/fast，代码完全相同。ASan 的检查由 make test 在小模型上做，走的是同一份代码。
FAST = $(BUILD)/fast
test-model-real: $(BUILD)/tok
	$(MAKE) BUILD=$(FAST) SAN= CFLAGS="$(subst -O1,-O2,$(CFLAGS))" $(FAST)/forward
	PYTHON=$(PYTHON) FORWARD=$(FAST)/forward sh tests/05-model/run_model.sh $(BUILD) real $(MODEL)

# 第 6 章：在真实模型上做带缓存的生成（贪心和按 generation_config.json 采样），与不带缓存的前向逐位对拍，例如
#   make test-gen-real MODEL=/data/models/Qwen3.8-27B PYTHON=$HOME/venvs/x/bin/python
# 和第 5 章一样用 -O2、不带 ASan 另建一份。
test-gen-real: $(BUILD)/tok
	$(MAKE) BUILD=$(FAST) SAN= CFLAGS="$(subst -O1,-O2,$(CFLAGS))" $(FAST)/forward $(FAST)/generate
	PYTHON=$(PYTHON) FORWARD=$(FAST)/forward GENERATE=$(FAST)/generate sh tests/06-generate/run_gen.sh $(BUILD) real $(MODEL)

# ---------- 第 7 章：单卡 GPU ----------
# GPU 版放在 $(BUILD)/gpu，工具的名字、用法、输出与 CPU 版完全相同，只是在 GPU 上算。
# 第 3 到 6 章的测试脚本原样拿 GPU 版再跑一遍，再加上 GPU 与 CPU 的逐层比较（tests/07-gpu/run_gpu.sh）。
# CUDA 的运行库与 ASan 默认的设置冲突（ASan 占住了一段 CUDA 要用的地址），要关掉 protect_shadow_gap。
GPUB    = $(BUILD)/gpu
GPU_ENV = ASAN_OPTIONS=protect_shadow_gap=0:detect_leaks=0
gpu:
	$(MAKE) BUILD=$(GPUB) GPU=1 all

test-gpu: all gpu
	$(GPU_ENV) PYTHON=$(PYTHON) sh tests/03-ops/run_ops.sh $(GPUB)
	$(GPU_ENV) PYTHON=$(PYTHON) sh tests/04-layer/run_layer.sh $(GPUB)
	$(GPU_ENV) PYTHON=$(PYTHON) sh tests/05-model/run_model.sh $(GPUB)
	$(GPU_ENV) PYTHON=$(PYTHON) sh tests/06-generate/run_gen.sh $(GPUB)
	$(GPU_ENV) PYTHON=$(PYTHON) sh tests/08-serve/run_batch.sh $(GPUB)
	$(GPU_ENV) PYTHON=$(PYTHON) sh tests/08-serve/run_serve.sh $(GPUB)
	$(GPU_ENV) PYTHON=$(PYTHON) sh tests/07-gpu/run_gpu.sh $(BUILD) $(GPUB)

# 第 7 章：在真实模型上验收 GPU 版，例如
#   make test-gpu-real MODEL=/data/models/Qwen3.8-27B PYTHON=$HOME/venvs/x/bin/python
# 依次是：第 4 章的四层逐步对拍、第 5 章的整模贪心、第 6 章的缓存与采样（都换成 GPU 版），
# 最后 GPU 与 CPU 逐层比较。和第 5、6 章一样，跑整个模型的那几份用 -O2、不带 ASan 另建。
GPU_FAST = $(GPUB)/fast
test-gpu-real: $(BUILD)/tok
	$(MAKE) BUILD=$(GPUB) GPU=1 $(GPUB)/tok $(GPUB)/layer
	$(MAKE) BUILD=$(GPU_FAST) GPU=1 SAN= CFLAGS="$(subst -O1,-O2,$(CFLAGS))" $(GPU_FAST)/forward $(GPU_FAST)/generate
	$(MAKE) BUILD=$(FAST) SAN= CFLAGS="$(subst -O1,-O2,$(CFLAGS))" $(FAST)/forward
	$(GPU_ENV) PYTHON=$(PYTHON) sh tests/04-layer/run_layer.sh $(GPUB) real $(MODEL)
	PYTHON=$(PYTHON) FORWARD=$(GPU_FAST)/forward sh tests/05-model/run_model.sh $(GPUB) real $(MODEL)
	PYTHON=$(PYTHON) FORWARD=$(GPU_FAST)/forward GENERATE=$(GPU_FAST)/generate sh tests/06-generate/run_gen.sh $(GPUB) real $(MODEL)
	PYTHON=$(PYTHON) sh tests/07-gpu/run_gpu.sh $(FAST) $(GPU_FAST) real $(MODEL) $(BUILD)/tok

# 第 8 章：在真实模型上验收服务端（GPU 版），例如
#   make test-serve-real MODEL=/data/models/Qwen3.8-27B PYTHON=$HOME/venvs/x/bin/python
# 几个问题同时发给 serve，再逐个单独交给 generate，生成的 token 必须逐个相同。和第 7 章一样用 -O2、不带 ASan 另建。
test-serve-real:
	$(MAKE) BUILD=$(GPU_FAST) GPU=1 SAN= CFLAGS="$(subst -O1,-O2,$(CFLAGS))" $(GPU_FAST)/serve $(GPU_FAST)/generate $(GPU_FAST)/tok
	$(PYTHON) tests/08-serve/serve_real.py $(GPU_FAST) $(MODEL)

# 第 2 章：全部 Unicode 码点的穷举对拍，几分钟
test-unicode: $(BUILD)/tok
	sh tests/02-tokenizer/run_tok.sh $(BUILD) unicode

# 第 2 章：在真实模型的分词器上验收，例如
#   make test-real TOKENIZER=/data/models/Qwen3.8-27B/tokenizer.json PYTHON=$HOME/venvs/x/bin/python
test-real: $(BUILD)/tok
	PYTHON=$(PYTHON) sh tests/02-tokenizer/run_tok.sh $(BUILD) real $(TOKENIZER)

# 重新生成 Unicode 数据表。需要 Unicode 9.0 和 16.0 的 UnicodeData.txt 与
# DerivedNormalizationProps.txt，下载地址见 tools/gen_unicode.c。
UCD9  ?= ucd/9.0.0
UCD16 ?= ucd/16.0.0
unicode-tables: tools/gen_unicode.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INC) -o $(BUILD)/gen_unicode tools/gen_unicode.c
	$(BUILD)/gen_unicode $(UCD9) $(UCD16) > $(CH2)/unicode_tables.c

clean:
	rm -rf $(BUILD)

.PHONY: all gpu test test-gpu test-gpu-real test-serve-real test-unicode test-real test-ops-real test-layer-real test-model-real test-gen-real unicode-tables clean
