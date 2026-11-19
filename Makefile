CC      ?= cc
CFLAGS  ?= -std=c17 -O1 -g -Wall -Wextra -Wpedantic -Wconversion -Werror -Isrc
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
ifeq ($(GPU),1)
BACKEND  = src/gpu_model.c
GPU_OBJ  = $(BUILD)/gpu_ops.o
DEFS     = -DUSE_GPU
LIBS     = -L$(CUDA)/lib64 -Wl,-rpath,$(CUDA)/lib64 -lcudart -lstdc++
else
BACKEND  = src/gpu_none.c
GPU_OBJ  =
DEFS     =
LIBS     =
endif

TOK_SRC = src/tok.c src/json.c src/unicode.c src/unicode_tables.c src/chat.c src/sha256.c

all: $(BUILD)/st_dump $(BUILD)/tok $(BUILD)/ops $(BUILD)/layer $(BUILD)/forward $(BUILD)/generate

$(BUILD)/st_dump: tools/st_dump.c src/st.c src/st.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(SAN) -o $@ tools/st_dump.c src/st.c

$(BUILD)/tok: tools/tok.c $(TOK_SRC) src/tok.h src/json.h src/unicode.h src/unicode_tables.h src/chat.h src/sha256.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(SAN) -o $@ tools/tok.c $(TOK_SRC)

$(BUILD)/ops: tools/ops.c src/ops.c src/ops.h src/st.c src/st.h src/gpu_ops.h $(GPU_OBJ)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(DEFS) $(SAN) -o $@ tools/ops.c src/ops.c src/st.c $(GPU_OBJ) -lm $(LIBS)

MODEL_SRC = src/model.c src/ops.c src/st.c src/json.c $(BACKEND)
MODEL_HDR = src/model.h src/ops.h src/st.h src/json.h src/gpu.h src/gpu_ops.h

$(BUILD)/layer: tools/layer.c $(MODEL_SRC) $(MODEL_HDR) $(GPU_OBJ)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(DEFS) $(SAN) -o $@ tools/layer.c $(MODEL_SRC) $(GPU_OBJ) -lm $(LIBS)

$(BUILD)/forward: tools/forward.c $(MODEL_SRC) $(MODEL_HDR) $(GPU_OBJ)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(DEFS) $(SAN) -o $@ tools/forward.c $(MODEL_SRC) $(GPU_OBJ) -lm $(LIBS)

GEN_SRC = $(MODEL_SRC) src/gen.c src/sample.c
GEN_HDR = $(MODEL_HDR) src/gen.h src/sample.h

$(BUILD)/generate: tools/generate.c $(GEN_SRC) $(GEN_HDR) $(GPU_OBJ)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(DEFS) $(SAN) -o $@ tools/generate.c $(GEN_SRC) $(GPU_OBJ) -lm $(LIBS)

# 第 7 章：GPU 内核由 nvcc 编译成目标文件，再和其余的 C 代码链接在一起
$(BUILD)/gpu_ops.o: src/gpu_ops.cu src/gpu_ops.h
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) -c -o $@ src/gpu_ops.cu

# 第 1 到 6 章的快速测试
test: $(BUILD)/st_dump $(BUILD)/tok $(BUILD)/ops $(BUILD)/layer $(BUILD)/forward $(BUILD)/generate
	sh tests/run.sh $(BUILD)
	sh tests/run_tok.sh $(BUILD)
	PYTHON=$(PYTHON) sh tests/run_ops.sh $(BUILD)
	PYTHON=$(PYTHON) sh tests/run_layer.sh $(BUILD)
	PYTHON=$(PYTHON) sh tests/run_model.sh $(BUILD)
	PYTHON=$(PYTHON) sh tests/run_gen.sh $(BUILD)

# 第 3 章：拿真实模型的权重当算子输入，例如
#   make test-ops-real MODEL=/data/models/Qwen3.8-27B
test-ops-real: $(BUILD)/ops
	PYTHON=$(PYTHON) sh tests/run_ops.sh $(BUILD) real $(MODEL)

# 第 4 章：在真实模型上做单层对拍，例如
#   make test-layer-real MODEL=/data/models/Qwen3.8-27B PYTHON=$HOME/venvs/x/bin/python
# 第二层对拍需要 PYTHON 里装了 torch 和 transformers，没装就只做第一层。
test-layer-real: $(BUILD)/layer $(BUILD)/tok
	PYTHON=$(PYTHON) sh tests/run_layer.sh $(BUILD) real $(MODEL)

# 第 5 章：在真实模型上做整模前向和贪心解码的对拍，例如
#   make test-model-real MODEL=/data/models/Qwen3.8-27B PYTHON=$HOME/venvs/x/bin/python
# 27B 每个 token 要算约 260 亿次乘加，开着 ASan 太慢，所以这里另外用 -O2、不带 ASan 构建一份到
# $(BUILD)/fast，代码完全相同。ASan 的检查由 make test 在小模型上做，走的是同一份代码。
FAST = $(BUILD)/fast
test-model-real: $(BUILD)/tok
	$(MAKE) BUILD=$(FAST) SAN= CFLAGS="$(subst -O1,-O2,$(CFLAGS))" $(FAST)/forward
	PYTHON=$(PYTHON) FORWARD=$(FAST)/forward sh tests/run_model.sh $(BUILD) real $(MODEL)

# 第 6 章：在真实模型上做带缓存的生成（贪心和按 generation_config.json 采样），与不带缓存的前向逐位对拍，例如
#   make test-gen-real MODEL=/data/models/Qwen3.8-27B PYTHON=$HOME/venvs/x/bin/python
# 和第 5 章一样用 -O2、不带 ASan 另建一份。
test-gen-real: $(BUILD)/tok
	$(MAKE) BUILD=$(FAST) SAN= CFLAGS="$(subst -O1,-O2,$(CFLAGS))" $(FAST)/forward $(FAST)/generate
	PYTHON=$(PYTHON) FORWARD=$(FAST)/forward GENERATE=$(FAST)/generate sh tests/run_gen.sh $(BUILD) real $(MODEL)

# ---------- 第 7 章：单卡 GPU ----------
# GPU 版放在 $(BUILD)/gpu，工具的名字、用法、输出与 CPU 版完全相同，只是在 GPU 上算。
# 第 3 到 6 章的测试脚本原样拿 GPU 版再跑一遍，再加上 GPU 与 CPU 的逐层比较（tests/run_gpu.sh）。
# CUDA 的运行库与 ASan 默认的设置冲突（ASan 占住了一段 CUDA 要用的地址），要关掉 protect_shadow_gap。
GPUB    = $(BUILD)/gpu
GPU_ENV = ASAN_OPTIONS=protect_shadow_gap=0:detect_leaks=0
gpu:
	$(MAKE) BUILD=$(GPUB) GPU=1 all

test-gpu: all gpu
	$(GPU_ENV) PYTHON=$(PYTHON) sh tests/run_ops.sh $(GPUB)
	$(GPU_ENV) PYTHON=$(PYTHON) sh tests/run_layer.sh $(GPUB)
	$(GPU_ENV) PYTHON=$(PYTHON) sh tests/run_model.sh $(GPUB)
	$(GPU_ENV) PYTHON=$(PYTHON) sh tests/run_gen.sh $(GPUB)
	$(GPU_ENV) PYTHON=$(PYTHON) sh tests/run_gpu.sh $(BUILD) $(GPUB)

# 第 7 章：在真实模型上验收 GPU 版，例如
#   make test-gpu-real MODEL=/data/models/Qwen3.8-27B PYTHON=$HOME/venvs/x/bin/python
# 依次是：第 4 章的四层逐步对拍、第 5 章的整模贪心、第 6 章的缓存与采样（都换成 GPU 版），
# 最后 GPU 与 CPU 逐层比较。和第 5、6 章一样，跑整个模型的那几份用 -O2、不带 ASan 另建。
GPU_FAST = $(GPUB)/fast
test-gpu-real: $(BUILD)/tok
	$(MAKE) BUILD=$(GPUB) GPU=1 $(GPUB)/tok $(GPUB)/layer
	$(MAKE) BUILD=$(GPU_FAST) GPU=1 SAN= CFLAGS="$(subst -O1,-O2,$(CFLAGS))" $(GPU_FAST)/forward $(GPU_FAST)/generate
	$(MAKE) BUILD=$(FAST) SAN= CFLAGS="$(subst -O1,-O2,$(CFLAGS))" $(FAST)/forward
	$(GPU_ENV) PYTHON=$(PYTHON) sh tests/run_layer.sh $(GPUB) real $(MODEL)
	PYTHON=$(PYTHON) FORWARD=$(GPU_FAST)/forward sh tests/run_model.sh $(GPUB) real $(MODEL)
	PYTHON=$(PYTHON) FORWARD=$(GPU_FAST)/forward GENERATE=$(GPU_FAST)/generate sh tests/run_gen.sh $(GPUB) real $(MODEL)
	PYTHON=$(PYTHON) sh tests/run_gpu.sh $(FAST) $(GPU_FAST) real $(MODEL) $(BUILD)/tok

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

.PHONY: all gpu test test-gpu test-gpu-real test-unicode test-real test-ops-real test-layer-real test-model-real test-gen-real unicode-tables clean
