/* layer — 第 4 章的命令行工具：算模型的一层，把每一步的中间结果都打印出来，给测试用。
 *
 *   layer <config.json> <层号> <token id，逗号分隔> <分片.safetensors> [更多分片 ...]
 *
 * 流程：嵌入（token 的位置依次是 0, 1, 2, ...）→ 第 <层号> 层 → 最终归一化。
 * 最后一步本来在第 64 层之后才做，这里接在这一层后面，只是为了让它的权重也得到对拍。
 *
 * 输出：每一步一段，第一行是“名字 [行,列]”，之后每行一个元素，用 %a 格式。
 * tests/04-layer/layer_ref.py 和 tests/04-layer/layer_official.py 读这个输出。
 *
 * 第 7 章：make gpu 用 -DUSE_GPU 另建一份（build/gpu/layer），在 GPU 上算，用法和输出格式完全相同。 */
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu.h"
#include "model.h"
#include "ops.h"
#include "st.h"

static void die(const char *msg) {
    fprintf(stderr, "错误：%s\n", msg);
    exit(1);
}

static void print(void *ctx, const char *name, const float *x, size_t rows, size_t cols) {
    (void)ctx;
    printf("%s [%zu,%zu]\n", name, rows, cols);
    for (size_t i = 0; i < rows * cols; i++) printf("%a\n", (double)x[i]);
}

/* 解析一个非负十进制整数，不接受符号、空白和多余的字符 */
static int parse_u(const char *s, const char *end, int64_t *out) {
    if (s == end || end - s > 10) return -1;
    int64_t v = 0;
    for (; s < end; s++) {
        if (*s < '0' || *s > '9') return -1;
        v = v * 10 + (*s - '0');
    }
    *out = v;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "用法：%s <config.json> <层号> <id,id,...> <分片> ...\n", argv[0]); return 2; }
    char err[512];
    model_cfg cfg;
    if (model_cfg_load(&cfg, argv[1], err, sizeof err)) die(err);

    int64_t layer;
    if (parse_u(argv[2], argv[2] + strlen(argv[2]), &layer)) die("层号应为非负整数");

    size_t n = 1;
    for (const char *p = argv[3]; *p; p++) n += *p == ',';
    int32_t *ids = malloc(n * sizeof *ids), *pos = malloc(n * sizeof *pos);
    if (!ids || !pos) die("内存不足");
    const char *p = argv[3];
    for (size_t t = 0; t < n; t++) {
        const char *e = strchr(p, ',');
        if (!e) e = p + strlen(p);
        int64_t v;
        if (parse_u(p, e, &v) || v > INT32_MAX) die("token id 应为逗号分隔的非负整数");
        ids[t] = (int32_t)v;
        pos[t] = (int32_t)t;
        p = *e ? e + 1 : e;
    }

    st_set S;
    if (st_set_open(&S, (const char *const *)(argv + 4), (size_t)(argc - 4), err, sizeof err)) die(err);
    const st_tensor *emb = st_set_find(&S, "model.language_model.embed_tokens.weight");
    if (!emb) die("找不到嵌入表 model.language_model.embed_tokens.weight");
    if (emb->dtype != ST_BF16 || emb->ndim != 2 || emb->shape[0] != cfg.vocab || emb->shape[1] != cfg.hidden)
        die("嵌入表应为 BF16 [vocab_size, hidden_size]");

    layer_w w;
    if (layer_load(&w, &cfg, &S, (size_t)layer, err, sizeof err)) die(err);
    float *fnorm = final_norm_load(&cfg, &S, err, sizeof err);
    if (!fnorm) die(err);

    size_t H = cfg.hidden;
    float *h = malloc(n * H * sizeof *h), *y = malloc(n * H * sizeof *y);
    if (!h || !y) die("内存不足");
    if (op_embed(h, emb->data, cfg.vocab, H, ids, n, err, sizeof err)) die(err);
    print(NULL, "embed", h, n, H);
#ifdef USE_GPU
    /* 第 7 章：GPU 版只把这一层放到 GPU 上算，嵌入和最终归一化仍在 CPU 上（它们由 forward 的测试查） */
    if (gpu_layer_forward(h, n, pos, &cfg, &w, print, NULL, err, sizeof err)) die(err);
#else
    if (layer_forward(h, n, pos, &cfg, &w, print, NULL, err, sizeof err)) die(err);
#endif
    op_rmsnorm(y, h, fnorm, n, H, cfg.eps);
    print(NULL, "final_norm", y, n, H);
    if (fflush(stdout) || ferror(stdout)) die("写输出出错");

    free(h); free(y); free(fnorm); free(ids); free(pos);
    layer_free(&w);
    st_set_close(&S);
    return 0;
}
