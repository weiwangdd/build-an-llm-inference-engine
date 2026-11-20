/* forward — 第 5 章的命令行工具：整个模型的前向和最朴素的贪心解码，给测试用。
 *
 *   forward <config.json> <token id，逗号分隔> <生成个数 m> <all|layers> <分片.safetensors> [更多分片 ...]
 *
 * m = 0：对给定的 token 做一次前向。
 * m > 0：贪心解码 m 个 token。每一步都从头对“提示词加上已经生成的 token”做一次完整的前向，
 *        取最后一行 logits 里分数最大的 token 接在后面（没有缓存，缓存是第 6 章的事）。
 *        一共做 m 次前向，最后一次的长度是 n + m - 1。
 *
 * 输出：最后一次前向的中间结果，格式同第 4 章的 layer 工具（“名字 [行,列]”，之后每行一个 %a）。
 *   all     每一层的每一步都打印，名字是 layers.<i>.<步骤名>
 *   layers  只打印 embed、每一层的输出 layers.<i>.out、final_norm 和 logits（真实模型用，否则输出太多）
 * m > 0 时最后再打印一段 “greedy [1,m]”，每行一个生成的 token id（十进制）。
 * tests/05-model/model_ref.py 和 tests/05-model/model_official.py 读这个输出。
 *
 * 第 7 章：make gpu 用 -DUSE_GPU 另建一份（build/gpu/forward），在 GPU 上算，用法和输出格式完全相同。 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu.h"
#include "model.h"
#include "st.h"

static void die(const char *msg) {
    fprintf(stderr, "错误：%s\n", msg);
    exit(1);
}

static int only_layers;   /* 1 表示只打印每一层的输出 */

static void print(void *ctx, const char *name, const float *x, size_t rows, size_t cols) {
    (void)ctx;
    if (only_layers && !strncmp(name, "layers.", 7)) {
        const char *dot = strrchr(name, '.');
        if (strcmp(dot, ".out")) return;
    }
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
    if (argc < 6) {
        fprintf(stderr, "用法：%s <config.json> <id,id,...> <生成个数> <all|layers> <分片> ...\n", argv[0]);
        return 2;
    }
    char err[512];
    int64_t m;
    if (parse_u(argv[3], argv[3] + strlen(argv[3]), &m) || m > 100000) die("生成个数应为不超过 100000 的非负整数");
    if (!strcmp(argv[4], "layers")) only_layers = 1;
    else if (strcmp(argv[4], "all")) die("输出方式只能是 all 或 layers");

    size_t n = 1;
    for (const char *p = argv[2]; *p; p++) n += *p == ',';
    size_t total = n + (size_t)m;                    /* 生成完之后的长度 */
    int32_t *ids = malloc(total * sizeof *ids);
    if (!ids) die("内存不足");
    const char *p = argv[2];
    for (size_t t = 0; t < n; t++) {
        const char *e = strchr(p, ',');
        if (!e) e = p + strlen(p);
        int64_t v;
        if (parse_u(p, e, &v) || v > INT32_MAX) die("token id 应为逗号分隔的非负整数");
        ids[t] = (int32_t)v;
        p = *e ? e + 1 : e;
    }

    st_set S;
    if (st_set_open(&S, (const char *const *)(argv + 5), (size_t)(argc - 5), err, sizeof err)) die(err);
    model M;
    if (model_load(&M, argv[1], &S, err, sizeof err)) die(err);
#ifdef USE_GPU
    if (model_to_gpu(&M, err, sizeof err)) die(err);      /* 第 7 章：GPU 版 */
#endif

    size_t V = M.cfg.vocab, steps = m > 0 ? (size_t)m : 1;      /* 前向的次数 */
    float *logits = malloc((n + steps - 1) * V * sizeof *logits);
    if (!logits) die("内存不足");
    for (size_t s = 0; s < steps; s++) {
        size_t len = n + s;
        int last = s + 1 == steps;                  /* 只打印最后一次前向 */
        if (model_forward(&M, ids, len, logits, last ? print : NULL, NULL, err, sizeof err)) die(err);
        if (m > 0 && logits_argmax(logits + (len - 1) * V, V, &ids[len], err, sizeof err)) die(err);
    }
    if (m > 0) {
        printf("greedy [1,%" PRId64 "]\n", m);
        for (size_t t = n; t < total; t++) printf("%" PRId32 "\n", ids[t]);
    }
    if (fflush(stdout) || ferror(stdout)) die("写输出出错");

    free(logits); free(ids);
    model_free(&M);
    st_set_close(&S);
    return 0;
}
