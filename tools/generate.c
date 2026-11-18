/* generate — 第 6 章的命令行工具：带缓存的生成循环（采样、停止条件）。
 *
 *   generate <config.json> <generation_config.json> <token id，逗号分隔> <max_new> [选项 ...] <分片 ...>
 *
 * 采样参数和结束符默认取 generation_config.json 里的，可以用选项覆盖：
 *   --greedy | --sample      贪心解码 / 采样（覆盖 do_sample）
 *   --temperature T  --top-k K  --top-p P
 *   --seed S                 随机数种子（非负整数），默认 0
 *   --chunk C                提示词每次最多交给模型 C 个 token，默认一次全给
 *   --no-cache               不用缓存，每一步都从头前向（第 5 章的做法）
 *   --dump                   打印每一步的 logits 和分布（测试用）
 *
 * 输出：每生成一个 token 打印一行 “token <id>”，最后一行 “stop eos” 或 “stop length”。
 * --dump 时，每个 token 之前还有：
 *   step <k>                 k 从 0 数起
 *   logits <V>               之后 V 行，选这个 token 时看的那一行 logits，%a 格式
 *   dist <K>                 之后 K 行 “<id> <概率 %a>”，采样的分布，按 id 从小到大
 *   draw <u %a>              这一步的随机数（只在采样时有）
 * tests/gen_ref.py 和 tests/gen_official.py 读这个输出。带不带缓存、提示词怎么分块，输出都应当逐字节相同。 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gen.h"
#include "model.h"
#include "st.h"

static void die(const char *msg) {
    fprintf(stderr, "错误：%s\n", msg);
    exit(1);
}

/* 解析一个非负十进制整数，不接受符号、空白和多余的字符 */
static int parse_u(const char *s, const char *end, uint64_t max, uint64_t *out) {
    if (s == end || end - s > 20) return -1;
    uint64_t v = 0;
    for (; s < end; s++) {
        if (*s < '0' || *s > '9') return -1;
        unsigned d = (unsigned)(*s - '0');
        if (v > (max - d) / 10) return -1;
        v = v * 10 + d;
    }
    *out = v;
    return 0;
}

static uint64_t num_u(const char *s, uint64_t max, const char *what) {
    uint64_t v;
    if (parse_u(s, s + strlen(s), max, &v)) {
        fprintf(stderr, "错误：%s 应为不超过 %" PRIu64 " 的非负整数\n", what, max);
        exit(1);
    }
    return v;
}

static double num_d(const char *s, const char *what) {
    char *end;
    double v = strtod(s, &end);
    if (end == s || *end) { fprintf(stderr, "错误：%s 应为数\n", what); exit(1); }
    return v;
}

static int dump;

static void on_token(void *ctx, size_t k, const float *logits, size_t V, const int32_t *ids, const double *probs,
                     size_t nk, double u, int32_t tok) {
    (void)ctx;
    if (dump) {
        printf("step %zu\nlogits %zu\n", k, V);
        for (size_t i = 0; i < V; i++) printf("%a\n", (double)logits[i]);
        printf("dist %zu\n", nk);
        for (size_t i = 0; i < nk; i++) printf("%" PRId32 " %a\n", ids[i], probs[i]);
        if (u >= 0) printf("draw %a\n", u);
    }
    printf("token %" PRId32 "\n", tok);
    fflush(stdout);                       /* 生成一个就输出一个 */
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "用法：%s <config.json> <generation_config.json> <id,id,...> <max_new> [选项] <分片> ...\n", argv[0]);
        return 2;
    }
    char err[512];
    gen_opts o = {0};
    o.max_new = (size_t)num_u(argv[4], 1000000, "max_new");
    if (o.max_new == 0) die("max_new 至少是 1");

    size_t n = 1;
    for (const char *p = argv[3]; *p; p++) n += *p == ',';
    int32_t *ids = malloc(n * sizeof *ids);
    if (!ids) die("内存不足");
    const char *p = argv[3];
    for (size_t t = 0; t < n; t++) {
        const char *e = strchr(p, ',');
        if (!e) e = p + strlen(p);
        uint64_t v;
        if (parse_u(p, e, INT32_MAX, &v)) die("token id 应为逗号分隔的非负整数");
        ids[t] = (int32_t)v;
        p = *e ? e + 1 : e;
    }

    /* 选项：先记下来，读完 generation_config.json 再覆盖 */
    int a = 5, greedy = -1, has_t = 0, has_k = 0, has_p = 0;
    double temp = 0, top_p = 0;
    size_t top_k = 0;
    for (; a < argc && !strncmp(argv[a], "--", 2); a++) {
        const char *opt = argv[a];
        int needs = strcmp(opt, "--greedy") && strcmp(opt, "--sample") && strcmp(opt, "--no-cache") && strcmp(opt, "--dump");
        if (needs && a + 1 >= argc) { fprintf(stderr, "错误：选项 %s 缺少参数\n", opt); exit(1); }
        if (!strcmp(opt, "--greedy")) greedy = 1;
        else if (!strcmp(opt, "--sample")) greedy = 0;
        else if (!strcmp(opt, "--no-cache")) o.no_cache = 1;
        else if (!strcmp(opt, "--dump")) dump = 1;
        else if (!strcmp(opt, "--temperature")) { temp = num_d(argv[++a], "temperature"); has_t = 1; }
        else if (!strcmp(opt, "--top-k")) { top_k = (size_t)num_u(argv[++a], 1u << 30, "top-k"); has_k = 1; }
        else if (!strcmp(opt, "--top-p")) { top_p = num_d(argv[++a], "top-p"); has_p = 1; }
        else if (!strcmp(opt, "--seed")) o.seed = num_u(argv[++a], UINT64_MAX, "seed");
        else if (!strcmp(opt, "--chunk")) {
            o.chunk = (size_t)num_u(argv[++a], 1u << 30, "chunk");
            if (o.chunk == 0) die("chunk 至少是 1");
        } else { fprintf(stderr, "错误：不认识的选项 %s\n", opt); exit(1); }
    }
    if (a == argc) die("缺少权重分片");

    st_set S;
    if (st_set_open(&S, (const char *const *)(argv + a), (size_t)(argc - a), err, sizeof err)) die(err);
    model M;
    if (model_load(&M, argv[1], &S, err, sizeof err)) die(err);
    gen_cfg G;
    if (gen_cfg_load(&G, argv[2], M.cfg.vocab, err, sizeof err)) die(err);
    if (greedy >= 0) G.sample.do_sample = !greedy;
    if (has_t) G.sample.temperature = temp;
    if (has_k) G.sample.top_k = top_k;
    if (has_p) G.sample.top_p = top_p;
    if (sample_cfg_check(&G.sample, err, sizeof err)) die(err);

    int32_t *out = malloc(o.max_new * sizeof *out);
    if (!out) die("内存不足");
    size_t nout;
    gen_stop why;
    if (generate(&M, &G, &o, ids, n, out, &nout, &why, on_token, NULL, err, sizeof err)) die(err);
    printf("stop %s\n", why == GEN_EOS ? "eos" : "length");
    if (fflush(stdout) || ferror(stdout)) die("写输出出错");

    free(out); free(ids);
    model_free(&M);
    st_set_close(&S);
    return 0;
}
