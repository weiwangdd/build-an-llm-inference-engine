/* batch — 第 8 章的命令行工具：把一组请求交给连续批处理的 engine，按到达时间陆续加入，一起生成。
 *
 *   batch <config.json> <generation_config.json> <请求清单> [--max-batch B] [--max-ctx C] [--dump]
 *         [--log 文件] <分片 ...>
 *
 * 请求清单每行一个请求：
 *   <到达步> <token id，逗号分隔> <max_new> [选项 ...]
 * 到达步是这个请求在 engine 的第几步之前加进去（从 0 数起）。选项与 generate 相同
 * （--greedy --sample --temperature --top-k --top-p --seed --chunk），另有
 *   --cancel-after K       生成 K 个 token 之后取消这个请求（模拟客户端断开）
 * 空行和 # 开头的行跳过。
 *
 * 输出：按清单的顺序，每个请求一段，先是一行 “req <i>”，之后与 generate 对这个请求的输出逐字节相同
 * （同样的选项，包括 --dump 时的 logits 和分布），被取消的最后一行是 “stop cancel”。
 * tests/run_batch.sh 拿它与 generate 逐个单独跑的结果比较。
 * --log 把每一步这一批的组成写进文件，每行 “step <s> <请求>:<token 数> ...”。
 *
 * --max-batch 默认 4，--max-ctx 默认 max_position_embeddings。 */
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "engine.h"
#include "gpu.h"
#include "model.h"
#include "st.h"

static void die(const char *msg) {
    fprintf(stderr, "错误：%s\n", msg);
    exit(1);
}

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

static uint64_t num_u(const char *s, uint64_t max, const char *what, size_t line) {
    uint64_t v;
    if (parse_u(s, s + strlen(s), max, &v)) {
        fprintf(stderr, "错误：请求清单第 %zu 行：%s 应为不超过 %" PRIu64 " 的非负整数\n", line, what, max);
        exit(1);
    }
    return v;
}

static double num_d(const char *s, const char *what, size_t line) {
    char *end;
    double v = strtod(s, &end);
    if (end == s || *end) { fprintf(stderr, "错误：请求清单第 %zu 行：%s 应为数\n", line, what); exit(1); }
    return v;
}

/* 每个请求的输出先攒在内存里，最后按清单顺序打印 */
typedef struct { char *s; size_t n, cap; } sbuf;

static void put(sbuf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int k = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (k < 0) die("格式化输出出错");
    if (b->n + (size_t)k + 1 > b->cap) {
        size_t nc = b->cap ? b->cap : 256;
        while (b->n + (size_t)k + 1 > nc) nc *= 2;
        char *ns = realloc(b->s, nc);
        if (!ns) die("内存不足");
        b->s = ns; b->cap = nc;
    }
    va_start(ap, fmt);
    vsnprintf(b->s + b->n, (size_t)k + 1, fmt, ap);
    va_end(ap);
    b->n += (size_t)k;
}

typedef struct {
    size_t   arrive;
    int32_t *ids;
    size_t   n;
    eng_params p;
    size_t   cancel_after;   /* 0 表示不取消 */
    uint64_t id;             /* engine 给的编号 */
    int      added, done, cancel_now;
    size_t   nout;
    sbuf     out;
} job;

static job *jobs;
static size_t njobs;
static int dump;

static job *find(uint64_t id) {
    for (size_t i = 0; i < njobs; i++)
        if (jobs[i].added && jobs[i].id == id) return &jobs[i];
    die("engine 报告了不认识的请求");
    return NULL;
}

static void on_token(void *ctx, uint64_t id, size_t k, const float *logits, size_t V, const int32_t *ids,
                     const double *probs, size_t nk, double u, int32_t tok) {
    (void)ctx;
    job *j = find(id);
    sbuf *b = &j->out;
    if (dump) {
        put(b, "step %zu\nlogits %zu\n", k, V);
        for (size_t i = 0; i < V; i++) put(b, "%a\n", (double)logits[i]);
        put(b, "dist %zu\n", nk);
        for (size_t i = 0; i < nk; i++) put(b, "%" PRId32 " %a\n", ids[i], probs[i]);
        if (u >= 0) put(b, "draw %a\n", u);
    }
    put(b, "token %" PRId32 "\n", tok);
    if (++j->nout == j->cancel_after) j->cancel_now = 1;
}

static void on_done(void *ctx, uint64_t id, eng_stop why, const char *msg) {
    (void)ctx;
    job *j = find(id);
    if (why == ENG_ERROR) { fprintf(stderr, "错误：请求 %" PRIu64 "：%s\n", id, msg); exit(1); }
    put(&j->out, "stop %s\n", why == ENG_EOS ? "eos" : "length");
    j->done = 1;
}

/* 解析请求清单的一行（会改写 line） */
static void parse_job(job *j, char *line, size_t ln, const gen_cfg *G) {
    char *tok[64];
    size_t nt = 0;
    for (char *p = strtok(line, " \t\r"); p; p = strtok(NULL, " \t\r")) {
        if (nt == 64) { fprintf(stderr, "错误：请求清单第 %zu 行太长\n", ln); exit(1); }
        tok[nt++] = p;
    }
    if (nt < 3) { fprintf(stderr, "错误：请求清单第 %zu 行应为 <到达步> <ids> <max_new> [选项]\n", ln); exit(1); }
    memset(j, 0, sizeof *j);
    j->arrive = (size_t)num_u(tok[0], 1u << 30, "到达步", ln);
    j->n = 1;
    for (const char *p = tok[1]; *p; p++) j->n += *p == ',';
    if (!(j->ids = malloc(j->n * sizeof *j->ids))) die("内存不足");
    const char *p = tok[1];
    for (size_t t = 0; t < j->n; t++) {
        const char *e = strchr(p, ',');
        if (!e) e = p + strlen(p);
        uint64_t v;
        if (parse_u(p, e, INT32_MAX, &v)) { fprintf(stderr, "错误：请求清单第 %zu 行：token id 应为逗号分隔的非负整数\n", ln); exit(1); }
        j->ids[t] = (int32_t)v;
        p = *e ? e + 1 : e;
    }
    j->p.max_new = (size_t)num_u(tok[2], 1000000, "max_new", ln);
    j->p.sample = G->sample;                     /* 默认取 generation_config.json，与 generate 相同 */
    for (size_t a = 3; a < nt; a++) {
        const char *o = tok[a];
        int needs = strcmp(o, "--greedy") && strcmp(o, "--sample");
        if (needs && a + 1 >= nt) { fprintf(stderr, "错误：请求清单第 %zu 行：选项 %s 缺少参数\n", ln, o); exit(1); }
        if (!strcmp(o, "--greedy")) j->p.sample.do_sample = 0;
        else if (!strcmp(o, "--sample")) j->p.sample.do_sample = 1;
        else if (!strcmp(o, "--temperature")) j->p.sample.temperature = num_d(tok[++a], "temperature", ln);
        else if (!strcmp(o, "--top-k")) j->p.sample.top_k = (size_t)num_u(tok[++a], 1u << 30, "top-k", ln);
        else if (!strcmp(o, "--top-p")) j->p.sample.top_p = num_d(tok[++a], "top-p", ln);
        else if (!strcmp(o, "--seed")) j->p.seed = num_u(tok[++a], UINT64_MAX, "seed", ln);
        else if (!strcmp(o, "--chunk")) {
            j->p.chunk = (size_t)num_u(tok[++a], 1u << 30, "chunk", ln);
            if (j->p.chunk == 0) { fprintf(stderr, "错误：请求清单第 %zu 行：chunk 至少是 1\n", ln); exit(1); }
        } else if (!strcmp(o, "--cancel-after")) {
            j->cancel_after = (size_t)num_u(tok[++a], 1000000, "cancel-after", ln);
            if (j->cancel_after == 0) { fprintf(stderr, "错误：请求清单第 %zu 行：cancel-after 至少是 1\n", ln); exit(1); }
        } else { fprintf(stderr, "错误：请求清单第 %zu 行：不认识的选项 %s\n", ln, o); exit(1); }
    }
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "用法：%s <config.json> <generation_config.json> <请求清单> [选项] <分片> ...\n", argv[0]);
        return 2;
    }
    char err[512];
    size_t max_batch = 4, max_ctx = 0;
    const char *logpath = NULL;
    int a = 4;
    for (; a < argc && !strncmp(argv[a], "--", 2); a++) {
        const char *o = argv[a];
        if (!strcmp(o, "--dump")) { dump = 1; continue; }
        if (a + 1 >= argc) { fprintf(stderr, "错误：选项 %s 缺少参数\n", o); exit(1); }
        if (!strcmp(o, "--max-batch")) max_batch = (size_t)num_u(argv[++a], 1u << 20, "max-batch", 0);
        else if (!strcmp(o, "--max-ctx")) max_ctx = (size_t)num_u(argv[++a], 1u << 30, "max-ctx", 0);
        else if (!strcmp(o, "--log")) logpath = argv[++a];
        else { fprintf(stderr, "错误：不认识的选项 %s\n", o); exit(1); }
    }
    if (a == argc) die("缺少权重分片");

    st_set S;
    if (st_set_open(&S, (const char *const *)(argv + a), (size_t)(argc - a), err, sizeof err)) die(err);
    model M;
    if (model_load(&M, argv[1], &S, err, sizeof err)) die(err);
#ifdef USE_GPU
    if (model_to_gpu(&M, err, sizeof err)) die(err);
#endif
    gen_cfg G;
    if (gen_cfg_load(&G, argv[2], M.cfg.vocab, err, sizeof err)) die(err);
    if (!max_ctx) max_ctx = M.cfg.max_pos;

    FILE *f = fopen(argv[3], "r");
    if (!f) die("打不开请求清单");
    char line[65536];
    size_t capj = 0;
    for (size_t ln = 1; fgets(line, sizeof line, f); ln++) {
        size_t len = strlen(line);
        if (len && line[len - 1] == '\n') line[--len] = 0;
        else if (!feof(f)) { fprintf(stderr, "错误：请求清单第 %zu 行太长\n", ln); exit(1); }
        if (!len || line[0] == '#') continue;
        if (njobs == capj) {
            capj = capj ? 2 * capj : 16;
            if (!(jobs = realloc(jobs, capj * sizeof *jobs))) die("内存不足");
        }
        parse_job(&jobs[njobs++], line, ln, &G);
    }
    fclose(f);
    if (!njobs) die("请求清单是空的");
    FILE *lg = logpath ? fopen(logpath, "w") : NULL;
    if (logpath && !lg) die("打不开日志文件");

    engine E;
    if (engine_init(&E, &M, &G, max_batch, max_ctx, on_token, on_done, NULL, err, sizeof err)) die(err);
    size_t left = njobs;
    for (size_t step = 0; left; step++) {
        for (size_t i = 0; i < njobs; i++)                 /* 到达步是 step 的请求按清单顺序加进去 */
            if (!jobs[i].added && jobs[i].arrive <= step) {
                if (engine_add(&E, jobs[i].ids, jobs[i].n, &jobs[i].p, &jobs[i].id, err, sizeof err)) {
                    fprintf(stderr, "错误：请求 %zu：%s\n", i, err);
                    exit(1);
                }
                jobs[i].added = 1;
            }
        if (engine_step(&E, err, sizeof err)) die(err);
        if (lg && E.last_nseq) {
            fprintf(lg, "step %zu", step);
            for (size_t b = 0; b < E.last_nseq; b++) {
                size_t r = 0;
                while (!(jobs[r].added && jobs[r].id == E.last_ids[b])) r++;
                fprintf(lg, " %zu:%zu", r, E.last_n[b]);
            }
            fprintf(lg, "\n");
        }
        for (size_t i = 0; i < njobs; i++) {
            job *j = &jobs[i];
            if (j->cancel_now && !j->done) {               /* 模拟客户端在这一步之后断开 */
                engine_cancel(&E, j->id);
                put(&j->out, "stop cancel\n");
                j->done = 1;
            }
        }
        left = 0;
        for (size_t i = 0; i < njobs; i++) left += !jobs[i].done;
    }
    for (size_t i = 0; i < njobs; i++) {
        printf("req %zu\n", i);
        if (jobs[i].out.n) fwrite(jobs[i].out.s, 1, jobs[i].out.n, stdout);
        free(jobs[i].out.s); free(jobs[i].ids);
    }
    if (fflush(stdout) || ferror(stdout)) die("写输出出错");
    if (lg && fclose(lg)) die("写日志出错");
    free(jobs);
    engine_free(&E);
    model_free(&M);
    st_set_close(&S);
    return 0;
}
