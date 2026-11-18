/* sample.c — 第 6 章：采样。规则以 transformers 5.23.0 的 generation/logits_process.py 为准。 */
#include "sample.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

static int fail(char *err, size_t cap, const char *fmt, ...) {
    if (err && cap) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, cap, fmt, ap);
        va_end(ap);
    }
    return -1;
}

/* ---------- 随机数 ---------- */

void rng_seed(rng64 *r, uint64_t seed) { r->s = seed; }

uint64_t rng_next(rng64 *r) {
    uint64_t z = (r->s += 0x9e3779b97f4a7c15u);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9u;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebu;
    return z ^ (z >> 31);
}

double rng_uniform(rng64 *r) { return (double)(rng_next(r) >> 11) * 0x1p-53; }

/* ---------- 分布 ---------- */

int sample_cfg_check(const sample_cfg *sc, char *err, size_t cap) {
    if (!sc->do_sample) return 0;
    if (!(sc->temperature > 0) || !isfinite(sc->temperature)) return fail(err, cap, "temperature 应为正数");
    if (!(sc->top_p > 0 && sc->top_p <= 1)) return fail(err, cap, "top_p 应在 (0, 1] 之间");
    return 0;
}

/* 排序用：分数从高到低，同分时 id 小的在前 */
static const double *g_z;
static int by_score(const void *a, const void *b) {
    int32_t i = *(const int32_t *)a, j = *(const int32_t *)b;
    if (g_z[i] != g_z[j]) return g_z[i] > g_z[j] ? -1 : 1;
    return (i > j) - (i < j);
}

int sample_dist(const float *row, size_t vocab, const sample_cfg *sc, int32_t *ids, double *probs, size_t *n,
                char *err, size_t cap) {
    if (sample_cfg_check(sc, err, cap)) return -1;
    size_t best = 0;
    for (size_t i = 0; i < vocab; i++) {
        if (!isfinite(row[i])) return fail(err, cap, "logits 的第 %zu 个元素是 %s", i, isnan(row[i]) ? "NaN" : "无穷大");
        if (row[i] > row[best]) best = i;           /* 严格大于：同样大时保留 id 小的 */
    }
    if (!sc->do_sample) {                           /* 贪心：只有一个 token，概率为 1 */
        ids[0] = (int32_t)best; probs[0] = 1; *n = 1;
        return 0;
    }

    double *z = malloc(vocab * sizeof *z), *e = malloc(vocab * sizeof *e);
    int32_t *order = malloc(vocab * sizeof *order);
    unsigned char *keep = calloc(vocab, 1);
    if (!z || !e || !order || !keep) { free(z); free(e); free(order); free(keep); return fail(err, cap, "内存不足"); }

    /* 1. 温度：z = logits / T */
    for (size_t i = 0; i < vocab; i++) { z[i] = (double)row[i] / sc->temperature; order[i] = (int32_t)i; }
    g_z = z;
    qsort(order, vocab, sizeof *order, by_score);

    /* 2. top-k：分数不低于第 k 名的都保留（与 transformers 相同，同分的不拆开） */
    size_t K = vocab;
    if (sc->top_k > 0 && sc->top_k < vocab) {
        double th = z[order[sc->top_k - 1]];
        K = sc->top_k;
        while (K < vocab && z[order[K]] >= th) K++;
    }

    /* 3. top-p：在这 K 个里，从高到低，排在它前面的累计概率还不到 top_p 的就保留。
     * transformers 的写法是从低到高累加、去掉累计概率不超过 1 - top_p 的，数学上是一回事。 */
    double m = z[order[0]], tot = 0;
    for (size_t r = 0; r < K; r++) { e[order[r]] = exp(z[order[r]] - m); tot += e[order[r]]; }
    size_t P = K;
    if (sc->top_p < 1) {
        double above = 0;
        for (P = 0; P < K && above / tot < sc->top_p; P++) above += e[order[P]];
    }

    /* 4. 剩下的 P 个重新归一化，按 id 从小到大输出 */
    double sum = 0;
    for (size_t r = 0; r < P; r++) { keep[order[r]] = 1; sum += e[order[r]]; }
    size_t k = 0;
    for (size_t i = 0; i < vocab; i++)
        if (keep[i]) { ids[k] = (int32_t)i; probs[k] = e[i] / sum; k++; }
    *n = k;
    free(z); free(e); free(order); free(keep);
    return 0;
}

int32_t sample_pick(const int32_t *ids, const double *probs, size_t n, double u) {
    double acc = 0;
    for (size_t i = 0; i < n; i++) {
        acc += probs[i];
        if (u < acc) return ids[i];
    }
    return ids[n - 1];
}
