/* ops.c — 第 3 章的算子。每个函数都是最直接的循环，内部用 double 累加。 */
#include "ops.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, sizeof f);   /* 不用指针强转，避免违反别名规则 */
    return f;
}

uint16_t bf16_at(const uint8_t *p, size_t i) {
    return (uint16_t)(p[2 * i] | p[2 * i + 1] << 8);
}

void op_matmul(float *y, const float *x, const uint8_t *w, size_t n, size_t in, size_t out) {
    for (size_t t = 0; t < n; t++) {
        const float *xt = x + t * in;
        for (size_t o = 0; o < out; o++) {
            const uint8_t *row = w + 2 * o * in;
            double acc = 0;
            for (size_t i = 0; i < in; i++) acc += (double)xt[i] * (double)bf16_to_f32(bf16_at(row, i));
            y[t * out + o] = (float)acc;
        }
    }
}

int op_embed(float *y, const uint8_t *table, size_t rows, size_t dim,
             const int32_t *ids, size_t n, char *err, size_t errcap) {
    for (size_t t = 0; t < n; t++) {
        if (ids[t] < 0 || (uint64_t)ids[t] >= rows) {
            if (err && errcap) snprintf(err, errcap, "第 %zu 个 token id %d 超出嵌入表范围 [0, %zu)", t, ids[t], rows);
            return -1;
        }
        const uint8_t *row = table + 2 * (size_t)ids[t] * dim;
        for (size_t i = 0; i < dim; i++) y[t * dim + i] = bf16_to_f32(bf16_at(row, i));
    }
    return 0;
}

/* 一行的 1/sqrt(sum(x^2) / div + eps) */
static double inv_rms(const float *x, size_t d, double div, double eps) {
    double ss = 0;
    for (size_t i = 0; i < d; i++) ss += (double)x[i] * (double)x[i];
    return 1.0 / sqrt(ss / div + eps);
}

void op_rmsnorm(float *y, const float *x, const float *w, size_t n, size_t d, double eps) {
    for (size_t t = 0; t < n; t++) {
        const float *xt = x + t * d;
        double r = inv_rms(xt, d, (double)d, eps);
        for (size_t i = 0; i < d; i++) y[t * d + i] = (float)((double)xt[i] * r * (1.0 + (double)w[i]));
    }
}

static double silu(double x) { return x / (1.0 + exp(-x)); }

void op_rmsnorm_gated(float *y, const float *x, const float *z, const float *w,
                      size_t n, size_t d, double eps) {
    for (size_t t = 0; t < n; t++) {
        const float *xt = x + t * d, *zt = z + t * d;
        double r = inv_rms(xt, d, (double)d, eps);
        for (size_t i = 0; i < d; i++)
            y[t * d + i] = (float)((double)xt[i] * r * (double)w[i] * silu((double)zt[i]));
    }
}

void op_l2norm(float *y, const float *x, size_t n, size_t d, double eps) {
    for (size_t t = 0; t < n; t++) {
        const float *xt = x + t * d;
        double r = inv_rms(xt, d, 1.0, eps);
        for (size_t i = 0; i < d; i++) y[t * d + i] = (float)((double)xt[i] * r);
    }
}

void op_silu(float *y, const float *x, size_t n) {
    for (size_t i = 0; i < n; i++) y[i] = (float)silu((double)x[i]);
}

void op_sigmoid(float *y, const float *x, size_t n) {
    for (size_t i = 0; i < n; i++) y[i] = (float)(1.0 / (1.0 + exp(-(double)x[i])));
}

/* ln(1 + e^x) 直接算，x 稍大 e^x 就溢出。改写成 max(x, 0) + ln(1 + e^-|x|)，两边数学上相等。 */
static double softplus(double v) { return fmax(v, 0.0) + log1p(exp(-fabs(v))); }

void op_softplus(float *y, const float *x, size_t n) {
    for (size_t i = 0; i < n; i++) y[i] = (float)softplus((double)x[i]);
}

void op_softmax(float *y, const float *x, size_t n, size_t d) {
    for (size_t t = 0; t < n; t++) {
        const float *xt = x + t * d;
        double m = -INFINITY, s = 0;
        for (size_t i = 0; i < d; i++) if ((double)xt[i] > m) m = (double)xt[i];
        for (size_t i = 0; i < d; i++) s += exp((double)xt[i] - m);
        for (size_t i = 0; i < d; i++) y[t * d + i] = (float)(exp((double)xt[i] - m) / s);
    }
}

void op_rope(float *y, const float *x, const int32_t *pos, size_t n, size_t heads,
             size_t head_dim, size_t rot, double theta) {
    size_t half = rot / 2;
    for (size_t t = 0; t < n; t++) {
        for (size_t h = 0; h < heads; h++) {
            const float *xh = x + (t * heads + h) * head_dim;
            float *yh = y + (t * heads + h) * head_dim;
            for (size_t i = 0; i < half; i++) {
                double f = (double)pos[t] * pow(theta, -(double)(2 * i) / (double)rot);
                double c = cos(f), s = sin(f), a = (double)xh[i], b = (double)xh[i + half];
                yh[i] = (float)(a * c - b * s);
                yh[i + half] = (float)(b * c + a * s);
            }
            for (size_t i = rot; i < head_dim; i++) yh[i] = xh[i];
        }
    }
}

void op_conv1d(float *y, const float *x, const float *w, size_t n, size_t c, size_t k) {
    op_conv1d_hist(y, x, NULL, 0, w, n, c, k);
}

/* 第 6 章：前面接上 nh 行历史。把 hist 和 x 看成一条 nh + n 行的序列，只算后 n 行的输出。
 * 求和的项和次序与“把整条序列一次交给 op_conv1d”完全相同，所以结果逐位相同。 */
void op_conv1d_hist(float *y, const float *x, const float *hist, size_t nh, const float *w,
                    size_t n, size_t c, size_t k) {
    for (size_t t = 0; t < n; t++) {
        for (size_t j = 0; j < c; j++) {
            double acc = 0;
            for (size_t i = 0; i < k; i++) {
                if (nh + t + i < k - 1) continue;         /* 对应的 token 在序列开头之前 */
                size_t src = nh + t + i - (k - 1);        /* 在“历史 + 新 token”里的行号 */
                const float *row = src < nh ? hist + src * c : x + (src - nh) * c;
                acc += (double)w[j * k + i] * (double)row[j];
            }
            y[t * c + j] = (float)acc;
        }
    }
}

/* ---------- 第 4 章 ---------- */

void op_add(float *y, const float *a, const float *b, size_t n) {
    for (size_t i = 0; i < n; i++) y[i] = a[i] + b[i];
}

void op_mul(float *y, const float *a, const float *b, size_t n) {
    for (size_t i = 0; i < n; i++) y[i] = a[i] * b[i];
}

void op_decay(float *g, const float *a, const float *A_log, const float *dt_bias, size_t n, size_t heads) {
    for (size_t t = 0; t < n; t++)
        for (size_t h = 0; h < heads; h++)
            g[t * heads + h] = (float)(-exp((double)A_log[h]) * softplus((double)a[t * heads + h] + (double)dt_bias[h]));
}

static int oom(char *err, size_t errcap) {
    if (err && errcap) snprintf(err, errcap, "内存不足");
    return -1;
}

int op_attention(float *y, const float *q, const float *k, const float *v,
                 size_t n, size_t heads, size_t kv, size_t d, double scale, char *err, size_t errcap) {
    return op_attention_past(y, q, k, v, n, 0, heads, kv, d, scale, err, errcap);
}

int op_attention_past(float *y, const float *q, const float *k, const float *v, size_t n, size_t past,
                      size_t heads, size_t kv, size_t d, double scale, char *err, size_t errcap) {
    double *p = malloc((past + n) * sizeof *p), *acc = malloc(d * sizeof *acc);
    if (!p || !acc) { free(p); free(acc); return oom(err, errcap); }
    size_t group = heads / kv;
    for (size_t t = 0; t < n; t++) {
        size_t last = past + t;                             /* 第 t 个新 token 在整条序列里的位置 */
        for (size_t h = 0; h < heads; h++) {
            const float *qh = q + (t * heads + h) * d;
            size_t g = h / group;
            double m = -INFINITY, sum = 0;
            for (size_t s = 0; s <= last; s++) {            /* 因果：只看位置 0 .. last */
                const float *ks = k + (s * kv + g) * d;
                double dot = 0;
                for (size_t i = 0; i < d; i++) dot += (double)qh[i] * (double)ks[i];
                p[s] = dot * scale;
                if (p[s] > m) m = p[s];
            }
            for (size_t s = 0; s <= last; s++) { p[s] = exp(p[s] - m); sum += p[s]; }
            for (size_t i = 0; i < d; i++) acc[i] = 0;
            for (size_t s = 0; s <= last; s++) {
                const float *vs = v + (s * kv + g) * d;
                for (size_t i = 0; i < d; i++) acc[i] += p[s] * (double)vs[i];
            }
            for (size_t i = 0; i < d; i++) y[(t * heads + h) * d + i] = (float)(acc[i] / sum);
        }
    }
    free(p); free(acc);
    return 0;
}

int op_gated_delta(float *y, const float *q, const float *k, const float *v, const float *g, const float *beta,
                   size_t n, size_t hk, size_t hv, size_t dk, size_t dv, char *err, size_t errcap) {
    double *S = calloc(hv * dk * dv, sizeof *S);           /* 状态的初值为 0 */
    if (!S) return oom(err, errcap);
    int rc = op_gated_delta_state(y, q, k, v, g, beta, S, n, hk, hv, dk, dv, err, errcap);
    free(S);
    return rc;
}

int op_gated_delta_state(float *y, const float *q, const float *k, const float *v, const float *g,
                         const float *beta, double *state, size_t n, size_t hk, size_t hv, size_t dk, size_t dv,
                         char *err, size_t errcap) {
    double *u = malloc(dv * sizeof *u);
    if (!u) return oom(err, errcap);
    size_t group = hv / hk;
    double scale = 1.0 / sqrt((double)dk);
    for (size_t j = 0; j < hv; j++) {                       /* 各个值头互不相干，一个一个算 */
        size_t kh = j / group;
        double *S = state + j * dk * dv;                    /* 第 j 个值头的状态，接着上次的往下算 */
        for (size_t t = 0; t < n; t++) {
            const float *qt = q + (t * hk + kh) * dk, *kt = k + (t * hk + kh) * dk;
            const float *vt = v + (t * hv + j) * dv;
            double decay = exp((double)g[t * hv + j]), b = (double)beta[t * hv + j];
            for (size_t i = 0; i < dk * dv; i++) S[i] *= decay;
            for (size_t c = 0; c < dv; c++) {               /* u = beta (v - Sᵀk)，S 按 [dk][dv] 存放 */
                double pred = 0;
                for (size_t r = 0; r < dk; r++) pred += S[r * dv + c] * (double)kt[r];
                u[c] = b * ((double)vt[c] - pred);
            }
            for (size_t r = 0; r < dk; r++)
                for (size_t c = 0; c < dv; c++) S[r * dv + c] += (double)kt[r] * u[c];
            for (size_t c = 0; c < dv; c++) {
                double o = 0;
                for (size_t r = 0; r < dk; r++) o += S[r * dv + c] * (double)qt[r];
                y[(t * hv + j) * dv + c] = (float)(o * scale);
            }
        }
    }
    free(u);
    return 0;
}
