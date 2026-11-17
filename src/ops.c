/* ops.c — 第 3 章的算子。每个函数都是最直接的循环，内部用 double 累加。 */
#include "ops.h"

#include <math.h>
#include <stdio.h>
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
void op_softplus(float *y, const float *x, size_t n) {
    for (size_t i = 0; i < n; i++) {
        double v = (double)x[i];
        y[i] = (float)(fmax(v, 0.0) + log1p(exp(-fabs(v))));
    }
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
    for (size_t t = 0; t < n; t++) {
        for (size_t j = 0; j < c; j++) {
            double acc = 0;
            for (size_t i = 0; i < k; i++) {
                if (t + i < k - 1) continue;              /* 对应的 token 在序列开头之前 */
                size_t src = t + i - (k - 1);
                acc += (double)w[j * k + i] * (double)x[src * c + j];
            }
            y[t * c + j] = (float)acc;
        }
    }
}
