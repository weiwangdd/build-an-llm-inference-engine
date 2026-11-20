/* gpu_model.c — 第 7 章：显存里的模型、缓存和前向。
 *
 * 这个文件是普通的 C（和其余代码一样用 gcc 加全部警告编译），只负责安排：分配显存、拷贝、
 * 按顺序调用 gpu_ops.cu 里的内核。下面的 full_attention、linear_attention、layer_run、model_run
 * 与 model.c 里的同名函数一行一行对应，步骤名和次序完全相同，区别只有两点：
 *   - 缓冲区在显存里（dmem / gpu_free 代替 malloc / free），算子换成 gop_xxx；
 *   - 有 trace 时，要先把这一步的结果从显存拷回内存，再交给回调。 */
#include "gpu.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu_ops.h"

static int fail(char *err, size_t cap, const char *fmt, ...) {
    if (err && cap) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, cap, fmt, ap);
        va_end(ap);
    }
    return -1;
}

/* 分配一块显存，失败返回 NULL（原因写进 err） */
static void *dmem(size_t bytes, char *err, size_t cap) {
    void *p;
    return gpu_alloc(&p, bytes, err, cap) ? NULL : p;
}

/* 有 trace 时，把显存里的 [rows, cols] 拷回内存交给回调 */
static int trace_dev(trace_fn T, void *ctx, const char *name, const float *d, size_t rows, size_t cols,
                     char *err, size_t cap) {
    if (!T) return 0;
    size_t len = rows * cols;
    float *h = malloc((len ? len : 1) * sizeof *h);
    if (!h) return fail(err, cap, "内存不足");
    int rc = gpu_get(h, d, len * sizeof *h, err, cap);
    if (!rc) T(ctx, name, h, rows, cols);
    free(h);
    return rc;
}
#define TR(name, x, rows, cols) do { if (trace_dev(T, ctx, name, x, rows, cols, err, cap)) goto done; } while (0)

static void mm(float *y, const float *x, const bf16_mat *w, size_t n) { gop_matmul(y, x, w->data, n, w->cols, w->rows); }

/* ---------- 前向（与 model.c 对应） ---------- */

/* 第 8 章：一批序列，同 model.c */
static layer_cache *seq_cache(const model_seq *s, size_t layer) { return s->kc ? &s->kc->layers[layer] : NULL; }
static size_t seq_past(const model_seq *s) { return s->kc ? s->kc->len : 0; }

static int full_attention(float *out, const float *x, size_t N, const int32_t *pos, const model_seq *seqs,
                          size_t nseq, size_t layer, const model_cfg *c, const layer_w *w,
                          trace_fn T, void *ctx, char *err, size_t cap) {
    size_t hd = c->head_dim, nh = c->heads, nkv = c->kv_heads, qd = nh * hd, kd = nkv * hd, n = N;
    float *qg = dmem(n * 2 * qd * sizeof *qg, err, cap), *q = dmem(n * qd * sizeof *q, err, cap);
    float *gate = dmem(n * qd * sizeof *gate, err, cap);
    float *k = dmem(n * kd * sizeof *k, err, cap), *v = dmem(n * kd * sizeof *v, err, cap);
    float *t1 = dmem(n * qd * sizeof *t1, err, cap), *t2 = dmem(n * qd * sizeof *t2, err, cap);
    int rc = -1;
    if (!qg || !q || !gate || !k || !v || !t1 || !t2) goto done;

    mm(qg, x, &w->q, n);                     TR("q_proj", qg, n, 2 * qd);
    /* 每个头 2·hd 维，前 hd 维是 q，后 hd 维是门控：按“n·nh 行、每行取 hd 个、源行距 2·hd”拷两次 */
    gpu_copy2d(q, hd, qg, 2 * hd, hd, n * nh);
    gpu_copy2d(gate, hd, qg + hd, 2 * hd, hd, n * nh);
    mm(k, x, &w->k, n);                      TR("k_proj", k, n, kd);
    mm(v, x, &w->v, n);                      TR("v_proj", v, n, kd);
    gop_rmsnorm(t1, q, w->q_norm, n * nh, hd, c->eps);         TR("q_norm", t1, n, qd);
    gop_rope(q, t1, pos, n, nh, hd, c->rot, c->theta);         TR("q_rope", q, n, qd);
    gop_rmsnorm(t1, k, w->k_norm, n * nkv, hd, c->eps);        TR("k_norm", t1, n, kd);
    gop_rope(k, t1, pos, n, nkv, hd, c->rot, c->theta);        TR("k_rope", k, n, kd);
    for (size_t b = 0, off = 0; b < nseq; off += seqs[b++].n) {
        size_t nb = seqs[b].n, past = seq_past(&seqs[b]);
        layer_cache *lc = seq_cache(&seqs[b], layer);
        const float *kall = k + off * kd, *vall = v + off * kd;
        if (lc) {
            gpu_copy(lc->k + past * kd, k + off * kd, nb * kd * sizeof *k);
            gpu_copy(lc->v + past * kd, v + off * kd, nb * kd * sizeof *v);
            kall = lc->k; vall = lc->v;
        }
        if (gop_attention_past(t1 + off * qd, q + off * qd, kall, vall, nb, past, nh, nkv, hd,
                               1.0 / sqrt((double)hd), err, cap)) goto done;
    }
    TR("attn", t1, n, qd);
    gop_sigmoid(t2, gate, n * qd);                             TR("gate_sigmoid", t2, n, qd);
    gop_mul(q, t1, t2, n * qd);                                TR("gated", q, n, qd);
    mm(out, q, &w->o, n);
    rc = 0;
done:
    gpu_free(qg); gpu_free(q); gpu_free(gate); gpu_free(k); gpu_free(v); gpu_free(t1); gpu_free(t2);
    return rc;
}

static int linear_attention(float *out, const float *x, size_t N, const model_seq *seqs, size_t nseq, size_t layer,
                            const model_cfg *c, const layer_w *w, trace_fn T, void *ctx, char *err, size_t cap) {
    size_t hk = c->lk_heads, hv = c->lv_heads, dk = c->lk_dim, dv = c->lv_dim, n = N;
    size_t kd = hk * dk, vd = hv * dv, cd = 2 * kd + vd, kw = c->conv_k - 1;
    float *qkv = dmem(n * cd * sizeof *qkv, err, cap), *cv = dmem(n * cd * sizeof *cv, err, cap);
    float *z = dmem(n * vd * sizeof *z, err, cap), *a = dmem(n * hv * sizeof *a, err, cap);
    float *b = dmem(n * hv * sizeof *b, err, cap);
    float *q = dmem(n * kd * sizeof *q, err, cap), *k = dmem(n * kd * sizeof *k, err, cap);
    float *v = dmem(n * vd * sizeof *v, err, cap);
    float *g = dmem(n * hv * sizeof *g, err, cap), *beta = dmem(n * hv * sizeof *beta, err, cap);
    float *core = dmem(n * vd * sizeof *core, err, cap), *tmp = dmem((kw ? kw : 1) * cd * sizeof *tmp, err, cap);
    double *S0 = NULL;                        /* 不用缓存时，门控 delta 规则从全零的状态出发 */
    int rc = -1;
    if (!qkv || !cv || !z || !a || !b || !q || !k || !v || !g || !beta || !core || !tmp) goto done;

    mm(qkv, x, &w->qkv, n);                  TR("qkv", qkv, n, cd);
    mm(z, x, &w->z, n);                      TR("z", z, n, vd);
    mm(b, x, &w->b, n);                      TR("b", b, n, hv);
    mm(a, x, &w->a, n);                      TR("a", a, n, hv);
    for (size_t s = 0, off = 0; s < nseq; off += seqs[s++].n) {
        size_t nb = seqs[s].n, past = seq_past(&seqs[s]);
        layer_cache *lc = seq_cache(&seqs[s], layer);
        size_t nh = lc ? (past < kw ? past : kw) : 0;
        const float *xs = qkv + off * cd;
        gop_conv1d_hist(cv + off * cd, xs, lc ? lc->conv : NULL, nh, w->conv, nb, cd, c->conv_k);
        if (lc) {                             /* 把“历史 + 新 token”的最后 kw 行留作下一次的历史 */
            size_t total = nh + nb, keep = total < kw ? total : kw;
            for (size_t r = 0; r < keep; r++) {
                size_t src = total - keep + r;
                gpu_copy(tmp + r * cd, src < nh ? lc->conv + src * cd : xs + (src - nh) * cd, cd * sizeof *tmp);
            }
            gpu_copy(lc->conv, tmp, keep * cd * sizeof *tmp);
        }
    }
    TR("conv", cv, n, cd);
    gop_silu(qkv, cv, n * cd);                                 TR("conv_silu", qkv, n, cd);
    gpu_copy2d(cv, kd, qkv, cd, kd, n);                        /* cv 暂存 q */
    gpu_copy2d(cv + n * kd, kd, qkv + kd, cd, kd, n);          /* 和 k */
    gpu_copy2d(v, vd, qkv + 2 * kd, cd, vd, n);
    gop_l2norm(q, cv, n * hk, dk, 1e-6);                       TR("q_l2", q, n, kd);
    gop_l2norm(k, cv + n * kd, n * hk, dk, 1e-6);              TR("k_l2", k, n, kd);
    gop_sigmoid(beta, b, n * hv);                              TR("beta", beta, n, hv);
    gop_decay(g, a, w->A_log, w->dt_bias, n, hv);              TR("g", g, n, hv);
    for (size_t s = 0, off = 0; s < nseq; off += seqs[s++].n) {
        layer_cache *lc = seq_cache(&seqs[s], layer);
        if (!lc) {
            if (!S0 && !(S0 = dmem(hv * dk * dv * sizeof *S0, err, cap))) goto done;
            gpu_zero(S0, hv * dk * dv * sizeof *S0);
        }
        gop_gated_delta_state(core + off * vd, q + off * kd, k + off * kd, v + off * vd, g + off * hv,
                              beta + off * hv, lc ? lc->S : S0, seqs[s].n, hk, hv, dk, dv);
    }
    TR("delta", core, n, vd);
    gop_rmsnorm_gated(v, core, z, w->gnorm, n * hv, dv, c->eps); TR("gnorm", v, n, vd);
    mm(out, v, &w->out, n);
    rc = 0;
done:
    gpu_free(qkv); gpu_free(cv); gpu_free(z); gpu_free(a); gpu_free(b); gpu_free(q); gpu_free(k); gpu_free(v);
    gpu_free(g); gpu_free(beta); gpu_free(core); gpu_free(tmp); gpu_free(S0);
    return rc;
}

/* h、pos 都在显存里 */
static int layer_run(float *h, size_t n, const int32_t *pos, const model_seq *seqs, size_t nseq, size_t layer,
                     const model_cfg *c, const layer_w *w, trace_fn T, void *ctx, char *err, size_t cap) {
    size_t H = c->hidden, I = c->inter;
    float *x = dmem(n * H * sizeof *x, err, cap), *y = dmem(n * H * sizeof *y, err, cap);
    float *gt = dmem(n * I * sizeof *gt, err, cap), *up = dmem(n * I * sizeof *up, err, cap);
    float *act = dmem(n * I * sizeof *act, err, cap);
    int rc = -1;
    if (!x || !y || !gt || !up || !act) goto done;

    gop_rmsnorm(x, h, w->in_norm, n, H, c->eps);               TR("attn_in", x, n, H);
    if (w->full ? full_attention(y, x, n, pos, seqs, nseq, layer, c, w, T, ctx, err, cap)
                : linear_attention(y, x, n, seqs, nseq, layer, c, w, T, ctx, err, cap)) goto done;
    TR("attn_out", y, n, H);
    gop_add(h, h, y, n * H);                                   TR("resid", h, n, H);

    gop_rmsnorm(x, h, w->post_norm, n, H, c->eps);             TR("mlp_in", x, n, H);
    mm(gt, x, &w->gate, n);                                    TR("gate", gt, n, I);
    mm(up, x, &w->up, n);                                      TR("up", up, n, I);
    gop_silu(act, gt, n * I);                                  TR("gate_silu", act, n, I);
    gop_mul(gt, act, up, n * I);                               TR("act", gt, n, I);
    mm(y, gt, &w->down, n);                                    TR("mlp_out", y, n, H);
    gop_add(h, h, y, n * H);                                   TR("out", h, n, H);
    /* 内核是异步的，到这里 GPU 可能还在算。等它算完，这一层出的错（比如访问越界）就在这一层报出来。 */
    rc = gpu_sync(err, cap);
done:
    gpu_free(x); gpu_free(y); gpu_free(gt); gpu_free(up); gpu_free(act);
    return rc;
}

/* 把步骤名加上 “layers.<i>.” 前缀，同 model.c */
typedef struct { trace_fn f; void *ctx; size_t layer; } prefix_ctx;

static void prefixed(void *p, const char *name, const float *x, size_t rows, size_t cols) {
    const prefix_ctx *pc = p;
    char nm[128];
    snprintf(nm, sizeof nm, "layers.%zu.%s", pc->layer, name);
    pc->f(pc->ctx, nm, x, rows, cols);
}

int gpu_model_run(const model *g, const model_seq *seqs, size_t nseq, float *logits,
                  trace_fn T, void *ctx, char *err, size_t cap) {
    const model_cfg *c = &g->cfg;
    size_t H = c->hidden, n = 0;
    for (size_t b = 0; b < nseq; b++) n += seqs[b].n;
    float *h = NULL, *x = NULL, *dlogits = NULL;
    int32_t *pos = malloc(n * sizeof *pos), *ids = malloc(n * sizeof *ids), *dids = NULL, *dpos = NULL;
    int rc = -1;
    if (!pos || !ids) { fail(err, cap, "内存不足"); goto done; }
    for (size_t b = 0, r = 0; b < nseq; b++)
        for (size_t t = 0; t < seqs[b].n; t++, r++) {
            ids[r] = seqs[b].ids[t];
            /* 内核里没法报错，所以 id 在这里先查一遍（与 op_embed 的检查相同） */
            if (ids[r] < 0 || (uint64_t)ids[r] >= c->vocab) {
                fail(err, cap, "第 %zu 个 token id %d 超出嵌入表范围 [0, %zu)", r, ids[r], c->vocab);
                goto done;
            }
            pos[r] = (int32_t)(seq_past(&seqs[b]) + t);
        }
    if (!(h = dmem(n * H * sizeof *h, err, cap)) || !(x = dmem(n * H * sizeof *x, err, cap)) ||
        !(dlogits = dmem(n * c->vocab * sizeof *dlogits, err, cap)) || !(dids = dmem(n * sizeof *dids, err, cap)) ||
        !(dpos = dmem(n * sizeof *dpos, err, cap)))
        goto done;
    if (gpu_put(dids, ids, n * sizeof *ids, err, cap) || gpu_put(dpos, pos, n * sizeof *pos, err, cap)) goto done;

    gop_embed(h, g->embed, H, dids, n);
    TR("embed", h, n, H);
    for (size_t i = 0; i < c->n_layers; i++) {
        prefix_ctx pc = {T, ctx, i};
        if (layer_run(h, n, dpos, seqs, nseq, i, c, &g->layers[i], T ? prefixed : NULL, &pc, err, cap)) goto done;
    }
    gop_rmsnorm(x, h, g->final_norm, n, H, c->eps);   TR("final_norm", x, n, H);
    mm(dlogits, x, &g->lm_head, n);
    if (gpu_get(logits, dlogits, n * c->vocab * sizeof *logits, err, cap)) goto done;
    if (T) T(ctx, "logits", logits, n, c->vocab);
    rc = 0;
done:
    gpu_free(h); gpu_free(x); gpu_free(dlogits); gpu_free(dids); gpu_free(dpos);
    free(pos); free(ids);
    return rc;
}

/* ---------- 把权重拷进显存 ---------- */

static int put_vec(float **d, const float *h, size_t n, char *err, size_t cap) {
    if (!(*d = dmem(n * sizeof **d, err, cap))) return -1;
    return gpu_put(*d, h, n * sizeof **d, err, cap);
}

static int put_mat(bf16_mat *d, const bf16_mat *h, char *err, size_t cap) {
    size_t bytes = 2 * h->rows * h->cols;
    uint8_t *p = dmem(bytes, err, cap);
    d->data = p; d->rows = h->rows; d->cols = h->cols;
    return p ? gpu_put(p, h->data, bytes, err, cap) : -1;
}

static void layer_gpu_free(layer_w *d) {
    float *v[] = {d->in_norm, d->post_norm, d->q_norm, d->k_norm, d->conv, d->A_log, d->dt_bias, d->gnorm};
    const bf16_mat *m[] = {&d->gate, &d->up, &d->down, &d->q, &d->k, &d->v, &d->o,
                           &d->qkv, &d->z, &d->a, &d->b, &d->out};
    for (size_t i = 0; i < sizeof v / sizeof v[0]; i++) gpu_free(v[i]);
    for (size_t i = 0; i < sizeof m / sizeof m[0]; i++) gpu_free((void *)m[i]->data);
    memset(d, 0, sizeof *d);
}

/* 把 CPU 上已经检查过的一层（h）拷进显存（d）。各向量的长度与 layer_load 里的相同。 */
static int layer_upload(layer_w *d, const model_cfg *c, const layer_w *h, char *err, size_t cap) {
    memset(d, 0, sizeof *d);
    d->full = h->full;
    size_t H = c->hidden, hd = c->head_dim, hv = c->lv_heads;
    size_t cd = 2 * c->lk_heads * c->lk_dim + hv * c->lv_dim;
#define V(f, len) put_vec(&d->f, h->f, len, err, cap)
#define M(f) put_mat(&d->f, &h->f, err, cap)
    int bad = V(in_norm, H) || V(post_norm, H) || M(gate) || M(up) || M(down);
    if (!bad && h->full)
        bad = M(q) || M(k) || M(v) || M(o) || V(q_norm, hd) || V(k_norm, hd);
    else if (!bad)
        bad = M(qkv) || M(z) || M(a) || M(b) || M(out) || V(conv, cd * c->conv_k) || V(A_log, hv) ||
              V(dt_bias, hv) || V(gnorm, c->lv_dim);
#undef V
#undef M
    if (bad) { layer_gpu_free(d); return -1; }
    return 0;
}

void gpu_model_free(model *g) {
    if (!g) return;
    if (g->layers)
        for (size_t i = 0; i < g->cfg.n_layers; i++) layer_gpu_free(&g->layers[i]);
    free(g->layers);
    gpu_free((void *)g->embed);
    gpu_free((void *)g->lm_head.data);
    gpu_free(g->final_norm);
    free(g);
}

int model_to_gpu(model *m, char *err, size_t cap) {
    if (m->gpu) return fail(err, cap, "模型已经在 GPU 上了");
    if (gpu_init(err, cap)) return -1;
    const model_cfg *c = &m->cfg;
    model *g = calloc(1, sizeof *g);
    if (!g) return fail(err, cap, "内存不足");
    g->cfg = *c;
    size_t eb = 2 * c->vocab * c->hidden;
    uint8_t *emb = dmem(eb, err, cap);
    g->embed = emb;
    if (!emb || gpu_put(emb, m->embed, eb, err, cap)) goto bad;
    if (put_mat(&g->lm_head, &m->lm_head, err, cap) || put_vec(&g->final_norm, m->final_norm, c->hidden, err, cap))
        goto bad;
    if (!(g->layers = calloc(c->n_layers, sizeof *g->layers))) { fail(err, cap, "内存不足"); goto bad; }
    for (size_t i = 0; i < c->n_layers; i++)
        if (layer_upload(&g->layers[i], c, &m->layers[i], err, cap)) goto bad;
    if (gpu_sync(err, cap)) goto bad;
    m->gpu = g;
    return 0;
bad:
    gpu_model_free(g);
    return -1;
}

/* ---------- 缓存 ---------- */

void gpu_cache_free(model_cache *kc) {
    if (!kc->layers) return;
    for (size_t i = 0; i < kc->n_layers; i++) {
        layer_cache *lc = &kc->layers[i];
        gpu_free(lc->k); gpu_free(lc->v); gpu_free(lc->conv); gpu_free(lc->S);
        memset(lc, 0, sizeof *lc);
    }
}

int gpu_cache_alloc(model_cache *kc, const model_cfg *c, char *err, size_t cap) {
    size_t kd = c->kv_heads * c->head_dim, cd = 2 * c->lk_heads * c->lk_dim + c->lv_heads * c->lv_dim;
    for (size_t i = 0; i < c->n_layers; i++) {
        layer_cache *lc = &kc->layers[i];
        if (c->full[i]) {
            if (!(lc->k = dmem(kc->cap * kd * sizeof *lc->k, err, cap)) ||
                !(lc->v = dmem(kc->cap * kd * sizeof *lc->v, err, cap))) goto bad;
        } else {
            size_t ns = c->lv_heads * c->lk_dim * c->lv_dim;
            if (!(lc->conv = dmem((c->conv_k > 1 ? c->conv_k - 1 : 1) * cd * sizeof *lc->conv, err, cap)) ||
                !(lc->S = dmem(ns * sizeof *lc->S, err, cap))) goto bad;
            gpu_zero(lc->S, ns * sizeof *lc->S);
        }
    }
    return gpu_sync(err, cap);
bad:
    gpu_cache_free(kc);
    return -1;
}

/* ---------- 一层（第 4 章的 layer 工具用） ---------- */

int gpu_layer_forward(float *h, size_t n, const int32_t *pos, const model_cfg *c, const layer_w *w,
                      trace_fn T, void *ctx, char *err, size_t cap) {
    if (gpu_init(err, cap)) return -1;
    layer_w d;
    model_seq one = {NULL, NULL, n};
    if (layer_upload(&d, c, w, err, cap)) return -1;
    size_t bytes = n * c->hidden * sizeof *h;
    float *dh = dmem(bytes, err, cap);
    int32_t *dpos = dmem(n * sizeof *dpos, err, cap);
    int rc = -1;
    if (dh && dpos && !gpu_put(dh, h, bytes, err, cap) && !gpu_put(dpos, pos, n * sizeof *pos, err, cap) &&
        !layer_run(dh, n, dpos, &one, 1, 0, c, &d, T, ctx, err, cap))
        rc = gpu_get(h, dh, bytes, err, cap);
    gpu_free(dh); gpu_free(dpos);
    layer_gpu_free(&d);
    return rc;
}
