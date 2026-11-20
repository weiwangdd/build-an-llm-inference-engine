/* model.c — 第 4 章：配置、一层的权重、一层的前向。公式以 transformers 5.23.0 的
 * models/qwen3_5/modeling_qwen3_5.py 为准（Qwen3_5DecoderLayer、Qwen3_5Attention、Qwen3_5GatedDeltaNet）。 */
#include "model.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu.h"
#include "json.h"
#include "ops.h"

static int fail(char *err, size_t cap, const char *fmt, ...) {
    if (err && cap) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, cap, fmt, ap);
        va_end(ap);
    }
    return -1;
}

/* ---------- config.json ---------- */

static int str_is(const jval *v, const char *s) { return v && v->type == J_STR && !strcmp(v->str, s); }

/* 正整数字段 */
static int get_size(const jval *o, const char *key, size_t *out, char *err, size_t cap) {
    int64_t v;
    const jval *j = json_get(o, key);
    if (!j) return fail(err, cap, "config.json 缺少字段 %s", key);
    if (j->type != J_NUM || json_int(j, &v) || v <= 0 || v > (int64_t)1 << 40)
        return fail(err, cap, "config.json 的 %s 应为正整数", key);
    *out = (size_t)v;
    return 0;
}

/* 有限的实数字段 */
static int get_num(const jval *o, const char *key, double *out, char *err, size_t cap) {
    const jval *j = json_get(o, key);
    if (!j) return fail(err, cap, "config.json 缺少字段 %s", key);
    char *end;
    if (j->type != J_NUM || (*out = strtod(j->str, &end), *end) || !isfinite(*out))
        return fail(err, cap, "config.json 的 %s 应为数", key);
    return 0;
}

/* 字段要么不存在，要么等于给定的值（用于“只是记录、我们只支持这一种”的字段） */
static int want_str(const jval *o, const char *key, const char *val, int required, char *err, size_t cap) {
    const jval *j = json_get(o, key);
    if (!j && !required) return 0;
    if (!str_is(j, val)) return fail(err, cap, "config.json 的 %s 只支持 \"%s\"", key, val);
    return 0;
}

static int want_false(const jval *o, const char *key, char *err, size_t cap) {
    const jval *j = json_get(o, key);
    if (!j || j->type != J_FALSE) return fail(err, cap, "config.json 的 %s 只支持 false", key);
    return 0;
}

static int parse_cfg(model_cfg *c, const jval *root, char *err, size_t cap) {
    memset(c, 0, sizeof *c);
    if (root->type != J_OBJ) return fail(err, cap, "config.json 应为一个对象");
    if (want_str(root, "model_type", "qwen3_5", 1, err, cap)) return -1;
    const jval *t = json_get(root, "text_config");
    if (!t || t->type != J_OBJ) return fail(err, cap, "config.json 缺少 text_config");
    /* 输出层复用嵌入表的模型（tie_word_embeddings 为 true）没有单独的 lm_head.weight，我们不支持。
     * 这个字段在顶层和 text_config 里各有一份，哪一份都不能是 true。 */
    const jval *tie0 = json_get(root, "tie_word_embeddings"), *tie1 = json_get(t, "tie_word_embeddings");
    if ((tie0 && tie0->type != J_FALSE) || (tie1 && tie1->type != J_FALSE))
        return fail(err, cap, "config.json 的 tie_word_embeddings 只支持 false");

    if (get_size(t, "vocab_size", &c->vocab, err, cap) || get_size(t, "hidden_size", &c->hidden, err, cap) ||
        get_size(t, "intermediate_size", &c->inter, err, cap) ||
        get_size(t, "num_hidden_layers", &c->n_layers, err, cap) || get_num(t, "rms_norm_eps", &c->eps, err, cap) ||
        get_size(t, "num_attention_heads", &c->heads, err, cap) ||
        get_size(t, "num_key_value_heads", &c->kv_heads, err, cap) ||
        get_size(t, "head_dim", &c->head_dim, err, cap) ||
        get_size(t, "linear_num_key_heads", &c->lk_heads, err, cap) ||
        get_size(t, "linear_num_value_heads", &c->lv_heads, err, cap) ||
        get_size(t, "linear_key_head_dim", &c->lk_dim, err, cap) ||
        get_size(t, "linear_value_head_dim", &c->lv_dim, err, cap) ||
        get_size(t, "linear_conv_kernel_dim", &c->conv_k, err, cap) ||
        get_size(t, "max_position_embeddings", &c->max_pos, err, cap))
        return -1;
    if (want_str(t, "hidden_act", "silu", 1, err, cap) || want_false(t, "attention_bias", err, cap) ||
        want_str(t, "output_gate_type", "swish", 0, err, cap))
        return -1;
    /* attn_output_gate 在 transformers 里没人读，门控是无条件的（第 3 章）。只接受 true，免得以为关掉了。 */
    const jval *gate = json_get(t, "attn_output_gate");
    if (gate && gate->type != J_TRUE) return fail(err, cap, "config.json 的 attn_output_gate 只支持 true");
    if (!(c->eps >= 0)) return fail(err, cap, "config.json 的 rms_norm_eps 不能是负数");
    if (c->heads % c->kv_heads) return fail(err, cap, "num_attention_heads 应是 num_key_value_heads 的整数倍");
    if (c->lv_heads % c->lk_heads) return fail(err, cap, "linear_num_value_heads 应是 linear_num_key_heads 的整数倍");

    const jval *lt = json_get(t, "layer_types");
    if (!lt || lt->type != J_ARR) return fail(err, cap, "config.json 缺少 layer_types");
    if (lt->len != c->n_layers) return fail(err, cap, "layer_types 有 %zu 项，与 num_hidden_layers 不符", lt->len);
    if (c->n_layers > MODEL_MAX_LAYERS) return fail(err, cap, "层数超过 %d", MODEL_MAX_LAYERS);
    for (size_t i = 0; i < lt->len; i++) {
        if (str_is(&lt->items[i], "full_attention")) c->full[i] = 1;
        else if (!str_is(&lt->items[i], "linear_attention")) return fail(err, cap, "layer_types 第 %zu 项不认识", i);
    }

    /* RoPE 参数在 rope_parameters 里（顶层那个重复的 partial_rotary_factor 不读，与 transformers 一致） */
    const jval *rp = json_get(t, "rope_parameters");
    if (!rp || rp->type != J_OBJ) return fail(err, cap, "config.json 缺少 rope_parameters");
    if (want_str(rp, "rope_type", "default", 1, err, cap) || get_num(rp, "rope_theta", &c->theta, err, cap)) return -1;
    if (!(c->theta > 0)) return fail(err, cap, "rope_theta 必须是正数");
    double prf = 1.0;
    if (json_get(rp, "partial_rotary_factor") && get_num(rp, "partial_rotary_factor", &prf, err, cap)) return -1;
    double rot = (double)c->head_dim * prf;        /* transformers 写的是 int(head_dim * factor)，这里要求恰好是整数 */
    if (!(rot >= 2) || rot > (double)c->head_dim || rot != floor(rot) || fmod(rot, 2) != 0)
        return fail(err, cap, "head_dim × partial_rotary_factor 应是不超过 head_dim 的正偶数");
    c->rot = (size_t)rot;
    return 0;
}

int model_cfg_load(model_cfg *c, const char *path, char *err, size_t cap) {
    jval root;
    if (json_parse_file(&root, path, err, cap)) return -1;
    int rc = parse_cfg(c, &root, err, cap);
    json_free(&root);
    return rc;
}

/* ---------- 权重 ---------- */

#define PREFIX "model.language_model."

static const st_tensor *find(const st_set *s, const char *name, char *err, size_t cap) {
    const st_tensor *t = st_set_find(s, name);
    if (!t) fail(err, cap, "找不到张量 %s", name);
    return t;
}

static int get_mat(bf16_mat *m, const st_set *s, const char *name, size_t rows, size_t cols, char *err, size_t cap) {
    const st_tensor *t = find(s, name, err, cap);
    if (!t) return -1;
    if (t->dtype != ST_BF16 || t->ndim != 2 || t->shape[0] != rows || t->shape[1] != cols)
        return fail(err, cap, "张量 %s 应为 BF16 [%zu,%zu]", name, rows, cols);
    m->data = t->data; m->rows = rows; m->cols = cols;
    return 0;
}

/* 小向量：BF16 或 F32，元素个数为 len，形状可以是 [len]，也可以是 [len,1,k] 这类（由 dims 给出）。 */
static float *get_vec(const st_set *s, const char *name, const uint64_t *shape, int ndim, char *err, size_t cap) {
    const st_tensor *t = find(s, name, err, cap);
    if (!t) return NULL;
    size_t n = 1;
    for (int i = 0; i < ndim; i++) n *= (size_t)shape[i];
    if ((t->dtype != ST_BF16 && t->dtype != ST_F32) || t->ndim != ndim ||
        memcmp(t->shape, shape, sizeof *shape * (size_t)ndim)) {
        fail(err, cap, "张量 %s 的形状或类型不对", name);
        return NULL;
    }
    float *p = malloc(n * sizeof *p);
    if (!p) { fail(err, cap, "内存不足"); return NULL; }
    if (t->dtype == ST_BF16) for (size_t i = 0; i < n; i++) p[i] = bf16_to_f32(bf16_at(t->data, i));
    else memcpy(p, t->data, n * sizeof *p);
    return p;
}

void layer_free(layer_w *w) {
    float **v[] = {&w->in_norm, &w->post_norm, &w->q_norm, &w->k_norm, &w->conv, &w->A_log, &w->dt_bias, &w->gnorm};
    for (size_t i = 0; i < sizeof v / sizeof v[0]; i++) { free(*v[i]); *v[i] = NULL; }
}

int layer_load(layer_w *w, const model_cfg *c, const st_set *s, size_t layer, char *err, size_t cap) {
    memset(w, 0, sizeof *w);
    if (layer >= c->n_layers) return fail(err, cap, "第 %zu 层不存在（共 %zu 层）", layer, c->n_layers);
    w->full = c->full[layer];
    char nm[256];
#define N(suffix) (snprintf(nm, sizeof nm, PREFIX "layers.%zu." suffix, layer), nm)
#define VEC(dst, suffix, ...) do { uint64_t sh_[] = {__VA_ARGS__}; \
        if (!(dst = get_vec(s, N(suffix), sh_, (int)(sizeof sh_ / sizeof sh_[0]), err, cap))) goto bad; } while (0)
#define MAT(dst, suffix, r, k) do { if (get_mat(&dst, s, N(suffix), r, k, err, cap)) goto bad; } while (0)
    size_t H = c->hidden;
    VEC(w->in_norm, "input_layernorm.weight", H);
    VEC(w->post_norm, "post_attention_layernorm.weight", H);
    MAT(w->gate, "mlp.gate_proj.weight", c->inter, H);
    MAT(w->up, "mlp.up_proj.weight", c->inter, H);
    MAT(w->down, "mlp.down_proj.weight", H, c->inter);
    if (w->full) {
        size_t hd = c->head_dim;
        MAT(w->q, "self_attn.q_proj.weight", c->heads * hd * 2, H);   /* 每个头 hd 维的 q 加 hd 维的门控 */
        MAT(w->k, "self_attn.k_proj.weight", c->kv_heads * hd, H);
        MAT(w->v, "self_attn.v_proj.weight", c->kv_heads * hd, H);
        MAT(w->o, "self_attn.o_proj.weight", H, c->heads * hd);
        VEC(w->q_norm, "self_attn.q_norm.weight", hd);
        VEC(w->k_norm, "self_attn.k_norm.weight", hd);
    } else {
        size_t kd = c->lk_heads * c->lk_dim, vd = c->lv_heads * c->lv_dim, cd = 2 * kd + vd;
        MAT(w->qkv, "linear_attn.in_proj_qkv.weight", cd, H);
        MAT(w->z, "linear_attn.in_proj_z.weight", vd, H);
        MAT(w->b, "linear_attn.in_proj_b.weight", c->lv_heads, H);
        MAT(w->a, "linear_attn.in_proj_a.weight", c->lv_heads, H);
        MAT(w->out, "linear_attn.out_proj.weight", H, vd);
        VEC(w->conv, "linear_attn.conv1d.weight", cd, 1, c->conv_k);
        VEC(w->A_log, "linear_attn.A_log", c->lv_heads);
        VEC(w->dt_bias, "linear_attn.dt_bias", c->lv_heads);
        VEC(w->gnorm, "linear_attn.norm.weight", c->lv_dim);
    }
    return 0;
#undef N
#undef VEC
#undef MAT
bad:
    layer_free(w);
    return -1;
}

float *final_norm_load(const model_cfg *c, const st_set *s, char *err, size_t cap) {
    uint64_t sh[] = {c->hidden};
    return get_vec(s, PREFIX "norm.weight", sh, 1, err, cap);
}

/* ---------- 前向 ---------- */

static void tr(trace_fn f, void *ctx, const char *name, const float *x, size_t rows, size_t cols) {
    if (f) f(ctx, name, x, rows, cols);
}

static void mm(float *y, const float *x, const bf16_mat *w, size_t n) { op_matmul(y, x, w->data, n, w->cols, w->rows); }

/* 第 8 章：一次前向可以同时算好几个序列（批处理）。它们的 token 一个接一个排成一张 [N, hidden] 的表，
 * N 是各序列 token 数之和。矩阵乘、归一化、激活这些“每个 token 各算各的”的步骤对整张表做一次；
 * 只有要看前面 token 的那几步（注意力、卷积、门控 delta 规则）按序列分开，各用各的缓存。
 * 每个输出的求和的项和次序与只有一个序列时完全相同，所以结果逐位相同。
 * 下面 seqs 是这一批的序列，layer 是第几层；一个序列的 kc 为 NULL 表示不用缓存（第 4、5 章）。 */
static layer_cache *seq_cache(const model_seq *s, size_t layer) { return s->kc ? &s->kc->layers[layer] : NULL; }
static size_t seq_past(const model_seq *s) { return s->kc ? s->kc->len : 0; }

/* 全注意力。x 是归一化之后的输入 [N, hidden]，结果写进 out [N, hidden]。 */
/* 有缓存时（第 6 章）：序列前面已经有 past 个 token，它们 RoPE 之后的 k 和 v 在 lc->k、lc->v 的前 past 行；
 * 新 token 的 k、v 接着写进去，注意力对全部 past + n 行算。没有缓存时 past 是 0。 */
static int full_attention(float *out, const float *x, size_t N, const int32_t *pos, const model_seq *seqs,
                          size_t nseq, size_t layer, const model_cfg *c, const layer_w *w,
                          trace_fn T, void *ctx, char *err, size_t cap) {
    size_t hd = c->head_dim, nh = c->heads, nkv = c->kv_heads, qd = nh * hd, kd = nkv * hd, n = N;
    float *qg = malloc(n * 2 * qd * sizeof *qg), *q = malloc(n * qd * sizeof *q), *gate = malloc(n * qd * sizeof *gate);
    float *k = malloc(n * kd * sizeof *k), *v = malloc(n * kd * sizeof *v);
    float *t1 = malloc(n * qd * sizeof *t1), *t2 = malloc(n * qd * sizeof *t2);
    int rc = -1;
    if (!qg || !q || !gate || !k || !v || !t1 || !t2) { fail(err, cap, "内存不足"); goto done; }

    mm(qg, x, &w->q, n);                     tr(T, ctx, "q_proj", qg, n, 2 * qd);
    /* q_proj 的输出按头排列，每个头 2·hd 维：前 hd 维是 q，后 hd 维是门控 */
    for (size_t t = 0; t < n; t++)
        for (size_t h = 0; h < nh; h++) {
            memcpy(q + (t * nh + h) * hd, qg + (t * nh + h) * 2 * hd, hd * sizeof *q);
            memcpy(gate + (t * nh + h) * hd, qg + (t * nh + h) * 2 * hd + hd, hd * sizeof *q);
        }
    mm(k, x, &w->k, n);                      tr(T, ctx, "k_proj", k, n, kd);
    mm(v, x, &w->v, n);                      tr(T, ctx, "v_proj", v, n, kd);
    op_rmsnorm(t1, q, w->q_norm, n * nh, hd, c->eps);          tr(T, ctx, "q_norm", t1, n, qd);
    op_rope(q, t1, pos, n, nh, hd, c->rot, c->theta);          tr(T, ctx, "q_rope", q, n, qd);
    op_rmsnorm(t1, k, w->k_norm, n * nkv, hd, c->eps);         tr(T, ctx, "k_norm", t1, n, kd);
    op_rope(k, t1, pos, n, nkv, hd, c->rot, c->theta);         tr(T, ctx, "k_rope", k, n, kd);
    /* 每个序列各看各的：新 token 的 k、v 接在自己的缓存后面，再对自己的全部 token 算注意力 */
    for (size_t b = 0, off = 0; b < nseq; off += seqs[b++].n) {
        size_t nb = seqs[b].n, past = seq_past(&seqs[b]);
        layer_cache *lc = seq_cache(&seqs[b], layer);
        const float *kall = k + off * kd, *vall = v + off * kd;
        if (lc) {
            memcpy(lc->k + past * kd, k + off * kd, nb * kd * sizeof *k);
            memcpy(lc->v + past * kd, v + off * kd, nb * kd * sizeof *v);
            kall = lc->k; vall = lc->v;
        }
        if (op_attention_past(t1 + off * qd, q + off * qd, kall, vall, nb, past, nh, nkv, hd,
                              1.0 / sqrt((double)hd), err, cap)) goto done;
    }
    tr(T, ctx, "attn", t1, n, qd);
    op_sigmoid(t2, gate, n * qd);                              tr(T, ctx, "gate_sigmoid", t2, n, qd);
    op_mul(q, t1, t2, n * qd);                                 tr(T, ctx, "gated", q, n, qd);
    mm(out, q, &w->o, n);
    rc = 0;
done:
    free(qg); free(q); free(gate); free(k); free(v); free(t1); free(t2);
    return rc;
}

/* 线性注意力（门控 delta 规则）。 */
/* 有缓存时（第 6 章）：卷积接着 lc->conv 里前面几个 token 的输入算，门控 delta 规则从 lc->S 出发，
 * 算完把两者更新成包含新 token 之后的样子。没有缓存时从零开始。 */
static int linear_attention(float *out, const float *x, size_t N, const model_seq *seqs, size_t nseq, size_t layer,
                            const model_cfg *c, const layer_w *w, trace_fn T, void *ctx, char *err, size_t cap) {
    size_t hk = c->lk_heads, hv = c->lv_heads, dk = c->lk_dim, dv = c->lv_dim, n = N;
    size_t kd = hk * dk, vd = hv * dv, cd = 2 * kd + vd;
    float *qkv = malloc(n * cd * sizeof *qkv), *cv = malloc(n * cd * sizeof *cv);
    float *z = malloc(n * vd * sizeof *z), *a = malloc(n * hv * sizeof *a), *b = malloc(n * hv * sizeof *b);
    float *q = malloc(n * kd * sizeof *q), *k = malloc(n * kd * sizeof *k), *v = malloc(n * vd * sizeof *v);
    float *g = malloc(n * hv * sizeof *g), *beta = malloc(n * hv * sizeof *beta);
    float *core = malloc(n * vd * sizeof *core);
    size_t kw = c->conv_k - 1;                /* 卷积要往前看 kw 个 token */
    float *tmp = malloc((kw ? kw : 1) * cd * sizeof *tmp);
    int rc = -1;
    if (!qkv || !cv || !z || !a || !b || !q || !k || !v || !g || !beta || !core || !tmp) {
        fail(err, cap, "内存不足"); goto done;
    }

    mm(qkv, x, &w->qkv, n);                  tr(T, ctx, "qkv", qkv, n, cd);
    mm(z, x, &w->z, n);                      tr(T, ctx, "z", z, n, vd);
    mm(b, x, &w->b, n);                      tr(T, ctx, "b", b, n, hv);
    mm(a, x, &w->a, n);                      tr(T, ctx, "a", a, n, hv);
    for (size_t s = 0, off = 0; s < nseq; off += seqs[s++].n) {     /* 卷积按序列分开 */
        size_t nb = seqs[s].n, past = seq_past(&seqs[s]);
        layer_cache *lc = seq_cache(&seqs[s], layer);
        size_t nh = lc ? (past < kw ? past : kw) : 0;
        const float *xs = qkv + off * cd;
        op_conv1d_hist(cv + off * cd, xs, lc ? lc->conv : NULL, nh, w->conv, nb, cd, c->conv_k);
        if (lc) {                             /* 把“历史 + 新 token”的最后 kw 行留作下一次的历史 */
            size_t total = nh + nb, keep = total < kw ? total : kw;
            for (size_t r = 0; r < keep; r++) {
                size_t src = total - keep + r;
                memcpy(tmp + r * cd, src < nh ? lc->conv + src * cd : xs + (src - nh) * cd, cd * sizeof *tmp);
            }
            memcpy(lc->conv, tmp, keep * cd * sizeof *tmp);
        }
    }
    tr(T, ctx, "conv", cv, n, cd);
    op_silu(qkv, cv, n * cd);                                  tr(T, ctx, "conv_silu", qkv, n, cd);
    /* 卷积之后再切开：每个 token 的 cd 维依次是 q（kd 维）、k（kd 维）、v（vd 维） */
    for (size_t t = 0; t < n; t++) {
        memcpy(cv + t * kd, qkv + t * cd, kd * sizeof *cv);                   /* cv 暂存 q */
        memcpy(cv + n * kd + t * kd, qkv + t * cd + kd, kd * sizeof *cv);     /* 和 k */
        memcpy(v + t * vd, qkv + t * cd + 2 * kd, vd * sizeof *v);
    }
    op_l2norm(q, cv, n * hk, dk, 1e-6);                        tr(T, ctx, "q_l2", q, n, kd);
    op_l2norm(k, cv + n * kd, n * hk, dk, 1e-6);               tr(T, ctx, "k_l2", k, n, kd);
    op_sigmoid(beta, b, n * hv);                               tr(T, ctx, "beta", beta, n, hv);
    op_decay(g, a, w->A_log, w->dt_bias, n, hv);               tr(T, ctx, "g", g, n, hv);
    for (size_t s = 0, off = 0; s < nseq; off += seqs[s++].n) {     /* 门控 delta 规则也按序列分开 */
        size_t nb = seqs[s].n;
        layer_cache *lc = seq_cache(&seqs[s], layer);
        float *ys = core + off * vd;
        const float *qs = q + off * kd, *ks = k + off * kd, *vs = v + off * vd, *gs = g + off * hv, *bs = beta + off * hv;
        if (lc ? op_gated_delta_state(ys, qs, ks, vs, gs, bs, lc->S, nb, hk, hv, dk, dv, err, cap)
               : op_gated_delta(ys, qs, ks, vs, gs, bs, nb, hk, hv, dk, dv, err, cap)) goto done;
    }
    tr(T, ctx, "delta", core, n, vd);
    op_rmsnorm_gated(v, core, z, w->gnorm, n * hv, dv, c->eps); tr(T, ctx, "gnorm", v, n, vd);
    mm(out, v, &w->out, n);
    rc = 0;
done:
    free(qkv); free(cv); free(z); free(a); free(b); free(q); free(k); free(v); free(g); free(beta); free(core);
    free(tmp);
    return rc;
}

/* 一层的前向。h 是这一批全部 N 个 token 的 [N, hidden]，就地更新；pos[r] 是第 r 行在它那个序列里的位置。
 * 只有一个不带缓存的序列时就是第 4 章的 layer_forward。 */
static int layer_run(float *h, size_t n, const int32_t *pos, const model_seq *seqs, size_t nseq, size_t layer,
                     const model_cfg *c, const layer_w *w, trace_fn T, void *ctx, char *err, size_t cap) {
    size_t H = c->hidden, I = c->inter;
    float *x = malloc(n * H * sizeof *x), *y = malloc(n * H * sizeof *y);
    float *gt = malloc(n * I * sizeof *gt), *up = malloc(n * I * sizeof *up), *act = malloc(n * I * sizeof *act);
    int rc = -1;
    if (!x || !y || !gt || !up || !act) { fail(err, cap, "内存不足"); goto done; }

    /* h = h + 注意力(RMSNorm(h)) */
    op_rmsnorm(x, h, w->in_norm, n, H, c->eps);                tr(T, ctx, "attn_in", x, n, H);
    if (w->full ? full_attention(y, x, n, pos, seqs, nseq, layer, c, w, T, ctx, err, cap)
                : linear_attention(y, x, n, seqs, nseq, layer, c, w, T, ctx, err, cap)) goto done;
    tr(T, ctx, "attn_out", y, n, H);
    op_add(h, h, y, n * H);                                    tr(T, ctx, "resid", h, n, H);

    /* h = h + 前馈(RMSNorm(h))，前馈(x) = down(silu(gate(x)) * up(x)) */
    op_rmsnorm(x, h, w->post_norm, n, H, c->eps);              tr(T, ctx, "mlp_in", x, n, H);
    mm(gt, x, &w->gate, n);                                    tr(T, ctx, "gate", gt, n, I);
    mm(up, x, &w->up, n);                                      tr(T, ctx, "up", up, n, I);
    op_silu(act, gt, n * I);                                   tr(T, ctx, "gate_silu", act, n, I);
    op_mul(gt, act, up, n * I);                                tr(T, ctx, "act", gt, n, I);
    mm(y, gt, &w->down, n);                                    tr(T, ctx, "mlp_out", y, n, H);
    op_add(h, h, y, n * H);                                    tr(T, ctx, "out", h, n, H);
    rc = 0;
done:
    free(x); free(y); free(gt); free(up); free(act);
    return rc;
}

int layer_forward(float *h, size_t n, const int32_t *pos, const model_cfg *c, const layer_w *w,
                  trace_fn T, void *ctx, char *err, size_t cap) {
    model_seq one = {NULL, NULL, n};
    return layer_run(h, n, pos, &one, 1, 0, c, w, T, ctx, err, cap);
}

/* ---------- 第 5 章：整个模型 ---------- */

void model_free(model *m) {
    if (m->layers)
        for (size_t i = 0; i < m->cfg.n_layers; i++) layer_free(&m->layers[i]);
    free(m->layers);
    free(m->final_norm);
    gpu_model_free(m->gpu);
    m->layers = NULL;
    m->final_norm = NULL;
    m->gpu = NULL;
}

int model_load(model *m, const char *config_path, const st_set *s, char *err, size_t cap) {
    memset(m, 0, sizeof *m);
    if (model_cfg_load(&m->cfg, config_path, err, cap)) return -1;
    const model_cfg *c = &m->cfg;
    if (c->vocab > INT32_MAX) return fail(err, cap, "vocab_size 太大，token id 要能放进 int32");
    bf16_mat emb;
    if (get_mat(&emb, s, PREFIX "embed_tokens.weight", c->vocab, c->hidden, err, cap)) return -1;
    m->embed = emb.data;
    /* 输出层不在 model.language_model. 下面，是顶层的 lm_head.weight */
    if (get_mat(&m->lm_head, s, "lm_head.weight", c->vocab, c->hidden, err, cap)) return -1;
    m->layers = calloc(c->n_layers, sizeof *m->layers);
    if (!m->layers) return fail(err, cap, "内存不足");
    for (size_t i = 0; i < c->n_layers; i++)
        if (layer_load(&m->layers[i], c, s, i, err, cap)) goto bad;
    if (!(m->final_norm = final_norm_load(c, s, err, cap))) goto bad;
    return 0;
bad:
    model_free(m);
    return -1;
}

/* 把第 4 章 layer_forward 报告的步骤名加上 “layers.<i>.” 前缀，再交给调用方 */
typedef struct { trace_fn f; void *ctx; size_t layer; } prefix_ctx;

static void prefixed(void *p, const char *name, const float *x, size_t rows, size_t cols) {
    const prefix_ctx *pc = p;
    char nm[128];
    snprintf(nm, sizeof nm, "layers.%zu.%s", pc->layer, name);
    pc->f(pc->ctx, nm, x, rows, cols);
}

/* 整个模型，对一批序列（第 8 章）。只有一个序列且不带缓存时是第 5 章的 model_forward（位置从 0 开始）；
 * 带缓存时（第 6 章）每个序列的新 token 接在它缓存里那 kc->len 个后面，位置从 kc->len 开始。
 * logits 依次是各序列的行，共 Σn 行。 */
static int model_run(const model *m, const model_seq *seqs, size_t nseq, float *logits,
                     trace_fn T, void *ctx, char *err, size_t cap) {
    if (m->gpu) return gpu_model_run(m->gpu, seqs, nseq, logits, T, ctx, err, cap);   /* 第 7 章 */
    const model_cfg *c = &m->cfg;
    size_t H = c->hidden, n = 0;
    for (size_t b = 0; b < nseq; b++) n += seqs[b].n;
    float *h = malloc(n * H * sizeof *h), *x = malloc(n * H * sizeof *x);
    int32_t *pos = malloc(n * sizeof *pos), *ids = malloc(n * sizeof *ids);
    int rc = -1;
    if (!h || !x || !pos || !ids) { fail(err, cap, "内存不足"); goto done; }
    for (size_t b = 0, r = 0; b < nseq; b++)          /* 各序列的 token 排成一张表 */
        for (size_t t = 0; t < seqs[b].n; t++, r++) {
            ids[r] = seqs[b].ids[t];
            pos[r] = (int32_t)(seq_past(&seqs[b]) + t);
        }

    if (op_embed(h, m->embed, c->vocab, H, ids, n, err, cap)) goto done;
    tr(T, ctx, "embed", h, n, H);
    for (size_t i = 0; i < c->n_layers; i++) {        /* 每一层的输出就是下一层的输入 */
        prefix_ctx pc = {T, ctx, i};
        if (layer_run(h, n, pos, seqs, nseq, i, c, &m->layers[i], T ? prefixed : NULL, &pc, err, cap)) goto done;
    }
    op_rmsnorm(x, h, m->final_norm, n, H, c->eps);    tr(T, ctx, "final_norm", x, n, H);
    mm(logits, x, &m->lm_head, n);                    tr(T, ctx, "logits", logits, n, c->vocab);
    rc = 0;
done:
    free(h); free(x); free(pos); free(ids);
    return rc;
}

int model_forward(const model *m, const int32_t *ids, size_t n, float *logits,
                  trace_fn T, void *ctx, char *err, size_t cap) {
    model_seq one = {NULL, ids, n};
    return model_run(m, &one, 1, logits, T, ctx, err, cap);
}

/* ---------- 第 6 章：缓存 ---------- */

void model_cache_free(model_cache *kc) {
    if (kc->gpu) gpu_cache_free(kc);
    else if (kc->layers)
        for (size_t i = 0; i < kc->n_layers; i++) {
            layer_cache *lc = &kc->layers[i];
            free(lc->k); free(lc->v); free(lc->conv); free(lc->S);
        }
    free(kc->layers);
    memset(kc, 0, sizeof *kc);
}

int model_cache_init(model_cache *kc, const model *m, size_t cap, char *err, size_t errcap) {
    const model_cfg *c = &m->cfg;
    memset(kc, 0, sizeof *kc);
    if (cap == 0 || cap > c->max_pos)
        return fail(err, errcap, "缓存容量 %zu 应在 1 到 max_position_embeddings（%zu）之间", cap, c->max_pos);
    kc->layers = calloc(c->n_layers, sizeof *kc->layers);
    if (!kc->layers) return fail(err, errcap, "内存不足");
    kc->n_layers = c->n_layers;
    kc->cap = cap;
    if (m->gpu) {                  /* 第 7 章：缓存建在显存里 */
        kc->gpu = 1;
        if (gpu_cache_alloc(kc, c, err, errcap)) { model_cache_free(kc); return -1; }
        return 0;
    }
    size_t kd = c->kv_heads * c->head_dim;
    size_t cd = 2 * c->lk_heads * c->lk_dim + c->lv_heads * c->lv_dim;
    for (size_t i = 0; i < c->n_layers; i++) {
        layer_cache *lc = &kc->layers[i];
        if (c->full[i]) {          /* 全注意力：每个 token 一行 k、一行 v，随序列变长 */
            lc->k = malloc(cap * kd * sizeof *lc->k);
            lc->v = malloc(cap * kd * sizeof *lc->v);
            if (!lc->k || !lc->v) goto oom;
        } else {                   /* 线性注意力：大小固定，与序列多长无关 */
            lc->conv = malloc((c->conv_k > 1 ? c->conv_k - 1 : 1) * cd * sizeof *lc->conv);
            lc->S = calloc(c->lv_heads * c->lk_dim * c->lv_dim, sizeof *lc->S);
            if (!lc->conv || !lc->S) goto oom;
        }
    }
    return 0;
oom:
    model_cache_free(kc);
    return fail(err, errcap, "内存不足（缓存）");
}

int model_step_batch(const model *m, model_seq *seqs, size_t nseq, float *logits,
                     trace_fn T, void *ctx, char *err, size_t cap) {
    if (nseq == 0) return fail(err, cap, "一批至少要有一个序列");
    for (size_t b = 0; b < nseq; b++) {
        model_cache *kc = seqs[b].kc;
        if (!kc) return fail(err, cap, "第 %zu 个序列没有缓存", b);
        if (kc->broken) return fail(err, cap, "第 %zu 个序列的缓存在上一次出错时已经损坏，不能接着用", b);
        if (kc->n_layers != m->cfg.n_layers) return fail(err, cap, "第 %zu 个序列的缓存与模型的层数不符", b);
        if (kc->gpu != (m->gpu != NULL))
            return fail(err, cap, "第 %zu 个序列的缓存与模型不在同一个设备上（一个在 GPU，一个在 CPU）", b);
        if (seqs[b].n == 0) return fail(err, cap, "第 %zu 个序列至少要给一个 token", b);
        if (seqs[b].n > kc->cap - kc->len)
            return fail(err, cap, "第 %zu 个序列的缓存已有 %zu 个 token，再放 %zu 个会超过容量 %zu",
                        b, kc->len, seqs[b].n, kc->cap);
        for (size_t j = 0; j < b; j++)         /* 同一个缓存出现两次，两个序列会互相覆盖 */
            if (seqs[j].kc == kc) return fail(err, cap, "第 %zu 个和第 %zu 个序列用的是同一个缓存", j, b);
    }
    if (model_run(m, seqs, nseq, logits, T, ctx, err, cap)) {
        /* 前面几层可能已经写进了新 token，后面几层没有 */
        for (size_t b = 0; b < nseq; b++) seqs[b].kc->broken = 1;
        return -1;
    }
    for (size_t b = 0; b < nseq; b++) seqs[b].kc->len += seqs[b].n;
    return 0;
}

int model_step(const model *m, model_cache *kc, const int32_t *ids, size_t n, float *logits,
               trace_fn T, void *ctx, char *err, size_t cap) {
    model_seq one = {kc, ids, n};
    return model_step_batch(m, &one, 1, logits, T, ctx, err, cap);
}

int logits_argmax(const float *row, size_t vocab, int32_t *out, char *err, size_t cap) {
    size_t best = 0;
    for (size_t i = 0; i < vocab; i++) {
        if (isnan(row[i])) return fail(err, cap, "logits 的第 %zu 个元素是 NaN", i);
        if (row[i] > row[best]) best = i;           /* 严格大于：同样大时保留前面那个 */
    }
    *out = (int32_t)best;
    return 0;
}
