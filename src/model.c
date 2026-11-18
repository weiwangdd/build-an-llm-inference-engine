/* model.c — 第 4 章：配置、一层的权重、一层的前向。公式以 transformers 5.23.0 的
 * models/qwen3_5/modeling_qwen3_5.py 为准（Qwen3_5DecoderLayer、Qwen3_5Attention、Qwen3_5GatedDeltaNet）。 */
#include "model.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
        get_size(t, "linear_conv_kernel_dim", &c->conv_k, err, cap))
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
    FILE *f = fopen(path, "rb");
    if (!f) return fail(err, cap, "打不开 %s", path);
    char *buf = NULL;
    size_t n = 0, capb = 0;
    for (;;) {
        if (n == capb) {
            char *nb = realloc(buf, capb = capb ? 2 * capb : 8192);
            if (!nb) { free(buf); fclose(f); return fail(err, cap, "内存不足"); }
            buf = nb;
        }
        size_t r = fread(buf + n, 1, capb - n, f);
        n += r;
        if (r == 0) break;
    }
    int bad = ferror(f);
    fclose(f);
    if (bad) { free(buf); return fail(err, cap, "读 %s 出错", path); }
    jval root;
    int rc = json_parse(&root, buf, n, err, cap);
    free(buf);
    if (rc) return -1;
    rc = parse_cfg(c, &root, err, cap);
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

/* 全注意力。x 是归一化之后的输入 [n, hidden]，结果写进 out [n, hidden]。 */
static int full_attention(float *out, const float *x, size_t n, const int32_t *pos, const model_cfg *c,
                          const layer_w *w, trace_fn T, void *ctx, char *err, size_t cap) {
    size_t hd = c->head_dim, nh = c->heads, nkv = c->kv_heads, qd = nh * hd, kd = nkv * hd;
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
    if (op_attention(t1, q, k, v, n, nh, nkv, hd, 1.0 / sqrt((double)hd), err, cap)) goto done;
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
static int linear_attention(float *out, const float *x, size_t n, const model_cfg *c, const layer_w *w,
                            trace_fn T, void *ctx, char *err, size_t cap) {
    size_t hk = c->lk_heads, hv = c->lv_heads, dk = c->lk_dim, dv = c->lv_dim;
    size_t kd = hk * dk, vd = hv * dv, cd = 2 * kd + vd;
    float *qkv = malloc(n * cd * sizeof *qkv), *cv = malloc(n * cd * sizeof *cv);
    float *z = malloc(n * vd * sizeof *z), *a = malloc(n * hv * sizeof *a), *b = malloc(n * hv * sizeof *b);
    float *q = malloc(n * kd * sizeof *q), *k = malloc(n * kd * sizeof *k), *v = malloc(n * vd * sizeof *v);
    float *g = malloc(n * hv * sizeof *g), *beta = malloc(n * hv * sizeof *beta);
    float *core = malloc(n * vd * sizeof *core);
    int rc = -1;
    if (!qkv || !cv || !z || !a || !b || !q || !k || !v || !g || !beta || !core) { fail(err, cap, "内存不足"); goto done; }

    mm(qkv, x, &w->qkv, n);                  tr(T, ctx, "qkv", qkv, n, cd);
    mm(z, x, &w->z, n);                      tr(T, ctx, "z", z, n, vd);
    mm(b, x, &w->b, n);                      tr(T, ctx, "b", b, n, hv);
    mm(a, x, &w->a, n);                      tr(T, ctx, "a", a, n, hv);
    op_conv1d(cv, qkv, w->conv, n, cd, c->conv_k);             tr(T, ctx, "conv", cv, n, cd);
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
    if (op_gated_delta(core, q, k, v, g, beta, n, hk, hv, dk, dv, err, cap)) goto done;
    tr(T, ctx, "delta", core, n, vd);
    op_rmsnorm_gated(v, core, z, w->gnorm, n * hv, dv, c->eps); tr(T, ctx, "gnorm", v, n, vd);
    mm(out, v, &w->out, n);
    rc = 0;
done:
    free(qkv); free(cv); free(z); free(a); free(b); free(q); free(k); free(v); free(g); free(beta); free(core);
    return rc;
}

int layer_forward(float *h, size_t n, const int32_t *pos, const model_cfg *c, const layer_w *w,
                  trace_fn T, void *ctx, char *err, size_t cap) {
    size_t H = c->hidden, I = c->inter;
    float *x = malloc(n * H * sizeof *x), *y = malloc(n * H * sizeof *y);
    float *gt = malloc(n * I * sizeof *gt), *up = malloc(n * I * sizeof *up), *act = malloc(n * I * sizeof *act);
    int rc = -1;
    if (!x || !y || !gt || !up || !act) { fail(err, cap, "内存不足"); goto done; }

    /* h = h + 注意力(RMSNorm(h)) */
    op_rmsnorm(x, h, w->in_norm, n, H, c->eps);                tr(T, ctx, "attn_in", x, n, H);
    if (w->full ? full_attention(y, x, n, pos, c, w, T, ctx, err, cap)
                : linear_attention(y, x, n, c, w, T, ctx, err, cap)) goto done;
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

/* ---------- 第 5 章：整个模型 ---------- */

void model_free(model *m) {
    if (m->layers)
        for (size_t i = 0; i < m->cfg.n_layers; i++) layer_free(&m->layers[i]);
    free(m->layers);
    free(m->final_norm);
    m->layers = NULL;
    m->final_norm = NULL;
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

int model_forward(const model *m, const int32_t *ids, size_t n, float *logits,
                  trace_fn T, void *ctx, char *err, size_t cap) {
    const model_cfg *c = &m->cfg;
    size_t H = c->hidden;
    float *h = malloc(n * H * sizeof *h), *x = malloc(n * H * sizeof *x);
    int32_t *pos = malloc(n * sizeof *pos);
    int rc = -1;
    if (!h || !x || !pos) { fail(err, cap, "内存不足"); goto done; }
    for (size_t t = 0; t < n; t++) pos[t] = (int32_t)t;

    if (op_embed(h, m->embed, c->vocab, H, ids, n, err, cap)) goto done;
    tr(T, ctx, "embed", h, n, H);
    for (size_t i = 0; i < c->n_layers; i++) {        /* 每一层的输出就是下一层的输入 */
        prefix_ctx pc = {T, ctx, i};
        if (layer_forward(h, n, pos, c, &m->layers[i], T ? prefixed : NULL, &pc, err, cap)) goto done;
    }
    op_rmsnorm(x, h, m->final_norm, n, H, c->eps);    tr(T, ctx, "final_norm", x, n, H);
    mm(logits, x, &m->lm_head, n);                    tr(T, ctx, "logits", logits, n, c->vocab);
    rc = 0;
done:
    free(h); free(x); free(pos);
    return rc;
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
