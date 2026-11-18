/* gen.c — 第 6 章：generation_config.json 和生成循环。 */
#include "gen.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "json.h"

static int fail(char *err, size_t cap, const char *fmt, ...) {
    if (err && cap) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, cap, fmt, ap);
        va_end(ap);
    }
    return -1;
}

/* ---------- generation_config.json ---------- */

static const char *const KNOWN[] = {"bos_token_id", "do_sample", "eos_token_id", "pad_token_id",
                                    "temperature", "top_k", "top_p", "transformers_version"};

static int token_id(const jval *v, size_t vocab, int32_t *out) {
    int64_t x;
    if (!v || v->type != J_NUM || json_int(v, &x) || x < 0 || (uint64_t)x >= vocab) return -1;
    *out = (int32_t)x;
    return 0;
}

static int parse_gen(gen_cfg *g, const jval *root, size_t vocab, char *err, size_t cap) {
    memset(g, 0, sizeof *g);
    if (root->type != J_OBJ) return fail(err, cap, "generation_config.json 应为一个对象");
    for (size_t i = 0; i < root->len; i++) {
        size_t k = 0;
        while (k < sizeof KNOWN / sizeof KNOWN[0] && strcmp(root->keys[i].str, KNOWN[k])) k++;
        if (k == sizeof KNOWN / sizeof KNOWN[0])
            return fail(err, cap, "generation_config.json 的字段 %s 不支持", root->keys[i].str);
    }
    int32_t id;
    const char *ids[] = {"bos_token_id", "pad_token_id"};          /* 只检查，不用 */
    for (size_t i = 0; i < 2; i++)
        if (json_get(root, ids[i]) && token_id(json_get(root, ids[i]), vocab, &id))
            return fail(err, cap, "generation_config.json 的 %s 应为小于词表大小的非负整数", ids[i]);

    const jval *eos = json_get(root, "eos_token_id");
    if (!eos) return fail(err, cap, "generation_config.json 缺少 eos_token_id");
    if (eos->type == J_ARR) {
        if (eos->len == 0 || eos->len > GEN_MAX_EOS)
            return fail(err, cap, "eos_token_id 应有 1 到 %d 个", GEN_MAX_EOS);
        for (size_t i = 0; i < eos->len; i++)
            if (token_id(&eos->items[i], vocab, &g->eos[g->n_eos++]))
                return fail(err, cap, "eos_token_id 的第 %zu 项应为小于词表大小的非负整数", i);
    } else if (token_id(eos, vocab, &g->eos[g->n_eos++])) {
        return fail(err, cap, "eos_token_id 应为小于词表大小的非负整数或这样的数组");
    }

    sample_cfg *s = &g->sample;
    const jval *ds = json_get(root, "do_sample");
    if (!ds || (ds->type != J_TRUE && ds->type != J_FALSE)) return fail(err, cap, "generation_config.json 的 do_sample 应为 true 或 false");
    s->do_sample = ds->type == J_TRUE;
    s->temperature = 1; s->top_k = 0; s->top_p = 1;
    const jval *t = json_get(root, "temperature"), *k = json_get(root, "top_k"), *p = json_get(root, "top_p");
    if (s->do_sample && (!t || !k || !p))
        return fail(err, cap, "do_sample 为 true 时 temperature、top_k、top_p 都要给出");
    char *end;
    if (t && (t->type != J_NUM || (s->temperature = strtod(t->str, &end), *end) || !(s->temperature > 0) ||
              !isfinite(s->temperature)))
        return fail(err, cap, "temperature 应为正数");
    int64_t kv;
    if (k && (k->type != J_NUM || json_int(k, &kv) || kv < 0)) return fail(err, cap, "top_k 应为非负整数");
    if (k) s->top_k = (size_t)kv;
    if (p && (p->type != J_NUM || (s->top_p = strtod(p->str, &end), *end) || !(s->top_p > 0 && s->top_p <= 1)))
        return fail(err, cap, "top_p 应在 (0, 1] 之间");
    const jval *tv = json_get(root, "transformers_version");
    if (tv && tv->type != J_STR) return fail(err, cap, "transformers_version 应为字符串");
    return 0;
}

int gen_cfg_load(gen_cfg *g, const char *path, size_t vocab, char *err, size_t cap) {
    jval root;
    if (json_parse_file(&root, path, err, cap)) return -1;
    int rc = parse_gen(g, &root, vocab, err, cap);
    json_free(&root);
    return rc;
}

/* ---------- 生成循环 ---------- */

static int is_eos(const gen_cfg *g, int32_t t) {
    for (size_t i = 0; i < g->n_eos; i++) if (g->eos[i] == t) return 1;
    return 0;
}

int generate(const model *m, const gen_cfg *g, const gen_opts *o, const int32_t *prompt, size_t n,
             int32_t *out, size_t *nout, gen_stop *why, gen_fn cb, void *ctx, char *err, size_t cap) {
    const model_cfg *c = &m->cfg;
    size_t V = c->vocab;
    *nout = 0;
    if (n == 0) return fail(err, cap, "提示词至少要有一个 token");
    if (o->max_new == 0) return fail(err, cap, "max_new 至少是 1");
    if (o->max_new > c->max_pos || n > c->max_pos - o->max_new)
        return fail(err, cap, "提示词 %zu 个 token 加上最多生成 %zu 个，超过 max_position_embeddings（%zu）",
                    n, o->max_new, c->max_pos);
    if (sample_cfg_check(&g->sample, err, cap)) return -1;

    size_t total = n + o->max_new, chunk = o->chunk ? o->chunk : n;
    int32_t *seq = malloc(total * sizeof *seq), *kids = malloc(V * sizeof *kids);
    double *probs = malloc(V * sizeof *probs);
    /* 不用缓存时每次前向都要放下全部 token 的 logits；用缓存时一次最多 chunk 个 */
    size_t rows = o->no_cache ? total : (chunk > 1 ? chunk : 1);
    float *logits = malloc(rows * V * sizeof *logits);
    model_cache kc = {0};
    int rc = -1;
    if (!seq || !kids || !probs || !logits) { fail(err, cap, "内存不足"); goto done; }
    memcpy(seq, prompt, n * sizeof *seq);
    if (!o->no_cache && model_cache_init(&kc, c, total, err, cap)) goto done;

    rng64 r;
    rng_seed(&r, o->seed);
    size_t len = n;                     /* seq 里已有的 token 数 */
    const float *last = NULL;           /* 选下一个 token 时看的那一行 */
    if (!o->no_cache) {                 /* 预填充：提示词分块交给模型，只有最后一块的最后一行有用 */
        for (size_t s = 0; s < n; s += chunk) {
            size_t k = n - s < chunk ? n - s : chunk;
            if (model_step(m, &kc, seq + s, k, logits, NULL, NULL, err, cap)) goto done;
            last = logits + (k - 1) * V;
        }
    }
    for (;;) {
        if (o->no_cache) {              /* 第 5 章的做法：每一步都从头算 */
            if (model_forward(m, seq, len, logits, NULL, NULL, err, cap)) goto done;
            last = logits + (len - 1) * V;
        }
        size_t nk;
        if (sample_dist(last, V, &g->sample, kids, probs, &nk, err, cap)) goto done;
        double u = g->sample.do_sample ? rng_uniform(&r) : -1;
        int32_t tok = g->sample.do_sample ? sample_pick(kids, probs, nk, u) : kids[0];
        if (cb) cb(ctx, *nout, last, V, kids, probs, nk, u, tok);
        out[(*nout)++] = tok;
        seq[len++] = tok;
        if (is_eos(g, tok)) { *why = GEN_EOS; break; }
        if (*nout == o->max_new) { *why = GEN_LENGTH; break; }
        if (!o->no_cache) {             /* 解码：只把新 token 交给模型 */
            if (model_step(m, &kc, &tok, 1, logits, NULL, NULL, err, cap)) goto done;
            last = logits;
        }
    }
    rc = 0;
done:
    if (!o->no_cache) model_cache_free(&kc);
    free(seq); free(kids); free(probs); free(logits);
    return rc;
}
