/* engine.c — 第 8 章：连续批处理。 */
#include "engine.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(char *err, size_t cap, const char *fmt, ...) {
    if (err && cap) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, cap, fmt, ap);
        va_end(ap);
    }
    return -1;
}

struct eng_req {
    uint64_t   id;
    eng_params p;
    int32_t   *prompt;
    size_t     n;           /* 提示词的 token 数 */
    size_t     fed;         /* 提示词已经交给模型几个 */
    int        running;     /* 0：还在排队；1：在这一批里，有缓存 */
    model_cache kc;
    rng64      rng;
    size_t     nout;        /* 已经生成了几个 */
    int32_t    last;        /* 上一步选出的 token，下一步交给模型 */
};

static void req_free(eng_req *r) {
    if (!r) return;
    if (r->running) model_cache_free(&r->kc);
    free(r->prompt);
    free(r);
}

int engine_init(engine *e, const model *m, const gen_cfg *g, size_t max_batch, size_t max_ctx,
                eng_token_fn on_token, eng_done_fn on_done, void *ctx, char *err, size_t cap) {
    memset(e, 0, sizeof *e);
    if (max_batch == 0) return fail(err, cap, "max_batch 至少是 1");
    if (max_ctx < 2 || max_ctx > m->cfg.max_pos)
        return fail(err, cap, "max_ctx 应在 2 到 max_position_embeddings（%zu）之间", m->cfg.max_pos);
    e->m = m; e->g = g; e->max_batch = max_batch; e->max_ctx = max_ctx;
    e->on_token = on_token; e->on_done = on_done; e->ctx = ctx;
    e->kids = malloc(m->cfg.vocab * sizeof *e->kids);
    e->probs = malloc(m->cfg.vocab * sizeof *e->probs);
    e->last_ids = malloc(max_batch * sizeof *e->last_ids);
    e->last_n = malloc(max_batch * sizeof *e->last_n);
    if (!e->kids || !e->probs || !e->last_ids || !e->last_n) { engine_free(e); return fail(err, cap, "内存不足"); }
    return 0;
}

void engine_free(engine *e) {
    for (size_t i = 0; i < e->nreq; i++) req_free(e->reqs[i]);
    free(e->reqs); free(e->logits); free(e->kids); free(e->probs); free(e->last_ids); free(e->last_n);
    memset(e, 0, sizeof *e);
}

size_t engine_pending(const engine *e) { return e->nreq; }

int engine_add(engine *e, const int32_t *prompt, size_t n, const eng_params *p, uint64_t *id,
               char *err, size_t cap) {
    const model_cfg *c = &e->m->cfg;
    if (n == 0) return fail(err, cap, "提示词至少要有一个 token");
    if (p->max_new == 0) return fail(err, cap, "max_new 至少是 1");
    if (p->max_new > e->max_ctx || n > e->max_ctx - p->max_new)
        return fail(err, cap, "提示词 %zu 个 token 加上最多生成 %zu 个，超过上限 %zu", n, p->max_new, e->max_ctx);
    for (size_t t = 0; t < n; t++)
        if (prompt[t] < 0 || (uint64_t)prompt[t] >= c->vocab)
            return fail(err, cap, "第 %zu 个 token id %d 超出词表范围 [0, %zu)", t, prompt[t], c->vocab);
    if (sample_cfg_check(&p->sample, err, cap)) return -1;
    if (e->nreq == e->capreq) {
        size_t nc = e->capreq ? 2 * e->capreq : 16;
        eng_req **nr = realloc(e->reqs, nc * sizeof *nr);
        if (!nr) return fail(err, cap, "内存不足");
        e->reqs = nr; e->capreq = nc;
    }
    eng_req *r = calloc(1, sizeof *r);
    if (!r || !(r->prompt = malloc(n * sizeof *r->prompt))) { free(r); return fail(err, cap, "内存不足"); }
    memcpy(r->prompt, prompt, n * sizeof *prompt);
    r->id = e->next_id++;
    r->p = *p;
    r->n = n;
    rng_seed(&r->rng, p->seed);
    e->reqs[e->nreq++] = r;
    *id = r->id;
    return 0;
}

/* 把第 i 个请求从表里拿掉（保持其余的顺序）并释放 */
static void remove_at(engine *e, size_t i) {
    req_free(e->reqs[i]);
    memmove(e->reqs + i, e->reqs + i + 1, (e->nreq - i - 1) * sizeof *e->reqs);
    e->nreq--;
}

int engine_cancel(engine *e, uint64_t id) {
    for (size_t i = 0; i < e->nreq; i++)
        if (e->reqs[i]->id == id) { remove_at(e, i); return 0; }
    return -1;
}

/* 结束一个请求：先调用回调，再拿掉 */
static void finish(engine *e, size_t i, eng_stop why, const char *msg) {
    if (e->on_done) e->on_done(e->ctx, e->reqs[i]->id, why, msg ? msg : "");
    remove_at(e, i);
}

static int is_eos(const gen_cfg *g, int32_t t) {
    for (size_t i = 0; i < g->n_eos; i++) if (g->eos[i] == t) return 1;
    return 0;
}

int engine_step(engine *e, char *err, size_t cap) {
    const model_cfg *c = &e->m->cfg;
    size_t V = c->vocab;
    char msg[512];
    e->last_nseq = 0;

    /* 一、按到达顺序把排队的请求补进来，直到这一批有 max_batch 个。补进来时才建缓存。 */
    size_t running = 0;
    for (size_t i = 0; i < e->nreq; i++) running += (size_t)e->reqs[i]->running;
    for (size_t i = 0; i < e->nreq && running < e->max_batch; i++) {
        eng_req *r = e->reqs[i];
        if (r->running) continue;
        if (model_cache_init(&r->kc, e->m, r->n + r->p.max_new, msg, sizeof msg)) {
            finish(e, i--, ENG_ERROR, msg);
            continue;
        }
        r->running = 1;
        running++;
    }
    if (running == 0) return 0;

    /* 二、每个请求交上这一步的 token */
    model_seq *seqs = malloc(running * sizeof *seqs);
    size_t *idx = malloc(running * sizeof *idx), rows = 0, ns = 0;
    if (!seqs || !idx) { free(seqs); free(idx); return fail(err, cap, "内存不足"); }
    for (size_t i = 0; i < e->nreq; i++) {
        eng_req *r = e->reqs[i];
        if (!r->running) continue;
        model_seq *s = &seqs[ns];
        s->kc = &r->kc;
        if (r->fed < r->n) {                       /* 预填充：提示词的下一块 */
            size_t left = r->n - r->fed, k = r->p.chunk && r->p.chunk < left ? r->p.chunk : left;
            s->ids = r->prompt + r->fed;
            s->n = k;
        } else {                                   /* 解码：上一步选出的 token */
            s->ids = &r->last;
            s->n = 1;
        }
        e->last_ids[ns] = r->id;
        e->last_n[ns] = s->n;
        idx[ns++] = i;
        rows += s->n;
    }
    e->last_nseq = ns;
    if (rows > e->logit_rows) {
        float *nl = realloc(e->logits, rows * V * sizeof *nl);
        if (!nl) { free(seqs); free(idx); return fail(err, cap, "内存不足"); }
        e->logits = nl; e->logit_rows = rows;
    }

    /* 三、一次前向 */
    int bad = model_step_batch(e->m, seqs, ns, e->logits, NULL, NULL, msg, sizeof msg);

    /* 四、预填充完的请求按顺序各选一个 token。停下的先记下来，最后再拿掉，免得下标变动。 */
    int *stop = calloc(ns, sizeof *stop);         /* 0：继续；1 + eng_stop：停下 */
    char (*why)[512] = bad ? NULL : calloc(ns, sizeof *why);
    if (!stop || (!bad && !why)) { free(seqs); free(idx); free(stop); free(why); return fail(err, cap, "内存不足"); }
    for (size_t b = 0, off = 0; b < ns; off += seqs[b++].n) {
        eng_req *r = e->reqs[idx[b]];
        if (bad) { stop[b] = 1 + ENG_ERROR; continue; }
        if (r->fed < r->n) {
            r->fed += seqs[b].n;
            if (r->fed < r->n) continue;           /* 提示词还没交完，这一步不选 */
        }
        const float *last = e->logits + (off + seqs[b].n - 1) * V;   /* 这个序列的最后一行 */
        size_t nk;
        if (sample_dist(last, V, &r->p.sample, e->kids, e->probs, &nk, why[b], sizeof why[b])) {
            stop[b] = 1 + ENG_ERROR;
            continue;
        }
        double u = r->p.sample.do_sample ? rng_uniform(&r->rng) : -1;
        int32_t tok = r->p.sample.do_sample ? sample_pick(e->kids, e->probs, nk, u) : e->kids[0];
        if (e->on_token) e->on_token(e->ctx, r->id, r->nout, last, V, e->kids, e->probs, nk, u, tok);
        r->nout++;
        r->last = tok;
        if (is_eos(e->g, tok)) stop[b] = 1 + ENG_EOS;
        else if (r->nout == r->p.max_new) stop[b] = 1 + ENG_LENGTH;
    }
    for (size_t b = 0; b < ns; b++)               /* 先按顺序通知 */
        if (stop[b] && e->on_done)
            e->on_done(e->ctx, e->reqs[idx[b]]->id, (eng_stop)(stop[b] - 1), bad ? msg : stop[b] - 1 == ENG_ERROR ? why[b] : "");
    for (size_t b = ns; b-- > 0;)                 /* 再倒着拿掉，前面的下标不受影响 */
        if (stop[b]) remove_at(e, idx[b]);
    free(stop); free(why);
    free(seqs); free(idx);
    return 0;
}
