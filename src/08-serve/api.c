/* api.c — 第 8 章：OpenAI 兼容接口的请求解析和应答格式。 */
#include "api.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "chat.h"
#include "json.h"
#include "unicode.h"

/* ---------- 缓冲区 ---------- */

void sb_put(sbuf *b, const void *s, size_t n) {
    if (b->oom) return;
    if (b->n + n + 1 > b->cap) {
        size_t nc = b->cap ? b->cap : 256;
        while (b->n + n + 1 > nc) nc *= 2;
        char *ns = realloc(b->s, nc);
        if (!ns) { b->oom = 1; return; }
        b->s = ns; b->cap = nc;
    }
    if (n) memcpy(b->s + b->n, s, n);
    b->n += n;
    b->s[b->n] = 0;
}

void sb_puts(sbuf *b, const char *s) { sb_put(b, s, strlen(s)); }

void sb_printf(sbuf *b, const char *fmt, ...) {
    char tmp[256];
    va_list ap;
    va_start(ap, fmt);
    int k = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if (k < 0 || (size_t)k >= sizeof tmp) { b->oom = 1; return; }   /* 只用来写短的东西 */
    sb_put(b, tmp, (size_t)k);
}

void sb_json_str(sbuf *b, const char *s, size_t n) {
    sb_put(b, "\"", 1);
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"') sb_put(b, "\\\"", 2);
        else if (c == '\\') sb_put(b, "\\\\", 2);
        else if (c == '\n') sb_put(b, "\\n", 2);
        else if (c == '\r') sb_put(b, "\\r", 2);
        else if (c == '\t') sb_put(b, "\\t", 2);
        else if (c < 0x20 || c == 0x7f) sb_printf(b, "\\u%04x", c);   /* 其余控制字符 */
        else sb_put(b, &s[i], 1);                                       /* UTF-8 原样输出 */
    }
    sb_put(b, "\"", 1);
}

void sb_free(sbuf *b) { free(b->s); memset(b, 0, sizeof *b); }

/* ---------- 请求 ---------- */

static int fail(int *status, int code, char *err, size_t cap, const char *fmt, ...) {
    *status = code;
    if (err && cap) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, cap, fmt, ap);
        va_end(ap);
    }
    return -1;
}

static const char *const FIELDS[] = {"model", "messages", "stream", "stream_options", "max_tokens",
                                     "max_completion_tokens", "temperature", "top_p", "top_k", "seed", "n",
                                     "chat_template_kwargs"};
static const char *const MSG_FIELDS[] = {"role", "content", "reasoning_content", "tool_calls"};
static const char *const KWARGS[] = {"enable_thinking", "reasoning_effort", "preserve_thinking"};

static int known(const char *k, const char *const *list, size_t n) {
    for (size_t i = 0; i < n; i++) if (!strcmp(k, list[i])) return 1;
    return 0;
}
#define KNOWN(k, list) known(k, list, sizeof list / sizeof list[0])

static int get_bool(const jval *v, int *out) {
    if (v->type != J_TRUE && v->type != J_FALSE) return -1;
    *out = v->type == J_TRUE;
    return 0;
}

static int get_real(const jval *v, double *out) {
    char *end;
    if (v->type != J_NUM) return -1;
    *out = strtod(v->str, &end);
    return *end || !isfinite(*out) ? -1 : 0;
}

static int get_count(const jval *v, int64_t lo, int64_t *out) {
    return v->type != J_NUM || json_int(v, out) || *out < lo ? -1 : 0;
}

/* chat_render 要的请求对象：{"messages": ..., "add_generation_prompt": true, 以及 chat_template_kwargs 里的} */
static char K_MESSAGES[] = "messages", K_AGP[] = "add_generation_prompt";

static int render(const jval *msgs, const jval *kw, api_chat *out, int *status, char *err, size_t cap) {
    size_t nk = 2 + (kw ? kw->len : 0);
    jval *keys = calloc(nk, sizeof *keys), *items = calloc(nk, sizeof *items);
    if (!keys || !items) { free(keys); free(items); return fail(status, 500, err, cap, "内存不足"); }
    keys[0] = (jval){.type = J_STR, .str = K_MESSAGES, .len = strlen(K_MESSAGES)};
    items[0] = *msgs;                        /* 浅拷贝，指针仍指向原来的树 */
    keys[1] = (jval){.type = J_STR, .str = K_AGP, .len = strlen(K_AGP)};
    items[1] = (jval){.type = J_TRUE};
    for (size_t i = 0; kw && i < kw->len; i++) { keys[2 + i] = kw->keys[i]; items[2 + i] = kw->items[i]; }
    jval req = {.type = J_OBJ, .len = nk, .keys = keys, .items = items};
    int rc = chat_render(&req, &out->prompt, &out->prompt_len, err, cap);
    free(keys); free(items);
    if (rc) *status = 400;
    return rc;
}

static int parse(const jval *root, const char *model_name, const sample_cfg *def, api_chat *out,
                 int *status, char *err, size_t cap) {
    if (root->type != J_OBJ) return fail(status, 400, err, cap, "请求正文应为 JSON 对象");
    for (size_t i = 0; i < root->len; i++)
        if (!KNOWN(root->keys[i].str, FIELDS))
            return fail(status, 400, err, cap, "不支持字段 %s", root->keys[i].str);

    const jval *v = json_get(root, "model");
    if (!v || v->type != J_STR) return fail(status, 400, err, cap, "缺少字符串类型的 model");
    if (strcmp(v->str, model_name) || strlen(model_name) != v->len)
        return fail(status, 404, err, cap, "没有模型 %s，这里只有 %s", v->str, model_name);

    const jval *msgs = json_get(root, "messages");
    if (!msgs || msgs->type != J_ARR) return fail(status, 400, err, cap, "缺少数组类型的 messages");
    for (size_t i = 0; i < msgs->len; i++) {
        const jval *m = &msgs->items[i];
        if (m->type != J_OBJ) return fail(status, 400, err, cap, "messages[%zu] 应为对象", i);
        for (size_t k = 0; k < m->len; k++)
            if (!KNOWN(m->keys[k].str, MSG_FIELDS))
                return fail(status, 400, err, cap, "messages[%zu] 不支持字段 %s", i, m->keys[k].str);
    }

    out->sample = *def;
    if ((v = json_get(root, "stream")) && get_bool(v, &out->stream))
        return fail(status, 400, err, cap, "stream 应为 true 或 false");
    if ((v = json_get(root, "stream_options"))) {
        if (!out->stream) return fail(status, 400, err, cap, "stream_options 只能在 stream 为 true 时给出");
        if (v->type != J_OBJ) return fail(status, 400, err, cap, "stream_options 应为对象");
        for (size_t k = 0; k < v->len; k++)
            if (strcmp(v->keys[k].str, "include_usage"))
                return fail(status, 400, err, cap, "stream_options 不支持字段 %s", v->keys[k].str);
        const jval *u = json_get(v, "include_usage");
        if (u && get_bool(u, &out->include_usage))
            return fail(status, 400, err, cap, "include_usage 应为 true 或 false");
    }
    const jval *mt = json_get(root, "max_tokens"), *mct = json_get(root, "max_completion_tokens");
    if (mt && mct) return fail(status, 400, err, cap, "max_tokens 和 max_completion_tokens 只能给一个");
    if (mt || mct) {
        int64_t x;
        if (get_count(mt ? mt : mct, 1, &x) || x > (int64_t)1 << 30)
            return fail(status, 400, err, cap, "%s 应为正整数", mt ? "max_tokens" : "max_completion_tokens");
        out->max_tokens = (size_t)x;
    }
    if ((v = json_get(root, "temperature"))) {
        double t;
        if (get_real(v, &t) || t < 0 || t > 2) return fail(status, 400, err, cap, "temperature 应在 0 到 2 之间");
        if (t == 0) out->sample.do_sample = 0;            /* OpenAI 的约定：0 就是贪心 */
        else { out->sample.do_sample = 1; out->sample.temperature = t; }
    }
    if ((v = json_get(root, "top_p"))) {
        double p;
        if (get_real(v, &p) || !(p > 0 && p <= 1)) return fail(status, 400, err, cap, "top_p 应在 (0, 1] 之间");
        out->sample.top_p = p;
    }
    if ((v = json_get(root, "top_k"))) {
        int64_t k;
        if (get_count(v, 0, &k) || k > (int64_t)1 << 30) return fail(status, 400, err, cap, "top_k 应为非负整数");
        out->sample.top_k = (size_t)k;
    }
    if ((v = json_get(root, "seed"))) {
        int64_t s;
        if (get_count(v, 0, &s)) return fail(status, 400, err, cap, "seed 应为非负整数");
        out->seed = (uint64_t)s;
    }
    if ((v = json_get(root, "n"))) {
        int64_t k;
        if (v->type != J_NUM || json_int(v, &k) || k != 1) return fail(status, 400, err, cap, "n 只支持 1");
    }
    if (sample_cfg_check(&out->sample, err, cap)) { *status = 400; return -1; }

    const jval *kw = json_get(root, "chat_template_kwargs");
    if (kw) {
        if (kw->type != J_OBJ) return fail(status, 400, err, cap, "chat_template_kwargs 应为对象");
        for (size_t k = 0; k < kw->len; k++)
            if (!KNOWN(kw->keys[k].str, KWARGS))
                return fail(status, 400, err, cap, "chat_template_kwargs 不支持字段 %s", kw->keys[k].str);
    }
    return render(msgs, kw, out, status, err, cap);
}

int api_parse_chat(const char *body, size_t n, const char *model_name, const sample_cfg *def,
                   api_chat *out, int *status, char *err, size_t cap) {
    memset(out, 0, sizeof *out);
    jval root;
    char jerr[256];
    if (json_parse(&root, body, n, jerr, sizeof jerr)) return fail(status, 400, err, cap, "正文不是合法的 JSON：%s", jerr);
    int rc = parse(&root, model_name, def, out, status, err, cap);
    json_free(&root);
    if (rc) api_chat_free(out);
    return rc;
}

void api_chat_free(api_chat *c) { free(c->prompt); c->prompt = NULL; }

/* ---------- 增量解码 ---------- */

/* 从 s[i] 开始的 n - i 个字节，是不是一个合法字符的开头、只是还没完（后面再来几个字节就可能是合法的）。
 * 每个字节的合法范围与 unicode.c 的 utf8_lossy 相同。 */
static int incomplete(const uint8_t *s, size_t i, size_t n) {
    uint8_t b = s[i], lo = 0x80, hi = 0xBF;
    size_t need;
    if (b >= 0xC2 && b <= 0xDF) need = 1;
    else if (b >= 0xE0 && b <= 0xEF) { need = 2; if (b == 0xE0) lo = 0xA0; if (b == 0xED) hi = 0x9F; }
    else if (b >= 0xF0 && b <= 0xF4) { need = 3; if (b == 0xF0) lo = 0x90; if (b == 0xF4) hi = 0x8F; }
    else return 0;
    if (n - i - 1 >= need) return 0;                  /* 字节够了，不是“没完” */
    for (size_t k = i + 1; k < n; k++) {
        uint8_t l = k == i + 1 ? lo : 0x80, h = k == i + 1 ? hi : 0xBF;
        if (s[k] < l || s[k] > h) return 0;
    }
    return 1;
}

void utf8_stream(uint8_t *pend, size_t *n, sbuf *out, int final) {
    size_t keep = *n;                                 /* 从哪里开始留下 */
    if (!final)                                       /* 末尾最多 3 个字节可能是半个字 */
        for (size_t back = 1; back <= 3 && back <= *n; back++)
            if (incomplete(pend, *n - back, *n)) { keep = *n - back; break; }
    size_t ln;
    uint8_t *l = utf8_lossy(pend, keep, &ln);
    if (!l) { out->oom = 1; return; }
    sb_put(out, l, ln);
    free(l);
    memmove(pend, pend + keep, *n - keep);
    *n -= keep;
}

/* ---------- 应答 ---------- */

void api_error(sbuf *b, int status, const char *msg) {
    const char *type = status == 404 ? "not_found_error" : status >= 500 ? "server_error" : "invalid_request_error";
    sb_puts(b, "{\"error\":{\"message\":");
    sb_json_str(b, msg, strlen(msg));
    sb_printf(b, ",\"type\":\"%s\",\"param\":null,\"code\":%d}}", type, status);
}

static void head(sbuf *b, const char *id, const char *object, long created, const char *model) {
    sb_puts(b, "{\"id\":");
    sb_json_str(b, id, strlen(id));
    sb_printf(b, ",\"object\":\"%s\",\"created\":%ld,\"model\":", object, created);
    sb_json_str(b, model, strlen(model));
}

static void usage(sbuf *b, size_t p, size_t c) {
    sb_printf(b, "\"usage\":{\"prompt_tokens\":%zu,\"completion_tokens\":%zu,\"total_tokens\":%zu}", p, c, p + c);
}

void api_completion(sbuf *b, const char *id, long created, const char *model, const char *content, size_t len,
                    const char *finish, size_t prompt_tokens, size_t completion_tokens) {
    head(b, id, "chat.completion", created, model);
    sb_puts(b, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    sb_json_str(b, content, len);
    sb_printf(b, "},\"finish_reason\":\"%s\"}],", finish);
    usage(b, prompt_tokens, completion_tokens);
    sb_puts(b, "}");
}

void api_chunk(sbuf *b, const char *id, long created, const char *model, const char *role,
               const char *content, size_t len, const char *finish) {
    sb_puts(b, "data: ");
    head(b, id, "chat.completion.chunk", created, model);
    sb_puts(b, ",\"choices\":[{\"index\":0,\"delta\":{");
    if (role) sb_printf(b, "\"role\":\"%s\"%s", role, content ? "," : "");
    if (content) { sb_puts(b, "\"content\":"); sb_json_str(b, content, len); }
    sb_puts(b, "},\"finish_reason\":");
    if (finish) sb_printf(b, "\"%s\"", finish); else sb_puts(b, "null");
    sb_puts(b, "}]}\n\n");
}

void api_usage_chunk(sbuf *b, const char *id, long created, const char *model, size_t prompt_tokens,
                     size_t completion_tokens) {
    sb_puts(b, "data: ");
    head(b, id, "chat.completion.chunk", created, model);
    sb_puts(b, ",\"choices\":[],");
    usage(b, prompt_tokens, completion_tokens);
    sb_puts(b, "}\n\n");
}

void api_models(sbuf *b, const char *model, long created) {
    sb_puts(b, "{\"object\":\"list\",\"data\":[{\"id\":");
    sb_json_str(b, model, strlen(model));
    sb_printf(b, ",\"object\":\"model\",\"created\":%ld,\"owned_by\":\"local\"}]}", created);
}
