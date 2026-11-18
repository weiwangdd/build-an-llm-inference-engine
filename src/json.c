/* json.c — 按 RFC 8259 解析 JSON。原则同第 1 章：不合规就报错，不猜。 */
#include "json.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *s, *p, *end;
    char *err; size_t errcap;
} P;

static int fail(P *ps, const char *fmt, ...) {
    if (ps->err && ps->errcap) {
        int line = 1;
        for (const char *q = ps->s; q < ps->p && q < ps->end; q++) line += *q == '\n';
        int k = snprintf(ps->err, ps->errcap, "JSON 第 %d 行：", line);
        if (k >= 0 && (size_t)k < ps->errcap) {
            va_list ap; va_start(ap, fmt);
            vsnprintf(ps->err + k, ps->errcap - (size_t)k, fmt, ap);
            va_end(ap);
        }
    }
    return -1;
}

static void ws(P *ps) {
    while (ps->p < ps->end && (*ps->p == ' ' || *ps->p == '\t' || *ps->p == '\n' || *ps->p == '\r')) ps->p++;
}

static int hex4(P *ps, unsigned *out) {
    if (ps->end - ps->p < 4) return fail(ps, "\\u 转义不完整");
    unsigned v = 0;
    for (int i = 0; i < 4; i++) {
        char c = *ps->p++;
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
        else return fail(ps, "\\u 转义含非十六进制字符");
    }
    *out = v;
    return 0;
}

static size_t put_utf8(char *d, unsigned cp) {
    if (cp < 0x80)    { d[0] = (char)cp; return 1; }
    if (cp < 0x800)   { d[0] = (char)(0xC0 | cp >> 6); d[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) { d[0] = (char)(0xE0 | cp >> 12); d[1] = (char)(0x80 | (cp >> 6 & 0x3F));
                        d[2] = (char)(0x80 | (cp & 0x3F)); return 3; }
    d[0] = (char)(0xF0 | cp >> 18); d[1] = (char)(0x80 | (cp >> 12 & 0x3F));
    d[2] = (char)(0x80 | (cp >> 6 & 0x3F)); d[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/* 原文里的非 ASCII 字节必须是合法 UTF-8。只检查结构，码点范围检查与 unicode.c 相同。 */
static int utf8_len(const unsigned char *p, const char *end) {
    unsigned c = p[0], v; int n; unsigned min;
    if (c < 0xC0) return 0;
    if (c < 0xE0) { n = 2; v = c & 0x1F; min = 0x80; }
    else if (c < 0xF0) { n = 3; v = c & 0x0F; min = 0x800; }
    else if (c < 0xF8) { n = 4; v = c & 0x07; min = 0x10000; }
    else return 0;
    if (end - (const char *)p < n) return 0;
    for (int i = 1; i < n; i++) { if ((p[i] & 0xC0) != 0x80) return 0; v = v << 6 | (p[i] & 0x3F); }
    if (v < min || v > 0x10FFFF || (v >= 0xD800 && v <= 0xDFFF)) return 0;
    return n;
}

static int parse_string(P *ps, jval *out) {
    ps->p++;                                     /* 跳过开头的引号 */
    const char *q = ps->p;
    while (q < ps->end && *q != '"') q += (*q == '\\' && q + 1 < ps->end) ? 2 : 1;
    char *buf = malloc((size_t)(q - ps->p) + 1); /* 解码后不会比原文长 */
    if (!buf) return fail(ps, "内存不足");
    size_t n = 0;
    while (ps->p < ps->end) {
        unsigned char c = (unsigned char)*ps->p;
        if (c == '"') {
            ps->p++; buf[n] = 0;
            out->type = J_STR; out->str = buf; out->len = n;
            return 0;
        }
        if (c < 0x20) { free(buf); return fail(ps, "字符串里有未转义的控制字符"); }
        if (c >= 0x80) {
            int k = utf8_len((const unsigned char *)ps->p, ps->end);
            if (!k) { free(buf); return fail(ps, "非法 UTF-8"); }
            memcpy(buf + n, ps->p, (size_t)k); n += (size_t)k; ps->p += k;
            continue;
        }
        ps->p++;
        if (c != '\\') { buf[n++] = (char)c; continue; }
        if (ps->p >= ps->end) break;
        char e = *ps->p++;
        unsigned cp, lo;
        switch (e) {
        case '"': case '\\': case '/': buf[n++] = e; break;
        case 'b': buf[n++] = '\b'; break;
        case 'f': buf[n++] = '\f'; break;
        case 'n': buf[n++] = '\n'; break;
        case 'r': buf[n++] = '\r'; break;
        case 't': buf[n++] = '\t'; break;
        case 'u':
            if (hex4(ps, &cp)) { free(buf); return -1; }
            if (cp >= 0xD800 && cp <= 0xDBFF) {  /* 代理对 */
                if (ps->end - ps->p < 2 || ps->p[0] != '\\' || ps->p[1] != 'u') { free(buf); return fail(ps, "代理对不完整"); }
                ps->p += 2;
                if (hex4(ps, &lo)) { free(buf); return -1; }
                if (lo < 0xDC00 || lo > 0xDFFF) { free(buf); return fail(ps, "代理对不完整"); }
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            } else if (cp >= 0xDC00 && cp <= 0xDFFF) { free(buf); return fail(ps, "孤立的低代理"); }
            n += put_utf8(buf + n, cp);
            break;
        default: free(buf); return fail(ps, "未知的转义 \\%c", e);
        }
    }
    free(buf);
    return fail(ps, "字符串没有结束");
}

/* 数字：-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)? */
static int parse_number(P *ps, jval *out) {
    const char *b = ps->p, *p = ps->p, *e = ps->end;
#define DIG(x) ((x) < e && *(x) >= '0' && *(x) <= '9')
    if (p < e && *p == '-') p++;
    if (!DIG(p)) return fail(ps, "非法数字");
    if (*p == '0') p++; else while (DIG(p)) p++;
    if (p < e && *p == '.') { p++; if (!DIG(p)) return fail(ps, "非法数字"); while (DIG(p)) p++; }
    if (p < e && (*p == 'e' || *p == 'E')) {
        p++;
        if (p < e && (*p == '+' || *p == '-')) p++;
        if (!DIG(p)) return fail(ps, "非法数字");
        while (DIG(p)) p++;
    }
#undef DIG
    out->type = J_NUM; out->len = (size_t)(p - b);
    out->str = malloc(out->len + 1);
    if (!out->str) return fail(ps, "内存不足");
    memcpy(out->str, b, out->len); out->str[out->len] = 0;
    ps->p = p;
    return 0;
}

static int lit(P *ps, const char *w, jtype t, jval *out) {
    size_t n = strlen(w);
    if ((size_t)(ps->end - ps->p) < n || memcmp(ps->p, w, n)) return fail(ps, "期望一个值");
    ps->p += n; out->type = t;
    return 0;
}

/* 用一个可增长数组收集元素 */
static int push(P *ps, jval **a, size_t *n, size_t *cap, const jval *v) {
    if (*n == *cap) {
        size_t nc = *cap ? *cap * 2 : 4;
        jval *t = realloc(*a, nc * sizeof **a);
        if (!t) return fail(ps, "内存不足");
        *a = t; *cap = nc;
    }
    (*a)[(*n)++] = *v;
    return 0;
}

static int key_cmp(const void *x, const void *y) {
    const jval *a = *(const jval *const *)x, *b = *(const jval *const *)y;
    size_t n = a->len < b->len ? a->len : b->len;
    int c = memcmp(a->str, b->str, n);
    return c ? c : (a->len > b->len) - (a->len < b->len);
}

static int parse_value(P *ps, jval *out, int depth);

static int parse_container(P *ps, jval *out, int depth) {
    int obj = *ps->p == '{';
    char close = obj ? '}' : ']';
    ps->p++;
    out->type = obj ? J_OBJ : J_ARR;
    size_t cap = 0, kcap = 0, kn = 0;
    ws(ps);
    if (ps->p < ps->end && *ps->p == close) { ps->p++; return 0; }
    for (;;) {
        jval k = {0}, v = {0};
        if (obj) {
            ws(ps);
            if (ps->p >= ps->end || *ps->p != '"') return fail(ps, "期望字符串作为键");
            if (parse_string(ps, &k)) return -1;
            ws(ps);
            if (ps->p >= ps->end || *ps->p != ':') { json_free(&k); return fail(ps, "期望 ':'"); }
            ps->p++;
        }
        if (parse_value(ps, &v, depth + 1)) { json_free(&k); json_free(&v); return -1; }
        /* 键和值成对加入，保证出错时 json_free 也能按 len 正确释放 */
        if (obj && push(ps, &out->keys, &kn, &kcap, &k)) { json_free(&k); json_free(&v); return -1; }
        if (push(ps, &out->items, &out->len, &cap, &v)) {
            json_free(&v);
            if (obj) json_free(&out->keys[--kn]);
            return -1;
        }
        ws(ps);
        if (ps->p < ps->end && *ps->p == ',') { ps->p++; continue; }
        if (ps->p < ps->end && *ps->p == close) { ps->p++; break; }
        return fail(ps, "期望 ',' 或 '%c'", close);
    }
    size_t n = out->len;
    if (obj && n > 1) {                          /* 检查重复键：排序后看相邻的 */
        const jval **ks = malloc(n * sizeof *ks);
        if (!ks) return fail(ps, "内存不足");
        for (size_t i = 0; i < n; i++) ks[i] = &out->keys[i];
        qsort(ks, n, sizeof *ks, key_cmp);
        int dup = 0;
        for (size_t i = 1; i < n && !dup; i++)
            if (key_cmp(&ks[i - 1], &ks[i]) == 0) dup = fail(ps, "对象里有重复的键 \"%s\"", ks[i]->str);
        free(ks);
        if (dup) return -1;
    }
    return 0;
}

static int parse_value(P *ps, jval *out, int depth) {
    if (depth > 128) return fail(ps, "嵌套过深");
    ws(ps);
    if (ps->p >= ps->end) return fail(ps, "期望一个值，却到了末尾");
    switch (*ps->p) {
    case '"': return parse_string(ps, out);
    case '{': case '[': return parse_container(ps, out, depth);
    case 't': return lit(ps, "true", J_TRUE, out);
    case 'f': return lit(ps, "false", J_FALSE, out);
    case 'n': return lit(ps, "null", J_NULL, out);
    default:  return parse_number(ps, out);
    }
}

int json_parse(jval *root, const char *s, size_t n, char *err, size_t errcap) {
    P ps = {s, s, s + n, err, errcap};
    memset(root, 0, sizeof *root);
    if (parse_value(&ps, root, 0)) { json_free(root); return -1; }
    ws(&ps);
    if (ps.p != ps.end) { json_free(root); return fail(&ps, "值之后还有多余内容"); }
    return 0;
}

void json_free(jval *v) {
    if (v->type == J_ARR || v->type == J_OBJ) {
        for (size_t i = 0; i < v->len; i++) {
            json_free(&v->items[i]);
            if (v->type == J_OBJ) json_free(&v->keys[i]);
        }
        free(v->items); free(v->keys);
    }
    free(v->str);
    memset(v, 0, sizeof *v);
}

const jval *json_get(const jval *v, const char *key) {
    if (!v || v->type != J_OBJ) return NULL;
    size_t kl = strlen(key);
    for (size_t i = 0; i < v->len; i++)
        if (v->keys[i].len == kl && memcmp(v->keys[i].str, key, kl) == 0) return &v->items[i];
    return NULL;
}

int json_int(const jval *v, int64_t *out) {
    if (!v || v->type != J_NUM) return -1;
    const char *p = v->str;
    int neg = *p == '-';
    if (neg) p++;
    uint64_t x = 0;
    for (; *p; p++) {
        if (*p < '0' || *p > '9') return -1;
        unsigned d = (unsigned)(*p - '0');
        if (x > (UINT64_C(1) << 63) / 10 || x * 10 > (UINT64_C(1) << 63) - d) return -1;
        x = x * 10 + d;
    }
    if (!neg && x > (uint64_t)INT64_MAX) return -1;
    *out = neg ? (int64_t)(0 - x) : (int64_t)x;
    return 0;
}

int json_parse_file(jval *root, const char *path, char *err, size_t cap) {
    FILE *f = fopen(path, "rb");
    if (!f) { if (err && cap) snprintf(err, cap, "打不开 %s", path); return -1; }
    char *buf = NULL;
    size_t n = 0, capb = 0;
    for (;;) {
        if (n == capb) {
            char *nb = realloc(buf, capb = capb ? 2 * capb : 8192);
            if (!nb) { free(buf); fclose(f); if (err && cap) snprintf(err, cap, "内存不足"); return -1; }
            buf = nb;
        }
        size_t r = fread(buf + n, 1, capb - n, f);
        n += r;
        if (r == 0) break;
    }
    int bad = ferror(f);
    fclose(f);
    if (bad) { free(buf); if (err && cap) snprintf(err, cap, "读 %s 出错", path); return -1; }
    int rc = json_parse(root, buf, n, err, cap);
    free(buf);
    return rc;
}
