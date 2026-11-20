/* st.c — safetensors 读取器实现。原则：任何不符合格式的输入都报错，绝不猜测。 */
#define _POSIX_C_SOURCE 200809L
#include "st.h"

#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static const struct { const char *name; size_t size; } DTYPES[ST_DTYPE_COUNT] = {
    [ST_BOOL] = {"BOOL", 1}, [ST_U8]  = {"U8", 1},  [ST_I8]   = {"I8", 1},
    [ST_I16]  = {"I16", 2},  [ST_U16] = {"U16", 2}, [ST_F16]  = {"F16", 2},
    [ST_BF16] = {"BF16", 2}, [ST_I32] = {"I32", 4}, [ST_U32]  = {"U32", 4},
    [ST_F32]  = {"F32", 4},  [ST_I64] = {"I64", 8}, [ST_U64]  = {"U64", 8},
    [ST_F64]  = {"F64", 8},
};

const char *st_dtype_name(st_dtype t) { return DTYPES[t].name; }
size_t      st_dtype_size(st_dtype t) { return DTYPES[t].size; }

/* ---------- 一个只够解析 safetensors 头的 JSON 解析器 ---------- */

typedef struct {
    const char *p, *end;
    char *err; size_t errcap;
} parser;

static int fail(parser *ps, const char *fmt, ...) {
    if (ps->err && ps->errcap) {
        va_list ap; va_start(ap, fmt);
        vsnprintf(ps->err, ps->errcap, fmt, ap);
        va_end(ap);
    }
    return -1;
}

static void skip_ws(parser *ps) {
    while (ps->p < ps->end &&
           (*ps->p == ' ' || *ps->p == '\t' || *ps->p == '\n' || *ps->p == '\r'))
        ps->p++;
}

static int expect(parser *ps, char c) {
    skip_ws(ps);
    if (ps->p >= ps->end || *ps->p != c)
        return fail(ps, "JSON 头：期望 '%c'", c);
    ps->p++;
    return 0;
}

static int peek(parser *ps) {
    skip_ws(ps);
    return ps->p < ps->end ? (unsigned char)*ps->p : -1;
}

static int hex4(parser *ps, unsigned *out) {
    if (ps->end - ps->p < 4) return fail(ps, "JSON 头：\\u 转义不完整");
    unsigned v = 0;
    for (int i = 0; i < 4; i++) {
        char c = *ps->p++;
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
        else return fail(ps, "JSON 头：\\u 转义含非十六进制字符");
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

/* 解析一个 JSON 字符串，返回 malloc 出来的 UTF-8 副本。 */
static int parse_string(parser *ps, char **out) {
    if (expect(ps, '"')) return -1;
    /* 解码后的长度不会超过原文长度 */
    char *buf = malloc((size_t)(ps->end - ps->p) + 1);
    if (!buf) return fail(ps, "内存不足");
    size_t n = 0;
    while (ps->p < ps->end) {
        unsigned char c = (unsigned char)*ps->p++;
        if (c == '"') { buf[n] = 0; *out = buf; return 0; }
        if (c < 0x20) break;                      /* 字符串内不允许裸控制字符 */
        if (c != '\\') { buf[n++] = (char)c; continue; }
        if (ps->p >= ps->end) break;
        char e = *ps->p++;
        switch (e) {
        case '"': case '\\': case '/': buf[n++] = e; break;
        case 'b': buf[n++] = '\b'; break;
        case 'f': buf[n++] = '\f'; break;
        case 'n': buf[n++] = '\n'; break;
        case 'r': buf[n++] = '\r'; break;
        case 't': buf[n++] = '\t'; break;
        case 'u': {
            unsigned cp, lo;
            if (hex4(ps, &cp)) { free(buf); return -1; }
            if (cp >= 0xD800 && cp <= 0xDBFF) {   /* 代理对的高半部分 */
                if (ps->end - ps->p < 2 || ps->p[0] != '\\' || ps->p[1] != 'u') goto bad;
                ps->p += 2;
                if (hex4(ps, &lo)) { free(buf); return -1; }
                if (lo < 0xDC00 || lo > 0xDFFF) goto bad;
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            } else if (cp >= 0xDC00 && cp <= 0xDFFF) goto bad;
            if (cp == 0) goto bad;                /* 名字里不接受 NUL */
            n += put_utf8(buf + n, cp);
            break;
        }
        default: goto bad;
        }
    }
bad:
    free(buf);
    return fail(ps, "JSON 头：字符串非法");
}

/* 解析一个非负十进制整数（safetensors 头里的数只有这一种）。 */
static int parse_u64(parser *ps, uint64_t *out) {
    skip_ws(ps);
    if (ps->p >= ps->end || *ps->p < '0' || *ps->p > '9')
        return fail(ps, "JSON 头：期望非负整数");
    uint64_t v = 0;
    while (ps->p < ps->end && *ps->p >= '0' && *ps->p <= '9') {
        unsigned d = (unsigned)(*ps->p++ - '0');
        if (v > (UINT64_MAX - d) / 10) return fail(ps, "JSON 头：整数溢出");
        v = v * 10 + d;
    }
    *out = v;
    return 0;
}

/* 跳过任意一个 JSON 值（用于 __metadata__ 和未知字段）。 */
static int skip_value(parser *ps, int depth) {
    if (depth > 64) return fail(ps, "JSON 头：嵌套过深");
    int c = peek(ps);
    if (c == '"') { char *s; if (parse_string(ps, &s)) return -1; free(s); return 0; }
    if (c == '{' || c == '[') {
        char close = c == '{' ? '}' : ']';
        ps->p++;
        if (peek(ps) == close) { ps->p++; return 0; }
        for (;;) {
            if (c == '{') {
                char *k; if (parse_string(ps, &k)) return -1; free(k);
                if (expect(ps, ':')) return -1;
            }
            if (skip_value(ps, depth + 1)) return -1;
            int d = peek(ps);
            if (d == ',') { ps->p++; continue; }
            if (d == close) { ps->p++; return 0; }
            return fail(ps, "JSON 头：期望 ',' 或 '%c'", close);
        }
    }
    /* 数字、true、false、null：读到分隔符为止 */
    const char *start = ps->p;
    while (ps->p < ps->end && !strchr(",]} \t\r\n", *ps->p)) ps->p++;
    if (ps->p == start) return fail(ps, "JSON 头：期望一个值");
    return 0;
}

/* 解析 {"dtype":..,"shape":[..],"data_offsets":[b,e]}。 */
static int parse_entry(parser *ps, st_tensor *t, uint64_t *begin, uint64_t *end) {
    int have_dtype = 0, have_shape = 0, have_off = 0;
    if (expect(ps, '{')) return -1;
    if (peek(ps) != '}') for (;;) {
        char *key;
        if (parse_string(ps, &key)) return -1;
        int rc = expect(ps, ':');
        if (rc == 0 && strcmp(key, "dtype") == 0) {
            char *v;
            rc = parse_string(ps, &v);
            if (rc == 0) {
                int found = 0;
                for (int i = 0; i < ST_DTYPE_COUNT; i++)
                    if (strcmp(v, DTYPES[i].name) == 0) { t->dtype = (st_dtype)i; found = 1; }
                if (!found) rc = fail(ps, "张量 %s：不支持的 dtype \"%s\"", t->name, v);
                free(v);
                have_dtype = 1;
            }
        } else if (rc == 0 && strcmp(key, "shape") == 0) {
            rc = expect(ps, '[');
            t->ndim = 0;
            if (rc == 0 && peek(ps) != ']') for (;;) {
                if (t->ndim == ST_MAX_DIMS) { rc = fail(ps, "张量 %s：维数超过 %d", t->name, ST_MAX_DIMS); break; }
                if ((rc = parse_u64(ps, &t->shape[t->ndim]))) break;
                t->ndim++;
                if (peek(ps) == ',') { ps->p++; continue; }
                break;
            }
            if (rc == 0) rc = expect(ps, ']');
            have_shape = 1;
        } else if (rc == 0 && strcmp(key, "data_offsets") == 0) {
            if ((rc = expect(ps, '[')) == 0 && (rc = parse_u64(ps, begin)) == 0 &&
                (rc = expect(ps, ',')) == 0 && (rc = parse_u64(ps, end)) == 0)
                rc = expect(ps, ']');
            have_off = 1;
        } else if (rc == 0) {
            rc = skip_value(ps, 0);
        }
        free(key);
        if (rc) return -1;
        if (peek(ps) == ',') { ps->p++; continue; }
        break;
    }
    if (expect(ps, '}')) return -1;
    if (!have_dtype || !have_shape || !have_off)
        return fail(ps, "张量 %s：缺少 dtype、shape 或 data_offsets", t->name);
    return 0;
}

static int by_name(const void *a, const void *b) {
    return strcmp(((const st_tensor *)a)->name, ((const st_tensor *)b)->name);
}

static int parse_header(st_file *f, parser *ps, const uint8_t *data, uint64_t data_size) {
    size_t cap = 0;
    if (expect(ps, '{')) return -1;
    if (peek(ps) != '}') for (;;) {
        char *key;
        if (parse_string(ps, &key)) return -1;
        if (expect(ps, ':')) { free(key); return -1; }
        if (strcmp(key, "__metadata__") == 0) {
            free(key);
            if (skip_value(ps, 0)) return -1;
        } else {
            if (f->count == cap) {
                cap = cap ? cap * 2 : 64;
                st_tensor *nt = realloc(f->tensors, cap * sizeof *nt);
                if (!nt) { free(key); return fail(ps, "内存不足"); }
                f->tensors = nt;
            }
            st_tensor *t = &f->tensors[f->count++];
            memset(t, 0, sizeof *t);
            t->name = key;
            uint64_t b = 0, e = 0;
            if (parse_entry(ps, t, &b, &e)) return -1;
            if (b > e || e > data_size)
                return fail(ps, "张量 %s：data_offsets 越界", t->name);
            /* 字节数必须恰好等于 元素数 × 元素大小，乘法要防溢出 */
            uint64_t want = st_dtype_size(t->dtype);
            for (int i = 0; i < t->ndim; i++) {
                if (t->shape[i] && want > UINT64_MAX / t->shape[i])
                    return fail(ps, "张量 %s：形状乘积溢出", t->name);
                want *= t->shape[i];
            }
            if (want != e - b)
                return fail(ps, "张量 %s：字节数与 shape、dtype 不符", t->name);
            t->data = data + b;
            t->nbytes = e - b;
        }
        if (peek(ps) == ',') { ps->p++; continue; }
        break;
    }
    if (expect(ps, '}')) return -1;
    skip_ws(ps);
    if (ps->p != ps->end) return fail(ps, "JSON 头：末尾有多余内容");

    qsort(f->tensors, f->count, sizeof *f->tensors, by_name);
    for (size_t i = 1; i < f->count; i++)
        if (strcmp(f->tensors[i - 1].name, f->tensors[i].name) == 0)
            return fail(ps, "张量名重复：%s", f->tensors[i].name);
    return 0;
}

int st_open(st_file *f, const char *path, char *err, size_t errcap) {
    memset(f, 0, sizeof *f);
    parser ps = { .err = err, .errcap = errcap };

    int fd = open(path, O_RDONLY);
    if (fd < 0) return fail(&ps, "打不开文件 %s", path);
    struct stat sb;
    if (fstat(fd, &sb) != 0 || sb.st_size < 8) {
        close(fd);
        return fail(&ps, "文件太短，不是 safetensors：%s", path);
    }
    f->map_size = (size_t)sb.st_size;
    f->map = mmap(NULL, f->map_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (f->map == MAP_FAILED) { f->map = NULL; return fail(&ps, "mmap 失败：%s", path); }

    const uint8_t *base = f->map;
    uint64_t hlen = 0;
    for (int i = 7; i >= 0; i--) hlen = hlen << 8 | base[i];   /* 小端，不依赖主机字节序 */
    if (hlen > f->map_size - 8) {
        st_close(f);
        return fail(&ps, "头长度 %llu 超出文件大小", (unsigned long long)hlen);
    }
    ps.p = (const char *)base + 8;
    ps.end = ps.p + hlen;
    if (parse_header(f, &ps, base + 8 + hlen, f->map_size - 8 - hlen)) {
        st_close(f);
        return -1;
    }
    return 0;
}

void st_close(st_file *f) {
    for (size_t i = 0; i < f->count; i++) free(f->tensors[i].name);
    free(f->tensors);
    if (f->map) munmap(f->map, f->map_size);
    memset(f, 0, sizeof *f);
}

const st_tensor *st_find(const st_file *f, const char *name) {
    st_tensor key = { .name = (char *)name };
    return bsearch(&key, f->tensors, f->count, sizeof *f->tensors, by_name);
}

/* ---------- 多个分片 ---------- */

static int by_name_ptr(const void *a, const void *b) {
    return strcmp((*(const st_tensor *const *)a)->name, (*(const st_tensor *const *)b)->name);
}

int st_set_open(st_set *s, const char *const *paths, size_t n, char *err, size_t errcap) {
    memset(s, 0, sizeof *s);
    parser ps = { .err = err, .errcap = errcap };
    s->files = calloc(n ? n : 1, sizeof *s->files);
    if (!s->files) return fail(&ps, "内存不足");
    size_t total = 0;
    for (size_t i = 0; i < n; i++) {
        if (st_open(&s->files[i], paths[i], err, errcap)) { st_set_close(s); return -1; }
        s->nfiles = i + 1;
        total += s->files[i].count;
    }
    s->tensors = malloc((total ? total : 1) * sizeof *s->tensors);
    if (!s->tensors) { st_set_close(s); return fail(&ps, "内存不足"); }
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < s->files[i].count; j++)
            s->tensors[s->count++] = &s->files[i].tensors[j];
    qsort(s->tensors, s->count, sizeof *s->tensors, by_name_ptr);
    for (size_t i = 1; i < s->count; i++)
        if (strcmp(s->tensors[i - 1]->name, s->tensors[i]->name) == 0) {
            fail(&ps, "张量名在多个分片中重复：%s", s->tensors[i]->name);
            st_set_close(s);
            return -1;
        }
    return 0;
}

void st_set_close(st_set *s) {
    for (size_t i = 0; i < s->nfiles; i++) st_close(&s->files[i]);
    free(s->files);
    free(s->tensors);
    memset(s, 0, sizeof *s);
}

const st_tensor *st_set_find(const st_set *s, const char *name) {
    st_tensor key = { .name = (char *)name };
    const st_tensor *kp = &key;
    const st_tensor **hit = bsearch(&kp, s->tensors, s->count, sizeof *s->tensors, by_name_ptr);
    return hit ? *hit : NULL;
}
