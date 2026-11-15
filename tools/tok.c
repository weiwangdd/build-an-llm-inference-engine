/* tok — 分词器的命令行工具，主要给测试用。输出格式与 tests/tok_ref.py 完全相同。
 *
 *   tok encode  tokenizer.json texts.jsonl   每行一个 JSON 字符串，输出每行的 id 序列
 *   tok decode  tokenizer.json ids.txt       每行一串 id，输出解码结果的十六进制：
 *                                            先是保留特殊 token 的，再是跳过特殊 token 的
 *   tok nfc     texts.jsonl                  输出 NFC 归一化结果的十六进制
 *   tok pretok  texts.jsonl                  归一化并预分词，每段输出一个十六进制串
 *   tok roundtrip tokenizer.json texts.jsonl 编码后再解码（保留特殊 token），输出十六进制；
 *                                            结果应当等于原文的 NFC 归一化
 *
 * 文本都以十六进制输出，这样换行、空格和非法 UTF-8 都不会让逐行比对出歧义。
 * 解码结果先按 Rust 的 from_utf8_lossy 规则把非法字节换成 U+FFFD，因为 Python 那边
 * 拿到的是 str，只能这样比。 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "json.h"
#include "tok.h"
#include "unicode.h"

static void hex(const uint8_t *s, size_t n) {
    for (size_t i = 0; i < n; i++) printf("%02x", s[i]);
}

static void die(const char *what, const char *err) { fprintf(stderr, "错误：%s%s\n", what, err); exit(1); }

/* 读一行（任意长度）。返回 NULL 表示文件结束。 */
static char *read_line(FILE *f, size_t *len) {
    size_t cap = 256, n = 0;
    char *b = malloc(cap);
    int c;
    while (b && (c = fgetc(f)) != EOF && c != '\n') {
        if (n + 1 == cap) { char *nb = realloc(b, cap *= 2); if (!nb) { free(b); b = NULL; break; } b = nb; }
        b[n++] = (char)c;
    }
    if (!b) die("内存不足", "");
    if (n == 0 && c == EOF) { free(b); return NULL; }
    b[n] = 0; *len = n;
    return b;
}

/* 读 JSONL 的一行：必须是一个 JSON 字符串。 */
static int next_text(FILE *f, jval *v, size_t lineno) {
    size_t n; char err[256];
    char *line = read_line(f, &n);
    if (!line) return 0;
    if (json_parse(v, line, n, err, sizeof err)) die("输入文件：", err);
    if (v->type != J_STR) { fprintf(stderr, "错误：输入第 %zu 行不是 JSON 字符串\n", lineno); exit(1); }
    free(line);
    return 1;
}

static uint32_t *to_cps(const jval *v, size_t *n) {
    uint32_t *cp = malloc((v->len ? v->len : 1) * sizeof *cp);
    size_t i = 0, k = 0;
    if (!cp) die("内存不足", "");
    while (i < v->len)
        if (utf8_next((const uint8_t *)v->str, v->len, &i, &cp[k++])) die("输入不是合法 UTF-8", "");
    *n = k;
    return cp;
}

static void put_cps(const uint32_t *cp, size_t n) {
    uint8_t b[4];
    for (size_t i = 0; i < n; i++) hex(b, utf8_put(b, cp[i]));
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "用法：%s encode|decode <tokenizer.json> <输入>  或  %s nfc|pretok <输入>\n", argv[0], argv[0]);
        return 2;
    }
    const char *mode = argv[1];
    int need_tok = !strcmp(mode, "encode") || !strcmp(mode, "decode") || !strcmp(mode, "roundtrip");
    if (need_tok && argc != 4) return 2;
    tok *t = NULL;
    char err[512];
    if (need_tok && tok_load(&t, argv[2], err, sizeof err)) die("", err);
    FILE *in = fopen(argv[need_tok ? 3 : 2], "rb");
    if (!in) die("打不开输入文件", "");

    jval v;
    if (!strcmp(mode, "encode") || !strcmp(mode, "roundtrip")) {
        int rt = !strcmp(mode, "roundtrip");
        for (size_t ln = 1; next_text(in, &v, ln); ln++) {
            int32_t *ids; size_t n;
            if (tok_encode(t, (const uint8_t *)v.str, v.len, &ids, &n, err, sizeof err)) die("", err);
            if (!rt) for (size_t i = 0; i < n; i++) printf(i ? " %d" : "%d", ids[i]);
            else {
                uint8_t *o; size_t on;
                if (tok_decode(t, ids, n, 0, &o, &on, err, sizeof err)) die("", err);
                hex(o, on);
                free(o);
            }
            putchar('\n');
            free(ids); json_free(&v);
        }
    } else if (!strcmp(mode, "decode")) {
        size_t len;
        for (char *line; (line = read_line(in, &len)); free(line)) {
            int32_t *ids = malloc((len / 2 + 1) * sizeof *ids);
            size_t n = 0;
            for (char *p = line, *e; *p; p = e) {
                long x = strtol(p, &e, 10);
                if (e == p) break;
                ids[n++] = (int32_t)x;
            }
            for (int skip = 0; skip < 2; skip++) {
                uint8_t *o, *l; size_t on, ln;
                if (tok_decode(t, ids, n, skip, &o, &on, err, sizeof err)) die("", err);
                l = utf8_lossy(o, on, &ln);
                hex(l, ln);
                putchar(skip ? '\n' : ' ');
                free(o); free(l);
            }
            free(ids);
        }
    } else if (!strcmp(mode, "nfc") || !strcmp(mode, "pretok")) {
        int pre = !strcmp(mode, "pretok");
        for (size_t ln = 1; next_text(in, &v, ln); ln++) {
            size_t n, nn;
            uint32_t *cp = to_cps(&v, &n), *nc;
            if (uc_nfc(cp, n, &nc, &nn)) die("内存不足", "");
            if (!pre) put_cps(nc, nn);
            else {
                size_t *ends = malloc((nn ? nn : 1) * sizeof *ends);
                size_t k = tok_pretokenize(nc, nn, ends);
                for (size_t p = 0, s = 0; p < k; s = ends[p++]) {
                    if (p) putchar(' ');
                    put_cps(nc + s, ends[p] - s);
                }
                free(ends);
            }
            putchar('\n');
            free(cp); free(nc); json_free(&v);
        }
    } else die("未知模式 ", mode);
    fclose(in);
    tok_free(t);
    return 0;
}
