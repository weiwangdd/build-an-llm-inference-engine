/* unicode.c — UTF-8、字符类别查询与 NFC 归一化。数据表见 unicode_tables.c。 */
#include "unicode.h"
#include "unicode_tables.h"

#include <stdlib.h>
#include <string.h>

/* ---------- UTF-8 ---------- */

int utf8_next(const uint8_t *s, size_t n, size_t *i, uint32_t *cp) {
    size_t k = *i;
    if (k >= n) return -1;
    uint8_t b = s[k];
    uint32_t c, min;
    int len;
    if (b < 0x80)      { *cp = b; *i = k + 1; return 0; }
    else if (b < 0xC0) return -1;                 /* 孤立的后续字节 */
    else if (b < 0xE0) { c = b & 0x1Fu; len = 2; min = 0x80; }
    else if (b < 0xF0) { c = b & 0x0Fu; len = 3; min = 0x800; }
    else if (b < 0xF8) { c = b & 0x07u; len = 4; min = 0x10000; }
    else return -1;
    if (n - k < (size_t)len) return -1;
    for (int j = 1; j < len; j++) {
        if ((s[k + (size_t)j] & 0xC0) != 0x80) return -1;
        c = c << 6 | (s[k + (size_t)j] & 0x3Fu);
    }
    if (c < min || c > 0x10FFFF || (c >= 0xD800 && c <= 0xDFFF)) return -1;
    *cp = c; *i = k + (size_t)len;
    return 0;
}

size_t utf8_put(uint8_t *o, uint32_t c) {
    if (c < 0x80)    { o[0] = (uint8_t)c; return 1; }
    if (c < 0x800)   { o[0] = (uint8_t)(0xC0 | c >> 6); o[1] = (uint8_t)(0x80 | (c & 0x3F)); return 2; }
    if (c < 0x10000) { o[0] = (uint8_t)(0xE0 | c >> 12); o[1] = (uint8_t)(0x80 | (c >> 6 & 0x3F));
                       o[2] = (uint8_t)(0x80 | (c & 0x3F)); return 3; }
    o[0] = (uint8_t)(0xF0 | c >> 18); o[1] = (uint8_t)(0x80 | (c >> 12 & 0x3F));
    o[2] = (uint8_t)(0x80 | (c >> 6 & 0x3F)); o[3] = (uint8_t)(0x80 | (c & 0x3F));
    return 4;
}

/* 非法片段的“最大有效前缀”规则（Unicode 第 3 章 U+FFFD 替换的推荐做法）：
 * 从当前字节开始，只要还可能是某个合法字符的开头就继续吃，一旦不可能就停，
 * 吃掉的这一段换成一个 U+FFFD。第二个字节的合法范围取决于第一个字节。 */
uint8_t *utf8_lossy(const uint8_t *s, size_t n, size_t *on) {
    uint8_t *o = malloc(n * 3 + 1);
    if (!o) return NULL;
    size_t i = 0, m = 0;
    while (i < n) {
        size_t j = i;
        uint32_t cp;
        if (utf8_next(s, n, &j, &cp) == 0) { memcpy(o + m, s + i, j - i); m += j - i; i = j; continue; }
        uint8_t b = s[i];
        int need; uint8_t lo = 0x80, hi = 0xBF;
        if (b >= 0xC2 && b <= 0xDF) need = 1;
        else if (b >= 0xE0 && b <= 0xEF) { need = 2; if (b == 0xE0) lo = 0xA0; if (b == 0xED) hi = 0x9F; }
        else if (b >= 0xF0 && b <= 0xF4) { need = 3; if (b == 0xF0) lo = 0x90; if (b == 0xF4) hi = 0x8F; }
        else need = 0;
        size_t k = i + 1;
        for (int t = 0; t < need && k < n; t++, k++) {
            uint8_t lim_lo = t == 0 ? lo : 0x80, lim_hi = t == 0 ? hi : 0xBF;
            if (s[k] < lim_lo || s[k] > lim_hi) break;
        }
        o[m++] = 0xEF; o[m++] = 0xBF; o[m++] = 0xBD;
        i = k;
    }
    *on = m;
    return o;
}

/* ---------- 查表 ---------- */

static int range_lookup(const uc_range *r, size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < r[mid].lo) hi = mid;
        else if (cp > r[mid].hi) lo = mid + 1;
        else return r[mid].v;
    }
    return 0;
}

int uc_class(uint32_t cp) { return range_lookup(UC_CLASS, UC_CLASS_N, cp); }
int uc_ccc(uint32_t cp)   { return range_lookup(UC_CCC, UC_CCC_N, cp); }

/* Unicode 的 White_Space 属性只有 25 个字符，几十年没变过，直接写出来。 */
int uc_is_space(uint32_t c) {
    return (c >= 0x09 && c <= 0x0D) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F ||
           c == 0x205F || c == 0x3000;
}

static const uc_decomp *find_decomp(uint32_t cp) {
    size_t lo = 0, hi = UC_DECOMP_N;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < UC_DECOMP[mid].cp) hi = mid;
        else if (cp > UC_DECOMP[mid].cp) lo = mid + 1;
        else return &UC_DECOMP[mid];
    }
    return NULL;
}

/* 查 a + b 能否合成；不能返回 0。 */
static uint32_t find_comp(uint32_t a, uint32_t b) {
    size_t lo = 0, hi = UC_COMP_N;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        const uc_pair *p = &UC_COMP[mid];
        if (a < p->a || (a == p->a && b < p->b)) hi = mid;
        else if (a > p->a || b > p->b) lo = mid + 1;
        else return p->c;
    }
    return 0;
}

/* ---------- NFC ---------- */

/* 韩文音节是按公式排列的，不在表里：音节 = 0xAC00 + (初声*21 + 中声)*28 + 终声 */
enum { SBASE = 0xAC00, LBASE = 0x1100, VBASE = 0x1161, TBASE = 0x11A7,
       LCOUNT = 19, VCOUNT = 21, TCOUNT = 28, NCOUNT = VCOUNT * TCOUNT, SCOUNT = LCOUNT * NCOUNT };

int uc_nfc(const uint32_t *in, size_t n, uint32_t **out, size_t *on) {
    /* 一个码点完全分解后最多 18 个码点（Unicode 的稳定性保证），按 18 倍预留空间 */
    uint32_t *d = malloc(sizeof(uint32_t) * (n * 18 + 1));
    if (!d) return -1;

    /* 第一步：完全规范分解 */
    size_t m = 0;
    for (size_t i = 0; i < n; i++) {
        uint32_t c = in[i];
        if (c >= SBASE && c < SBASE + SCOUNT) {
            uint32_t s = c - SBASE;
            d[m++] = LBASE + s / NCOUNT;
            d[m++] = VBASE + s % NCOUNT / TCOUNT;
            if (s % TCOUNT) d[m++] = TBASE + s % TCOUNT;
        } else {
            const uc_decomp *e = find_decomp(c);
            if (!e) d[m++] = c;
            else for (uint32_t k = 0; k < e->len; k++) d[m++] = UC_DECOMP_DATA[e->off + k];
        }
    }

    /* 第二步：规范排序。每一段连续的非起始字符（组合类不为 0）按组合类做稳定排序。 */
    for (size_t i = 0; i < m;) {
        if (uc_ccc(d[i]) == 0) { i++; continue; }
        size_t j = i;
        while (j < m && uc_ccc(d[j]) != 0) j++;
        for (size_t a = i + 1; a < j; a++)          /* 插入排序是稳定的 */
            for (size_t b = a; b > i && uc_ccc(d[b - 1]) > uc_ccc(d[b]); b--) {
                uint32_t t = d[b]; d[b] = d[b - 1]; d[b - 1] = t;
            }
        i = j;
    }

    /* 第三步：规范合成。从左到右，每个字符尝试与最近的起始字符合成，
     * 前提是两者之间没有“阻挡”：中间字符的组合类都比它小（且不为 0）。 */
    size_t r = 0;            /* 结果长度，原地写回 d */
    long starter = -1;       /* 最近的起始字符在结果中的位置 */
    int last = -1;           /* 结果中最后一个字符的组合类；-1 表示还没有字符 */
    for (size_t i = 0; i < m; i++) {
        uint32_t c = d[i];
        int cc = uc_ccc(c);
        if (starter >= 0 && (last < cc || (last == 0 && (size_t)starter == r - 1))) {
            uint32_t a = d[starter], x = 0;
            if (a >= LBASE && a < LBASE + LCOUNT && c >= VBASE && c < VBASE + VCOUNT)
                x = SBASE + ((a - LBASE) * VCOUNT + (c - VBASE)) * TCOUNT;
            else if (a >= SBASE && a < SBASE + SCOUNT && (a - SBASE) % TCOUNT == 0 &&
                     c > TBASE && c < TBASE + TCOUNT)
                x = a + (c - TBASE);
            else
                x = find_comp(a, c);
            if (x) { d[starter] = x; continue; }    /* 合成成功：c 被吃掉，last 不变 */
        }
        if (cc == 0) starter = (long)r;
        last = cc;
        d[r++] = c;
    }
    *out = d; *on = r;
    return 0;
}
