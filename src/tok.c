/* tok.c — BPE 分词器。 */
#include "tok.h"
#include "json.h"
#include "unicode.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Qwen 的预分词正则。我们不实现通用正则引擎，而是在 match_at 里手写这一条，
 * 所以加载时要求文件里的正则与它一字不差。 */
static const char QWEN_PATTERN[] =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}"
    "| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

typedef struct {
    int32_t  id;
    uint8_t *s;  size_t len;       /* 原文中要匹配的内容 */
    int      special, normalized;  /* normalized 为真：在归一化之后的文本里匹配 */
} added_tok;

typedef struct { uint64_t key; int32_t rank, id; } merge_slot;   /* key = 左 id << 32 | 右 id */

struct tok {
    size_t    n;                   /* id 上界 */
    uint8_t **bytes;               /* 每个 id 对应的字节串；NULL 表示这个 id 没有定义 */
    uint32_t *blen;
    uint8_t  *special;
    int32_t  *vh; size_t vcap;     /* 词表的哈希表：字节串 → id，开放寻址，-1 表示空 */
    merge_slot *mh; size_t mcap;   /* 合并规则的哈希表：(左, 右) → (优先级, 新 id) */
    int32_t   byte_id[256];        /* 单字节 token 的 id */
    added_tok *added; size_t nadded;
};

static int fail(char *err, size_t cap, const char *fmt, ...) {
    if (err && cap) { va_list ap; va_start(ap, fmt); vsnprintf(err, cap, fmt, ap); va_end(ap); }
    return -1;
}

/* ---------- 字节与字符的对应（GPT-2 的“byte-level”技巧） ----------
 *
 * BPE 在字节上工作，但 tokenizer.json 要把每个 token 写成 JSON 字符串。有些字节
 * （控制字符、空格、不完整的 UTF-8 片段）没法好好地写进字符串，于是 GPT-2 把 256 个
 * 字节一一映射到 256 个“看得见”的 Unicode 字符：可打印的那 188 个字节映射到同码点的
 * 字符，其余 68 个依次映射到 U+0100 起的字符。空格 0x20 因此变成了 'Ġ'（U+0120）。
 * 我们在加载时就把映射逆转回去，此后程序里只有字节。 */
static int byte_of_char[324];        /* 码点 → 字节，-1 表示不在映射里 */

static void init_byte_map(void) {
    int next = 256;
    for (int c = 0; c < 324; c++) byte_of_char[c] = -1;
    for (int b = 0; b < 256; b++) {
        int printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        byte_of_char[printable ? b : next++] = b;
    }
}

/* 把 JSON 里的 token 字符串（UTF-8）逆映射成原始字节，写进 out（长度不超过 n）。 */
static int unmap_token(const char *s, size_t n, uint8_t *out, size_t *on) {
    size_t i = 0, m = 0;
    uint32_t cp;
    while (i < n) {
        if (utf8_next((const uint8_t *)s, n, &i, &cp) || cp >= 324 || byte_of_char[cp] < 0) return -1;
        out[m++] = (uint8_t)byte_of_char[cp];
    }
    *on = m;
    return 0;
}

/* ---------- 哈希表 ---------- */

static uint64_t fnv1a(const uint8_t *p, size_t n) {
    uint64_t h = 0xcbf29ce484222325u;
    for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 0x100000001b3u;
    return h;
}

static uint64_t mix(uint64_t x) {           /* splitmix64 的收尾步骤，把相近的 key 打散 */
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9u;
    x ^= x >> 27; x *= 0x94d049bb133111ebu;
    return x ^ (x >> 31);
}

static int32_t vocab_find(const tok *t, const uint8_t *s, size_t n) {
    for (size_t i = fnv1a(s, n) & (t->vcap - 1);; i = (i + 1) & (t->vcap - 1)) {
        int32_t id = t->vh[i];
        if (id < 0) return -1;
        if (t->blen[id] == n && memcmp(t->bytes[id], s, n) == 0) return id;
    }
}

static const merge_slot *merge_find(const tok *t, int32_t a, int32_t b) {
    uint64_t key = (uint64_t)(uint32_t)a << 32 | (uint32_t)b;
    for (size_t i = mix(key) & (t->mcap - 1);; i = (i + 1) & (t->mcap - 1)) {
        if (t->mh[i].id < 0) return NULL;
        if (t->mh[i].key == key) return &t->mh[i];
    }
}

static size_t pow2_at_least(size_t n) { size_t c = 16; while (c < n) c <<= 1; return c; }

/* ---------- 加载 ---------- */

static int read_file(const char *path, char **buf, size_t *n, char *err, size_t cap) {
    FILE *f = fopen(path, "rb");
    if (!f) return fail(err, cap, "打不开 %s", path);
    size_t sz = 0, c = 1 << 20;
    char *b = malloc(c);
    for (size_t r; b && (r = fread(b + sz, 1, c - sz, f)) > 0;) {
        sz += r;
        if (sz == c) { char *nb = realloc(b, c *= 2); if (!nb) { free(b); b = NULL; } else b = nb; }
    }
    int bad = ferror(f);
    fclose(f);
    if (!b) return fail(err, cap, "内存不足");
    if (bad) { free(b); return fail(err, cap, "读取 %s 出错", path); }
    *buf = b; *n = sz;
    return 0;
}

/* 检查 v 是否是指定的布尔值或 null（missing_ok 为真时字段可以不存在）。 */
static int is_false(const jval *v) { return v && v->type == J_FALSE; }
static int is_null_or_empty(const jval *v) {
    return !v || v->type == J_NULL || (v->type == J_STR && v->len == 0);
}
static int str_is(const jval *v, const char *s) {
    return v && v->type == J_STR && v->len == strlen(s) && memcmp(v->str, s, v->len) == 0;
}

/* 检查归一化、预分词、解码器等配置是不是 Qwen 的那一套。 */
static int check_pipeline(const jval *root, char *err, size_t cap) {
    const jval *nm = json_get(root, "normalizer");
    if (!nm || !str_is(json_get(nm, "type"), "NFC"))
        return fail(err, cap, "只支持 normalizer = NFC");

    const jval *pt = json_get(root, "pre_tokenizer");
    const jval *seq = json_get(pt, "pretokenizers");
    if (!str_is(json_get(pt, "type"), "Sequence") || !seq || seq->type != J_ARR || seq->len != 2)
        return fail(err, cap, "只支持 pre_tokenizer = Sequence[Split, ByteLevel]");
    const jval *sp = &seq->items[0], *bl = &seq->items[1];
    if (!str_is(json_get(sp, "type"), "Split") ||
        !str_is(json_get(json_get(sp, "pattern"), "Regex"), QWEN_PATTERN) ||
        !str_is(json_get(sp, "behavior"), "Isolated") || !is_false(json_get(sp, "invert")))
        return fail(err, cap, "预分词的 Split 不是 Qwen 的正则（本实现只支持这一条）");
    if (!str_is(json_get(bl, "type"), "ByteLevel") || !is_false(json_get(bl, "add_prefix_space")) ||
        !is_false(json_get(bl, "use_regex")))
        return fail(err, cap, "只支持 ByteLevel(add_prefix_space=false, use_regex=false)");

    if (!str_is(json_get(json_get(root, "decoder"), "type"), "ByteLevel"))
        return fail(err, cap, "只支持 decoder = ByteLevel");
    const jval *pp = json_get(root, "post_processor");
    if (pp && pp->type != J_NULL && !str_is(json_get(pp, "type"), "ByteLevel"))
        return fail(err, cap, "只支持 post_processor = ByteLevel 或 null（不添加额外 token）");

    const jval *m = json_get(root, "model");
    if (!str_is(json_get(m, "type"), "BPE")) return fail(err, cap, "只支持 BPE 模型");
    const jval *dr = json_get(m, "dropout");
    if (dr && dr->type != J_NULL) return fail(err, cap, "不支持 BPE dropout");
    if (!is_null_or_empty(json_get(m, "unk_token")) ||
        !is_null_or_empty(json_get(m, "continuing_subword_prefix")) ||
        !is_null_or_empty(json_get(m, "end_of_word_suffix")))
        return fail(err, cap, "不支持 unk_token / continuing_subword_prefix / end_of_word_suffix");
    const char *flags[] = {"fuse_unk", "byte_fallback", "ignore_merges"};
    for (int i = 0; i < 3; i++) {
        const jval *f = json_get(m, flags[i]);
        if (f && f->type != J_FALSE) return fail(err, cap, "只支持 %s = false", flags[i]);
    }
    return 0;
}

static int set_id(tok *t, int64_t id, const uint8_t *s, size_t n, char *err, size_t cap) {
    if (id < 0 || (uint64_t)id >= t->n) return fail(err, cap, "id %lld 越界", (long long)id);
    if (t->bytes[id]) return fail(err, cap, "id %lld 重复", (long long)id);
    t->bytes[id] = malloc(n ? n : 1);
    if (!t->bytes[id]) return fail(err, cap, "内存不足");
    memcpy(t->bytes[id], s, n);
    t->blen[id] = (uint32_t)n;
    return 0;
}

static int load_vocab(tok *t, const jval *vocab, const jval *added, char *err, size_t cap) {
    /* 先找出最大 id，决定数组大小 */
    int64_t maxid = -1, id;
    for (size_t i = 0; i < vocab->len; i++) {
        if (json_int(&vocab->items[i], &id) || id < 0 || id > INT32_MAX - 1)
            return fail(err, cap, "词表条目 \"%s\" 的 id 不合法", vocab->keys[i].str);
        if (id > maxid) maxid = id;
    }
    for (size_t i = 0; i < added->len; i++) {
        if (json_int(json_get(&added->items[i], "id"), &id) || id < 0 || id > INT32_MAX - 1)
            return fail(err, cap, "added_tokens 第 %zu 项的 id 不合法", i);
        if (id > maxid) maxid = id;
    }
    t->n = (size_t)(maxid + 1);
    t->bytes = calloc(t->n, sizeof *t->bytes);
    t->blen = calloc(t->n, sizeof *t->blen);
    t->special = calloc(t->n, 1);
    t->vcap = pow2_at_least(vocab->len * 2);
    t->vh = malloc(t->vcap * sizeof *t->vh);
    if (!t->bytes || !t->blen || !t->special || !t->vh) return fail(err, cap, "内存不足");
    memset(t->vh, 0xFF, t->vcap * sizeof *t->vh);      /* 全部置为 -1 */

    uint8_t buf[1024];
    for (size_t i = 0; i < vocab->len; i++) {
        const jval *k = &vocab->keys[i];
        size_t n;
        if (k->len > sizeof buf || unmap_token(k->str, k->len, buf, &n) || n == 0)
            return fail(err, cap, "词表条目 \"%s\" 不是合法的 byte-level 字符串", k->str);
        json_int(&vocab->items[i], &id);
        if (vocab_find(t, buf, n) >= 0) return fail(err, cap, "词表里有两个 token 对应同样的字节");
        if (set_id(t, id, buf, n, err, cap)) return -1;
        size_t h = fnv1a(buf, n) & (t->vcap - 1);
        while (t->vh[h] >= 0) h = (h + 1) & (t->vcap - 1);
        t->vh[h] = (int32_t)id;
    }
    for (int b = 0; b < 256; b++) {
        uint8_t c = (uint8_t)b;
        if ((t->byte_id[b] = vocab_find(t, &c, 1)) < 0)
            return fail(err, cap, "词表缺少单字节 token 0x%02x，有些文本将无法编码", b);
    }

    /* 特殊 token 和其他“附加 token”：编码时按原文匹配，不经过 BPE */
    t->added = calloc(added->len ? added->len : 1, sizeof *t->added);
    if (!t->added) return fail(err, cap, "内存不足");
    for (size_t i = 0; i < added->len; i++) {
        const jval *a = &added->items[i], *c = json_get(a, "content");
        const jval *sp = json_get(a, "special"), *nz = json_get(a, "normalized");
        if (!c || c->type != J_STR || c->len == 0) return fail(err, cap, "added_tokens 第 %zu 项没有内容", i);
        if (!sp || !nz || (sp->type != J_TRUE && sp->type != J_FALSE) || (nz->type != J_TRUE && nz->type != J_FALSE))
            return fail(err, cap, "added_tokens \"%s\" 缺少 special / normalized", c->str);
        const char *opts[] = {"single_word", "lstrip", "rstrip"};
        for (int k = 0; k < 3; k++)
            if (!is_false(json_get(a, opts[k])))
                return fail(err, cap, "added_tokens \"%s\"：只支持 %s = false", c->str, opts[k]);
        json_int(json_get(a, "id"), &id);
        added_tok *at = &t->added[t->nadded++];
        at->id = (int32_t)id;
        at->special = sp->type == J_TRUE;
        at->normalized = nz->type == J_TRUE;
        /* 要在归一化之后的文本里匹配的 token，内容本身也要先归一化 */
        uint32_t *cps = malloc(c->len * sizeof *cps), *nc = NULL;
        size_t ncp = 0, j = 0, nn = 0;
        if (!cps) return fail(err, cap, "内存不足");
        while (j < c->len) utf8_next((const uint8_t *)c->str, c->len, &j, &cps[ncp++]);
        if (at->normalized) { if (uc_nfc(cps, ncp, &nc, &nn)) { free(cps); return fail(err, cap, "内存不足"); } }
        else { nc = cps; cps = NULL; nn = ncp; }
        free(cps);
        at->s = malloc(nn * 4 + 1);
        if (!at->s) { free(nc); return fail(err, cap, "内存不足"); }
        for (size_t q = 0; q < nn; q++) at->len += utf8_put(at->s + at->len, nc[q]);
        free(nc);
        /* 解码时 tokenizers 把附加 token 也交给 ByteLevel 解码器：如果每个字符都在
         * 字节映射里，就按映射还原成字节；只要有一个字符不在，整个 token 原样输出。
         * Qwen 的附加 token 都是 ASCII，两种情况结果相同，但我们照规则实现。 */
        uint8_t tmp[1024]; size_t tn;
        const uint8_t *db = at->s; size_t dn = at->len;
        if (at->len <= sizeof tmp && unmap_token((const char *)at->s, at->len, tmp, &tn) == 0) { db = tmp; dn = tn; }
        if (id < (int64_t)t->n && t->bytes[id]) {
            /* 已经在词表里：tokenizers 允许这种情况，此时两边必须是同一个 token */
            if (dn != t->blen[id] || memcmp(db, t->bytes[id], dn))
                return fail(err, cap, "added_tokens \"%s\" 的 id 与词表里的另一个 token 冲突", c->str);
        } else if (set_id(t, id, db, dn, err, cap)) return -1;
        t->special[id] = (uint8_t)at->special;
    }
    return 0;
}

static int load_merges(tok *t, const jval *merges, char *err, size_t cap) {
    t->mcap = pow2_at_least(merges->len * 2);
    t->mh = malloc(t->mcap * sizeof *t->mh);
    if (!t->mh) return fail(err, cap, "内存不足");
    for (size_t i = 0; i < t->mcap; i++) t->mh[i].id = -1;

    uint8_t a[1024], b[1024], ab[2048];
    for (size_t r = 0; r < merges->len; r++) {
        const jval *m = &merges->items[r];
        const char *sa, *sb; size_t la, lb;
        /* 两种写法：新版是 ["左", "右"]，旧版是 "左 右"（byte-level 的 token 里不会有空格） */
        if (m->type == J_ARR && m->len == 2 && m->items[0].type == J_STR && m->items[1].type == J_STR) {
            sa = m->items[0].str; la = m->items[0].len; sb = m->items[1].str; lb = m->items[1].len;
        } else if (m->type == J_STR) {
            const char *sp = memchr(m->str, ' ', m->len);
            if (!sp || memchr(sp + 1, ' ', m->len - (size_t)(sp + 1 - m->str)))
                return fail(err, cap, "第 %zu 条合并规则 \"%s\" 格式不对", r, m->str);
            sa = m->str; la = (size_t)(sp - m->str); sb = sp + 1; lb = m->len - la - 1;
        } else return fail(err, cap, "第 %zu 条合并规则格式不对", r);
        size_t na, nb;
        if (la > sizeof a || lb > sizeof b || unmap_token(sa, la, a, &na) || unmap_token(sb, lb, b, &nb))
            return fail(err, cap, "第 %zu 条合并规则含非 byte-level 字符", r);
        memcpy(ab, a, na); memcpy(ab + na, b, nb);
        int32_t ia = vocab_find(t, a, na), ib = vocab_find(t, b, nb), iab = vocab_find(t, ab, na + nb);
        if (ia < 0 || ib < 0 || iab < 0)
            return fail(err, cap, "第 %zu 条合并规则用到了词表里没有的 token", r);
        if (merge_find(t, ia, ib)) return fail(err, cap, "第 %zu 条合并规则重复", r);
        uint64_t key = (uint64_t)(uint32_t)ia << 32 | (uint32_t)ib;
        size_t h = mix(key) & (t->mcap - 1);
        while (t->mh[h].id >= 0) h = (h + 1) & (t->mcap - 1);
        t->mh[h] = (merge_slot){key, (int32_t)r, iab};
    }
    return 0;
}

int tok_load(tok **out, const char *path, char *err, size_t cap) {
    init_byte_map();
    char *buf; size_t n;
    if (read_file(path, &buf, &n, err, cap)) return -1;
    jval root;
    int rc = json_parse(&root, buf, n, err, cap);
    free(buf);
    if (rc) return -1;
    tok *t = calloc(1, sizeof *t);
    if (!t) { json_free(&root); return fail(err, cap, "内存不足"); }
    const jval *model = json_get(&root, "model");
    const jval *vocab = json_get(model, "vocab"), *merges = json_get(model, "merges");
    const jval *added = json_get(&root, "added_tokens");
    if (check_pipeline(&root, err, cap)) rc = -1;
    else if (!vocab || vocab->type != J_OBJ || !merges || merges->type != J_ARR || !added || added->type != J_ARR)
        rc = fail(err, cap, "缺少 model.vocab / model.merges / added_tokens");
    else rc = load_vocab(t, vocab, added, err, cap) || load_merges(t, merges, err, cap) ? -1 : 0;
    json_free(&root);
    if (rc) { tok_free(t); return -1; }
    *out = t;
    return 0;
}

void tok_free(tok *t) {
    if (!t) return;
    for (size_t i = 0; t->bytes && i < t->n; i++) free(t->bytes[i]);
    for (size_t i = 0; i < t->nadded; i++) free(t->added[i].s);
    free(t->bytes); free(t->blen); free(t->special); free(t->vh); free(t->mh); free(t->added);
    free(t);
}

size_t tok_size(const tok *t) { return t->n; }
int tok_is_special(const tok *t, int32_t id) { return id >= 0 && (size_t)id < t->n && t->special[id]; }

int32_t tok_find(const tok *t, const char *s) {
    size_t n = strlen(s);
    for (size_t i = 0; i < t->nadded; i++)
        if (t->added[i].len == n && memcmp(t->added[i].s, s, n) == 0) return t->added[i].id;
    return vocab_find(t, (const uint8_t *)s, n);
}

/* ---------- 预分词 ----------
 *
 * 正则的 7 个分支按顺序尝试，第一个能匹配的分支获胜（这是回溯式正则引擎的规则，
 * 不是“最长匹配”）。因为任何字符都至少能被某个分支匹配，所以每次匹配都从当前位置开始，
 * 整段文本被切成首尾相接的若干段。match_at 返回从 i 开始的这一段的终点。 */

static int is_crlf(uint32_t c) { return c == '\r' || c == '\n'; }
static int is_lm(uint32_t c) { return uc_class(c) == UC_LM; }
static int is_num(uint32_t c) { return uc_class(c) == UC_N; }
/* [^\s\p{L}\p{M}\p{N}]：既不是空白、也不是字母、标记、数字 */
static int is_punct(uint32_t c) { return !uc_is_space(c) && uc_class(c) == UC_OTHER; }

/* (?i:...) 里的大小写不敏感。Unicode 的大小写折叠里，能折叠成 s 的除了 S 还有 ſ（U+017F）；
 * t、r、e、v、m、l、d 只有大小写两种。 */
static uint32_t fold(uint32_t c) {
    if (c >= 'A' && c <= 'Z') return c + 32;
    if (c == 0x17F) return 's';
    return c;
}

static size_t match_at(const uint32_t *c, size_t n, size_t i) {
    /* 1. (?i:'s|'t|'re|'ve|'m|'ll|'d) */
    if (c[i] == '\'' && i + 1 < n) {
        uint32_t x = fold(c[i + 1]), y = i + 2 < n ? fold(c[i + 2]) : 0;
        if (x == 's' || x == 't' || x == 'm' || x == 'd') return i + 2;
        if ((x == 'r' && y == 'e') || (x == 'v' && y == 'e') || (x == 'l' && y == 'l')) return i + 3;
    }
    /* 2. [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ ：可选的一个前导字符，后面跟一串字母或标记 */
    {
        size_t j = n;
        if (is_lm(c[i])) j = i;
        else if (!is_crlf(c[i]) && !is_num(c[i]) && i + 1 < n && is_lm(c[i + 1])) j = i + 1;
        if (j < n) { while (j < n && is_lm(c[j])) j++; return j; }
    }
    /* 3. \p{N} ：一次只取一个数字字符 */
    if (is_num(c[i])) return i + 1;
    /* 4.  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]* ：可选的一个空格，一串标点符号，再带上紧跟的换行 */
    {
        size_t j = n;
        if (c[i] == ' ' && i + 1 < n && is_punct(c[i + 1])) j = i + 1;
        else if (is_punct(c[i])) j = i;
        if (j < n) {
            while (j < n && is_punct(c[j])) j++;
            while (j < n && is_crlf(c[j])) j++;
            return j;
        }
    }
    /* 剩下三个分支都从空白开始。先找出这一串空白 [i, j)。 */
    size_t j = i;
    while (j < n && uc_is_space(c[j])) j++;
    /* 走到这里 c[i] 一定是空白：字母、标记、数字、标点都已经被前面的分支接走了 */
    /* 5. \s*[\r\n]+ ：\s* 先吃掉整串空白，再回退到最后一个换行，[\r\n]+ 匹配它 */
    for (size_t k = j; k > i; k--)
        if (is_crlf(c[k - 1])) return k;
    /* 6. \s+(?!\S) ：后面不能紧跟非空白。到了文本末尾就整串都要；
     *    否则回退一个字符，把最后一个空白留给下一段（于是 "  x" 切成 " " 和 " x"） */
    if (j == n) return j;
    if (j - 1 > i) return j - 1;
    /* 7. \s+ */
    return j;
}

size_t tok_pretokenize(const uint32_t *c, size_t n, size_t *ends) {
    size_t k = 0;
    for (size_t i = 0; i < n;) ends[k++] = i = match_at(c, n, i);
    return k;
}

/* ---------- BPE ---------- */

typedef struct { int32_t *v; size_t n, cap; } ivec;

static int ivec_push(ivec *a, int32_t x) {
    if (a->n == a->cap) {
        size_t nc = a->cap ? a->cap * 2 : 64;
        int32_t *t = realloc(a->v, nc * sizeof *t);
        if (!t) return -1;
        a->v = t; a->cap = nc;
    }
    a->v[a->n++] = x;
    return 0;
}

/* 对一段字节做 BPE：先把每个字节变成单字节 token，然后反复找出相邻两个 token 中
 * 优先级最高（rank 最小）的那一对合并，相同时取最左边的，直到没有可合并的为止。
 * 这是最朴素的 O(n²) 写法，第二卷再优化。 */
static int bpe(const tok *t, const uint8_t *s, size_t n, ivec *out) {
    int32_t *sym = malloc((n ? n : 1) * sizeof *sym);
    if (!sym) return -1;
    for (size_t i = 0; i < n; i++) sym[i] = t->byte_id[s[i]];
    size_t m = n;
    for (;;) {
        int32_t best_rank = INT32_MAX, best_id = -1;
        size_t best = 0;
        for (size_t i = 0; i + 1 < m; i++) {
            const merge_slot *ms = merge_find(t, sym[i], sym[i + 1]);
            if (ms && ms->rank < best_rank) { best_rank = ms->rank; best_id = ms->id; best = i; }
        }
        if (best_id < 0) break;
        sym[best] = best_id;
        memmove(sym + best + 1, sym + best + 2, (m - best - 2) * sizeof *sym);
        m--;
    }
    int rc = 0;
    for (size_t i = 0; i < m && rc == 0; i++) rc = ivec_push(out, sym[i]);
    free(sym);
    return rc;
}

/* 在 [s, s+n) 里从 from 开始找附加 token 的最左匹配；同一位置有多个时取最长的。
 * want_norm 选择找哪一类（归一化前匹配的，还是归一化后匹配的）。找不到返回 n。 */
static size_t find_added(const tok *t, const uint8_t *s, size_t n, size_t from, int want_norm, const added_tok **hit) {
    for (size_t i = from; i < n; i++) {
        const added_tok *best = NULL;
        for (size_t k = 0; k < t->nadded; k++) {
            const added_tok *a = &t->added[k];
            if (a->normalized != want_norm || a->len > n - i || memcmp(s + i, a->s, a->len)) continue;
            if (!best || a->len > best->len) best = a;
        }
        if (best) { *hit = best; return i; }
    }
    return n;
}

/* 处理一段已归一化、不含附加 token 的文本：预分词，然后每段做 BPE。 */
static int encode_plain(const tok *t, const uint32_t *cp, size_t n, ivec *out) {
    size_t *ends = malloc((n ? n : 1) * sizeof *ends);
    uint8_t *buf = malloc(n * 4 + 1);
    int rc = ends && buf ? 0 : -1;
    size_t k = rc ? 0 : tok_pretokenize(cp, n, ends);
    for (size_t p = 0, start = 0; p < k && rc == 0; start = ends[p++]) {
        size_t len = 0;
        for (size_t i = start; i < ends[p]; i++) len += utf8_put(buf + len, cp[i]);
        rc = bpe(t, buf, len, out);
    }
    free(ends); free(buf);
    return rc;
}

/* 一段归一化之前的文本（不含“归一化前匹配”的附加 token）：解码成码点，NFC，
 * 再在归一化后的文本里找“归一化后匹配”的附加 token，其余部分交给 encode_plain。 */
static int encode_segment(const tok *t, const uint8_t *s, size_t n, ivec *out) {
    uint32_t *cp = malloc((n ? n : 1) * sizeof *cp), *nc = NULL;
    if (!cp) return -1;
    size_t ncp = 0, i = 0, nn;
    while (i < n) utf8_next(s, n, &i, &cp[ncp++]);          /* 调用方已检查过 UTF-8 */
    int rc = uc_nfc(cp, ncp, &nc, &nn);
    free(cp);
    if (rc) return -1;
    /* 归一化后的文本转回 UTF-8 才能按字节匹配；同时记下每个字节对应第几个码点 */
    uint8_t *u = malloc(nn * 4 + 1);
    size_t *at = malloc((nn * 4 + 1) * sizeof *at), un = 0;
    if (!u || !at) rc = -1;
    for (size_t q = 0; rc == 0 && q < nn; q++) {
        size_t w = utf8_put(u + un, nc[q]);
        for (size_t z = 0; z < w; z++) at[un + z] = q;
        un += w;
    }
    at[un] = nn;
    for (size_t pos = 0; rc == 0 && pos < un;) {
        const added_tok *hit = NULL;
        size_t m = find_added(t, u, un, pos, 1, &hit);
        rc = encode_plain(t, nc + at[pos], at[m] - at[pos], out);
        if (rc == 0 && hit) { rc = ivec_push(out, hit->id); m += hit->len; }
        pos = m;
    }
    free(u); free(at); free(nc);
    return rc;
}

int tok_encode(const tok *t, const uint8_t *s, size_t n, int32_t **ids, size_t *nids, char *err, size_t cap) {
    for (size_t i = 0; i < n;) {
        uint32_t c;
        if (utf8_next(s, n, &i, &c)) return fail(err, cap, "输入在第 %zu 字节处不是合法的 UTF-8", i);
    }
    ivec out = {0};
    int rc = 0;
    /* 第一步：在原文里找“归一化前匹配”的附加 token（Qwen 的全部特殊 token 都是这一类），
     * 它们直接变成 id，不参与归一化和 BPE。中间的普通文本交给 encode_segment。 */
    for (size_t pos = 0; rc == 0 && pos < n;) {
        const added_tok *hit = NULL;
        size_t m = find_added(t, s, n, pos, 0, &hit);
        rc = encode_segment(t, s + pos, m - pos, &out);
        if (rc == 0 && hit) { rc = ivec_push(&out, hit->id); m += hit->len; }
        pos = m;
    }
    if (rc) { free(out.v); return fail(err, cap, "内存不足"); }
    *ids = out.v ? out.v : malloc(1);
    *nids = out.n;
    return 0;
}

int tok_decode(const tok *t, const int32_t *ids, size_t n, int skip_special,
               uint8_t **out, size_t *on, char *err, size_t cap) {
    size_t len = 0;
    for (size_t i = 0; i < n; i++) {
        if (ids[i] < 0 || (size_t)ids[i] >= t->n || !t->bytes[ids[i]])
            return fail(err, cap, "第 %zu 个 id %d 不在词表里", i, ids[i]);
        if (!(skip_special && t->special[ids[i]])) len += t->blen[ids[i]];
    }
    uint8_t *o = malloc(len + 1);
    if (!o) return fail(err, cap, "内存不足");
    size_t m = 0;
    for (size_t i = 0; i < n; i++) {
        if (skip_special && t->special[ids[i]]) continue;
        memcpy(o + m, t->bytes[ids[i]], t->blen[ids[i]]);
        m += t->blen[ids[i]];
    }
    *out = o; *on = m;
    return 0;
}
