/* gen_unicode — 从 Unicode 字符数据库（UCD）的文本文件生成 src/02-tokenizer/unicode_tables.c。
 *
 * 用法：gen_unicode <UCD 9.0 目录> <UCD 16.0 目录> > src/02-tokenizer/unicode_tables.c
 * 9.0 目录里要有 UnicodeData.txt 和 DerivedNormalizationProps.txt，16.0 目录里只用 UnicodeData.txt。
 *
 * 为什么要两个版本：官方 tokenizers 库的 NFC 归一化内置的是 Unicode 9.0 的数据，
 * 而它的正则引擎用的是 Unicode 16.0 的字符类别。要和它逐字节一致，只能各用各的版本。
 * 详见第 2 章“两个 Unicode 版本”一节。
 *
 * 数据文件来自 Unicode 官方：
 *   https://www.unicode.org/Public/9.0.0/ucd/   和   https://www.unicode.org/Public/16.0.0/ucd/
 * 写作时我们的环境访问不了 unicode.org，用的是 ICU 仓库里的副本（github.com/unicode-org/icu，
 * icu4c/source/data/unidata/，标签 release-58-2 对应 9.0，release-76-1 对应 16.0）。
 * 本仓库提交的 src/02-tokenizer/unicode_tables.c 由以下文件生成（SHA-256）：
 *   9.0  UnicodeData.txt                68dfc414d28257b9b5d6ddbb8b466c768c00ebdf6cbf7784364a9b6cad55ee8f
 *   9.0  DerivedNormalizationProps.txt  ba369a14c8d0ea8b7e9063991347ac698f915c393bcd5bbf26fac14f2b814ba1
 *   16.0 UnicodeData.txt                ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f
 *
 * 生成的文件已经提交进仓库，平时编译不需要运行本程序：make unicode-tables UCD9=... UCD16=... */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NCP 0x110000

static uint8_t  ccc[NCP];
static uint8_t  cls[NCP];          /* 0 其他，1 字母或标记（L/M），2 数字（N） */
static uint32_t *dec[NCP];         /* 规范分解（只有一层），dec[c][0] 是长度 */
static uint8_t  excl[NCP];         /* Full_Composition_Exclusion */

static void die(const char *m, const char *p) { fprintf(stderr, "gen_unicode：%s %s\n", m, p); exit(1); }

static FILE *open_in(const char *dir, const char *name, char *path) {
    snprintf(path, 1024, "%s/%s", dir, name);
    FILE *f = fopen(path, "r");
    if (!f) die("打不开", path);
    return f;
}

/* 把 UnicodeData.txt 的一行切成 15 个字段（原地改写）。 */
static int split(char *line, char **fld) {
    int n = 0;
    fld[n++] = line;
    for (char *p = line; *p && n < 15; p++)
        if (*p == ';') { *p = 0; fld[n++] = p + 1; }
    return n;
}

/* 读 UnicodeData.txt。want_cls 为真时取类别，want_norm 为真时取组合类和分解。 */
static void read_unicodedata(const char *dir, int want_cls, int want_norm) {
    char path[1024], line[1024], *f[15];
    FILE *fp = open_in(dir, "UnicodeData.txt", path);
    long first = -1;
    while (fgets(line, sizeof line, fp)) {
        line[strcspn(line, "\r\n")] = 0;
        if (split(line, f) != 15) die("字段数不对", path);
        long cp = strtol(f[0], NULL, 16);
        if (cp < 0 || cp >= NCP) die("码点越界", path);
        /* 大段的汉字、韩文音节等只写首尾两行：<..., First> 和 <..., Last> */
        long lo = cp;
        if (strstr(f[1], ", First>")) { first = cp; continue; }
        if (strstr(f[1], ", Last>")) { if (first < 0) die("Last 前没有 First", path); lo = first; first = -1; }
        for (long c = lo; c <= cp; c++) {
            if (want_cls) cls[c] = (f[2][0] == 'L' || f[2][0] == 'M') ? 1 : f[2][0] == 'N' ? 2 : 0;
            if (want_norm) ccc[c] = (uint8_t)atoi(f[3]);
        }
        if (want_norm && f[5][0] && f[5][0] != '<') {   /* 带 <tag> 的是兼容分解，NFC 不用 */
            uint32_t tmp[8]; int n = 0;
            for (char *p = f[5]; *p;) {
                char *e; long v = strtol(p, &e, 16);
                if (e == p || n == 8) die("分解格式不对", path);
                tmp[n++] = (uint32_t)v; p = e;
                while (*p == ' ') p++;
            }
            dec[cp] = malloc(sizeof(uint32_t) * (size_t)(n + 1));
            dec[cp][0] = (uint32_t)n;
            memcpy(dec[cp] + 1, tmp, sizeof(uint32_t) * (size_t)n);
        }
    }
    fclose(fp);
}

static void read_exclusions(const char *dir) {
    char path[1024], line[1024];
    FILE *fp = open_in(dir, "DerivedNormalizationProps.txt", path);
    while (fgets(line, sizeof line, fp)) {
        line[strcspn(line, "#\r\n")] = 0;
        char *semi = strchr(line, ';');
        if (!semi) continue;
        char *prop = semi + 1;
        while (*prop == ' ') prop++;
        size_t pl = strcspn(prop, " ;");
        if (pl != strlen("Full_Composition_Exclusion") || strncmp(prop, "Full_Composition_Exclusion", pl)) continue;
        char *e; long lo = strtol(line, &e, 16), hi = lo;
        if (e[0] == '.' && e[1] == '.') hi = strtol(e + 2, NULL, 16);
        for (long c = lo; c <= hi; c++) excl[c] = 1;
    }
    fclose(fp);
}

/* 递归展开成完全分解，写进 out，返回长度。 */
static int full_decomp(uint32_t c, uint32_t *out) {
    if (!dec[c]) { out[0] = c; return 1; }
    int n = 0;
    for (uint32_t i = 1; i <= dec[c][0]; i++) n += full_decomp(dec[c][i], out + n);
    return n;
}

/* 把一个按码点索引的字节数组压成区间表：只输出非零值的区间。 */
static void emit_ranges(const char *name, const uint8_t *v) {
    printf("const uc_range %s[] = {\n", name);
    size_t n = 0;
    for (long c = 0; c < NCP;) {
        if (!v[c]) { c++; continue; }
        long s = c;
        while (c + 1 < NCP && v[c + 1] == v[s]) c++;
        printf("    {0x%04lX, 0x%04lX, %u},\n", s, c, v[s]);
        n++; c++;
    }
    printf("};\nconst size_t %s_N = %zu;\n\n", name, n);
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "用法：%s <UCD 9.0 目录> <UCD 16.0 目录>\n", argv[0]); return 2; }
    read_unicodedata(argv[1], 0, 1);
    read_exclusions(argv[1]);
    read_unicodedata(argv[2], 1, 0);

    printf("/* 由 tools/gen_unicode.c 生成，不要手改。\n"
           " * 组合类、分解、合成：Unicode 9.0；字符类别：Unicode 16.0。 */\n"
           "#include \"unicode_tables.h\"\n\n");
    emit_ranges("UC_CLASS", cls);
    emit_ranges("UC_CCC", ccc);

    /* 完全分解：按码点升序，数据平铺在一个数组里 */
    static uint32_t buf[64];
    size_t nd = 0, off = 0;
    printf("const uc_decomp UC_DECOMP[] = {\n");
    for (uint32_t c = 0; c < NCP; c++) if (dec[c]) {
        int n = full_decomp(c, buf);
        printf("    {0x%04X, %zu, %d},\n", c, off, n);
        off += (size_t)n; nd++;
    }
    printf("};\nconst size_t UC_DECOMP_N = %zu;\n\nconst uint32_t UC_DECOMP_DATA[] = {\n", nd);
    for (uint32_t c = 0; c < NCP; c++) if (dec[c]) {
        int n = full_decomp(c, buf);
        printf("   ");
        for (int i = 0; i < n; i++) printf(" 0x%04X,", buf[i]);
        printf("\n");
    }
    printf("};\n\n");

    /* 合成表：分解恰好是两个码点、且不在排除表里的字符（“主合成字符”）。按 (a, b) 升序。 */
    size_t np = 0;
    typedef struct { uint32_t a, b, c; } pr;
    pr *pairs = malloc(sizeof(pr) * NCP / 16);
    for (uint32_t c = 0; c < NCP; c++)
        if (dec[c] && dec[c][0] == 2 && !excl[c]) pairs[np++] = (pr){dec[c][1], dec[c][2], c};
    for (size_t i = 1; i < np; i++)               /* 插入排序，一千来项足够 */
        for (size_t j = i; j > 0 && (pairs[j - 1].a > pairs[j].a ||
                    (pairs[j - 1].a == pairs[j].a && pairs[j - 1].b > pairs[j].b)); j--) {
            pr t = pairs[j]; pairs[j] = pairs[j - 1]; pairs[j - 1] = t;
        }
    printf("const uc_pair UC_COMP[] = {\n");
    for (size_t i = 0; i < np; i++) printf("    {0x%04X, 0x%04X, 0x%04X},\n", pairs[i].a, pairs[i].b, pairs[i].c);
    printf("};\nconst size_t UC_COMP_N = %zu;\n", np);
    return 0;
}
