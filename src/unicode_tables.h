/* unicode_tables.h — 由 tools/gen_unicode.c 生成的 Unicode 数据表的声明。 */
#ifndef UNICODE_TABLES_H
#define UNICODE_TABLES_H

#include <stddef.h>
#include <stdint.h>

typedef struct { uint32_t lo, hi; uint8_t v; } uc_range;          /* [lo, hi] 上的值都是 v */
typedef struct { uint32_t cp; uint32_t off; uint8_t len; } uc_decomp; /* 分解在 UC_DECOMP_DATA[off..off+len) */
typedef struct { uint32_t a, b, c; } uc_pair;                    /* a + b 合成为 c */

extern const uc_range  UC_CLASS[];  extern const size_t UC_CLASS_N;   /* 1 = L 或 M，2 = N */
extern const uc_range  UC_CCC[];    extern const size_t UC_CCC_N;     /* 规范组合类 */
extern const uc_decomp UC_DECOMP[]; extern const size_t UC_DECOMP_N;  /* 完全规范分解（不含韩文音节） */
extern const uint32_t  UC_DECOMP_DATA[];
extern const uc_pair   UC_COMP[];   extern const size_t UC_COMP_N;    /* 主合成对 */

#endif
