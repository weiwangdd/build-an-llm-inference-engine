/* st_dump — 逐个张量打印：校验和、dtype、形状、名字。
 * 输出格式与 tests/ref_dump.py 完全相同，两者 diff 为空即验收通过。 */
#include <inttypes.h>
#include <stdio.h>
#include "st.h"

static uint64_t fnv1a64(const uint8_t *p, uint64_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint64_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ull; }
    return h;
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "用法：%s 文件.safetensors\n", argv[0]); return 2; }
    st_file f;
    char err[256];
    if (st_open(&f, argv[1], err, sizeof err)) { fprintf(stderr, "错误：%s\n", err); return 1; }
    for (size_t i = 0; i < f.count; i++) {
        const st_tensor *t = &f.tensors[i];
        printf("%016" PRIx64 " %s [", fnv1a64(t->data, t->nbytes), st_dtype_name(t->dtype));
        for (int d = 0; d < t->ndim; d++) printf(d ? ",%" PRIu64 : "%" PRIu64, t->shape[d]);
        printf("] ");
        /* 名字按 JSON 规则转义输出，避免换行等字符破坏逐行比对 */
        for (const unsigned char *s = (const unsigned char *)t->name; *s; s++) {
            if (*s == '\\') fputs("\\\\", stdout);
            else if (*s < 0x20) printf("\\u%04x", *s);
            else putchar(*s);
        }
        putchar('\n');
    }
    st_close(&f);
    return 0;
}
