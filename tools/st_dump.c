/* st_dump — 把一个或多个分片合并后，逐个张量打印：CRC-32、dtype、形状、名字。
 * 输出格式与 tests/ref_dump.py 完全相同，两者 diff 为空即验收通过。 */
#include <inttypes.h>
#include <stdio.h>
#include "st.h"

/* CRC-32（IEEE 802.3，与 zlib 的 crc32 相同）：多项式 0xEDB88320，按字节查表。 */
static uint32_t crc_table[256];

static void crc32_init(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        crc_table[i] = c;
    }
}

static uint32_t crc32(const uint8_t *p, uint64_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (uint64_t i = 0; i < n; i++) c = crc_table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "用法：%s 分片1.safetensors [分片2.safetensors ...]\n", argv[0]); return 2; }
    st_set s;
    char err[512];
    if (st_set_open(&s, (const char *const *)(argv + 1), (size_t)(argc - 1), err, sizeof err)) {
        fprintf(stderr, "错误：%s\n", err);
        return 1;
    }
    crc32_init();
    for (size_t i = 0; i < s.count; i++) {
        const st_tensor *t = s.tensors[i];
        printf("%08" PRIx32 " %s [", crc32(t->data, t->nbytes), st_dtype_name(t->dtype));
        for (int d = 0; d < t->ndim; d++) printf(d ? ",%" PRIu64 : "%" PRIu64, t->shape[d]);
        printf("] ");
        /* 名字按 JSON 规则转义输出，避免换行等字符破坏逐行比对 */
        for (const unsigned char *c = (const unsigned char *)t->name; *c; c++) {
            if (*c == '\\') fputs("\\\\", stdout);
            else if (*c < 0x20) printf("\\u%04x", *c);
            else putchar(*c);
        }
        putchar('\n');
    }
    st_set_close(&s);
    return 0;
}
