/* unicode.h — 分词器用到的 Unicode 工具：UTF-8 编解码、字符类别、NFC 归一化。 */
#ifndef UNICODE_H
#define UNICODE_H

#include <stddef.h>
#include <stdint.h>

/* 从 s[*i] 起解码一个 UTF-8 字符，成功时前移 *i 并返回 0。
 * 严格模式：过长编码、代理区码点、超过 U+10FFFF、截断的序列都返回 -1。 */
int    utf8_next(const uint8_t *s, size_t n, size_t *i, uint32_t *cp);
/* 把 cp 编码成 UTF-8 写进 out（至少 4 字节），返回字节数。 */
size_t utf8_put(uint8_t *out, uint32_t cp);
/* 把任意字节串转成合法 UTF-8：每个非法片段换成 U+FFFD，规则与 Rust 的
 * String::from_utf8_lossy 相同。返回 malloc 出来的缓冲区，*on 是长度。 */
uint8_t *utf8_lossy(const uint8_t *s, size_t n, size_t *on);

enum { UC_OTHER = 0, UC_LM = 1, UC_N = 2 };
int  uc_class(uint32_t cp);     /* 预分词正则里的 [\p{L}\p{M}] 和 \p{N} */
int  uc_is_space(uint32_t cp);  /* 正则里的 \s */
int  uc_ccc(uint32_t cp);       /* 规范组合类 */

/* NFC 归一化。输入输出都是码点数组；*out 由 malloc 分配。成功返回 0。 */
int  uc_nfc(const uint32_t *in, size_t n, uint32_t **out, size_t *on);

#endif
