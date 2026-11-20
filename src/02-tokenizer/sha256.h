/* sha256.h — SHA-256（FIPS 180-4）。用来确认对话模板文件与我们手写实现的那一份逐字节相同。 */
#ifndef SHA256_H
#define SHA256_H

#include <stddef.h>
#include <stdint.h>

void sha256(const uint8_t *data, size_t n, uint8_t out[32]);

#endif
