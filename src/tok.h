/* tok.h — Qwen 系列模型的 BPE 分词器，读 Hugging Face 的 tokenizer.json。
 *
 * 编码的流水线（与 Hugging Face tokenizers 库相同）：
 *   原文 → 切出特殊 token → NFC 归一化 → 预分词（一条正则把文本切成小段）
 *        → 每段按字节做 BPE 合并 → token id
 * 解码：id → 字节 → 拼起来。
 *
 * 只支持 Qwen 用到的那一种配置。tokenizer.json 里任何别的写法都会被拒绝，而不是被忽略。 */
#ifndef TOK_H
#define TOK_H

#include <stddef.h>
#include <stdint.h>

typedef struct tok tok;

int  tok_load(tok **out, const char *path, char *err, size_t errcap);
void tok_free(tok *t);

/* id 的上界（最大 id + 1）。 */
size_t tok_size(const tok *t);
/* 某个 id 是不是特殊 token（如 <|im_start|>）。 */
int    tok_is_special(const tok *t, int32_t id);
/* 按内容查特殊 token 或普通 token 的 id，找不到返回 -1。 */
int32_t tok_find(const tok *t, const char *content);

/* 编码 UTF-8 文本。*ids 由 malloc 分配。text 不是合法 UTF-8 时失败。 */
int  tok_encode(const tok *t, const uint8_t *text, size_t n, int32_t **ids, size_t *nids,
                char *err, size_t errcap);
/* 解码。skip_special 为真时跳过特殊 token。输出是原始字节，不保证是合法 UTF-8：
 * 一个汉字可能被切在两个 token 里，单独解码其中一个就只有半个字。 */
int  tok_decode(const tok *t, const int32_t *ids, size_t n, int skip_special,
                uint8_t **out, size_t *on, char *err, size_t errcap);

/* 以下两个函数把流水线的中间步骤单独暴露出来，供测试逐步对拍。 */

/* 预分词：把码点序列切成小段，第 k 段是 [ends[k-1], ends[k])。返回段数。
 * ends 至少要有 n 个位置。 */
size_t tok_pretokenize(const uint32_t *cp, size_t n, size_t *ends);

#endif
