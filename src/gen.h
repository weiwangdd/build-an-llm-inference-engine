/* gen.h — 第 6 章：生成循环。
 *
 *   提示词（可以分几块）→ 模型 → 最后一行 logits → 采样 → 新 token → 模型（只算这一个）→ …
 *
 * 直到生成了结束符（eos），或者生成了 max_new 个。用到第 6 章的缓存（model_step），
 * 也可以不用缓存、每一步都从头算（第 5 章的做法），给测试当标准答案。 */
#ifndef GEN_H
#define GEN_H

#include <stddef.h>
#include <stdint.h>
#include "model.h"
#include "sample.h"

#define GEN_MAX_EOS 16

/* generation_config.json 里与生成有关的部分 */
typedef struct {
    sample_cfg sample;
    int32_t eos[GEN_MAX_EOS];       /* 结束符，生成其中任何一个就停 */
    size_t n_eos;
} gen_cfg;

/* 读 generation_config.json。只接受下面这些字段，别的（如 repetition_penalty）一律报错：
 *   eos_token_id（整数或整数数组，必需）、do_sample（必需）；do_sample 为 true 时 temperature、top_k、top_p
 *   也必需（transformers 不同版本的默认值不同，不猜）；bos_token_id、pad_token_id、transformers_version 只记录不用。
 * eos 的 id 要小于 vocab。 */
int gen_cfg_load(gen_cfg *g, const char *path, size_t vocab, char *err, size_t errcap);

typedef struct {
    size_t   max_new;     /* 最多生成几个 token，至少 1 */
    size_t   chunk;       /* 提示词每次最多交给模型几个 token；0 表示一次全给 */
    int      no_cache;    /* 1：不用缓存，每一步都对全部 token 从头前向（第 5 章的做法） */
    uint64_t seed;        /* 随机数种子；贪心解码不用随机数 */
} gen_opts;

typedef enum { GEN_EOS, GEN_LENGTH } gen_stop;

/* 每选出一个 token 调用一次。k 从 0 数起；logits 是选它时看的那一行；ids、probs、nkept 是采样的分布
 * （sample_dist 的输出）；u 是这一步用的随机数，贪心解码时是 -1。 */
typedef void (*gen_fn)(void *ctx, size_t k, const float *logits, size_t vocab,
                       const int32_t *ids, const double *probs, size_t nkept, double u, int32_t tok);

/* 从 n 个 token 的提示词出发生成。out 至少要有 o->max_new 个位置，生成的个数写进 *nout，
 * 为什么停下写进 *why。cb 可以为 NULL。提示词加上 max_new 超过 max_position_embeddings 时报错。 */
int generate(const model *m, const gen_cfg *g, const gen_opts *o, const int32_t *prompt, size_t n,
             int32_t *out, size_t *nout, gen_stop *why, gen_fn cb, void *ctx, char *err, size_t errcap);

#endif
