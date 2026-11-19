/* engine.h — 第 8 章：同时为好几个请求生成（连续批处理）。
 *
 * 第 6 章的 generate 一次只管一个请求：预填充提示词，然后一个 token 一个 token 地解码，直到停下。
 * 服务端同时有很多请求，有的刚到、有的生成到一半。engine 把它们放在一起，每调用一次 engine_step，
 * 就把所有正在进行的请求各往前推一步，用一次 model_step_batch 一起算：
 *   还在预填充的请求交上提示词的下一块（最多 chunk 个 token），
 *   已经在解码的请求交上上一步选出的那一个 token。
 * 一个请求停下（结束符、达到个数、被取消）就离开这一批，排队的请求马上补进来，不用等整批结束。
 * 这叫“连续批处理”（continuous batching），也叫按迭代调度。
 *
 * 每个请求的采样参数、随机数种子、缓存都是自己的。批处理的前向与单独算逐位相同（model_step_batch），
 * 采样的代码和次序与 generate 相同，所以每个请求生成的 token、看到的 logits、用的随机数，
 * 都与把它单独交给 generate 时逐位相同，不管同时还有哪些请求、它们什么时候来、什么时候走。
 *
 * 只有一个线程，engine 的函数都不能同时调用。 */
#ifndef ENGINE_H
#define ENGINE_H

#include <stddef.h>
#include <stdint.h>
#include "gen.h"
#include "model.h"
#include "sample.h"

typedef struct {
    sample_cfg sample;    /* 这个请求的采样参数 */
    uint64_t   seed;      /* 随机数种子 */
    size_t     max_new;   /* 最多生成几个 token，至少 1 */
    size_t     chunk;     /* 提示词每一步最多交给模型几个 token；0 表示一次全给 */
} eng_params;

typedef enum { ENG_EOS, ENG_LENGTH, ENG_ERROR } eng_stop;

/* 选出一个 token 时调用。参数的含义与 gen_fn（gen.h）相同，多了请求的编号 id。 */
typedef void (*eng_token_fn)(void *ctx, uint64_t id, size_t k, const float *logits, size_t vocab,
                             const int32_t *ids, const double *probs, size_t nkept, double u, int32_t tok);
/* 请求结束时调用一次（被 engine_cancel 取消的除外）。why 为 ENG_ERROR 时 msg 是原因，否则为空串。
 * 回调返回之后这个请求就不存在了。 */
typedef void (*eng_done_fn)(void *ctx, uint64_t id, eng_stop why, const char *msg);

typedef struct eng_req eng_req;

typedef struct {
    const model  *m;
    const gen_cfg *g;           /* 结束符 */
    size_t max_batch;           /* 同时进行的请求最多几个 */
    size_t max_ctx;             /* 每个请求的提示词加生成的 token 最多几个，决定缓存的大小 */
    eng_req **reqs;             /* 全部请求（排队的和进行中的），按到达顺序 */
    size_t nreq, capreq;
    uint64_t next_id;
    float   *logits;            /* 一步的 logits，[Σn, vocab] */
    size_t   logit_rows;
    int32_t *kids;              /* 采样用的临时空间，[vocab] */
    double  *probs;
    eng_token_fn on_token;
    eng_done_fn  on_done;
    void        *ctx;
    /* 最近一次 engine_step 的这一批：每个序列是哪个请求、交了几个 token（测试和日志用） */
    uint64_t *last_ids;
    size_t   *last_n;
    size_t    last_nseq;
} engine;

/* max_batch 至少 1；max_ctx 在 2 到 max_position_embeddings 之间。 */
int  engine_init(engine *e, const model *m, const gen_cfg *g, size_t max_batch, size_t max_ctx,
                 eng_token_fn on_token, eng_done_fn on_done, void *ctx, char *err, size_t errcap);
void engine_free(engine *e);

/* 加一个请求，排在队尾，编号写进 *id。检查不通过（提示词为空、token id 越界、
 * 提示词加 max_new 超过 max_ctx、采样参数不合法）时报错，什么都不变。prompt 会被复制。 */
int  engine_add(engine *e, const int32_t *prompt, size_t n, const eng_params *p, uint64_t *id,
                char *err, size_t errcap);

/* 取消一个请求（比如客户端断开了）：马上从队列或这一批里拿掉，释放它的缓存，不再调用回调。
 * 没有这个请求（已经结束）时什么都不做，返回 -1。 */
int  engine_cancel(engine *e, uint64_t id);

/* 还有没有请求（排队或进行中）。 */
size_t engine_pending(const engine *e);

/* 往前推一步：把排队的请求补进这一批（最多 max_batch 个），一起算一次前向，给每个预填充完的请求选一个 token。
 * 没有请求时什么都不做。这一步的前向出错时，这一批的请求都以 ENG_ERROR 结束，engine 本身还能接着用。
 * 只有内存不足、连回调都没法调用时才返回 -1。 */
int  engine_step(engine *e, char *err, size_t errcap);

#endif
