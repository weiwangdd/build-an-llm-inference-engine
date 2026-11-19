/* model.h — 第 4 章：读 config.json，加载一层的权重，算一层的前向。
 *
 * 只支持 Qwen3.8（transformers 里的 model_type qwen3_5）的纯文本部分。
 * config.json 里任何我们不认识或不支持的取值都报错，而不是猜一个默认值。 */
#ifndef MODEL_H
#define MODEL_H

#include <stddef.h>
#include <stdint.h>
#include "st.h"

#define MODEL_MAX_LAYERS 256

typedef struct {
    size_t vocab, hidden, inter, n_layers;
    double eps;                       /* rms_norm_eps */
    unsigned char full[MODEL_MAX_LAYERS]; /* layer_types：1 是全注意力，0 是线性注意力 */
    /* 全注意力 */
    size_t heads, kv_heads, head_dim, rot;
    double theta;
    /* 线性注意力 */
    size_t lk_heads, lv_heads, lk_dim, lv_dim, conv_k;
    size_t max_pos;                   /* max_position_embeddings：位置的上限（第 6 章用） */
} model_cfg;

/* 读 config.json（多模态模型读其中的 text_config）。成功返回 0，失败返回 -1 并把原因写进 err。 */
int model_cfg_load(model_cfg *c, const char *path, char *err, size_t errcap);

/* 一个 BF16 矩阵，[rows, cols]，数据指向 mmap 区域。 */
typedef struct { const uint8_t *data; size_t rows, cols; } bf16_mat;

typedef struct {
    int full;
    float *in_norm, *post_norm;              /* [hidden]，1 + w 形式 */
    bf16_mat gate, up, down;                 /* 前馈层 */
    /* 全注意力 */
    bf16_mat q, k, v, o;
    float *q_norm, *k_norm;                  /* [head_dim]，1 + w 形式 */
    /* 线性注意力 */
    bf16_mat qkv, z, a, b, out;
    float *conv;                             /* [conv_dim, conv_k] */
    float *A_log, *dt_bias;                  /* [lv_heads] */
    float *gnorm;                            /* [lv_dim]，w 形式 */
} layer_w;

/* 从权重表里取第 layer 层的全部张量，检查名字、dtype 和形状。 */
int  layer_load(layer_w *w, const model_cfg *c, const st_set *s, size_t layer, char *err, size_t errcap);
void layer_free(layer_w *w);

/* 最终归一化的权重（model.norm），[hidden]。返回 malloc 出来的数组，失败返回 NULL。 */
float *final_norm_load(const model_cfg *c, const st_set *s, char *err, size_t errcap);

/* 每算完一步就调用一次，把这一步的结果交给调用方（测试用它输出中间张量）。
 * 张量看作 [rows, cols]，rows 是 token 数。可以为 NULL。 */
typedef void (*trace_fn)(void *ctx, const char *name, const float *x, size_t rows, size_t cols);

/* 一层的前向：h 是 [n, hidden]，就地更新为这一层的输出。pos[t] 是第 t 个 token 的位置。
 * 成功返回 0，失败（内存不足）返回 -1。 */
int layer_forward(float *h, size_t n, const int32_t *pos, const model_cfg *c, const layer_w *w,
                  trace_fn trace, void *ctx, char *err, size_t errcap);

/* ---------- 第 5 章：整个模型 ---------- */

typedef struct model {
    model_cfg cfg;
    const uint8_t *embed;        /* 嵌入表，BF16 [vocab, hidden]，指向 mmap 区域 */
    layer_w *layers;             /* [n_layers] */
    float *final_norm;           /* [hidden]，1 + w 形式 */
    bf16_mat lm_head;            /* 输出层，[vocab, hidden] */
    struct model *gpu;           /* 第 7 章：model_to_gpu（gpu.h）之后指向显存里的那一份，否则为 NULL */
} model;

/* 读配置，加载嵌入表、每一层、最终归一化和输出层，逐个检查名字、dtype 和形状。
 * 权重仍然留在 s 的 mmap 里，s 要比 m 活得久。
 * 之后可以用 model_to_gpu（gpu.h）把模型搬到 GPU 上，下面的函数就都在 GPU 上算（第 7 章）。 */
int  model_load(model *m, const char *config_path, const st_set *s, char *err, size_t errcap);
void model_free(model *m);

/* 整个模型的前向：ids 是 n 个 token，位置依次是 0 .. n-1。
 * logits 是 [n, vocab]，第 t 行是“看过第 0 .. t 个 token 之后，下一个 token 的分数”。
 * trace 收到的名字是 embed、layers.<i>.<第 4 章的步骤名>、final_norm、logits。 */
int  model_forward(const model *m, const int32_t *ids, size_t n, float *logits,
                   trace_fn trace, void *ctx, char *err, size_t errcap);

/* 贪心解码选哪个 token：分数最大的那个；有几个同样大时取 id 最小的。
 * 有 NaN 时报错返回 -1（NaN 和谁比都是假，不报错的话结果取决于它出现在哪）。 */
int  logits_argmax(const float *row, size_t vocab, int32_t *out, char *err, size_t errcap);

/* ---------- 第 6 章：缓存 ----------
 *
 * 生成时每次只给新来的 token，前面那些 token 留下的东西存在缓存里：
 *   全注意力层   每个 token RoPE 之后的 k 和 v（“KV 缓存”），随序列变长
 *   线性注意力层 卷积要往前看的最近 conv_k-1 个 token 的输入，和门控 delta 规则的状态矩阵，
 *                大小固定
 * 分几次交给 model_step 与一次交给 model_forward，logits 逐位相同。 */

typedef struct {
    float *k, *v;            /* 全注意力层：[cap, kv_heads * head_dim] */
    float *conv;             /* 线性注意力层：[conv_k - 1, 卷积的通道数]，旧的在前 */
    double *S;               /* 线性注意力层：[lv_heads, lk_dim, lv_dim] */
} layer_cache;

typedef struct {
    size_t n_layers;
    size_t cap, len;         /* 最多能放多少个 token、已经放了多少个 */
    int broken;              /* model_step 中途出错后为 1，缓存不能再用 */
    int gpu;                 /* 第 7 章：1 表示 layers 里的指针指向显存（给 GPU 上的模型用） */
    layer_cache *layers;
} model_cache;

/* 给模型 m 建一个能放 cap 个 token 的空缓存。cap 不能超过 max_position_embeddings。
 * m 在 GPU 上时缓存也建在显存里（第 7 章）。 */
int  model_cache_init(model_cache *kc, const model *m, size_t cap, char *err, size_t errcap);
void model_cache_free(model_cache *kc);

/* 把 n 个新 token 接在缓存里的 kc->len 个 token 后面算一遍，位置是 kc->len .. kc->len + n - 1。
 * logits 是 [n, vocab]，与 model_forward 对“缓存里的 token 加上这 n 个”算出的最后 n 行逐位相同。
 * 成功后 kc->len 增加 n。放不下时报错，缓存不变；中途出错时缓存标记为损坏。 */
int  model_step(const model *m, model_cache *kc, const int32_t *ids, size_t n, float *logits,
                trace_fn trace, void *ctx, char *err, size_t errcap);

/* ---------- 第 8 章：批处理 ----------
 *
 * 服务端同时有好几个请求在生成。每个请求有自己的缓存，各自接上几个新 token（预填充时是提示词的一块，
 * 解码时是上一步选出的一个），一次前向一起算：矩阵乘对全部 token 做一次，权重只读一遍。
 * 每个序列的 logits 与单独调用 model_step 逐位相同。 */

typedef struct {
    model_cache *kc;         /* 这个序列的缓存（model_forward 内部用 NULL 表示不用缓存） */
    const int32_t *ids;      /* 新 token */
    size_t n;                /* 新 token 的个数，至少 1 */
} model_seq;

/* 一批 nseq 个序列一起往前算一步。logits 依次是第 0 个序列的 n 行、第 1 个序列的 n 行……共 Σn 行。
 * 每个缓存只能出现一次。成功后每个缓存的 len 增加各自的 n；检查不通过时什么都不变；
 * 中途出错时这一批的缓存都标记为损坏。trace 收到的张量是全部 Σn 行。 */
int  model_step_batch(const model *m, model_seq *seqs, size_t nseq, float *logits,
                      trace_fn trace, void *ctx, char *err, size_t errcap);

#endif
