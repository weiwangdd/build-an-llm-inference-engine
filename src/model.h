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

typedef struct {
    model_cfg cfg;
    const uint8_t *embed;        /* 嵌入表，BF16 [vocab, hidden]，指向 mmap 区域 */
    layer_w *layers;             /* [n_layers] */
    float *final_norm;           /* [hidden]，1 + w 形式 */
    bf16_mat lm_head;            /* 输出层，[vocab, hidden] */
} model;

/* 读配置，加载嵌入表、每一层、最终归一化和输出层，逐个检查名字、dtype 和形状。
 * 权重仍然留在 s 的 mmap 里，s 要比 m 活得久。 */
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

#endif
