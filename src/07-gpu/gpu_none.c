/* gpu_none.c — 第 7 章：没有 CUDA 时代替 gpu_model.c 和 gpu_ops.cu。
 * model_to_gpu 一律报错，所以模型不会到 GPU 上，其余函数也就不会被调用。 */
#include "gpu.h"

#include <stdio.h>

static int no_cuda(char *err, size_t cap) {
    if (err && cap) snprintf(err, cap, "编译时没有启用 CUDA（GPU 版用 make gpu 构建）");
    return -1;
}

int model_to_gpu(model *m, char *err, size_t cap) { (void)m; return no_cuda(err, cap); }

int gpu_layer_forward(float *h, size_t n, const int32_t *pos, const model_cfg *c, const layer_w *w,
                      trace_fn trace, void *ctx, char *err, size_t cap) {
    (void)h; (void)n; (void)pos; (void)c; (void)w; (void)trace; (void)ctx;
    return no_cuda(err, cap);
}

int gpu_model_run(const model *g, const model_seq *seqs, size_t nseq, float *logits,
                  trace_fn trace, void *ctx, char *err, size_t cap) {
    (void)g; (void)seqs; (void)nseq; (void)logits; (void)trace; (void)ctx;
    return no_cuda(err, cap);
}

void gpu_model_free(model *g) { (void)g; }

int gpu_cache_alloc(model_cache *kc, const model_cfg *c, char *err, size_t cap) {
    (void)kc; (void)c;
    return no_cuda(err, cap);
}

void gpu_cache_free(model_cache *kc) { (void)kc; }
