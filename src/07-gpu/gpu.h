/* gpu.h — 第 7 章：把整个模型搬到一张 GPU 上算。
 *
 * 用法只多一步：model_load 之后调用 model_to_gpu，把权重拷进显存。之后照常调用第 5、6 章的
 * model_forward、model_cache_init、model_step，它们发现模型在 GPU 上，就转到这里的实现，
 * 每一步与 CPU 的同名步骤一一对应，trace 收到的步骤名、形状也完全相同。
 *
 * 显存里的模型仍用 model、layer_w、model_cache 这几个结构体，只是里面的指针指向显存。
 *
 * 没有 CUDA 的机器上编译的是 gpu_none.c，这些函数一律报错“编译时没有启用 CUDA”。 */
#ifndef GPU_H
#define GPU_H

#include <stddef.h>
#include <stdint.h>
#include "model.h"

/* 把 m 的全部权重拷进显存，m->gpu 指向显存里的那一份。m 的 CPU 那一份不动（仍要 m 来做检查和释放）。 */
int  model_to_gpu(model *m, char *err, size_t errcap);

/* 一层的前向在 GPU 上算（第 4 章 layer 工具用）：h 在内存里，先拷上去，算完拷回来。
 * 这一层的权重每次调用都拷一遍。 */
int  gpu_layer_forward(float *h, size_t n, const int32_t *pos, const model_cfg *c, const layer_w *w,
                       trace_fn trace, void *ctx, char *err, size_t errcap);

/* ---- 下面几个由 model.c 调用 ---- */

/* g 是 model_to_gpu 建的显存里的模型。ids、logits 在内存里。seqs 是一批序列（第 8 章），
 * 只有一个序列且它的 kc 为 NULL 时不用缓存。 */
int  gpu_model_run(const model *g, const model_seq *seqs, size_t nseq, float *logits,
                   trace_fn trace, void *ctx, char *err, size_t errcap);
void gpu_model_free(model *g);
/* 在显存里分配 kc 每一层的缓存（kc->n_layers、kc->cap 已经填好）。 */
int  gpu_cache_alloc(model_cache *kc, const model_cfg *c, char *err, size_t errcap);
void gpu_cache_free(model_cache *kc);

#endif
