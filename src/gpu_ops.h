/* gpu_ops.h — 第 7 章：GPU 上的算子，以及显存的分配和拷贝。
 *
 * 每个 gop_xxx 与 ops.h 里的 op_xxx 一一对应，参数的含义和顺序都相同，区别只有一个：
 * 所有指针都指向显存（显卡自己的内存，CPU 不能直接读写）。
 *
 * 写法和第 3 章一样朴素：一个线程算一个输出（或一行、一列），内部用 double，
 * 求和的项和次序与 ops.c 完全相同，并且编译时关掉 FMA（见 Makefile 的 --fmad=false）。
 * 所以除了 exp、sin 这类数学函数（CUDA 的实现与 glibc 的不保证逐位相同）以外，
 * 每个结果都应当与 CPU 逐位相同。性能留到第二卷。
 *
 * 内核的启动是异步的：gop_xxx 返回时 GPU 可能还没算完，出错也要等到 gpu_sync 或 gpu_get 时才知道。
 * 这个头文件既给 C 用，也给 CUDA（C++）用。 */
#ifndef GPU_OPS_H
#define GPU_OPS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 选中 0 号 GPU（可以用环境变量 CUDA_VISIBLE_DEVICES 换一张），检查它支持 double。可以重复调用。 */
int  gpu_init(char *err, size_t errcap);

/* 显存的分配、释放、清零和拷贝。put：内存 → 显存；get：显存 → 内存，会等前面的内核全部算完，
 * 并检查它们有没有出错；copy、copy2d：显存 → 显存（copy2d 拷 rows 行，每行 width 个 float，
 * 源和目的的行距分别是 spitch、dpitch 个 float）。 */
int  gpu_alloc(void **p, size_t bytes, char *err, size_t errcap);
void gpu_free(void *p);
void gpu_zero(void *p, size_t bytes);
int  gpu_put(void *dst, const void *src, size_t bytes, char *err, size_t errcap);
int  gpu_get(void *dst, const void *src, size_t bytes, char *err, size_t errcap);
void gpu_copy(void *dst, const void *src, size_t bytes);
void gpu_copy2d(float *dst, size_t dpitch, const float *src, size_t spitch, size_t width, size_t rows);
/* 等前面的内核全部算完，检查有没有出错。 */
int  gpu_sync(char *err, size_t errcap);

/* ---------- 算子（与 ops.h 对应） ---------- */

void gop_matmul(float *y, const float *x, const uint8_t *w, size_t n, size_t in, size_t out);
/* 与 op_embed 不同，这里不检查 id 是否越界（内核里没法报错），调用方要先在 CPU 上检查。 */
void gop_embed(float *y, const uint8_t *table, size_t dim, const int32_t *ids, size_t n);
void gop_rmsnorm(float *y, const float *x, const float *w, size_t n, size_t d, double eps);
void gop_rmsnorm_gated(float *y, const float *x, const float *z, const float *w, size_t n, size_t d, double eps);
void gop_l2norm(float *y, const float *x, size_t n, size_t d, double eps);
void gop_silu(float *y, const float *x, size_t n);
void gop_sigmoid(float *y, const float *x, size_t n);
void gop_softplus(float *y, const float *x, size_t n);
void gop_softmax(float *y, const float *x, size_t n, size_t d);
void gop_rope(float *y, const float *x, const int32_t *pos, size_t n, size_t heads, size_t head_dim, size_t rot,
              double theta);
void gop_conv1d_hist(float *y, const float *x, const float *hist, size_t nh, const float *w, size_t n, size_t c,
                     size_t k);
void gop_add(float *y, const float *a, const float *b, size_t n);
void gop_mul(float *y, const float *a, const float *b, size_t n);
void gop_decay(float *g, const float *a, const float *A_log, const float *dt_bias, size_t n, size_t heads);
int  gop_attention_past(float *y, const float *q, const float *k, const float *v, size_t n, size_t past,
                        size_t heads, size_t kv, size_t d, double scale, char *err, size_t errcap);
void gop_gated_delta_state(float *y, const float *q, const float *k, const float *v, const float *g,
                           const float *beta, double *state, size_t n, size_t hk, size_t hv, size_t dk, size_t dv);

#ifdef __cplusplus
}
#endif

#endif
