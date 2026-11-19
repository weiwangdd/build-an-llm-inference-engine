/* gpu_ops.cu — 第 7 章：GPU 上的算子。
 *
 * 每个内核都是对应的 CPU 循环原样搬过来：CPU 上的“for 每个输出”变成“每个线程一个输出”，
 * 线程内部的循环与 ops.c 一字不差（double 累加，求和次序相同）。
 *
 * 内核用“网格跨步循环”写：线程 i 依次处理第 i、i + 线程总数、i + 2·线程总数 …… 个输出。
 * 这样启动多少个线程与输出有多少个无关，输出再多也不会超出 CUDA 对网格大小的限制。 */
#include "gpu_ops.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

static int cuda_fail(cudaError_t e, const char *what, char *err, size_t cap) {
    if (err && cap) snprintf(err, cap, "CUDA 出错（%s）：%s", what, cudaGetErrorString(e));
    return -1;
}

/* ---------- 设备与显存 ---------- */

static int inited;

extern "C" int gpu_init(char *err, size_t cap) {
    if (inited) return 0;
    int count = 0, pools = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess) return cuda_fail(e, "找 GPU", err, cap);
    if (count == 0) { if (err && cap) snprintf(err, cap, "没有可用的 GPU"); return -1; }
    if ((e = cudaSetDevice(0)) != cudaSuccess) return cuda_fail(e, "选 GPU", err, cap);
    cudaDeviceProp p;
    if ((e = cudaGetDeviceProperties(&p, 0)) != cudaSuccess) return cuda_fail(e, "读 GPU 属性", err, cap);
    if ((e = cudaDeviceGetAttribute(&pools, cudaDevAttrMemoryPoolsSupported, 0)) != cudaSuccess)
        return cuda_fail(e, "读 GPU 属性", err, cap);
    if (!pools) { if (err && cap) snprintf(err, cap, "GPU %s 不支持显存池（cudaMallocAsync）", p.name); return -1; }
    /* cudaMallocAsync 从一个显存池里分配。默认每次同步时把空闲的显存还给驱动，下次再要；
     * 我们每一层都要分配、释放一批临时缓冲区，所以让池子留着它们。 */
    cudaMemPool_t pool;
    uint64_t keep = UINT64_MAX;
    if ((e = cudaDeviceGetDefaultMemPool(&pool, 0)) != cudaSuccess ||
        (e = cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keep)) != cudaSuccess)
        return cuda_fail(e, "设置显存池", err, cap);
    inited = 1;
    return 0;
}

extern "C" int gpu_alloc(void **p, size_t bytes, char *err, size_t cap) {
    *p = NULL;
    cudaError_t e = cudaMallocAsync(p, bytes ? bytes : 1, 0);
    if (e != cudaSuccess) {
        *p = NULL;
        if (err && cap) snprintf(err, cap, "显存不足（要 %zu 字节）：%s", bytes, cudaGetErrorString(e));
        return -1;
    }
    return 0;
}

extern "C" void gpu_free(void *p) {
    if (p) cudaFreeAsync(p, 0);
}

extern "C" void gpu_zero(void *p, size_t bytes) {
    if (bytes) cudaMemsetAsync(p, 0, bytes, 0);
}

extern "C" int gpu_put(void *dst, const void *src, size_t bytes, char *err, size_t cap) {
    cudaError_t e = bytes ? cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) : cudaSuccess;
    return e == cudaSuccess ? 0 : cuda_fail(e, "拷到显存", err, cap);
}

extern "C" int gpu_get(void *dst, const void *src, size_t bytes, char *err, size_t cap) {
    /* cudaMemcpy 会等前面的内核算完；内核出的错也在这里报出来 */
    cudaError_t e = bytes ? cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) : cudaSuccess;
    if (e == cudaSuccess) e = cudaGetLastError();
    return e == cudaSuccess ? 0 : cuda_fail(e, "从显存拷回", err, cap);
}

extern "C" void gpu_copy(void *dst, const void *src, size_t bytes) {
    if (bytes) cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, 0);
}

extern "C" void gpu_copy2d(float *dst, size_t dpitch, const float *src, size_t spitch, size_t width, size_t rows) {
    if (width && rows)
        cudaMemcpy2DAsync(dst, dpitch * sizeof(float), src, spitch * sizeof(float), width * sizeof(float), rows,
                          cudaMemcpyDeviceToDevice, 0);
}

extern "C" int gpu_sync(char *err, size_t cap) {
    cudaError_t e = cudaDeviceSynchronize();
    if (e == cudaSuccess) e = cudaGetLastError();
    return e == cudaSuccess ? 0 : cuda_fail(e, "计算", err, cap);
}

/* ---------- 启动内核 ---------- */

#define THREADS 128   /* 每个线程块的线程数 */

/* 处理 n 个输出要几个线程块。超过 2^20 个块就不再加，多出来的输出由网格跨步循环接着处理。 */
static unsigned blocks(size_t n) {
    size_t b = (n + THREADS - 1) / THREADS;
    return (unsigned)(b > (1u << 20) ? (1u << 20) : b);
}

#define FOR_EACH(i, n) for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
                            i += (size_t)gridDim.x * blockDim.x)
#define LAUNCH(kernel, n, ...) do { if ((n) > 0) kernel<<<blocks(n), THREADS>>>(__VA_ARGS__); } while (0)

/* ---------- 数学函数，与 ops.c 相同的写法 ---------- */

__device__ static float bf16(const uint8_t *p, size_t i) {
    /* 显存是我们自己分配的，起点按 256 字节对齐，第 i 个 BF16 一定在偶数地址上 */
    uint16_t h = ((const uint16_t *)p)[i];
    return __uint_as_float((unsigned)h << 16);
}
__device__ static double d_silu(double x) { return x / (1.0 + exp(-x)); }
__device__ static double d_softplus(double v) { return fmax(v, 0.0) + log1p(exp(-fabs(v))); }

__device__ static double d_inv_rms(const float *x, size_t d, double div, double eps) {
    double ss = 0;
    for (size_t i = 0; i < d; i++) ss += (double)x[i] * (double)x[i];
    return 1.0 / sqrt(ss / div + eps);
}

/* ---------- 第 3 章的算子 ---------- */

/* 每个线程算一个 y[t][o]：w 的第 o 行与 x 的第 t 行逐项相乘，从 i = 0 加到 in - 1。 */
__global__ static void k_matmul(float *y, const float *x, const uint8_t *w, size_t n, size_t in, size_t out) {
    FOR_EACH(idx, n * out) {
        size_t t = idx / out, o = idx % out;
        const float *xt = x + t * in;
        const uint8_t *row = w + 2 * o * in;
        double acc = 0;
        for (size_t i = 0; i < in; i++) acc += (double)xt[i] * (double)bf16(row, i);
        y[idx] = (float)acc;
    }
}
extern "C" void gop_matmul(float *y, const float *x, const uint8_t *w, size_t n, size_t in, size_t out) {
    LAUNCH(k_matmul, n * out, y, x, w, n, in, out);
}

__global__ static void k_embed(float *y, const uint8_t *table, size_t dim, const int32_t *ids, size_t n) {
    FOR_EACH(idx, n * dim) {
        size_t t = idx / dim, i = idx % dim;
        y[idx] = bf16(table + 2 * (size_t)ids[t] * dim, i);
    }
}
extern "C" void gop_embed(float *y, const uint8_t *table, size_t dim, const int32_t *ids, size_t n) {
    LAUNCH(k_embed, n * dim, y, table, dim, ids, n);
}

/* 归一化：每个线程负责一整行，和 CPU 一样先求平方和，再逐个元素乘。 */
__global__ static void k_rmsnorm(float *y, const float *x, const float *w, size_t n, size_t d, double eps) {
    FOR_EACH(t, n) {
        const float *xt = x + t * d;
        double r = d_inv_rms(xt, d, (double)d, eps);
        for (size_t i = 0; i < d; i++) y[t * d + i] = (float)((double)xt[i] * r * (1.0 + (double)w[i]));
    }
}
extern "C" void gop_rmsnorm(float *y, const float *x, const float *w, size_t n, size_t d, double eps) {
    LAUNCH(k_rmsnorm, n, y, x, w, n, d, eps);
}

__global__ static void k_rmsnorm_gated(float *y, const float *x, const float *z, const float *w, size_t n, size_t d,
                                       double eps) {
    FOR_EACH(t, n) {
        const float *xt = x + t * d, *zt = z + t * d;
        double r = d_inv_rms(xt, d, (double)d, eps);
        for (size_t i = 0; i < d; i++)
            y[t * d + i] = (float)((double)xt[i] * r * (double)w[i] * d_silu((double)zt[i]));
    }
}
extern "C" void gop_rmsnorm_gated(float *y, const float *x, const float *z, const float *w, size_t n, size_t d,
                                  double eps) {
    LAUNCH(k_rmsnorm_gated, n, y, x, z, w, n, d, eps);
}

__global__ static void k_l2norm(float *y, const float *x, size_t n, size_t d, double eps) {
    FOR_EACH(t, n) {
        const float *xt = x + t * d;
        double r = d_inv_rms(xt, d, 1.0, eps);
        for (size_t i = 0; i < d; i++) y[t * d + i] = (float)((double)xt[i] * r);
    }
}
extern "C" void gop_l2norm(float *y, const float *x, size_t n, size_t d, double eps) {
    LAUNCH(k_l2norm, n, y, x, n, d, eps);
}

__global__ static void k_silu(float *y, const float *x, size_t n) {
    FOR_EACH(i, n) y[i] = (float)d_silu((double)x[i]);
}
__global__ static void k_sigmoid(float *y, const float *x, size_t n) {
    FOR_EACH(i, n) y[i] = (float)(1.0 / (1.0 + exp(-(double)x[i])));
}
__global__ static void k_softplus(float *y, const float *x, size_t n) {
    FOR_EACH(i, n) y[i] = (float)d_softplus((double)x[i]);
}
extern "C" void gop_silu(float *y, const float *x, size_t n) { LAUNCH(k_silu, n, y, x, n); }
extern "C" void gop_sigmoid(float *y, const float *x, size_t n) { LAUNCH(k_sigmoid, n, y, x, n); }
extern "C" void gop_softplus(float *y, const float *x, size_t n) { LAUNCH(k_softplus, n, y, x, n); }

__global__ static void k_softmax(float *y, const float *x, size_t n, size_t d) {
    FOR_EACH(t, n) {
        const float *xt = x + t * d;
        double m = -INFINITY, s = 0;
        for (size_t i = 0; i < d; i++) if ((double)xt[i] > m) m = (double)xt[i];
        for (size_t i = 0; i < d; i++) s += exp((double)xt[i] - m);
        for (size_t i = 0; i < d; i++) y[t * d + i] = (float)(exp((double)xt[i] - m) / s);
    }
}
extern "C" void gop_softmax(float *y, const float *x, size_t n, size_t d) { LAUNCH(k_softmax, n, y, x, n, d); }

/* RoPE：每个线程负责一个头里的一个维度 i。i 在前半（< rot/2）时算一对 y[i]、y[i+rot/2]；
 * i 在 rot 之后时原样复制；其余的 i 已由前半的线程算过，什么也不做。 */
__global__ static void k_rope(float *y, const float *x, const int32_t *pos, size_t n, size_t heads, size_t head_dim,
                              size_t rot, double theta) {
    size_t half = rot / 2;
    FOR_EACH(idx, n * heads * head_dim) {
        size_t i = idx % head_dim, th = idx / head_dim, t = th / heads;
        const float *xh = x + th * head_dim;
        float *yh = y + th * head_dim;
        if (i < half) {
            double f = (double)pos[t] * pow(theta, -(double)(2 * i) / (double)rot);
            double c = cos(f), s = sin(f), a = (double)xh[i], b = (double)xh[i + half];
            yh[i] = (float)(a * c - b * s);
            yh[i + half] = (float)(b * c + a * s);
        } else if (i >= rot) {
            yh[i] = xh[i];
        }
    }
}
extern "C" void gop_rope(float *y, const float *x, const int32_t *pos, size_t n, size_t heads, size_t head_dim,
                         size_t rot, double theta) {
    LAUNCH(k_rope, n * heads * head_dim, y, x, pos, n, heads, head_dim, rot, theta);
}

__global__ static void k_conv1d_hist(float *y, const float *x, const float *hist, size_t nh, const float *w, size_t n,
                                     size_t c, size_t k) {
    FOR_EACH(idx, n * c) {
        size_t t = idx / c, j = idx % c;
        double acc = 0;
        for (size_t i = 0; i < k; i++) {
            if (nh + t + i < k - 1) continue;
            size_t src = nh + t + i - (k - 1);
            const float *row = src < nh ? hist + src * c : x + (src - nh) * c;
            acc += (double)w[j * k + i] * (double)row[j];
        }
        y[idx] = (float)acc;
    }
}
extern "C" void gop_conv1d_hist(float *y, const float *x, const float *hist, size_t nh, const float *w, size_t n,
                                size_t c, size_t k) {
    LAUNCH(k_conv1d_hist, n * c, y, x, hist, nh, w, n, c, k);
}

/* ---------- 第 4 章的算子 ---------- */

__global__ static void k_add(float *y, const float *a, const float *b, size_t n) { FOR_EACH(i, n) y[i] = a[i] + b[i]; }
__global__ static void k_mul(float *y, const float *a, const float *b, size_t n) { FOR_EACH(i, n) y[i] = a[i] * b[i]; }
extern "C" void gop_add(float *y, const float *a, const float *b, size_t n) { LAUNCH(k_add, n, y, a, b, n); }
extern "C" void gop_mul(float *y, const float *a, const float *b, size_t n) { LAUNCH(k_mul, n, y, a, b, n); }

__global__ static void k_decay(float *g, const float *a, const float *A_log, const float *dt_bias, size_t n,
                               size_t heads) {
    FOR_EACH(idx, n * heads) {
        size_t h = idx % heads;
        g[idx] = (float)(-exp((double)A_log[h]) * d_softplus((double)a[idx] + (double)dt_bias[h]));
    }
}
extern "C" void gop_decay(float *g, const float *a, const float *A_log, const float *dt_bias, size_t n, size_t heads) {
    LAUNCH(k_decay, n * heads, g, a, A_log, dt_bias, n, heads);
}

/* 注意力：一个线程块负责一个（新 token t，查询头 h），块里的线程分三段合作：
 *   1. 各线程分头算 p[s] = scale · q·k[s]（每个点积仍由一个线程从 i = 0 加到 d - 1）
 *   2. 0 号线程独自求最大值 m，大家分头算 e^(p[s] − m)，0 号线程再从 s = 0 起依次加出总和
 *   3. 线程 i 负责输出的第 i 维：从 s = 0 起依次加 p[s] · v[s][i]，最后除以总和
 * 每个求和都只由一个线程按 CPU 的次序做，所以结果与 CPU 相同。p 放在显存里（序列可能很长，放不进共享内存）。 */
__global__ static void k_attention(float *y, const float *q, const float *k, const float *v, double *pbuf, size_t n,
                                   size_t past, size_t heads, size_t kv, size_t d, double scale) {
    __shared__ double m_sh, sum_sh;
    size_t group = heads / kv, len = past + n;
    for (size_t row = blockIdx.x; row < n * heads; row += gridDim.x) {     /* row = t · heads + h */
        size_t t = row / heads, h = row % heads, g = h / group, last = past + t;
        const float *qh = q + row * d;
        double *p = pbuf + row * len;
        for (size_t s = threadIdx.x; s <= last; s += blockDim.x) {
            const float *ks = k + (s * kv + g) * d;
            double dot = 0;
            for (size_t i = 0; i < d; i++) dot += (double)qh[i] * (double)ks[i];
            p[s] = dot * scale;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            double m = -INFINITY;
            for (size_t s = 0; s <= last; s++) if (p[s] > m) m = p[s];
            m_sh = m;
        }
        __syncthreads();
        for (size_t s = threadIdx.x; s <= last; s += blockDim.x) p[s] = exp(p[s] - m_sh);
        __syncthreads();
        if (threadIdx.x == 0) {
            double sum = 0;
            for (size_t s = 0; s <= last; s++) sum += p[s];
            sum_sh = sum;
        }
        __syncthreads();
        for (size_t i = threadIdx.x; i < d; i += blockDim.x) {
            double acc = 0;
            for (size_t s = 0; s <= last; s++) acc += p[s] * (double)v[(s * kv + g) * d + i];
            y[row * d + i] = (float)(acc / sum_sh);
        }
        __syncthreads();      /* 下一轮会改 m_sh、sum_sh，等这一轮都用完 */
    }
}
extern "C" int gop_attention_past(float *y, const float *q, const float *k, const float *v, size_t n, size_t past,
                                  size_t heads, size_t kv, size_t d, double scale, char *err, size_t cap) {
    if (n == 0) return 0;
    void *p;
    if (gpu_alloc(&p, n * heads * (past + n) * sizeof(double), err, cap)) return -1;
    size_t rows = n * heads;
    k_attention<<<(unsigned)(rows > (1u << 20) ? (1u << 20) : rows), THREADS>>>(y, q, k, v, (double *)p, n, past,
                                                                               heads, kv, d, scale);
    gpu_free(p);              /* 按顺序执行：内核算完之后才真正释放 */
    return 0;
}

/* 门控 delta 规则：状态矩阵 S 的每一列（固定值头 j 和列 c）的更新只用到这一列自己：
 *   S[:,c] *= e^g；pred = Σ_r S[r][c] k[r]；u = beta (v[c] − pred)；S[:,c] += k u；y[c] = Σ_r S[r][c] q[r] / √dk
 * 所以每个线程负责一列，按 token 顺序往下算，各列之间互不等待。 */
__global__ static void k_gated_delta(float *y, const float *q, const float *k, const float *v, const float *g,
                                     const float *beta, double *state, size_t n, size_t hk, size_t hv, size_t dk,
                                     size_t dv) {
    size_t group = hv / hk;
    double scale = 1.0 / sqrt((double)dk);
    FOR_EACH(idx, hv * dv) {
        size_t j = idx / dv, c = idx % dv, kh = j / group;
        double *S = state + j * dk * dv;
        for (size_t t = 0; t < n; t++) {
            const float *qt = q + (t * hk + kh) * dk, *kt = k + (t * hk + kh) * dk;
            double decay = exp((double)g[t * hv + j]), b = (double)beta[t * hv + j];
            for (size_t r = 0; r < dk; r++) S[r * dv + c] *= decay;
            double pred = 0;
            for (size_t r = 0; r < dk; r++) pred += S[r * dv + c] * (double)kt[r];
            double u = b * ((double)v[(t * hv + j) * dv + c] - pred);
            for (size_t r = 0; r < dk; r++) S[r * dv + c] += (double)kt[r] * u;
            double o = 0;
            for (size_t r = 0; r < dk; r++) o += S[r * dv + c] * (double)qt[r];
            y[(t * hv + j) * dv + c] = (float)(o * scale);
        }
    }
}
extern "C" void gop_gated_delta_state(float *y, const float *q, const float *k, const float *v, const float *g,
                                      const float *beta, double *state, size_t n, size_t hk, size_t hv, size_t dk,
                                      size_t dv) {
    if (n > 0) LAUNCH(k_gated_delta, hv * dv, y, q, k, v, g, beta, state, n, hk, hv, dk, dv);
}
