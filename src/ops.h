/* ops.h — 第 3 章：CPU 上的基本算子。
 *
 * 约定：
 *   - 激活值（输入和输出）一律是 float，按行主序连续存放。
 *   - 大矩阵（嵌入表、线性层的权重）直接用 safetensors 里的 BF16 原始字节，
 *     按小端逐字节读出，不要求对齐，也不事先转换成 float（27B 的权重转成 float 要 100 多 GB）。
 *   - 小向量（归一化的权重、卷积核）由调用方先转成 float 再传进来。
 *   - 内部一律用 double 计算，最后舍入成 float。这样每个结果都几乎等于“精确值舍入到 float”，
 *     后面各章（包括 GPU 版）都拿它当标尺。
 *   - 输出缓冲区不能与输入重叠。
 *
 * 形状参数的合法性由调用方保证（见 tools/ops.c）；只有运行时才知道是否合法的输入
 * （例如 token id）由算子自己检查。 */
#ifndef OPS_H
#define OPS_H

#include <stddef.h>
#include <stdint.h>

/* BF16 就是 float 的高 16 位：1 位符号、8 位指数、7 位尾数。转换是精确的。 */
float    bf16_to_f32(uint16_t h);
/* 读 BF16 数组 p 的第 i 个元素（小端，不要求对齐）。 */
uint16_t bf16_at(const uint8_t *p, size_t i);

/* 线性层：y[t][o] = sum_i x[t][i] * w[o][i]。x 是 [n, in]，w 是 BF16 的 [out, in]，y 是 [n, out]。
 * 注意 w 的存放方式是“每个输出一行”，与 PyTorch 的 nn.Linear 相同。 */
void op_matmul(float *y, const float *x, const uint8_t *w, size_t n, size_t in, size_t out);

/* 嵌入：y[t] = table[ids[t]]。table 是 BF16 的 [rows, dim]。id 越界时报错返回 -1。 */
int  op_embed(float *y, const uint8_t *table, size_t rows, size_t dim,
              const int32_t *ids, size_t n, char *err, size_t errcap);

/* RMSNorm（Qwen3.8 的“零中心”写法）：对 x 的每一行（长度 d），
 *   y = x / sqrt(mean(x^2) + eps) * (1 + w)
 * 权重存的是与 1 的差，所以全零的权重表示“不缩放”。 */
void op_rmsnorm(float *y, const float *x, const float *w, size_t n, size_t d, double eps);

/* 带门控的 RMSNorm（线性注意力层的输出用）：
 *   y = x / sqrt(mean(x^2) + eps) * w * silu(z)
 * 注意这里是 w 而不是 1 + w。 */
void op_rmsnorm_gated(float *y, const float *x, const float *z, const float *w,
                      size_t n, size_t d, double eps);

/* L2 归一化：y = x / sqrt(sum(x^2) + eps)，对每一行。 */
void op_l2norm(float *y, const float *x, size_t n, size_t d, double eps);

/* 逐元素激活函数，n 是元素个数。
 *   silu(x)     = x / (1 + e^-x)
 *   sigmoid(x)  = 1 / (1 + e^-x)
 *   softplus(x) = ln(1 + e^x) */
void op_silu(float *y, const float *x, size_t n);
void op_sigmoid(float *y, const float *x, size_t n);
void op_softplus(float *y, const float *x, size_t n);

/* softmax，对每一行（长度 d）：y_i = e^(x_i - m) / sum_j e^(x_j - m)，m 是这一行的最大值。 */
void op_softmax(float *y, const float *x, size_t n, size_t d);

/* 旋转位置编码（RoPE），只转每个头的前 rot 维（rot 为偶数，rot <= head_dim），其余原样复制。
 * x 和 y 是 [n, heads, head_dim]，pos[t] 是第 t 个 token 的位置。前 rot 维分成前后两半：
 *   f_i = pos * theta^(-2i/rot)，i = 0 .. rot/2-1
 *   y[i]         = x[i] cos f_i - x[i+rot/2] sin f_i
 *   y[i+rot/2]   = x[i+rot/2] cos f_i + x[i] sin f_i */
void op_rope(float *y, const float *x, const int32_t *pos, size_t n, size_t heads,
             size_t head_dim, size_t rot, double theta);

/* 逐通道的因果一维卷积：x 和 y 是 [n, c]（n 个 token，c 个通道），w 是 [c, k]。
 *   y[t][j] = sum_{i=0}^{k-1} w[j][i] * x[t-(k-1)+i][j]，t-(k-1)+i < 0 的项当作 0
 * 即 w[j][k-1] 乘当前 token，w[j][0] 乘往前数第 k-1 个 token。 */
void op_conv1d(float *y, const float *x, const float *w, size_t n, size_t c, size_t k);

/* ---------- 第 4 章加的算子 ---------- */

/* 逐元素加法、乘法：y[i] = a[i] + b[i]，y[i] = a[i] * b[i]。float 的一次加法、乘法本来就是正确舍入。
 * 例外：这两个算子的 y 可以就是 a（就地更新），残差连接就是这样用的。 */
void op_add(float *y, const float *a, const float *b, size_t n);
void op_mul(float *y, const float *a, const float *b, size_t n);

/* 线性注意力的衰减量（对数形式）：a 是 [n, heads]，A_log、dt_bias 是 [heads]，
 *   g[t][h] = -e^(A_log[h]) * softplus(a[t][h] + dt_bias[h])
 * 状态矩阵每一步乘 e^g，g <= 0，所以 e^g 在 0 和 1 之间。 */
void op_decay(float *g, const float *a, const float *A_log, const float *dt_bias, size_t n, size_t heads);

/* 因果的分组查询注意力。q、y 是 [n, heads, d]，k、v 是 [n, kv, d]，第 h 个查询头用第 h / (heads / kv) 组键值。
 *   p[t][s] = softmax_s( scale * q[t]·k[s] )，s 只取 0 .. t（因果）
 *   y[t]    = Σ_s p[t][s] v[s]
 * 内部用 double。成功返回 0，内存不足返回 -1。 */
int  op_attention(float *y, const float *q, const float *k, const float *v,
                  size_t n, size_t heads, size_t kv, size_t d, double scale, char *err, size_t errcap);

/* 门控 delta 规则（线性注意力的核心）。q、k 是 [n, hk, dk]，v、y 是 [n, hv, dv]，g、beta 是 [n, hv]。
 * 第 j 个值头用第 j / (hv / hk) 个键头。每个值头维护一个 dk × dv 的状态矩阵 S，初值为 0，逐 token 更新：
 *   S    = e^(g[t]) · S                  衰减：旧的记忆按比例淡去
 *   u    = beta[t] · (v[t] − Sᵀ k[t])    Sᵀk 是状态对 v 的“预测”，u 是按 beta 缩放的预测误差
 *   S    = S + k[t] uᵀ                   把误差写回状态
 *   y[t] = Sᵀ q[t] / √dk
 * 状态全程用 double 保存。成功返回 0，内存不足返回 -1。 */
int  op_gated_delta(float *y, const float *q, const float *k, const float *v, const float *g, const float *beta,
                    size_t n, size_t hk, size_t hv, size_t dk, size_t dv, char *err, size_t errcap);

/* ---------- 第 6 章：接着上一次往下算（缓存） ----------
 * 下面三个算子是上面三个的推广：把序列分几次交给它们，每次只给新来的 token，同时给出前面
 * 留下的东西，结果与把整条序列一次交给原来的算子逐位相同（求和的项和次序完全一样）。
 * 原来的三个就是它们在“前面什么都没有”时的特例。 */

/* 因果卷积，前面接上 nh 行历史 hist（[nh, c]，是紧挨在 x 前面的 nh 个 token 的输入，nh <= k-1）。 */
void op_conv1d_hist(float *y, const float *x, const float *hist, size_t nh, const float *w,
                    size_t n, size_t c, size_t k);

/* 注意力，前面已经有 past 个 token。q、y 是 n 个新 token 的 [n, heads, d]；k、v 是全部
 * past + n 个 token 的 [past + n, kv, d]（前 past 行来自缓存）。第 t 个新 token 的位置是 past + t，
 * 看位置 0 .. past + t。 */
int  op_attention_past(float *y, const float *q, const float *k, const float *v, size_t n, size_t past,
                       size_t heads, size_t kv, size_t d, double scale, char *err, size_t errcap);

/* 门控 delta 规则，从给定的状态出发。state 是 [hv, dk, dv] 的 double，进来时是处理完前面那些
 * token 之后的状态，出去时更新为处理完这 n 个 token 之后的状态。 */
int  op_gated_delta_state(float *y, const float *q, const float *k, const float *v, const float *g,
                          const float *beta, double *state, size_t n, size_t hk, size_t hv, size_t dk, size_t dv,
                          char *err, size_t errcap);

#endif
