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

#endif
