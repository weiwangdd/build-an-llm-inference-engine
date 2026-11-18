/* sample.h — 第 6 章：从一行 logits 选出下一个 token。
 *
 * 分成互不相干的两半：
 *   1. 确定的一半（sample_dist）：logits → 温度 → top-k → top-p → 保留下来的 token 和它们的概率。
 *      顺序和取舍规则与 transformers 的 TemperatureLogitsWarper、TopKLogitsWarper、TopPLogitsWarper 相同。
 *   2. 随机的一半（sample_pick）：拿一个 [0, 1) 里的随机数 u，按累计概率挑一个。
 *      随机数由我们自己的生成器（splitmix64）给出，同一个种子总是给出同一串数，所以整个采样可以复现。
 * 贪心解码是特例：分布里只有分数最大的那一个 token，概率为 1，不用随机数。 */
#ifndef SAMPLE_H
#define SAMPLE_H

#include <stddef.h>
#include <stdint.h>

/* splitmix64：64 位状态，每次加一个常数再搅拌。简单、快、统计性质足够好，种子相同输出就相同。 */
typedef struct { uint64_t s; } rng64;
void     rng_seed(rng64 *r, uint64_t seed);
uint64_t rng_next(rng64 *r);
/* [0, 1) 里的均匀随机数：取 rng_next 的高 53 位除以 2^53，所以是 2^-53 的整数倍。 */
double   rng_uniform(rng64 *r);

typedef struct {
    int    do_sample;     /* 0 表示贪心解码，下面三个参数不用 */
    double temperature;   /* 大于 0。logits 先除以它：小于 1 让分布更尖，大于 1 让分布更平 */
    size_t top_k;         /* 只保留分数最高的 top_k 个（与第 top_k 名同分的也保留）；0 表示不限 */
    double top_p;         /* 在剩下的里面，从高到低保留到累计概率达到 top_p 为止；1 表示不限 */
} sample_cfg;

/* 检查参数是否合法：temperature > 0，0 < top_p <= 1。 */
int sample_cfg_check(const sample_cfg *sc, char *err, size_t errcap);

/* 确定的一半。row 是一行 logits（vocab 个）。结果按 id 从小到大写进 ids 和 probs（各要 vocab 个位置），
 * 个数写进 *n，概率之和为 1（舍入误差内）。logits 里有 NaN 或无穷大时报错返回 -1。 */
int sample_dist(const float *row, size_t vocab, const sample_cfg *sc, int32_t *ids, double *probs, size_t *n,
                char *err, size_t errcap);

/* 随机的一半：按 id 从小到大累加概率，返回第一个使累计概率大于 u 的 token。
 * 舍入使得加到最后也不大于 u 时（u 极接近 1），返回最后一个。 */
int32_t sample_pick(const int32_t *ids, const double *probs, size_t n, double u);

#endif
