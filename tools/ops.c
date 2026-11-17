/* ops — 第 3 章算子的命令行工具，给测试用。
 *
 *   ops bf16                                     打印全部 65536 个 BF16 位模式转成 float 后的位模式（十六进制）
 *   ops <算子> [键=值 ...] 文件.safetensors [更多文件 ...]
 *
 * 输入张量从一个或多个 safetensors 文件里按名字取（几个文件像第 1 章的分片一样合并），
 * 默认名字是 x、w、z、ids、pos，可以用 x=<名字> 这样的参数换成别的，例如直接指向模型权重。
 * 标量参数 eps、theta、rot 没有默认值，用到就必须给出。
 *
 *   matmul         x[n,in] F32, w[out,in] BF16          -> [n,out]
 *   embed          ids[n] I32, w[rows,dim] BF16         -> [n,dim]
 *   rmsnorm        x[...,d] F32, w[d], eps              -> 与 x 同形
 *   rmsnorm_gated  x[...,d] F32, z 与 x 同形, w[d], eps -> 与 x 同形
 *   l2norm         x[...,d] F32, eps                    -> 与 x 同形
 *   silu | sigmoid | softplus | softmax   x F32          -> 与 x 同形（softmax 按最后一维）
 *   rope           x[n,heads,head_dim] F32, pos[n] I32, theta, rot -> 与 x 同形
 *   conv1d         x[n,c] F32, w[c,k] 或 [c,1,k]         -> [n,c]
 * 小向量 w 可以是 BF16 或 F32。
 *
 * 输出：第一行是形状，如 [2,5120]；之后每行一个元素，用 C 的 %a 格式（十六进制浮点，精确无损）。
 * 输出格式由 tests/ops_ref.py 读取并与参考值比较。 */
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ops.h"
#include "st.h"

static void die(const char *fmt, const char *a) {
    fputs("错误：", stderr);
    fprintf(stderr, fmt, a);
    fputc('\n', stderr);
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("内存不足%s", "");
    return p;
}

/* ---------- 参数 ---------- */

static const char *KEYS[] = {"x", "w", "z", "ids", "pos", "eps", "theta", "rot"};
#define NKEYS (sizeof KEYS / sizeof KEYS[0])
static const char *val[NKEYS];

static const char *arg(const char *k) {
    for (size_t i = 0; i < NKEYS; i++) if (!strcmp(KEYS[i], k)) return val[i] ? val[i] : (i < 5 ? k : NULL);
    return NULL;
}

static double num(const char *k) {
    const char *s = arg(k);
    if (!s) die("缺少参数 %s", k);
    char *end;
    double v = strtod(s, &end);
    if (end == s || *end || !isfinite(v)) die("参数 %s 不是有限的数", k);
    return v;
}

/* ---------- 取张量 ---------- */

static st_set S;

static const st_tensor *get(const char *key) {
    const st_tensor *t = st_set_find(&S, arg(key));
    if (!t) die("找不到张量 %s", arg(key));
    return t;
}

static uint64_t numel(const st_tensor *t) {
    uint64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    return n;
}

/* 激活值：必须是 F32，至少 mindim 维、至多 maxdim 维，且不能是空张量。 */
static const st_tensor *act(const char *key, int mindim, int maxdim) {
    const st_tensor *t = get(key);
    if (t->dtype != ST_F32) die("张量 %s 应为 F32", t->name);
    if (t->ndim < mindim || t->ndim > maxdim) die("张量 %s 的维数不对", t->name);
    if (numel(t) == 0) die("张量 %s 是空的", t->name);
    return t;
}

/* 把 F32 张量拷成对齐的 float 数组（mmap 里的数据不一定 4 字节对齐）。 */
static float *f32_copy(const st_tensor *t) {
    float *p = xmalloc((size_t)t->nbytes);
    memcpy(p, t->data, (size_t)t->nbytes);
    return p;
}

/* 小向量：BF16 或 F32，转成 float。 */
static float *vec(const st_tensor *t) {
    size_t n = (size_t)numel(t);
    float *p = xmalloc(n * sizeof *p);
    if (t->dtype == ST_BF16) for (size_t i = 0; i < n; i++) p[i] = bf16_to_f32(bf16_at(t->data, i));
    else if (t->dtype == ST_F32) memcpy(p, t->data, n * sizeof *p);
    else die("张量 %s 应为 BF16 或 F32", t->name);
    return p;
}

static int32_t *i32(const st_tensor *t, uint64_t want) {
    if (t->dtype != ST_I32 || t->ndim != 1 || t->shape[0] != want) die("张量 %s 应为 I32，形状与输入的 token 数一致", t->name);
    int32_t *p = xmalloc((size_t)t->nbytes);
    memcpy(p, t->data, (size_t)t->nbytes);
    return p;
}

static const st_tensor *matrix(const char *key, uint64_t cols) {
    const st_tensor *t = get(key);
    if (t->dtype != ST_BF16 || t->ndim != 2) die("张量 %s 应为二维 BF16", t->name);
    if (cols && t->shape[1] != cols) die("张量 %s 的列数与输入不符", t->name);
    return t;
}

/* ---------- 输出 ---------- */

static void print(const float *y, const uint64_t *shape, int ndim) {
    putchar('[');
    uint64_t n = 1;
    for (int i = 0; i < ndim; i++) { printf(i ? ",%" PRIu64 : "%" PRIu64, shape[i]); n *= shape[i]; }
    puts("]");
    for (uint64_t i = 0; i < n; i++) printf("%a\n", (double)y[i]);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "用法：%s <算子> [键=值 ...] 文件.safetensors ...\n", argv[0]); return 2; }
    const char *op = argv[1];

    if (!strcmp(op, "bf16")) {
        if (argc != 2) die("bf16 不带参数%s", "");
        for (uint32_t h = 0; h < 65536; h++) {
            float f = bf16_to_f32((uint16_t)h);
            uint32_t u;
            memcpy(&u, &f, sizeof u);
            printf("%08" PRIx32 "\n", u);
        }
        return 0;
    }

    int a = 2;
    for (; a < argc && strchr(argv[a], '='); a++) {
        size_t klen = (size_t)(strchr(argv[a], '=') - argv[a]);
        size_t i = 0;
        while (i < NKEYS && (strlen(KEYS[i]) != klen || strncmp(KEYS[i], argv[a], klen))) i++;
        if (i == NKEYS) die("不认识的参数 %s", argv[a]);
        if (val[i]) die("参数重复：%s", argv[a]);
        val[i] = argv[a] + klen + 1;
    }
    if (a == argc) die("没有给出输入文件%s", "");
    char err[512];
    if (st_set_open(&S, (const char *const *)(argv + a), (size_t)(argc - a), err, sizeof err)) die("%s", err);

    const st_tensor *xt = NULL;
    float *y = NULL;
    uint64_t shape[ST_MAX_DIMS];
    int ndim = 0;

    if (!strcmp(op, "matmul")) {
        xt = act("x", 2, 2);
        const st_tensor *w = matrix("w", xt->shape[1]);
        float *x = f32_copy(xt);
        shape[0] = xt->shape[0]; shape[1] = w->shape[0]; ndim = 2;
        y = xmalloc((size_t)(shape[0] * shape[1]) * sizeof *y);
        op_matmul(y, x, w->data, (size_t)shape[0], (size_t)xt->shape[1], (size_t)shape[1]);
        free(x);
    } else if (!strcmp(op, "embed")) {
        const st_tensor *w = matrix("w", 0);
        const st_tensor *it = get("ids");
        if (it->ndim != 1 || it->shape[0] == 0) die("张量 %s 应为非空的一维 I32", it->name);
        int32_t *ids = i32(it, it->shape[0]);
        shape[0] = it->shape[0]; shape[1] = w->shape[1]; ndim = 2;
        y = xmalloc((size_t)(shape[0] * shape[1]) * sizeof *y);
        if (op_embed(y, w->data, (size_t)w->shape[0], (size_t)w->shape[1], ids, (size_t)shape[0], err, sizeof err))
            die("%s", err);
        free(ids);
    } else if (!strcmp(op, "rmsnorm") || !strcmp(op, "rmsnorm_gated") || !strcmp(op, "l2norm")) {
        xt = act("x", 1, ST_MAX_DIMS);
        size_t d = (size_t)xt->shape[xt->ndim - 1], n = (size_t)numel(xt) / d;
        double eps = num("eps");
        if (eps < 0) die("eps 不能是负数%s", "");
        float *x = f32_copy(xt), *w = NULL, *z = NULL;
        y = xmalloc(n * d * sizeof *y);
        if (strcmp(op, "l2norm")) {
            const st_tensor *wt = get("w");
            if (wt->ndim != 1 || wt->shape[0] != d) die("张量 %s 的长度应等于输入的最后一维", wt->name);
            w = vec(wt);
        }
        if (!strcmp(op, "rmsnorm")) op_rmsnorm(y, x, w, n, d, eps);
        else if (!strcmp(op, "l2norm")) op_l2norm(y, x, n, d, eps);
        else {
            const st_tensor *zt = act("z", xt->ndim, xt->ndim);
            if (memcmp(zt->shape, xt->shape, sizeof(uint64_t) * (size_t)xt->ndim)) die("张量 %s 的形状应与输入相同", zt->name);
            z = f32_copy(zt);
            op_rmsnorm_gated(y, x, z, w, n, d, eps);
        }
        free(x); free(w); free(z);
    } else if (!strcmp(op, "silu") || !strcmp(op, "sigmoid") || !strcmp(op, "softplus") || !strcmp(op, "softmax")) {
        xt = act("x", 1, ST_MAX_DIMS);
        size_t n = (size_t)numel(xt);
        float *x = f32_copy(xt);
        y = xmalloc(n * sizeof *y);
        if (!strcmp(op, "silu")) op_silu(y, x, n);
        else if (!strcmp(op, "sigmoid")) op_sigmoid(y, x, n);
        else if (!strcmp(op, "softplus")) op_softplus(y, x, n);
        else { size_t d = (size_t)xt->shape[xt->ndim - 1]; op_softmax(y, x, n / d, d); }
        free(x);
    } else if (!strcmp(op, "rope")) {
        xt = act("x", 3, 3);
        int32_t *pos = i32(get("pos"), xt->shape[0]);
        double theta = num("theta"), r = num("rot");
        if (!(theta > 0)) die("theta 必须是正数%s", "");
        if (r != floor(r) || r < 2 || fmod(r, 2) != 0 || r > (double)xt->shape[2])
            die("rot 必须是正偶数，且不超过 head_dim%s", "");
        float *x = f32_copy(xt);
        y = xmalloc((size_t)numel(xt) * sizeof *y);
        op_rope(y, x, pos, (size_t)xt->shape[0], (size_t)xt->shape[1], (size_t)xt->shape[2], (size_t)r, theta);
        free(x); free(pos);
    } else if (!strcmp(op, "conv1d")) {
        xt = act("x", 2, 2);
        const st_tensor *wt = get("w");
        uint64_t c = xt->shape[1];
        if (!((wt->ndim == 2 || (wt->ndim == 3 && wt->shape[1] == 1)) && wt->shape[0] == c && wt->shape[wt->ndim - 1] > 0))
            die("张量 %s 应为 [c,k] 或 [c,1,k]，c 等于输入的通道数", wt->name);
        size_t k = (size_t)wt->shape[wt->ndim - 1];
        float *x = f32_copy(xt), *w = vec(wt);
        y = xmalloc((size_t)numel(xt) * sizeof *y);
        op_conv1d(y, x, w, (size_t)xt->shape[0], (size_t)c, k);
        free(x); free(w);
    } else {
        die("不认识的算子 %s", op);
    }
    if (xt && ndim == 0) { ndim = xt->ndim; memcpy(shape, xt->shape, sizeof(uint64_t) * (size_t)ndim); }
    print(y, shape, ndim);
    free(y);
    st_set_close(&S);
    return 0;
}
