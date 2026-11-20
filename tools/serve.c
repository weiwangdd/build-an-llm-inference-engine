/* serve — 第 8 章：OpenAI 兼容的 HTTP 服务端。
 *
 *   serve <config.json> <generation_config.json> <tokenizer.json> <chat_template.jinja> [选项 ...] <分片 ...>
 *
 * 选项：
 *   --host H         监听的 IPv4 地址，默认 127.0.0.1（只有本机能连）
 *   --port P         端口，默认 8000；0 表示让系统挑一个
 *   --name N         模型的名字，请求里的 model 必须是它，默认 qwen3.8-27b
 *   --max-batch B    同时生成的请求最多几个，默认 8
 *   --max-ctx C      每个请求的提示词加生成最多几个 token，默认 4096（决定每个请求的缓存有多大）
 *   --chunk K        预填充每步最多交给模型几个 token，默认 0（一次全给）
 *   --max-conns M    同时最多几个连接，默认 64
 *   --log-steps      每一步在 stderr 打印这一批由哪些请求组成
 *   --slow MS        测试用：每一步之后多等 MS 毫秒（模拟大模型的速度，好测取消）
 *
 * 接口见 src/08-serve/api.h，结构见 src/08-serve/server.h。Ctrl-C 或 SIGTERM 停止。
 * make gpu 另建一份 GPU 版（build/gpu/serve），用法相同。 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "chat.h"
#include "gen.h"
#include "gpu.h"
#include "model.h"
#include "server.h"
#include "st.h"
#include "tok.h"

static void die(const char *msg) {
    fprintf(stderr, "错误：%s\n", msg);
    exit(1);
}

static uint64_t num_u(const char *s, uint64_t max, const char *what) {
    uint64_t v = 0;
    if (!*s || strlen(s) > 20) goto bad;
    for (const char *p = s; *p; p++) {
        if (*p < '0' || *p > '9') goto bad;
        unsigned d = (unsigned)(*p - '0');
        if (v > (max - d) / 10) goto bad;
        v = v * 10 + d;
    }
    return v;
bad:
    fprintf(stderr, "错误：%s 应为不超过 %" PRIu64 " 的非负整数\n", what, max);
    exit(1);
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "用法：%s <config.json> <generation_config.json> <tokenizer.json> <chat_template.jinja> "
                        "[选项] <分片> ...\n", argv[0]);
        return 2;
    }
    char err[512];
    server_opts o = {.host = "127.0.0.1", .port = 8000, .model_name = "qwen3.8-27b", .max_batch = 8,
                     .max_ctx = 4096, .chunk = 0, .max_conns = 64};
    int a = 5;
    for (; a < argc && !strncmp(argv[a], "--", 2); a++) {
        const char *opt = argv[a];
        if (!strcmp(opt, "--log-steps")) { o.log_steps = 1; continue; }
        if (a + 1 >= argc) { fprintf(stderr, "错误：选项 %s 缺少参数\n", opt); exit(1); }
        const char *v = argv[++a];
        if (!strcmp(opt, "--host")) o.host = v;
        else if (!strcmp(opt, "--port")) o.port = (int)num_u(v, 65535, "port");
        else if (!strcmp(opt, "--name")) { if (!*v) die("--name 不能为空"); o.model_name = v; }
        else if (!strcmp(opt, "--max-batch")) o.max_batch = (size_t)num_u(v, 1u << 16, "max-batch");
        else if (!strcmp(opt, "--max-ctx")) o.max_ctx = (size_t)num_u(v, 1u << 30, "max-ctx");
        else if (!strcmp(opt, "--chunk")) o.chunk = (size_t)num_u(v, 1u << 30, "chunk");
        else if (!strcmp(opt, "--slow")) o.slow_ms = (unsigned)num_u(v, 60000, "slow");
        else if (!strcmp(opt, "--max-conns")) o.max_conns = (size_t)num_u(v, 1u << 16, "max-conns");
        else { fprintf(stderr, "错误：不认识的选项 %s\n", opt); exit(1); }
    }
    if (a == argc) die("缺少权重分片");

    /* 对话模板必须是我们照着写的那一份（第 2 章） */
    FILE *tf = fopen(argv[4], "rb");
    if (!tf) die("打不开对话模板文件");
    static char tbuf[1 << 16];
    size_t tn = fread(tbuf, 1, sizeof tbuf, tf);
    fclose(tf);
    if (chat_template_check(tbuf, tn, err, sizeof err)) die(err);
    tok *T;
    if (tok_load(&T, argv[3], err, sizeof err)) die(err);

    st_set S;
    if (st_set_open(&S, (const char *const *)(argv + a), (size_t)(argc - a), err, sizeof err)) die(err);
    model M;
    if (model_load(&M, argv[1], &S, err, sizeof err)) die(err);
    if (tok_size(T) > M.cfg.vocab) die("分词器的 id 超出了模型的词表");
#ifdef USE_GPU
    if (model_to_gpu(&M, err, sizeof err)) die(err);      /* 第 7 章：GPU 版 */
#endif
    gen_cfg G;
    if (gen_cfg_load(&G, argv[2], M.cfg.vocab, err, sizeof err)) die(err);

    int rc = server_run(&M, &G, T, &o, err, sizeof err);
    if (rc) fprintf(stderr, "错误：%s\n", err);
    model_free(&M);
    st_set_close(&S);
    tok_free(T);
    return rc ? 1 : 0;
}
