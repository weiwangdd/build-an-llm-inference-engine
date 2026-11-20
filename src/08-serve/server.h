/* server.h — 第 8 章：HTTP 服务端。
 *
 * 只有一个线程，一个循环反复做三件事：
 *   1. 用 poll 看哪些连接有动静：新连接来了、收到了请求的字节、可以接着发应答、客户端断开了；
 *   2. 收齐一个请求就解析（http.c、api.c），套上对话模板、分词，交给 engine 排队；
 *      客户端在生成途中断开，就把它的请求从 engine 里取消；
 *   3. engine 里还有请求，就调用一次 engine_step，所有在生成的请求各前进一个 token，
 *      新 token 解码成文字，追加到各自连接的发送缓冲区（流式的马上作为一个 SSE 事件发出）。
 * 没有锁，也没有线程之间的通信，所以批处理的结果完全确定：与 tools/batch 和逐个单独生成逐位相同。
 * 代价是 engine_step 算的时候（27B 上每步约 1 秒）不收发数据，连接在内核的缓冲区里等着。
 *
 * 每个连接只处理一个请求，应答发完就关闭（Connection: close）。 */
#ifndef SERVER_H
#define SERVER_H

#include <stddef.h>
#include "gen.h"
#include "model.h"
#include "tok.h"

typedef struct {
    const char *host;        /* 监听的 IPv4 地址，如 127.0.0.1 */
    int         port;        /* 0 表示让系统挑一个空闲端口 */
    const char *model_name;  /* 请求里的 model 必须是它 */
    size_t      max_batch;   /* 同时生成的请求最多几个，其余的排队 */
    size_t      max_ctx;     /* 每个请求的提示词加生成最多几个 token */
    size_t      chunk;       /* 预填充每步最多几个 token，0 表示一次全给 */
    size_t      max_conns;   /* 同时最多几个连接，超过的回 503 */
    int         log_steps;   /* 1：每一步在 stderr 打印这一批的组成 */
    unsigned    slow_ms;     /* 测试用：每一步之后多等这么多毫秒，让小模型像大模型那样慢，好测取消 */
} server_opts;

/* 开始监听后在 stdout 打印一行 “监听 http://<host>:<port>”，然后一直服务，直到收到 SIGINT 或 SIGTERM，
 * 取消所有请求、关闭所有连接后返回 0。每个请求的开始、结束、取消在 stderr 各打印一行。
 * 启动失败（比如端口被占用）返回 -1。 */
int server_run(const model *m, const gen_cfg *g, const tok *t, const server_opts *o, char *err, size_t errcap);

#endif
