/* server.c — 第 8 章：单线程的 HTTP 服务端。结构见 server.h。 */
#define _POSIX_C_SOURCE 200809L
#include "server.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>
#include "api.h"
#include "engine.h"
#include "http.h"

static int fail(char *err, size_t cap, const char *fmt, ...) {
    if (err && cap) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, cap, fmt, ap);
        va_end(ap);
    }
    return -1;
}

static volatile sig_atomic_t stop_flag;
static void on_signal(int sig) { (void)sig; stop_flag = 1; }

/* 一个连接的状态：
 *   C_READ   在收请求
 *   C_GEN    请求在 engine 里（排队或生成中）
 *   C_FLUSH  应答已经全部放进 out，发完就关
 *   C_DRAIN  应答发完了，关掉了发送方向，把客户端还在发的东西读完再关（直接关会让对方收到 RST，
 *            可能连已经发过去的应答都读不到） */
enum { C_READ, C_GEN, C_FLUSH, C_DRAIN };

typedef struct {
    int      fd;
    int      state;
    sbuf     in, out;
    size_t   sent;              /* out 里已经发出去的字节数 */
    int      sent_continue;
    time_t   deadline;          /* C_DRAIN 最多等到什么时候 */
    /* 生成 */
    uint64_t req;               /* engine 里的编号 */
    char     id[32];            /* 应答里的 id：chatcmpl-<编号> */
    long     created;
    int      stream, include_usage;
    size_t   prompt_tokens, ntok;
    sbuf     pend;              /* 已解码、还没输出的字节（末尾可能是半个字） */
    sbuf     text;              /* 非流式：攒到最后一起发 */
    sbuf     ids;               /* 生成的 token id，逗号分隔，写进日志 */
} conn;

typedef struct {
    const model *m;
    const gen_cfg *g;
    const tok *t;
    const server_opts *o;
    engine e;
    conn **c;
    size_t nc;
} srv;

static void logf_(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    fflush(stderr);
}

static void conn_free(conn *c) {
    close(c->fd);
    sb_free(&c->in); sb_free(&c->out); sb_free(&c->pend); sb_free(&c->text); sb_free(&c->ids);
    free(c);
}

/* 一个完整的（非流式）应答：状态行、头部、正文 */
static void respond(conn *c, int status, const char *ctype, const char *extra, const sbuf *body) {
    sb_printf(&c->out, "HTTP/1.1 %d %s\r\n", status, http_reason(status));
    sb_printf(&c->out, "Content-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n", ctype, body->n);
    if (extra) sb_puts(&c->out, extra);
    sb_puts(&c->out, "\r\n");
    sb_put(&c->out, body->s, body->n);
    c->state = C_FLUSH;
}

static void respond_error(conn *c, int status, const char *msg, const char *extra) {
    sbuf b = {0};
    api_error(&b, status, msg);
    respond(c, status, "application/json", extra, &b);
    sb_free(&b);
}

static conn *by_req(srv *s, uint64_t id) {
    for (size_t i = 0; i < s->nc; i++)
        if (s->c[i]->state == C_GEN && s->c[i]->req == id) return s->c[i];
    return NULL;
}

/* ---------- engine 的回调：新 token 和结束 ---------- */

static void emit(srv *s, conn *c, const sbuf *piece) {
    if (!piece->n) return;
    if (c->stream) api_chunk(&c->out, c->id, c->created, s->o->model_name, NULL, piece->s, piece->n, NULL);
    else sb_put(&c->text, piece->s, piece->n);
}

static void on_token(void *ctx, uint64_t id, size_t k, const float *logits, size_t V, const int32_t *kids,
                     const double *probs, size_t nk, double u, int32_t tok) {
    (void)k; (void)logits; (void)V; (void)kids; (void)probs; (void)nk; (void)u;
    srv *s = ctx;
    conn *c = by_req(s, id);
    if (!c) return;
    sb_printf(&c->ids, c->ntok ? ",%" PRId32 : "%" PRId32, tok);
    c->ntok++;
    uint8_t *bytes;
    size_t nb;
    char err[256];
    /* 特殊 token（结束符 <|im_end|> 等）不出现在文字里，与 tokenizers 的 skip_special_tokens 相同 */
    if (tok_decode(s->t, &tok, 1, 1, &bytes, &nb, err, sizeof err)) { c->pend.oom = 1; return; }
    sb_put(&c->pend, bytes, nb);
    free(bytes);
    if (c->pend.oom) return;
    sbuf piece = {0};
    utf8_stream((uint8_t *)c->pend.s, &c->pend.n, &piece, 0);
    emit(s, c, &piece);
    sb_free(&piece);
}

static void on_done(void *ctx, uint64_t id, eng_stop why, const char *msg) {
    srv *s = ctx;
    conn *c = by_req(s, id);
    if (!c) return;
    const char *finish = why == ENG_EOS ? "stop" : why == ENG_LENGTH ? "length" : "error";
    logf_("完成 id=%" PRIu64 " 原因=%s 提示词=%zu 生成=%zu token=%s%s%s", id, finish, c->prompt_tokens, c->ntok,
          c->ids.n ? c->ids.s : "", why == ENG_ERROR ? " 错误=" : "", why == ENG_ERROR ? msg : "");
    if (c->pend.oom || c->text.oom) { why = ENG_ERROR; msg = "内存不足"; }
    if (why == ENG_ERROR) {
        if (c->stream) {                      /* 头部已经发了，只能在流里报错 */
            sbuf b = {0};
            api_error(&b, 500, msg);
            sb_puts(&c->out, "data: ");
            sb_put(&c->out, b.s, b.n);
            sb_puts(&c->out, "\n\n");
            sb_free(&b);
            c->state = C_FLUSH;
        } else respond_error(c, 500, msg, NULL);
        return;
    }
    sbuf piece = {0};
    utf8_stream((uint8_t *)c->pend.s, &c->pend.n, &piece, 1);
    emit(s, c, &piece);
    sb_free(&piece);
    if (c->stream) {
        api_chunk(&c->out, c->id, c->created, s->o->model_name, NULL, NULL, 0, finish);
        if (c->include_usage) api_usage_chunk(&c->out, c->id, c->created, s->o->model_name, c->prompt_tokens, c->ntok);
        sb_puts(&c->out, "data: [DONE]\n\n");
        c->state = C_FLUSH;
    } else {
        sbuf b = {0};
        api_completion(&b, c->id, c->created, s->o->model_name, c->text.s ? c->text.s : "", c->text.n, finish,
                       c->prompt_tokens, c->ntok);
        respond(c, 200, "application/json", NULL, &b);
        sb_free(&b);
    }
}

/* ---------- 处理一个收齐的请求 ---------- */

static void chat(srv *s, conn *c, const char *body, size_t n) {
    api_chat q;
    int status;
    char err[512];
    if (api_parse_chat(body, n, s->o->model_name, &s->g->sample, &q, &status, err, sizeof err)) {
        respond_error(c, status, err, NULL);
        return;
    }
    int32_t *ids = NULL;
    size_t nids;
    if (tok_encode(s->t, (const uint8_t *)q.prompt, q.prompt_len, &ids, &nids, err, sizeof err)) {
        api_chat_free(&q);
        respond_error(c, 400, err, NULL);
        return;
    }
    api_chat_free(&q);
    size_t ctx = s->o->max_ctx;
    if (nids >= ctx) {
        snprintf(err, sizeof err, "提示词有 %zu 个 token，上下文最多 %zu 个", nids, ctx);
        free(ids);
        respond_error(c, 400, err, NULL);
        return;
    }
    eng_params p = {.sample = q.sample, .seed = q.seed, .chunk = s->o->chunk,
                    .max_new = q.max_tokens ? q.max_tokens : ctx - nids};
    if (engine_add(&s->e, ids, nids, &p, &c->req, err, sizeof err)) {
        free(ids);
        respond_error(c, 400, err, NULL);
        return;
    }
    free(ids);
    c->state = C_GEN;
    c->stream = q.stream;
    c->include_usage = q.include_usage;
    c->prompt_tokens = nids;
    c->created = (long)time(NULL);
    snprintf(c->id, sizeof c->id, "chatcmpl-%" PRIu64, c->req);     /* 与日志里的 id 相同 */
    logf_("开始 id=%" PRIu64 " 提示词=%zu max_new=%zu 流式=%d 采样=%d 种子=%" PRIu64, c->req, nids, p.max_new,
          c->stream, p.sample.do_sample, p.seed);
    if (c->stream) {
        sb_puts(&c->out, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                         "Connection: close\r\n\r\n");
        api_chunk(&c->out, c->id, c->created, s->o->model_name, "assistant", "", 0, NULL);
    }
}

static void handle(srv *s, conn *c, const http_req *r) {
    const char *body = c->in.s + r->head_len;
    if (!strcmp(r->path, "/v1/chat/completions")) {
        if (strcmp(r->method, "POST")) respond_error(c, 405, "这个路径只接受 POST", "Allow: POST\r\n");
        else chat(s, c, body, r->body_len);
    } else if (!strcmp(r->path, "/v1/models")) {
        if (strcmp(r->method, "GET")) { respond_error(c, 405, "这个路径只接受 GET", "Allow: GET\r\n"); return; }
        sbuf b = {0};
        api_models(&b, s->o->model_name, 0);
        respond(c, 200, "application/json", NULL, &b);
        sb_free(&b);
    } else {
        char msg[400];
        snprintf(msg, sizeof msg, "没有路径 %s", r->path);
        respond_error(c, 404, msg, NULL);
    }
}

/* ---------- 网络 ---------- */

static int nonblock(int fd) {
    int fl = fcntl(fd, F_GETFL);
    return fl < 0 ? -1 : fcntl(fd, F_SETFL, fl | O_NONBLOCK);
}

/* 客户端断开或出错：生成中的请求取消，连接关掉 */
static void drop(srv *s, size_t i, const char *why) {
    conn *c = s->c[i];
    if (c->state == C_GEN) {
        engine_cancel(&s->e, c->req);
        logf_("取消 id=%" PRIu64 " 原因=%s 已生成=%zu token=%s", c->req, why, c->ntok, c->ids.n ? c->ids.s : "");
    }
    conn_free(c);
    s->c[i] = s->c[--s->nc];
}

/* 收数据。返回 -1 表示连接要关掉。 */
static int on_readable(srv *s, conn *c) {
    char buf[65536];
    for (;;) {
        ssize_t k = recv(c->fd, buf, sizeof buf, 0);
        if (k == 0) return -1;                         /* 对方关了 */
        if (k < 0) return errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR ? 0 : -1;
        if (c->state != C_READ) continue;              /* 请求已经收齐，多出来的丢掉 */
        sb_put(&c->in, buf, (size_t)k);
        if (c->in.oom) return -1;
        http_req r;
        int status;
        char err[256];
        int rc = http_parse(c->in.s, c->in.n, &r, &status, err, sizeof err);
        if (rc < 0) respond_error(c, status, err, NULL);
        else if (rc == 1 && r.expect_continue && !c->sent_continue) {
            sb_puts(&c->out, "HTTP/1.1 100 Continue\r\n\r\n");
            c->sent_continue = 1;
        } else if (rc == 2) handle(s, c, &r);
        if (c->in.n > HTTP_MAX_HEAD + HTTP_MAX_BODY && c->state == C_READ)
            respond_error(c, 413, "请求太大", NULL);
    }
}

/* 发数据。返回 -1 表示连接要关掉。 */
static int on_writable(conn *c) {
    while (c->sent < c->out.n) {
        ssize_t k = send(c->fd, c->out.s + c->sent, c->out.n - c->sent, 0);
        if (k < 0) return errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR ? 0 : -1;
        c->sent += (size_t)k;
    }
    if (c->state == C_FLUSH) {                         /* 应答发完了：关掉发送方向，等对方关 */
        shutdown(c->fd, SHUT_WR);
        c->state = C_DRAIN;
        c->deadline = time(NULL) + 5;
    }
    return 0;
}

static int open_listener(const server_opts *o, int *port, char *err, size_t cap) {
    struct sockaddr_in a;
    memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)o->port);
    if (inet_pton(AF_INET, o->host, &a.sin_addr) != 1) return fail(err, cap, "监听地址 %s 不是 IPv4 地址", o->host);
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return fail(err, cap, "socket 失败：%s", strerror(errno));
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    socklen_t len = sizeof a;
    if (bind(fd, (struct sockaddr *)&a, sizeof a) || listen(fd, 128) || nonblock(fd) ||
        getsockname(fd, (struct sockaddr *)&a, &len)) {
        fail(err, cap, "监听 %s:%d 失败：%s", o->host, o->port, strerror(errno));
        close(fd);
        return -1;
    }
    *port = ntohs(a.sin_port);
    return fd;
}

int server_run(const model *m, const gen_cfg *g, const tok *t, const server_opts *o, char *err, size_t cap) {
    if (o->max_conns == 0) return fail(err, cap, "max_conns 至少是 1");
    srv s = {.m = m, .g = g, .t = t, .o = o};
    if (engine_init(&s.e, m, g, o->max_batch, o->max_ctx, on_token, on_done, &s, err, cap)) return -1;
    int port = 0, lfd = open_listener(o, &port, err, cap);
    if (lfd < 0) { engine_free(&s.e); return -1; }
    s.c = calloc(o->max_conns, sizeof *s.c);
    struct pollfd *pf = calloc(o->max_conns + 1, sizeof *pf);
    if (!s.c || !pf) { free(s.c); free(pf); close(lfd); engine_free(&s.e); return fail(err, cap, "内存不足"); }

    /* 写已关闭的连接时，默认会收到 SIGPIPE 信号而退出；忽略它，改为 send 返回 EPIPE */
    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_signal;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    printf("监听 http://%s:%d\n", o->host, port);
    fflush(stdout);

    size_t step = 0;
    while (!stop_flag) {
        /* 一、等事件。engine 里有活就不等（超时 0），只看一眼；没活就最多等 1 秒。 */
        pf[0] = (struct pollfd){.fd = lfd, .events = POLLIN};
        for (size_t i = 0; i < s.nc; i++)
            pf[i + 1] = (struct pollfd){.fd = s.c[i]->fd,
                                        .events = (short)(POLLIN | (s.c[i]->sent < s.c[i]->out.n ? POLLOUT : 0))};
        int k = poll(pf, s.nc + 1, engine_pending(&s.e) ? 0 : 1000);
        if (k < 0 && errno != EINTR) { fail(err, cap, "poll 失败：%s", strerror(errno)); break; }
        size_t nc = s.nc;               /* 下面会增删连接，先按这一轮 poll 时的个数处理 */

        /* 二、已有的连接：倒着处理，drop 把最后一个挪到第 i 个，不影响前面的 */
        time_t now = time(NULL);
        for (size_t i = nc; k > 0 && i-- > 0;) {
            conn *c = s.c[i];
            short re = pf[i + 1].revents;
            if ((re & POLLIN) && on_readable(&s, c)) { drop(&s, i, "客户端断开"); continue; }
            if ((re & (POLLERR | POLLNVAL)) || ((re & POLLHUP) && !(re & POLLIN))) { drop(&s, i, "连接出错"); continue; }
            if (on_writable(c)) { drop(&s, i, "发送失败"); continue; }
        }
        for (size_t i = s.nc; i-- > 0;)                 /* 还没发的（比如刚生成的 100 Continue、错误应答）发出去 */
            if (on_writable(s.c[i]) || (s.c[i]->state == C_DRAIN && now > s.c[i]->deadline)) drop(&s, i, "发送结束");
        /* 三、新连接 */
        if (k > 0 && (pf[0].revents & POLLIN))
            for (;;) {
                int fd = accept(lfd, NULL, NULL);
                if (fd < 0) break;
                conn *c = calloc(1, sizeof *c);
                if (!c || nonblock(fd)) { free(c); close(fd); continue; }
                c->fd = fd;
                if (s.nc == o->max_conns) {            /* 满了：回一个 503，借用临时的连接结构发出去 */
                    respond_error(c, 503, "连接太多，请稍后再试", NULL);
                    on_writable(c);
                    conn_free(c);
                    continue;
                }
                s.c[s.nc++] = c;
            }
        /* 四、往前算一步 */
        if (engine_pending(&s.e)) {
            if (engine_step(&s.e, err, cap)) break;
            if (o->log_steps && s.e.last_nseq) {
                sbuf b = {0};
                for (size_t i = 0; i < s.e.last_nseq; i++)
                    sb_printf(&b, " %" PRIu64 ":%zu", s.e.last_ids[i], s.e.last_n[i]);
                logf_("步 %zu 批=%s", step, b.s ? b.s + 1 : "");      /* + 1 跳过开头的空格 */
                sb_free(&b);
            }
            step++;
            if (o->slow_ms) {
                struct timespec ts = {(time_t)(o->slow_ms / 1000), (long)(o->slow_ms % 1000) * 1000000L};
                nanosleep(&ts, NULL);
            }
            for (size_t i = s.nc; i-- > 0;)            /* 新生成的文字马上发出去 */
                if (on_writable(s.c[i])) drop(&s, i, "发送失败");
        }
    }
    int rc = stop_flag ? 0 : -1;
    while (s.nc) drop(&s, s.nc - 1, "服务端停止");
    free(s.c); free(pf);
    close(lfd);
    engine_free(&s.e);
    logf_("停止");
    return rc;
}
