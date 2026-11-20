/* http.h — 第 8 章：HTTP/1.1 请求的解析（只解析，不碰网络）。
 *
 * 一个 HTTP 请求是这样一段文字：
 *   POST /v1/chat/completions HTTP/1.1\r\n        请求行：方法、路径、版本
 *   Host: localhost:8000\r\n                      头部：每行一个 “名字: 值”
 *   Content-Type: application/json\r\n
 *   Content-Length: 52\r\n                        正文有多少字节
 *   \r\n                                          空行，头部结束
 *   {"model":"qwen", "messages": [...]}           正文（body），恰好 Content-Length 个字节
 *
 * 只支持服务端用到的最小子集，其余一律拒绝，并给出对应的状态码：
 *   - 只认 HTTP/1.1（505）；每行必须以 \r\n 结束，不接受续行（400）；必须有 Host（400）。
 *   - 正文只能用 Content-Length 给出长度；Transfer-Encoding（比如分块传输）不支持（501）；
 *     POST 没有 Content-Length（411）；Content-Length 重复或不是十进制数（400）。
 *   - 头部最多 HTTP_MAX_HEAD 字节（431），正文最多 HTTP_MAX_BODY 字节（413）。
 *   - Expect 只认 100-continue（417）。
 * 每个连接只处理一个请求（应答之后关闭连接），所以不需要处理同一个连接上的下一个请求。 */
#ifndef HTTP_H
#define HTTP_H

#include <stddef.h>

#define HTTP_MAX_HEAD (16 * 1024)
#define HTTP_MAX_BODY (1024 * 1024)

typedef struct {
    char   method[16];
    char   path[256];
    size_t head_len;          /* 请求行加头部加空行的字节数，正文从这里开始 */
    size_t body_len;          /* Content-Length，没有时为 0 */
    int    expect_continue;   /* 客户端发了 Expect: 100-continue，要先回一句 100 Continue 才会发正文 */
} http_req;

/* 看 buf 里已经收到的 n 个字节。
 *   返回 0：头部还没收完，接着收；
 *   返回 1：头部完整，已填好 r，正文还没收齐（buf 里有 r->head_len + r->body_len 个字节时才齐）；
 *   返回 2：整个请求都收齐了；
 *   返回 -1：请求不合法，*status 是应答的状态码，原因写进 err。 */
int http_parse(const char *buf, size_t n, http_req *r, int *status, char *err, size_t errcap);

/* 状态码的标准说明（如 404 → "Not Found"）。 */
const char *http_reason(int status);

#endif
