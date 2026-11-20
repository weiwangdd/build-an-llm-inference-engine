/* http.c — 第 8 章：HTTP/1.1 请求的解析。规则见 http.h，依据 RFC 9110 和 RFC 9112。 */
#include "http.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static int bad(int *status, int code, char *err, size_t cap, const char *fmt, ...) {
    *status = code;
    if (err && cap) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, cap, fmt, ap);
        va_end(ap);
    }
    return -1;
}

/* RFC 9110 的 token 字符：方法和头部名字只能由它们组成 */
static int tchar(char c) {
    if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return 1;
    return c && strchr("!#$%&'*+-.^_`|~", c) != NULL;
}

/* 不区分大小写地比较头部名字（HTTP 规定头部名字不区分大小写） */
static int name_is(const char *s, size_t n, const char *want) {
    if (strlen(want) != n) return 0;
    for (size_t i = 0; i < n; i++) {
        char a = s[i], b = want[i];
        if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
        if (a != b) return 0;
    }
    return 1;
}

static int value_is(const char *s, size_t n, const char *want) { return name_is(s, n, want); }

int http_parse(const char *buf, size_t n, http_req *r, int *status, char *err, size_t cap) {
    /* 找头部的结尾：第一个 \r\n\r\n */
    size_t end = 0;
    for (size_t i = 0; i + 3 < n; i++)
        if (buf[i] == '\r' && buf[i + 1] == '\n' && buf[i + 2] == '\r' && buf[i + 3] == '\n') { end = i + 4; break; }
    if (!end) {
        /* 还没收完，但已经能看出不对的，马上拒绝，不要一直等一个永远不会来的 \r\n\r\n */
        for (size_t i = 0; i < n; i++)
            if (buf[i] == '\n' && (i == 0 || buf[i - 1] != '\r'))
                return bad(status, 400, err, cap, "行必须以 \\r\\n 结束");
        if (n >= HTTP_MAX_HEAD) return bad(status, 431, err, cap, "请求头超过 %d 字节", HTTP_MAX_HEAD);
        return 0;
    }
    if (end > HTTP_MAX_HEAD) return bad(status, 431, err, cap, "请求头超过 %d 字节", HTTP_MAX_HEAD);
    memset(r, 0, sizeof *r);
    r->head_len = end;

    /* 逐行处理。每行必须以 \r\n 结束，行内不能有单独的 \r、\n 或 NUL。 */
    size_t p = 0, line = 0;
    int has_host = 0, has_len = 0;
    while (p < end - 2) {
        size_t e = p;
        while (buf[e] != '\r' && buf[e] != '\n' && buf[e]) e++;
        if (buf[e] != '\r' || buf[e + 1] != '\n') return bad(status, 400, err, cap, "第 %zu 行没有以 \\r\\n 结束", line + 1);
        const char *s = buf + p;
        size_t len = e - p;
        if (line == 0) {                        /* 请求行：方法 SP 路径 SP HTTP/1.1 */
            size_t a = 0;
            while (a < len && tchar(s[a])) a++;
            if (a == 0 || a >= sizeof r->method || a >= len || s[a] != ' ')
                return bad(status, 400, err, cap, "请求行格式不对");
            memcpy(r->method, s, a);
            size_t b = a + 1;
            while (b < len && s[b] != ' ') b++;
            if (b == a + 1 || b >= len) return bad(status, 400, err, cap, "请求行格式不对");
            if (s[a + 1] != '/') return bad(status, 400, err, cap, "路径必须以 / 开头");
            if (b - a - 1 >= sizeof r->path) return bad(status, 414, err, cap, "路径太长");
            for (size_t i = a + 1; i < b; i++)
                if ((unsigned char)s[i] <= 0x20 || (unsigned char)s[i] >= 0x7f)
                    return bad(status, 400, err, cap, "路径里有不允许的字符");
            memcpy(r->path, s + a + 1, b - a - 1);
            const char *ver = s + b + 1;
            size_t vl = len - b - 1;
            if (vl != 8 || memcmp(ver, "HTTP/", 5) || ver[6] != '.' || ver[5] < '0' || ver[5] > '9' ||
                ver[7] < '0' || ver[7] > '9')
                return bad(status, 400, err, cap, "请求行格式不对");
            if (memcmp(ver, "HTTP/1.1", 8)) return bad(status, 505, err, cap, "只支持 HTTP/1.1，收到 %.8s", ver);
        } else {                                /* 头部：名字 ":" 可选空白 值 可选空白 */
            if (s[0] == ' ' || s[0] == '\t') return bad(status, 400, err, cap, "不支持头部续行（第 %zu 行）", line + 1);
            size_t a = 0;
            while (a < len && tchar(s[a])) a++;
            if (a == 0 || a >= len || s[a] != ':') return bad(status, 400, err, cap, "第 %zu 行不是“名字: 值”", line + 1);
            size_t v0 = a + 1, v1 = len;
            while (v0 < v1 && (s[v0] == ' ' || s[v0] == '\t')) v0++;
            while (v1 > v0 && (s[v1 - 1] == ' ' || s[v1 - 1] == '\t')) v1--;
            for (size_t i = v0; i < v1; i++)
                if ((unsigned char)s[i] < 0x20 && s[i] != '\t') return bad(status, 400, err, cap, "第 %zu 行的值里有控制字符", line + 1);
            const char *v = s + v0;
            size_t vn = v1 - v0;
            if (name_is(s, a, "host")) {
                if (has_host) return bad(status, 400, err, cap, "Host 出现了不止一次");
                has_host = 1;
            } else if (name_is(s, a, "content-length")) {
                if (has_len) return bad(status, 400, err, cap, "Content-Length 出现了不止一次");
                has_len = 1;
                if (vn == 0 || vn > 12) return bad(status, 400, err, cap, "Content-Length 应为十进制数");
                size_t x = 0;
                for (size_t i = 0; i < vn; i++) {
                    if (v[i] < '0' || v[i] > '9') return bad(status, 400, err, cap, "Content-Length 应为十进制数");
                    x = x * 10 + (size_t)(v[i] - '0');
                }
                r->body_len = x;
            } else if (name_is(s, a, "transfer-encoding")) {
                return bad(status, 501, err, cap, "不支持 Transfer-Encoding，请用 Content-Length 给出正文长度");
            } else if (name_is(s, a, "expect")) {
                if (!value_is(v, vn, "100-continue")) return bad(status, 417, err, cap, "Expect 只支持 100-continue");
                r->expect_continue = 1;
            }
            /* 其余的头部（Content-Type、User-Agent、Accept……）不影响我们的处理，不检查 */
        }
        p = e + 2;
        line++;
    }
    if (!has_host) return bad(status, 400, err, cap, "HTTP/1.1 的请求必须有 Host");
    if (!strcmp(r->method, "POST") && !has_len) return bad(status, 411, err, cap, "POST 请求必须有 Content-Length");
    if (r->body_len > HTTP_MAX_BODY) return bad(status, 413, err, cap, "正文超过 %d 字节", HTTP_MAX_BODY);
    return n - end >= r->body_len ? 2 : 1;
}

const char *http_reason(int s) {
    switch (s) {
    case 100: return "Continue";
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 411: return "Length Required";
    case 413: return "Content Too Large";
    case 414: return "URI Too Long";
    case 417: return "Expectation Failed";
    case 431: return "Request Header Fields Too Large";
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    case 503: return "Service Unavailable";
    case 505: return "HTTP Version Not Supported";
    default:  return "Unknown";
    }
}
