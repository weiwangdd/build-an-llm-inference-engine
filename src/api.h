/* api.h — 第 8 章：OpenAI 兼容接口的请求解析和应答格式（只处理 JSON 文本，不碰网络）。
 *
 * 只实现 POST /v1/chat/completions（对话补全）和 GET /v1/models（模型列表）的一个子集。
 * 请求里认识的字段：
 *   model                   必需，必须是服务端加载的那个模型的名字（否则 404）
 *   messages                必需，对话消息，交给第 2 章的对话模板。每条消息只认 role、content、
 *                           reasoning_content、tool_calls
 *   stream                  true 时流式应答（SSE，见下）
 *   stream_options          {"include_usage": true} 时流式应答的最后多一块用量统计（只在 stream 时可以给）
 *   max_tokens / max_completion_tokens   最多生成几个 token（只能给一个），不给时生成到上下文长度的上限
 *   temperature             0 到 2；0 表示贪心解码，其余照第 6 章的温度
 *   top_p                   (0, 1]
 *   top_k                   非负整数（不是 OpenAI 的字段，vLLM 等服务端都支持）
 *   seed                    非负整数，随机数种子；不给时是 0，所以同样的请求总是得到同样的回答
 *   n                       只能是 1
 *   chat_template_kwargs    对话模板的参数：enable_thinking、reasoning_effort、preserve_thinking
 *                           （同样不是 OpenAI 的字段，与 vLLM 的写法相同）
 * 没给的采样参数取 generation_config.json 里的。其余字段（tools、stop、logprobs、frequency_penalty……）
 * 一律以 400 拒绝，而不是悄悄忽略：客户端以为生效了，其实没有，这比报错更糟。
 *
 * 流式应答用 SSE（Server-Sent Events，服务器推送事件）：HTTP 应答的正文是一串事件，
 * 每个事件是一行 “data: <JSON>” 加一个空行，客户端边收边处理；最后一个事件是 “data: [DONE]”。 */
#ifndef API_H
#define API_H

#include <stddef.h>
#include <stdint.h>
#include "sample.h"

/* 可以增长的字节缓冲区。内存不足时 oom 置 1，之后的写入都忽略。 */
typedef struct { char *s; size_t n, cap; int oom; } sbuf;
void sb_put(sbuf *b, const void *s, size_t n);
void sb_puts(sbuf *b, const char *s);
void sb_printf(sbuf *b, const char *fmt, ...);
/* 把 [s, s+n) 写成 JSON 字符串（带引号）。s 必须是合法 UTF-8。 */
void sb_json_str(sbuf *b, const char *s, size_t n);
void sb_free(sbuf *b);

typedef struct {
    char      *prompt;          /* 套上对话模板之后的提示词文字（malloc），交给分词器 */
    size_t     prompt_len;
    int        stream;
    int        include_usage;
    size_t     max_tokens;      /* 0 表示请求里没给 */
    sample_cfg sample;
    uint64_t   seed;
} api_chat;

/* 解析 /v1/chat/completions 的正文。def 是 generation_config.json 里的采样参数。
 * 成功返回 0；失败返回 -1，*status 是应答的状态码（400 或 404），原因写进 err。 */
int  api_parse_chat(const char *body, size_t n, const char *model_name, const sample_cfg *def,
                    api_chat *out, int *status, char *err, size_t errcap);
void api_chat_free(api_chat *c);

/* 增量解码：pend 里是还没输出的字节（token 解码出来的，一个汉字可能被切在两个 token 里）。
 * 把能确定的部分换成合法 UTF-8（非法字节按 from_utf8_lossy 的规则换成 U+FFFD）追加到 out，
 * 末尾一个还可能被后面的 token 补全的半个字留在 pend 里。final 为真时全部输出。
 * 分几次输出的结果拼起来，与对全部字节一次做 from_utf8_lossy 相同。 */
void utf8_stream(uint8_t *pend, size_t *n, sbuf *out, int final);

/* ---- 应答的正文 ---- */
/* {"error": {...}} */
void api_error(sbuf *b, int status, const char *msg);
/* 非流式的完整应答 */
void api_completion(sbuf *b, const char *id, long created, const char *model, const char *content, size_t len,
                    const char *finish, size_t prompt_tokens, size_t completion_tokens);
/* 流式应答的一个事件（含 “data: ” 和空行）。role 不为 NULL 时 delta 里带 role；
 * finish 不为 NULL 时这是最后一块，delta 为空。 */
void api_chunk(sbuf *b, const char *id, long created, const char *model, const char *role,
               const char *content, size_t len, const char *finish);
void api_usage_chunk(sbuf *b, const char *id, long created, const char *model, size_t prompt_tokens,
                     size_t completion_tokens);
void api_models(sbuf *b, const char *model, long created);

#endif
