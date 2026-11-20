/* chat.c — 手写的 Qwen3.8 对话模板。
 *
 * 这个文件逐段对应 chat_template.jinja。每一段前面的注释引用了模板里对应的那几行，
 * 读者可以把两者并排对照。模板里每个标签都用了 {%- 和 {{-，模板文本里的空白和换行
 * 全部被吃掉，所以输出恰好是各个 {{ }} 表达式的值依次拼接起来，这让手写成为可能。
 *
 * 模板运行在 Jinja 里，值的语义是 Python 的：某个字段可能“未定义”（JSON 里没有这个键），
 * 也可能是 null、布尔、数字、字符串、数组、对象。下面的代码在每个用到字段的地方都按
 * Jinja 的规则处理这几种情况；遇到模板会报错的情况，我们也报错。 */
#include "chat.h"
#include "sha256.h"
#include "unicode.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Qwen3.8-27B 的 chat_template.jinja（9004 字节）的 SHA-256 */
static const char TEMPLATE_SHA256[] = "af16f654968d47d09a6c604e00370b12dbc21f76456f59c9e98b82e84d9f6375";

static int fail(char *err, size_t cap, const char *fmt, ...) {
    if (err && cap) { va_list ap; va_start(ap, fmt); vsnprintf(err, cap, fmt, ap); va_end(ap); }
    return -1;
}

int chat_template_check(const char *text, size_t n, char *err, size_t cap) {
    uint8_t h[32];
    char hex[65];
    sha256((const uint8_t *)text, n, h);
    for (int i = 0; i < 32; i++) snprintf(hex + 2 * i, 3, "%02x", h[i]);
    if (strcmp(hex, TEMPLATE_SHA256))
        return fail(err, cap, "对话模板不是 Qwen3.8-27B 的那一份（SHA-256 %s），本实现只支持那一份", hex);
    return 0;
}

/* ---------- 可增长的字符串 ---------- */

typedef struct { char *s; size_t n, cap; int oom; } sbuf;

static void put(sbuf *b, const char *s, size_t n) {
    if (b->oom) return;
    if (b->n + n + 1 > b->cap) {
        size_t c = b->cap ? b->cap : 256;
        while (c < b->n + n + 1) c *= 2;
        char *t = realloc(b->s, c);
        if (!t) { b->oom = 1; return; }
        b->s = t; b->cap = c;
    }
    memcpy(b->s + b->n, s, n);
    b->n += n;
    b->s[b->n] = 0;
}
static void puts_(sbuf *b, const char *s) { put(b, s, strlen(s)); }

/* ---------- Jinja / Python 的值语义 ---------- */

/* 字段：NULL 表示“未定义”。只有对象才有字段，对其他类型取字段也是未定义。 */
static const jval *field(const jval *v, const char *k) { return v && v->type == J_OBJ ? json_get(v, k) : NULL; }

static int is_str(const jval *v, const char *s) {
    return v && v->type == J_STR && v->len == strlen(s) && memcmp(v->str, s, v->len) == 0;
}

/* Python 的真值：未定义、null、false、0、空串、空数组、空对象为假 */
static int truthy(const jval *v) {
    if (!v) return 0;
    switch (v->type) {
    case J_NULL: case J_FALSE: return 0;
    case J_TRUE: return 1;
    case J_NUM: { for (const char *p = v->str; *p && *p != 'e' && *p != 'E'; p++) if (*p >= '1' && *p <= '9') return 1; return 0; }
    default: return v->len != 0;
    }
}

/* Python str.isspace() 为真的字符，Jinja 的 trim 过滤器用的就是 str.strip() */
static int py_space(uint32_t c) {
    return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 ||
           c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 ||
           c == 0x202F || c == 0x205F || c == 0x3000;
}

/* trim：去掉首尾空白，返回 [*b, *e) */
static void trim(const char *s, size_t n, size_t *b, size_t *e) {
    size_t i = 0, j = n, k;
    uint32_t c;
    while (i < n) { k = i; if (utf8_next((const uint8_t *)s, n, &k, &c) || !py_space(c)) break; i = k; }
    /* 从尾部往回找：UTF-8 的续字节是 10xxxxxx，回退到字符开头 */
    while (j > i) {
        size_t st = j - 1;
        while (st > i && ((unsigned char)s[st] & 0xC0) == 0x80) st--;
        k = st;
        if (utf8_next((const uint8_t *)s, n, &k, &c) || !py_space(c)) break;
        j = st;
    }
    *b = i; *e = j;
}

/* ---------- tojson：与 Python 的 json.dumps(x, ensure_ascii=False) 相同 ---------- */

/* Python 的 repr(float)：最短的、能精确读回的十进制表示。
 * 十进制指数在 [-4, 16) 之间用定点写法（整数也要带 .0），否则用科学计数法，指数至少两位。 */
static void py_float(sbuf *b, double x) {
    char buf[64];
    if (isinf(x)) { puts_(b, x > 0 ? "Infinity" : "-Infinity"); return; }
    if (isnan(x)) { puts_(b, "NaN"); return; }
    int prec = 1;
    for (; prec <= 17; prec++) {
        snprintf(buf, sizeof buf, "%.*e", prec - 1, x);
        if (strtod(buf, NULL) == x) break;
    }
    /* buf 形如 -d.ddde+XX：取出符号、数字串和指数 */
    char digits[32]; int nd = 0, neg = 0;
    const char *p = buf;
    if (*p == '-') { neg = 1; p++; }
    for (; *p && *p != 'e'; p++) if (*p != '.') digits[nd++] = *p;
    int ex = atoi(p + 1);
    while (nd > 1 && digits[nd - 1] == '0') nd--;
    digits[nd] = 0;
    if (neg) puts_(b, "-");
    if (ex >= -4 && ex < 16) {
        if (ex < 0) {
            puts_(b, "0.");
            for (int i = 0; i < -ex - 1; i++) puts_(b, "0");
            put(b, digits, (size_t)nd);
        } else if (nd <= ex + 1) {
            put(b, digits, (size_t)nd);
            for (int i = nd; i < ex + 1; i++) puts_(b, "0");
            puts_(b, ".0");
        } else {
            put(b, digits, (size_t)ex + 1);
            puts_(b, ".");
            put(b, digits + ex + 1, (size_t)(nd - ex - 1));
        }
    } else {
        put(b, digits, 1);
        if (nd > 1) { puts_(b, "."); put(b, digits + 1, (size_t)nd - 1); }
        snprintf(buf, sizeof buf, "e%c%02d", ex < 0 ? '-' : '+', ex < 0 ? -ex : ex);
        puts_(b, buf);
    }
}

static void json_str(sbuf *b, const char *s, size_t n) {
    puts_(b, "\"");
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        char e[8];
        switch (c) {
        case '"':  puts_(b, "\\\""); break;
        case '\\': puts_(b, "\\\\"); break;
        case '\n': puts_(b, "\\n"); break;
        case '\r': puts_(b, "\\r"); break;
        case '\t': puts_(b, "\\t"); break;
        case '\b': puts_(b, "\\b"); break;
        case '\f': puts_(b, "\\f"); break;
        default:
            if (c < 0x20) { snprintf(e, sizeof e, "\\u%04x", c); puts_(b, e); }
            else put(b, s + i, 1);
        }
    }
    puts_(b, "\"");
}

static void tojson(sbuf *b, const jval *v) {
    switch (v->type) {
    case J_NULL:  puts_(b, "null"); break;
    case J_TRUE:  puts_(b, "true"); break;
    case J_FALSE: puts_(b, "false"); break;
    case J_STR:   json_str(b, v->str, v->len); break;
    case J_NUM:
        if (strpbrk(v->str, ".eE")) py_float(b, strtod(v->str, NULL));
        else if (strcmp(v->str, "-0") == 0) puts_(b, "0");   /* Python 里 -0 是整数 0 */
        else puts_(b, v->str);                                /* 整数原样输出，Python 的 int 不限位数 */
        break;
    case J_ARR:
        puts_(b, "[");
        for (size_t i = 0; i < v->len; i++) { if (i) puts_(b, ", "); tojson(b, &v->items[i]); }
        puts_(b, "]");
        break;
    case J_OBJ:
        puts_(b, "{");
        for (size_t i = 0; i < v->len; i++) {
            if (i) puts_(b, ", ");
            json_str(b, v->keys[i].str, v->keys[i].len);
            puts_(b, ": ");
            tojson(b, &v->items[i]);
        }
        puts_(b, "}");
        break;
    }
}

/* ---------- 模板本体 ---------- */

typedef struct {
    sbuf out;
    int image_count, video_count;   /* {%- set image_count = namespace(value=0) %} */
    int add_vision_id;
    char *err; size_t cap;
} ctx;

/* {%- macro render_content(content, do_vision_count, is_system_content=false) %}
 * 结果写进 b。 */
static int render_content(ctx *c, sbuf *b, const jval *content, int do_count, int is_system) {
    /* {%- elif content is none or content is undefined %} {{- '' }} */
    if (!content || content->type == J_NULL) return 0;
    /* {%- if content is string %} {{- content }} */
    if (content->type == J_STR) { put(b, content->str, content->len); return 0; }
    /* {%- elif content is iterable and content is not mapping %} ：数组 */
    if (content->type != J_ARR) return fail(c->err, c->cap, "Unexpected content type.");
    for (size_t i = 0; i < content->len; i++) {
        const jval *item = &content->items[i];
        /* 模板假定每一项是对象。字符串项会让 'image' in item 变成子串判断，我们不支持 */
        if (item->type != J_OBJ) return fail(c->err, c->cap, "content 数组的元素必须是对象");
        const jval *type = field(item, "type");
        char tag[32];
        /* {%- if 'image' in item or 'image_url' in item or item.type == 'image' %} */
        if (field(item, "image") || field(item, "image_url") || is_str(type, "image")) {
            if (is_system) return fail(c->err, c->cap, "System message cannot contain images.");
            if (do_count) c->image_count++;
            if (c->add_vision_id) { snprintf(tag, sizeof tag, "Picture %d: ", c->image_count); puts_(b, tag); }
            puts_(b, "<|vision_start|><|image_pad|><|vision_end|>");
        /* {%- elif 'video' in item or item.type == 'video' %} */
        } else if (field(item, "video") || is_str(type, "video")) {
            if (is_system) return fail(c->err, c->cap, "System message cannot contain videos.");
            if (do_count) c->video_count++;
            if (c->add_vision_id) { snprintf(tag, sizeof tag, "Video %d: ", c->video_count); puts_(b, tag); }
            puts_(b, "<|vision_start|><|video_pad|><|vision_end|>");
        /* {%- elif 'text' in item %} {{- item.text }} */
        } else if (field(item, "text")) {
            const jval *t = field(item, "text");
            /* Jinja 会把非字符串转成 Python 的 str()，比如 True、None；我们不支持这种写法 */
            if (t->type != J_STR) return fail(c->err, c->cap, "content 里的 text 必须是字符串");
            put(b, t->str, t->len);
        } else return fail(c->err, c->cap, "Unexpected item type in content.");
    }
    return 0;
}

/* render_content(...)|trim，结果放进 *s（malloc），长度 *n */
static int content_trimmed(ctx *c, const jval *content, int do_count, int is_system, char **s, size_t *n) {
    sbuf t = {0};
    if (render_content(c, &t, content, do_count, is_system)) { free(t.s); return -1; }
    if (t.oom) { free(t.s); return fail(c->err, c->cap, "内存不足"); }
    size_t b, e;
    trim(t.s ? t.s : "", t.n, &b, &e);
    *s = malloc(e - b + 1);
    if (!*s) { free(t.s); return fail(c->err, c->cap, "内存不足"); }
    memcpy(*s, t.s ? t.s + b : "", e - b);
    (*s)[e - b] = 0; *n = e - b;
    free(t.s);
    return 0;
}

static int starts(const char *s, size_t n, const char *p) { size_t k = strlen(p); return n >= k && !memcmp(s, p, k); }
static int ends(const char *s, size_t n, const char *p) { size_t k = strlen(p); return n >= k && !memcmp(s + n - k, p, k); }

static const char TOOLS_INTRO[] = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
static const char TOOLS_FORMAT[] =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested "
    "within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the function call, "
    "but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current knowledge and "
    "do not tell the user about function calls\n</IMPORTANT>";
static const char EFFORT_XHIGH[] =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
    "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
static const char EFFORT_LOW[] =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
    "without unnecessary elaboration.";

/* 助手消息里的工具调用 */
static int render_tool_calls(ctx *c, const jval *calls, size_t content_len) {
    sbuf *o = &c->out;
    for (size_t i = 0; i < calls->len; i++) {
        const jval *tc = &calls->items[i];
        /* {%- if tool_call.function is defined %} {%- set tool_call = tool_call.function %} */
        if (field(tc, "function")) tc = field(tc, "function");
        /* '...' + tool_call.name：name 不是字符串时 Jinja 会报错 */
        const jval *name = field(tc, "name");
        if (!name || name->type != J_STR) return fail(c->err, c->cap, "工具调用缺少字符串类型的 name");
        if (i == 0) puts_(o, content_len ? "\n\n<tool_call>\n<function=" : "<tool_call>\n<function=");
        else puts_(o, "\n<tool_call>\n<function=");
        put(o, name->str, name->len);
        puts_(o, ">\n");
        /* {%- if tool_call.arguments is defined and tool_call.arguments != '' %}
         *     {%- for args_name, args_value in tool_call.arguments|items %} */
        const jval *args = field(tc, "arguments");
        if (args && !is_str(args, "")) {
            /* items 过滤器只接受映射；OpenAI 接口里常见的“JSON 字符串形式的参数”会让模板报错 */
            if (args->type != J_OBJ) return fail(c->err, c->cap, "Can only get item pairs from a mapping.");
            for (size_t k = 0; k < args->len; k++) {
                puts_(o, "<parameter=");
                put(o, args->keys[k].str, args->keys[k].len);
                puts_(o, ">\n");
                /* 字符串原样输出，其他值用 tojson */
                const jval *v = &args->items[k];
                if (v->type == J_STR) put(o, v->str, v->len); else tojson(o, v);
                puts_(o, "\n</parameter>\n");
            }
        }
        puts_(o, "</function>\n</tool_call>");
    }
    return 0;
}

static const char *const KNOWN[] = {"messages", "tools", "add_generation_prompt", "enable_thinking",
                                    "reasoning_effort", "preserve_thinking", "add_vision_id"};

int chat_render(const jval *req, char **out, size_t *len, char *err, size_t cap) {
    if (!req || req->type != J_OBJ) return fail(err, cap, "请求必须是 JSON 对象");
    for (size_t i = 0; i < req->len; i++) {
        int ok = 0;
        for (size_t k = 0; k < sizeof KNOWN / sizeof *KNOWN; k++) ok |= is_str(&req->keys[i], KNOWN[k]);
        if (!ok) return fail(err, cap, "不认识的字段 \"%s\"", req->keys[i].str);
    }
    const jval *msgs = field(req, "messages"), *tools = field(req, "tools");
    const jval *thinking = field(req, "enable_thinking"), *effort = field(req, "reasoning_effort");
    const jval *preserve = field(req, "preserve_thinking");
    if (!msgs || msgs->type != J_ARR) return fail(err, cap, "messages 必须是数组");
    for (size_t i = 0; i < msgs->len; i++)
        if (msgs->items[i].type != J_OBJ) return fail(err, cap, "第 %zu 条消息不是对象", i);
    /* 模板只把工具列表当作数组用；对象会被模板悄悄忽略，字符串会被逐字符输出，这两种我们都拒绝 */
    if (tools && tools->type != J_NULL && tools->type != J_ARR) return fail(err, cap, "tools 必须是数组");

    ctx c = {.add_vision_id = truthy(field(req, "add_vision_id")), .err = err, .cap = cap};
    sbuf *o = &c.out;
    char *content = NULL;
    size_t cl = 0;
    int rc = -1;

    /* {%- if not messages %} raise */
    if (msgs->len == 0) { fail(err, cap, "No messages provided."); goto done; }

    /* 思考强度说明：
     * {%- if enable_thinking is undefined or enable_thinking is true %}
     *     resolved_reasoning_effort = reasoning_effort|default('xhigh')
     * “is true”只对 JSON 的 true 成立；default 只替换未定义，null 不算未定义。 */
    const char *instr = "";
    if (!thinking || thinking->type == J_TRUE) {
        if (!effort || is_str(effort, "xhigh")) instr = EFFORT_XHIGH;
        else if (is_str(effort, "low")) instr = EFFORT_LOW;
        else if (!is_str(effort, "medium")) { fail(err, cap, "Unexpected reasoning effort."); goto done; }
    }

    const jval *first = &msgs->items[0];
    int first_is_system = is_str(field(first, "role"), "system");
    if (truthy(tools)) {
        /* {%- if tools and tools is iterable and tools is not mapping %} */
        puts_(o, "<|im_start|>system\n");
        if (*instr) { puts_(o, instr); puts_(o, "\n\n"); }
        puts_(o, TOOLS_INTRO);
        for (size_t i = 0; i < tools->len; i++) { puts_(o, "\n"); tojson(o, &tools->items[i]); }
        puts_(o, "\n</tools>");
        puts_(o, TOOLS_FORMAT);
        if (first_is_system) {
            if (content_trimmed(&c, field(first, "content"), 0, 1, &content, &cl)) goto done;
            if (cl) { puts_(o, "\n\n"); put(o, content, cl); }
            free(content); content = NULL;
        }
        puts_(o, "<|im_end|>\n");
    } else if (first_is_system) {
        if (content_trimmed(&c, field(first, "content"), 0, 1, &content, &cl)) goto done;
        if (cl) {
            puts_(o, "<|im_start|>system\n");
            if (*instr) { puts_(o, instr); puts_(o, "\n\n"); }
            put(o, content, cl);
            puts_(o, "<|im_end|>\n");
        } else if (*instr) { puts_(o, "<|im_start|>system\n"); puts_(o, instr); puts_(o, "<|im_end|>\n"); }
        free(content); content = NULL;
    } else if (*instr) {
        puts_(o, "<|im_start|>system\n"); puts_(o, instr); puts_(o, "<|im_end|>\n");
    }

    /* 从后往前找最后一条“真正的”用户提问：内容不是整个包在 <tool_response> 里的那种。
     * {%- for message in messages[::-1] %} ... {%- set ns.last_query_index = index %} */
    size_t n = msgs->len, last_query = n - 1;
    int multi_step = 1;
    for (size_t k = n; k-- > 0 && multi_step;) {
        if (!is_str(field(&msgs->items[k], "role"), "user")) continue;
        if (content_trimmed(&c, field(&msgs->items[k], "content"), 0, 0, &content, &cl)) goto done;
        if (!(starts(content, cl, "<tool_response>") && ends(content, cl, "</tool_response>"))) {
            multi_step = 0; last_query = k;
        }
        free(content); content = NULL;
    }
    if (multi_step) { fail(err, cap, "No user query found in messages."); goto done; }

    for (size_t i = 0; i < n; i++) {
        const jval *m = &msgs->items[i], *role = field(m, "role");
        if (content_trimmed(&c, field(m, "content"), 1, 0, &content, &cl)) goto done;
        if (is_str(role, "system")) {
            if (i != 0) { fail(err, cap, "System message must be at the beginning."); goto done; }
        } else if (is_str(role, "user")) {
            puts_(o, "<|im_start|>user\n"); put(o, content, cl); puts_(o, "<|im_end|>\n");
        } else if (is_str(role, "assistant")) {
            const jval *r = field(m, "reasoning_content");
            size_t rb = 0, re = 0;
            if (r && r->type == J_STR) trim(r->str, r->len, &rb, &re);
            puts_(o, "<|im_start|>assistant\n");
            /* {%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %} */
            if (!preserve || preserve->type == J_TRUE || i > last_query) {
                puts_(o, "<think>\n");
                if (re > rb) put(o, r->str + rb, re - rb);
                puts_(o, "\n</think>\n\n");
            }
            put(o, content, cl);
            /* {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %} */
            const jval *calls = field(m, "tool_calls");
            if (truthy(calls)) {
                if (calls->type != J_ARR) {
                    /* 非空对象会被模板跳过，非空字符串会让模板报错。两种都不是正常输入，我们拒绝 */
                    fail(err, cap, "tool_calls 必须是数组"); goto done;
                }
                if (render_tool_calls(&c, calls, cl)) goto done;
            }
            puts_(o, "<|im_end|>\n");
        } else if (is_str(role, "tool")) {
            /* {%- if loop.previtem and loop.previtem.role != "tool" %} {{- '<|im_start|>user' }}
             * 第一条消息就是工具结果时没有 previtem，于是不加这个开头（模板如此）。
             * previtem 若是空对象，Jinja 视为假；但空对象没有 role，在上一轮就已经报错了。 */
            if (i > 0 && !is_str(field(&msgs->items[i - 1], "role"), "tool")) puts_(o, "<|im_start|>user");
            puts_(o, "\n<tool_response>\n"); put(o, content, cl); puts_(o, "\n</tool_response>");
            /* 下一条不是工具结果，或者这是最后一条：结束这一轮 */
            if (i + 1 == n || !is_str(field(&msgs->items[i + 1], "role"), "tool")) puts_(o, "<|im_end|>\n");
        } else { fail(err, cap, "Unexpected message role."); goto done; }
        free(content); content = NULL;
    }

    if (truthy(field(req, "add_generation_prompt"))) {
        puts_(o, "<|im_start|>assistant\n");
        /* {%- if enable_thinking is defined and enable_thinking is false %} */
        puts_(o, thinking && thinking->type == J_FALSE ? "<think>\n\n</think>\n\n" : "<think>\n");
    }
    if (o->oom) { fail(err, cap, "内存不足"); goto done; }
    rc = 0;
done:
    free(content);
    if (rc) { free(o->s); return -1; }
    *out = o->s ? o->s : calloc(1, 1);
    *len = o->n;
    return *out ? 0 : fail(err, cap, "内存不足");
}
