/* json.h — 一个通用的小型 JSON 解析器，把整段文本解析成一棵树。
 *
 * 第 1 章的 safetensors 头只需要很少的 JSON 特性，所以那里的解析器是专用的。
 * tokenizer.json 用到了 JSON 的全部特性（布尔、null、嵌套对象），于是这里写一个通用的。 */
#ifndef JSON_H
#define JSON_H

#include <stddef.h>
#include <stdint.h>

typedef enum { J_NULL, J_FALSE, J_TRUE, J_NUM, J_STR, J_ARR, J_OBJ } jtype;

typedef struct jval jval;
struct jval {
    jtype  type;
    char  *str;       /* J_STR：解码后的 UTF-8，以 0 结尾，但中间可能有 \u0000，长度看 len */
                      /* J_NUM：原文，以 0 结尾 */
    size_t len;       /* J_STR / J_NUM：字节数；J_ARR / J_OBJ：元素个数 */
    jval  *items;     /* J_ARR：元素；J_OBJ：值 */
    jval  *keys;      /* J_OBJ：键（都是 J_STR），与 items 一一对应，保持文件中的顺序 */
};

/* 解析 [s, s+n)。成功返回 0；失败返回 -1，并把原因（含行号）写进 err。
 * 同一个对象里出现重复的键算失败。 */
int  json_parse(jval *root, const char *s, size_t n, char *err, size_t errcap);
void json_free(jval *v);
/* 读整个文件再解析（第 6 章从 model.c 挪过来，config.json 和 generation_config.json 共用）。 */
int  json_parse_file(jval *root, const char *path, char *err, size_t errcap);

/* 在对象里按键查找（线性查找，只用于小对象）；找不到或 v 不是对象返回 NULL。 */
const jval *json_get(const jval *v, const char *key);
/* 把 J_NUM 解析成整数；不是整数（带小数点、指数或超出范围）返回 -1。 */
int  json_int(const jval *v, int64_t *out);

#endif
