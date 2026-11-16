/* chat.h — Qwen3.8 的对话模板：把一组对话消息拼成模型训练时见过的那种文本。
 *
 * 官方的模板是一个 Jinja 文件（chat_template.jinja），由 Python 的 transformers 库执行。
 * 我们不实现 Jinja，而是把这一份模板的逻辑手写成 C，所以只认这一份模板：
 * chat_template_check 用 SHA-256 确认文件与我们照着写的那一份逐字节相同。
 *
 * 输入是一个 JSON 对象，字段与 transformers 的 apply_chat_template 的参数对应：
 *   messages              消息数组（必需）。每条消息是对象：role、content，助手消息还可以有
 *                         reasoning_content 和 tool_calls
 *   tools                 工具定义数组（可选）
 *   add_generation_prompt 是否在末尾加上“轮到助手说话”的开头（可选）
 *   enable_thinking       思考模式开关（可选）
 *   reasoning_effort      思考强度："xhigh"（默认）、"medium"、"low"（可选）
 *   preserve_thinking     是否保留历史轮次的思考内容（可选，默认保留）
 *   add_vision_id         是否给图片、视频编号（可选）
 * 不认识的字段一律拒绝。 */
#ifndef CHAT_H
#define CHAT_H

#include <stddef.h>
#include "json.h"

/* 模板文件内容是否就是我们实现的那一份。是返回 0。 */
int chat_template_check(const char *text, size_t n, char *err, size_t errcap);

/* 按模板渲染。成功返回 0，*out 由 malloc 分配，以 0 结尾，*len 是字节数。
 * 模板本身会报错的输入（如没有用户消息、系统消息不在开头）返回 -1，err 里是原因。 */
int chat_render(const jval *req, char **out, size_t *len, char *err, size_t errcap);

#endif
