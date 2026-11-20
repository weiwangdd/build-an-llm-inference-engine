# 第 8 章 服务层

正文：[book/08-serve.md](../../book/08-serve.md)

本章新增的文件：

- `engine.h`、`engine.c`：连续批处理（几个请求同时生成，一个结束就补进下一个）。
- `http.h`、`http.c`：HTTP/1.1 的一个严格子集。
- `api.h`、`api.c`：OpenAI 兼容的 `/v1/chat/completions` 请求解析和回答格式。
- `server.h`、`server.c`：单线程的 poll 事件循环，把连接、请求和引擎接起来。

后面章节的扩展：

- 批处理的前向 `model_step_batch` 在 `src/04-layer/model.c` 和 `src/07-gpu/gpu_model.c` 里。

命令行工具：`tools/batch.c`、`tools/serve.c`。测试：`tests/08-serve/`。
