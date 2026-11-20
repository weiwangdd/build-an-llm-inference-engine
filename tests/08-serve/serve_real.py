#!/usr/bin/env python3
"""第 8 章：在真实模型上验收服务端（make test-serve-real）。只用标准库，另用 tokenizers 库解码。

    tests/08-serve/serve_real.py <build 目录> <模型目录>

build 目录里要有 serve、generate、tok（make test-serve-real 用 GPU 版、-O2、不带 ASan 另建的那一份）。
  一、启动 serve，几个问题同时发过去（有流式有非流式，有贪心有采样），记下总用时；
  二、每个问题再单独交给 generate（同样的提示词和参数），逐个记下用时；
  三、要求每个问题服务端生成的 token 与单独生成的逐个相同，文字等于 tokenizers 的解码，
      并报告批处理的总用时与逐个单独生成的总用时。
环境变量 N（同时发几个，默认 4）、GEN（每个最多生成几个 token，默认 32）。
"""
import http.client, json, os, re, signal, subprocess, sys, tempfile, threading, time

from tokenizers import Tokenizer

B, M = sys.argv[1], sys.argv[2]
N, GEN_MAX = int(os.environ.get("N", 4)), int(os.environ.get("GEN", 32))
NAME = "qwen3.8-27b"
FILES = [f"{M}/{f}" for f in ("config.json", "generation_config.json", "tokenizer.json", "chat_template.jinja")]
SHARDS = sorted(os.path.join(M, f) for f in os.listdir(M) if f.endswith(".safetensors"))
TK = Tokenizer.from_file(FILES[2])
QUESTIONS = ["中国的首都是哪里？", "用一句话解释什么是大模型推理。", "Write a haiku about the sea.",
             "1 加 1 等于几？", "推荐一本关于编程的书。", "What is the capital of France?",
             "解释一下什么是缓存。", "Tell me a short joke."]
fails = 0


def body(i):
    b = {"model": NAME, "messages": [{"role": "user", "content": QUESTIONS[i % len(QUESTIONS)]}],
         "max_tokens": GEN_MAX, "chat_template_kwargs": {"enable_thinking": False}}
    if i % 2:
        b["seed"] = i                        # 按 generation_config.json 的参数采样
    else:
        b["temperature"] = 0                 # 贪心
    if i % 4 < 2:
        b["stream"] = True
    return b


def prompt_ids(b):
    req = dict(messages=b["messages"], add_generation_prompt=True, **b["chat_template_kwargs"])
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write(json.dumps(req, ensure_ascii=False) + "\n")
    out = subprocess.run([f"{B}/tok", "chat", FILES[2], FILES[3], f.name], capture_output=True, text=True, check=True)
    os.unlink(f.name)
    return [int(x) for x in out.stdout.split()[1:]]


def single(b):
    ids = prompt_ids(b)
    opts = ["--greedy"] if b.get("temperature") == 0 else ["--seed", str(b.get("seed", 0))]
    t0 = time.time()
    out = subprocess.run([f"{B}/generate", FILES[0], FILES[1], ",".join(map(str, ids)), str(b["max_tokens"]),
                          *opts, *SHARDS], capture_output=True, text=True, check=True)
    return ids, [int(l.split()[1]) for l in out.stdout.splitlines() if l.startswith("token ")], time.time() - t0


# 一、服务端
errpath = f"{B}/serve_real_err.txt"
errlog = open(errpath, "w")
t0 = time.time()
proc = subprocess.Popen([f"{B}/serve", *FILES, "--port", "0", "--name", NAME, "--max-batch", str(N),
                         "--log-steps", *SHARDS], stdout=subprocess.PIPE, stderr=errlog, text=True)
m = re.match(r"监听 http://127\.0\.0\.1:(\d+)", proc.stdout.readline())
if not m:
    print("FAIL 服务端没有启动：", open(errpath).read()[-2000:])
    sys.exit(1)
port = int(m.group(1))
print(f"服务端启动（加载权重）用时 {time.time() - t0:.0f} 秒")

results = [None] * N


def run(i):
    b = body(i)
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
    c.request("POST", "/v1/chat/completions", json.dumps(b, ensure_ascii=False).encode(),
              {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read().decode()
    if r.status != 200:
        results[i] = (None, data)
    elif b.get("stream"):
        ev = [json.loads(l[6:]) for l in data.split("\n\n") if l.startswith("data: {")]
        results[i] = (ev[0]["id"], "".join(e["choices"][0]["delta"].get("content", "") for e in ev if e["choices"]))
    else:
        j = json.loads(data)
        results[i] = (j["id"], j["choices"][0]["message"]["content"])


t0 = time.time()
th = [threading.Thread(target=run, args=(i,)) for i in range(N)]
for t in th:
    t.start()
for t in th:
    t.join()
t_batch = time.time() - t0
proc.send_signal(signal.SIGTERM)
proc.wait(timeout=120)
errlog.close()
log = open(errpath, encoding="utf-8").read()
steps = [l for l in log.splitlines() if l.startswith("步 ")]
widest = max((len(l.split("批=")[1].split()) for l in steps), default=0)
print(f"服务端：{N} 个请求同时处理，用时 {t_batch:.1f} 秒，{len(steps)} 步，一批最多 {widest} 个请求")

# 二、三、逐个单独生成并比较
t_single = 0.0
for i in range(N):
    b = body(i)
    ids, want, dt = single(b)
    t_single += dt
    rid, text = results[i]
    if rid is None:
        print(f"FAIL 问题 {i}：服务端报错 {text[:200]}")
        fails += 1
        continue
    n = int(rid.split("-")[1])
    mm = re.search(rf"完成 id={n} 原因=\S+ .*token=([\d,]*)", log)
    got = [int(x) for x in mm.group(1).split(",") if x] if mm else None
    shown = text.replace("\n", "\\n")
    if got == want and text == TK.decode(want, skip_special_tokens=True):
        print(f"ok   问题 {i}（{'流式' if b.get('stream') else '非流式'}，{'贪心' if b.get('temperature') == 0 else '采样'}）："
              f"{len(want)} 个 token 与单独生成逐个相同（单独 {dt:.0f} 秒）：{QUESTIONS[i % len(QUESTIONS)]} → {shown}")
    else:
        print(f"FAIL 问题 {i}：服务端 {got} 对单独生成 {want}")
        fails += 1
print(f"逐个单独生成共用时 {t_single:.0f} 秒（每次都要重新加载权重），服务端同时处理 {t_batch:.0f} 秒")
sys.exit(1 if fails else 0)
