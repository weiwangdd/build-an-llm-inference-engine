#!/usr/bin/env python3
"""第 8 章：服务端的端到端测试（只用标准库，另用 tokenizers 库独立地把 token 解码成文字）。

    tests/08-serve/serve_test.py <build 目录> <fixture 目录>

启动 build/serve，通过 HTTP 发请求，检查：
  一、结果对：每个请求的 token 与把同样的提示词交给 build/generate 单独生成的逐个相同（提示词由 build/tok chat
      套模板得到，两者都是前几章对拍过的）；应答的文字等于 tokenizers 库对这些 token 的解码；
      finish_reason、usage 与 token 一致。服务端在日志里写出每个请求生成的 token，按应答的 id 对上。
  二、批处理确实发生：几个请求同时在路上时，日志里有一步同时算了好几个请求。
  三、流式：SSE 的格式（data: 行加空行、第一块带 role、最后一块带 finish_reason、最后是 [DONE]、
      include_usage 的用量块），各块拼起来与非流式的文字相同。
  四、取消：流式收到几块后断开，服务端取消这个请求（日志），已生成的是单独生成的前缀，之后照常服务。
  五、拒绝：写坏的 HTTP 请求和不支持的字段，各自得到约定的状态码和 JSON 错误，服务端不崩溃。
  六、Expect: 100-continue。
  七、SIGTERM 之后正常退出（退出码 0；ASan 检查没有内存泄漏）。
每一项打印一行 ok 或 FAIL。
"""
import http.client, json, os, re, signal, socket, subprocess, sys, tempfile, threading, time

from tokenizers import Tokenizer

B, FX = sys.argv[1], sys.argv[2]
SERVE = os.environ.get("SERVE", f"{B}/serve")
NAME, CTX, BATCH = "tiny", 600, 4
FILES = [f"{FX}/{f}" for f in ("config.json", "generation_config.json", "tokenizer.json", "chat_template.jinja")]
TK = Tokenizer.from_file(f"{FX}/tokenizer.json")
GEN = json.load(open(f"{FX}/generation_config.json"))
EOS = set(GEN["eos_token_id"])
fails = 0


def ok(msg):
    print("ok  ", msg, flush=True)


def bad(msg):
    global fails
    fails += 1
    print("FAIL", msg, flush=True)


# ---------- 标准答案：build/tok chat 套模板，build/generate 单独生成 ----------

def prompt_ids(messages, kwargs):
    req = dict(messages=messages, add_generation_prompt=True, **kwargs)
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write(json.dumps(req, ensure_ascii=False) + "\n")
    out = subprocess.run([f"{B}/tok", "chat", FILES[2], FILES[3], f.name], capture_output=True, text=True, check=True)
    os.unlink(f.name)
    return [int(x) for x in out.stdout.split()[1:]]


def expected(body):
    ids = prompt_ids(body["messages"], body.get("chat_template_kwargs", {}))
    max_new = body.get("max_tokens", CTX - len(ids))
    opts = []
    t = body.get("temperature")
    if t == 0:
        opts.append("--greedy")
    elif t is not None:
        opts += ["--sample", "--temperature", repr(float(t))]
    if "top_p" in body:
        opts += ["--top-p", repr(float(body["top_p"]))]
    if "top_k" in body:
        opts += ["--top-k", str(body["top_k"])]
    opts += ["--seed", str(body.get("seed", 0))]
    out = subprocess.run([f"{B}/generate", FILES[0], FILES[1], ",".join(map(str, ids)), str(max_new), *opts,
                          f"{FX}/model.safetensors"], capture_output=True, text=True, check=True)
    toks = [int(l.split()[1]) for l in out.stdout.splitlines() if l.startswith("token ")]
    return ids, toks


# ---------- 启动服务端 ----------

errlog = open(f"{B}/serve_err.txt", "w")
env = dict(os.environ)
proc = subprocess.Popen([SERVE, *FILES, "--port", "0", "--name", NAME, "--max-batch", str(BATCH), "--max-ctx",
                         str(CTX), "--chunk", "50", "--max-conns", "16", "--log-steps", "--slow", "15",
                         f"{FX}/model.safetensors"], stdout=subprocess.PIPE, stderr=errlog, text=True, env=env)
line = proc.stdout.readline()
m = re.match(r"监听 http://127\.0\.0\.1:(\d+)", line)
if not m:
    print("FAIL 服务端没有启动：", line, open(f"{B}/serve_err.txt").read())
    sys.exit(1)
PORT = int(m.group(1))


def log():
    return open(f"{B}/serve_err.txt", encoding="utf-8").read()


def done_tokens(rid):
    """日志里请求 rid 的结束行：(原因, token 列表)"""
    for l in log().splitlines():
        m = re.match(rf"(完成|取消) id={rid} 原因=(\S+) .*token=([\d,]*)", l)
        if m:
            return m.group(1), m.group(2), [int(x) for x in m.group(3).split(",") if x]
    return None, None, None


def post(body, path="/v1/chat/completions"):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=120)
    c.request("POST", path, json.dumps(body, ensure_ascii=False).encode(), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, r.getheader("Content-Type"), data


def parse_sse(data):
    """把 SSE 正文拆成事件，检查格式：每个事件是一行 data: ...，后面一个空行"""
    text = data.decode("utf-8")
    if not text.endswith("\n\n"):
        raise ValueError("SSE 正文没有以空行结束")
    events = []
    for block in text[:-2].split("\n\n"):
        if not block.startswith("data: ") or "\n" in block:
            raise ValueError(f"不是一行 data: 事件：{block[:80]!r}")
        events.append(block[6:])
    if events[-1] != "[DONE]":
        raise ValueError("最后一个事件不是 [DONE]")
    return [json.loads(e) for e in events[:-1]]


def chat(body):
    """发一个对话请求，返回 (id, 文字, finish_reason, usage)；流式时还检查 SSE 的格式"""
    st, ctype, data = post(body)
    if st != 200:
        raise ValueError(f"状态码 {st}：{data[:200]!r}")
    if not body.get("stream"):
        if not ctype.startswith("application/json"):
            raise ValueError(f"Content-Type 是 {ctype}")
        j = json.loads(data)
        ch = j["choices"][0]
        if j["object"] != "chat.completion" or j["model"] != NAME or ch["message"]["role"] != "assistant":
            raise ValueError(f"应答格式不对：{j}")
        return j["id"], ch["message"]["content"], ch["finish_reason"], j["usage"]
    if not ctype.startswith("text/event-stream"):
        raise ValueError(f"Content-Type 是 {ctype}")
    ev = parse_sse(data)
    rid = ev[0]["id"]
    if any(e["id"] != rid or e["object"] != "chat.completion.chunk" or e["model"] != NAME for e in ev):
        raise ValueError("各块的 id、object、model 不一致")
    if ev[0]["choices"][0]["delta"] != {"role": "assistant", "content": ""}:
        raise ValueError(f"第一块应只有 role：{ev[0]}")
    usage = None
    if body.get("stream_options", {}).get("include_usage"):
        last = ev.pop()
        if last["choices"] != [] or "usage" not in last:
            raise ValueError(f"最后应是用量块：{last}")
        usage = last["usage"]
    fin = ev[-1]["choices"][0]
    if fin["delta"] != {} or fin["finish_reason"] is None:
        raise ValueError(f"最后一块应带 finish_reason：{fin}")
    mid = ev[1:-1]
    if any(e["choices"][0]["finish_reason"] is not None or set(e["choices"][0]["delta"]) != {"content"} or
           not e["choices"][0]["delta"]["content"] for e in mid):
        raise ValueError("中间的块应只有非空的 content")
    return rid, "".join(e["choices"][0]["delta"]["content"] for e in mid), fin["finish_reason"], usage


def check(name, body, result=None):
    """把服务端的结果与标准答案比较"""
    try:
        rid, text, finish, usage = result if result else chat(body)
    except Exception as e:  # noqa: BLE001
        bad(f"{name}：{e}")
        return None
    n = int(rid.split("-")[1])
    kind, why, toks = done_tokens(n)
    ids, want = expected(body)
    errs = []
    if kind != "完成":
        errs.append("日志里没有完成记录")
    elif toks != want:
        errs.append(f"token 与单独生成不同：{toks[:8]}… 对 {want[:8]}…")
    if text != TK.decode(want, skip_special_tokens=True):
        errs.append(f"文字与 tokenizers 的解码不同：{text[:20]!r}")
    stop = "stop" if want and want[-1] in EOS else "length"
    if finish != stop:
        errs.append(f"finish_reason 是 {finish}，应为 {stop}")
    if usage is not None and usage != {"prompt_tokens": len(ids), "completion_tokens": len(want),
                                       "total_tokens": len(ids) + len(want)}:
        errs.append(f"usage 不对：{usage}")
    if errs:
        bad(f"{name}：" + "；".join(errs))
    else:
        ok(f"{name}：提示词 {len(ids)} 个、生成 {len(want)} 个 token（{finish}），与单独生成逐个相同，文字与 tokenizers 解码相同")
    return text


def user(s):
    return [{"role": "user", "content": s}]


# ---------- 一、单个请求 ----------
c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30)
c.request("GET", "/v1/models")
r = c.getresponse()
j = json.loads(r.read())
c.close()
if r.status == 200 and [d["id"] for d in j["data"]] == [NAME]:
    ok("GET /v1/models 列出模型 " + NAME)
else:
    bad(f"GET /v1/models：{r.status} {j}")

check("非流式_贪心", {"model": NAME, "messages": user("你好，介绍一下你自己。"), "max_tokens": 12, "temperature": 0,
                  "chat_template_kwargs": {"enable_thinking": False}})
check("非流式_采样_默认参数", {"model": NAME, "messages": user("hello"), "max_tokens": 20, "seed": 3})
check("非流式_不给max_tokens", {"model": NAME, "messages": user("写一首诗"), "seed": 11,
                            "chat_template_kwargs": {"enable_thinking": False}})
b_stream = {"model": NAME, "messages": [{"role": "system", "content": "你是助手。"}] + user("讲个故事"),
            "max_tokens": 30, "seed": 5, "temperature": 0.8, "top_p": 0.9, "top_k": 50, "stream": True,
            "stream_options": {"include_usage": True}}
t1 = check("流式_带用量", b_stream)
b_plain = {k: v for k, v in b_stream.items() if k not in ("stream", "stream_options")}
t2 = check("同一请求_非流式", b_plain)
if t1 is not None and t1 == t2:
    ok("已知答案：流式各块拼起来与非流式的文字相同")
elif t1 is not None:
    bad("流式与非流式的文字不同")

# ---------- 二、同时来的请求：批处理 ----------
bodies = [
    {"model": NAME, "messages": user("第一个问题"), "max_tokens": 25, "seed": 1, "stream": True},
    {"model": NAME, "messages": user("第二个问题，长一点" * 20), "max_tokens": 15, "seed": 2},
    {"model": NAME, "messages": user("third"), "max_tokens": 20, "temperature": 0, "stream": True},
    {"model": NAME, "messages": user("第四个"), "max_tokens": 18, "seed": 4, "temperature": 1.3, "top_k": 0},
    {"model": NAME, "messages": user("第五个"), "max_tokens": 10, "seed": 5, "stream": True,
     "chat_template_kwargs": {"enable_thinking": False}},
    {"model": NAME, "messages": user("第一个问题"), "max_tokens": 25, "seed": 1},    # 与第一个相同
    {"model": NAME, "messages": user("最后一个"), "max_tokens": 12, "seed": 7, "top_p": 0.5},
]
results = [None] * len(bodies)


def run(i):
    time.sleep(0.03 * i)                     # 陆续到达
    try:
        results[i] = chat(bodies[i])
    except Exception as e:  # noqa: BLE001
        results[i] = e


before = len(log().splitlines())
th = [threading.Thread(target=run, args=(i,)) for i in range(len(bodies))]
for t in th:
    t.start()
for t in th:
    t.join()
texts = []
for i, rres in enumerate(results):
    if isinstance(rres, Exception):
        bad(f"同时_{i}：{rres}")
        texts.append(None)
    else:
        texts.append(check(f"同时_{i}", bodies[i], rres))
steps = [l for l in log().splitlines()[before:] if l.startswith("步 ")]
widest = max((len(l.split("批=")[1].split()) for l in steps), default=0)
mixed = sum(1 for l in steps if len({int(x.split(":")[1]) > 1 for x in l.split("批=")[1].split()}) == 2)
if widest >= 2 and mixed >= 1 and widest <= BATCH:
    ok(f"批处理：{len(bodies)} 个请求用了 {len(steps)} 步，一批最多 {widest} 个请求（上限 {BATCH}），{mixed} 步里预填充与解码同批")
else:
    bad(f"没有测到批处理：一批最多 {widest} 个，{mixed} 步预填充与解码同批")
if texts[0] is not None and texts[0] == texts[5]:
    ok("已知答案：同时到达的两个相同请求结果相同")

# ---------- 三、取消 ----------
# 挑一个单独生成时不会很快停下的种子
cbody = None
for seed in range(40):
    b = {"model": NAME, "messages": user("请一直说下去"), "max_tokens": 200, "seed": seed, "stream": True}
    if len(expected(b)[1]) >= 40:
        cbody = b
        break
if not cbody:
    bad("取消：找不到能生成 40 个 token 的种子")
else:
    s = socket.create_connection(("127.0.0.1", PORT))
    raw = json.dumps(cbody).encode()
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
              b"Content-Length: %d\r\n\r\n" % len(raw) + raw)
    got = b""
    while got.count(b"data: ") < 4:          # role 块加 3 块内容
        k = s.recv(65536)
        if not k:
            break
        got += k
    s.close()
    rid = int(re.search(rb'"id":"chatcmpl-(\d+)"', got).group(1))
    for _ in range(100):
        kind, why, toks = done_tokens(rid)
        if kind:
            break
        time.sleep(0.05)
    want = expected(cbody)[1]
    if kind == "取消" and 3 <= len(toks) < len(want) and toks == want[:len(toks)]:
        ok(f"取消：收到 3 块后断开，服务端在生成 {len(toks)} 个（共 {len(want)} 个）时取消，已生成的与单独生成的前缀相同")
    else:
        bad(f"取消：日志 {kind} {why}，已生成 {toks and len(toks)} 个")
    check("取消之后照常服务", {"model": NAME, "messages": user("还在吗"), "max_tokens": 8, "temperature": 0})

# ---------- 五、拒绝 ----------
good = json.dumps({"model": NAME, "messages": user("x"), "max_tokens": 2}).encode()


def req(body, extra=b"", method=b"POST", path=b"/v1/chat/completions", ver=b"HTTP/1.1", host=True, length=True):
    h = method + b" " + path + b" " + ver + b"\r\n"
    if host:
        h += b"Host: x\r\n"
    if length:
        h += b"Content-Length: %d\r\n" % len(body)
    return h + extra + b"\r\n" + body


def jbody(**kw):
    d = {"model": NAME, "messages": user("x"), "max_tokens": 2}
    d.update(kw)
    return json.dumps(d, ensure_ascii=False).encode()


big = json.dumps({"model": NAME, "messages": user("长" * 600)}).encode()
cases = [
    ("请求行不对", b"HELLO\r\nHost: x\r\n\r\n", 400),
    ("HTTP/1.0", req(good, ver=b"HTTP/1.0"), 505),
    ("没有Host", req(good, host=False), 400),
    ("POST没有Content-Length", req(good, length=False), 411),
    ("Content-Length不是数", req(good, length=False, extra=b"Content-Length: 12a\r\n"), 400),
    ("Content-Length重复", req(good, extra=b"Content-Length: %d\r\n" % len(good)), 400),
    ("分块传输", req(good, length=False, extra=b"Transfer-Encoding: chunked\r\n"), 501),
    ("正文太大", req(b"", length=False, extra=b"Content-Length: 2000000\r\n"), 413),
    ("头部太大", b"GET /v1/models HTTP/1.1\r\nHost: x\r\nX: " + b"a" * 20000 + b"\r\n\r\n", 431),
    ("头部续行", req(good, extra=b"X: a\r\n b\r\n"), 400),
    ("只有LF", b"GET /v1/models HTTP/1.1\nHost: x\n\n", 400),
    ("Expect不认识", req(good, extra=b"Expect: something\r\n"), 417),
    ("路径不存在", req(good, path=b"/v1/completions"), 404),
    ("方法不对", req(b"", method=b"GET", length=False), 405),
    ("models用POST", req(good, path=b"/v1/models"), 405),
    ("不是JSON", req(b"{model"), 400),
    ("JSON不是对象", req(b"[1,2]"), 400),
    ("重复的键", req(b'{"model":"tiny","model":"tiny","messages":[]}'), 400),
    ("不支持的字段", req(jbody(tools=[])), 400),
    ("不支持的字段stop", req(jbody(stop=["x"])), 400),
    ("模型名不对", req(jbody(model="gpt-4")), 404),
    ("缺少model", req(json.dumps({"messages": user("x")}).encode()), 400),
    ("缺少messages", req(json.dumps({"model": NAME}).encode()), 400),
    ("messages为空", req(jbody(messages=[])), 400),
    ("消息里不支持的字段", req(jbody(messages=[{"role": "user", "content": "x", "name": "a"}])), 400),
    ("角色不认识", req(jbody(messages=[{"role": "robot", "content": "x"}])), 400),
    ("没有用户消息", req(jbody(messages=[{"role": "system", "content": "x"}])), 400),
    ("n为2", req(jbody(n=2)), 400),
    ("温度为负", req(jbody(temperature=-0.5)), 400),
    ("温度大于2", req(jbody(temperature=2.5)), 400),
    ("温度不是数", req(jbody(temperature="hot")), 400),
    ("top_p为0", req(jbody(top_p=0)), 400),
    ("top_k为小数", req(jbody(top_k=1.5)), 400),
    ("seed为负", req(jbody(seed=-1)), 400),
    ("max_tokens为0", req(jbody(max_tokens=0)), 400),
    ("两种max_tokens", req(jbody(max_completion_tokens=3)), 400),
    ("stream不是布尔", req(jbody(stream=1)), 400),
    ("不流式却给stream_options", req(jbody(stream_options={"include_usage": True})), 400),
    ("stream_options不认识", req(jbody(stream=True, stream_options={"x": 1})), 400),
    ("模板参数不认识", req(jbody(chat_template_kwargs={"add_vision_id": True})), 400),
    ("思考强度不认识", req(jbody(chat_template_kwargs={"reasoning_effort": "max"})), 400),
    ("提示词太长", req(big), 400),
    ("提示词加max_tokens太长", req(jbody(max_tokens=590)), 400),
    ("非法UTF-8", req(b'{"model":"tiny","messages":[{"role":"user","content":"\xff"}]}'), 400),
]
nrej = 0
for name, raw, want in cases:
    try:
        s = socket.create_connection(("127.0.0.1", PORT), timeout=30)
        s.sendall(raw)
        got = b""
        while True:
            k = s.recv(65536)
            if not k:
                break
            got += k
        s.close()
        head, _, body = got.partition(b"\r\n\r\n")
        status = int(head.split(b" ")[1])
        err = json.loads(body)["error"]
        if status != want or err["code"] != want or not err["message"]:
            bad(f"拒绝 {name}：状态码 {status}，应为 {want}：{err}")
        else:
            nrej += 1
    except Exception as e:  # noqa: BLE001
        bad(f"拒绝 {name}：{e}")
ok(f"拒绝：{nrej}/{len(cases)} 个写坏的请求得到约定的状态码和 JSON 错误")
check("拒绝之后照常服务", {"model": NAME, "messages": user("再来"), "max_tokens": 6, "temperature": 0})

# ---------- 六、Expect: 100-continue ----------
s = socket.create_connection(("127.0.0.1", PORT), timeout=30)
body = jbody(temperature=0, max_tokens=4)
s.sendall(req(b"", length=False, extra=b"Content-Length: %d\r\nExpect: 100-continue\r\n" % len(body)))
first = s.recv(65536)
s.sendall(body)
rest = b""
while True:
    k = s.recv(65536)
    if not k:
        break
    rest += k
s.close()
if first == b"HTTP/1.1 100 Continue\r\n\r\n" and rest.startswith(b"HTTP/1.1 200 OK\r\n"):
    ok("Expect: 100-continue：先回 100 Continue，收到正文后回 200")
else:
    bad(f"Expect: 100-continue：{first[:40]!r} {rest[:40]!r}")

# ---------- 官方客户端（装了 openai 库才测） ----------
try:
    import openai
except ImportError:
    openai = None
    print("SKIP 官方 openai 客户端：没装 openai 库（pip install openai）")
if openai:
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1"
    cl = openai.OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="none")
    ob = {"model": NAME, "messages": user("用官方客户端问一句"), "max_tokens": 10, "seed": 9}
    try:
        r = cl.chat.completions.create(**ob)
        u = r.usage
        check(f"官方 openai {openai.__version__} 客户端_非流式", ob,
              (r.id, r.choices[0].message.content, r.choices[0].finish_reason,
               {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens}))
        ev = list(cl.chat.completions.create(**ob, stream=True))
        text = "".join(e.choices[0].delta.content or "" for e in ev if e.choices)
        check(f"官方 openai {openai.__version__} 客户端_流式", ob, (ev[0].id, text, ev[-1].choices[0].finish_reason, None))
        try:
            cl.chat.completions.create(**ob, stop=["x"])
            bad("官方客户端：不支持的字段没有被拒绝")
        except openai.BadRequestError:
            ok("官方客户端：不支持的字段 stop 得到 BadRequestError")
    except Exception as e:  # noqa: BLE001
        bad(f"官方 openai 客户端：{e}")

# ---------- 七、停止 ----------
proc.send_signal(signal.SIGTERM)
rc = proc.wait(timeout=60)
errlog.close()
if rc == 0 and log().rstrip().endswith("停止"):
    ok("SIGTERM 之后正常退出（ASan 没有报告内存泄漏）")
else:
    bad(f"退出码 {rc}：{log()[-500:]}")
sys.exit(1 if fails else 0)
