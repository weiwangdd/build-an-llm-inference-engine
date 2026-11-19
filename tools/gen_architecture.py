# 生成 book/images/qwen38-architecture.svg：Qwen3.8-27B 的架构图，按实现它的章节上色。
# 用法：python3 tools/gen_architecture.py [输出路径]
import html, sys
W, H = 1520, 2100
C = {  # 章: (填充, 描边, 名称)
 1: ("#DCE8F7", "#3D6FAE", "第 1 章 读取权重（safetensors）"),
 2: ("#DDF1D6", "#3E8E3E", "第 2 章 分词器与对话模板"),
 3: ("#FFF0C4", "#B5850A", "第 3 章 算子"),
 4: ("#FBDCCB", "#C55A22", "第 4 章 单层前向（层的组装、注意力、门控 delta 规则）"),
 5: ("#E7DDF6", "#6B47A6", "第 5 章 整模前向与贪心解码"),
 6: ("#D3EFEE", "#1C8584", "第 6 章 缓存、采样与停止"),
 8: ("#F6D6E4", "#B03A6E", "第 8 章 服务层（HTTP、OpenAI 兼容接口、连续批处理）"),
}
out = []
def esc(s): return html.escape(s, quote=False)
def text(x, y, s, size=15, anchor="middle", weight="normal", color="#222", italic=False):
    st = ' font-style="italic"' if italic else ''
    out.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{color}"{st}>{esc(s)}</text>')
def box(x, y, w, h, ch, lines, size=15, sub=None, rx=8, dash=False, fill=None, bold_first=True, lw=1.6):
    f, s, _ = C[ch]
    d = ' stroke-dasharray="6 4"' if dash else ''
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill or f}" stroke="{s}" stroke-width="{lw}"{d}/>')
    if isinstance(lines, str): lines = [lines]
    n = len(lines); lh = size * 1.35
    y0 = y + h / 2 - (n - 1) * lh / 2 + size * 0.35
    for i, l in enumerate(lines):
        sz = size if i == 0 else size - 2
        text(x + w / 2, y0 + i * lh, l, sz, weight="bold" if (i == 0 and bold_first) else "normal", color="#1a1a1a" if i == 0 else "#444")
def panel(x, y, w, h, ch, title, fillop=0.35, tx=16):
    f, s, _ = C[ch]
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{f}" fill-opacity="{fillop}" stroke="{s}" stroke-width="2.2"/>')
    text(x + tx, y + 26, title, 17, anchor="start", weight="bold", color=s)
def arrow(pts, color="#555", dash=False, w=1.6, head=True):
    d = "M" + " L".join(f"{a},{b}" for a, b in pts)
    da = ' stroke-dasharray="6 4"' if dash else ''
    mk = ' marker-end="url(#ah)"' if head else ''
    out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{w}"{da}{mk}/>')
def circ(x, y, sym, ch=4, r=13):
    f, s, _ = C[ch]
    out.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{f}" stroke="{s}" stroke-width="1.6"/>')
    text(x, y + 6, sym, 18, weight="bold", color=s)
def label(x, y, s, size=13, anchor="middle", color="#555"):
    text(x, y, s, size, anchor=anchor, color=color)

# ---------- 标题 ----------
text(W/2, 40, "Qwen3.8-27B 推理的完整路径：从提问到回答", 26, weight="bold")
text(W/2, 68, "颜色表示由哪一章实现；方括号里是每个 token 的张量形状；n 是 token 个数", 15, color="#555")

# ---------- 第 2 章：输入 ----------
y1 = 96
box(20, y1, 310, 70, 8, ["HTTP 请求", "POST /v1/chat/completions"])
arrow([(330, y1+35), (368, y1+35)])
box(370, y1, 200, 70, 2, ["提问（文本）", "用户的一段话"], bold_first=True, fill="#FFFFFF")
box(610, y1, 230, 70, 2, ["对话模板", "套上 <|im_start|> 等标记"])
box(880, y1, 330, 70, 2, ["分词器", "切出特殊 token → NFC → 预分词 → BPE"])
box(1250, y1, 210, 70, 2, ["token id 序列", "n 个整数，词表 248320"])
arrow([(570, y1+35), (608, y1+35)]); arrow([(840, y1+35), (878, y1+35)]); arrow([(1210, y1+35), (1248, y1+35)])

# ---------- 第 5 章：整模容器 ----------
px, py, pw, ph = 360, 196, 1130, 1270
panel(px, py, pw, ph, 5, "整个模型（一次前向）", 0.30)
arrow([(1355, y1+70), (1355, 150+80), (925, 230), (925, 248)])
# 嵌入
box(700, 250, 450, 58, 3, ["嵌入：按 id 查表", "embed_tokens [248320, 5120] → h [n, 5120]"])
arrow([(925, 308), (925, 340)])

# ---------- 第 4 章：一层 ----------
lx, ly, lw_, lh_ = 380, 342, 1090, 872
panel(lx, ly, lw_, lh_, 4, "一个解码层（共 64 层：16 组 × [3 个线性注意力层 + 1 个全注意力层]）", 0.30)
# 骨架列
sx = 410; sw = 210; cx = sx + sw/2
yy = 400
box(sx, yy, sw, 40, 4, "输入 h [n, 5120]", fill="#FFFFFF")
arrow([(cx, yy+40), (cx, yy+70)])
box(sx, yy+72, sw, 52, 3, ["RMSNorm", "input_layernorm，1+w 形式"])
arrow([(cx, yy+124), (cx, yy+156)])
box(sx, yy+158, sw, 92, 4, ["token 混合器", "按层的类型二选一：", "线性注意力 或 全注意力", ], size=15)
# 到两个混合器
arrow([(sx+sw, yy+204), (640, yy+204), (640, 381), (1259, 381), (1259, 390)], color=C[4][1])
arrow([(852, 381), (852, 390)], color=C[4][1])
label(1056, 376, "或", 14, color=C[4][1])
arrow([(cx, yy+250), (cx, yy+290)])
circ(cx, yy+304, "+")
# 残差旁路
arrow([(sx, yy+20), (sx-14, yy+20), (sx-14, yy+304), (cx-14, yy+304)], color="#888")
label(cx+20, yy+300, "残差相加", 13, anchor="start", color=C[4][1])
arrow([(cx, yy+317), (cx, yy+350)])
box(sx, yy+352, sw, 52, 3, ["RMSNorm", "post_attention_layernorm"])
arrow([(cx, yy+404), (cx, yy+430)])
# 前馈
fy = yy+432
box(sx, fy, 100, 50, 3, ["gate_proj", "→ 17408"], size=14)
box(sx+110, fy, 100, 50, 3, ["up_proj", "→ 17408"], size=14)
box(sx, fy+66, 100, 36, 3, "silu", size=14)
arrow([(sx+50, fy+50), (sx+50, fy+64)])
circ(cx, fy+132, "×")
arrow([(sx+50, fy+102), (sx+50, fy+132), (cx-14, fy+132)])
arrow([(sx+160, fy+50), (sx+160, fy+132), (cx+14, fy+132)])
arrow([(cx, fy+145), (cx, fy+170)])
box(sx, fy+172, sw, 50, 3, ["down_proj", "17408 → 5120"])
label(cx-8, fy-8, "前馈层 SwiGLU", 13, anchor="end", color=C[4][1])
arrow([(cx, fy+222), (cx, fy+250)])
circ(cx, fy+264, "+")
arrow([(sx, yy+330), (sx-14, yy+330), (sx-14, fy+264), (cx-14, fy+264)], color="#888")
label(cx+20, fy+260, "残差相加", 13, anchor="start", color=C[4][1])
arrow([(cx, fy+277), (cx, fy+302)])
box(sx, fy+304, sw, 54, 4, ["输出 h [n, 5120]", "即下一层的输入（共 64 层）"], fill="#FFFFFF")
# 64 层循环箭头


# 线性注意力子面板
ax, ay, aw, ah = 655, 392, 395, 535
out.append(f'<rect x="{ax}" y="{ay}" width="{aw}" height="{ah}" rx="10" fill="#FFFFFF" fill-opacity="0.7" stroke="{C[4][1]}" stroke-width="1.6"/>')
text(ax+aw/2, ay+24, "线性注意力层（Gated DeltaNet，48 层）", 16, weight="bold", color=C[4][1])
bx = ax+15; bw = aw-30
y = ay+40
box(bx, y, 175, 46, 3, ["in_proj_qkv", "→ 10240"], size=14); box(bx+185, y, 85, 46, 3, ["in_proj_z", "→ 6144"], size=13); box(bx+278, y, 87, 46, 3, ["in_proj_a/b", "→ 48 各"], size=13)
y += 60; arrow([(bx+87, y-14), (bx+87, y)])
box(bx, y, 175, 46, 3, ["因果卷积 k=4 + silu", "逐通道"], size=14)
y += 60; arrow([(bx+87, y-14), (bx+87, y)])
box(bx, y, 175, 46, 3, ["切开 q、k、v，L2 归一化", "q,k 16 头×128；v 48 头×128"], size=13)
box(bx+185, y-60, 180, 106, 3, ["β = sigmoid(b)", "g = −e^A_log ·", "softplus(a + dt_bias)", "（衰减的对数）"], size=13)
arrow([(bx+320, ay+86), (bx+320, ay+98)])
y += 60; arrow([(bx+87, y-14), (bx+87, y)]); arrow([(bx+275, y-14), (bx+275, y+20), (bx+bw, y+20)], head=False)
box(bx, y, bw, 58, 4, ["门控 delta 规则", "S ← e^g·S；S ← S + β·k(v − Sᵀk)ᵀ；y = Sᵀq/√128"], size=14)
y += 72; arrow([(bx+bw/2, y-14), (bx+bw/2, y)])
box(bx, y, bw, 46, 3, ["门控 RMSNorm（w 形式）× silu(z)", "每个值头 128 维"], size=14)
y += 60; arrow([(bx+bw/2, y-14), (bx+bw/2, y)])
box(bx, y, bw, 40, 3, ["out_proj  6144 → 5120"], size=14)
y += 54
box(bx, y, bw, 80, 6, ["缓存（每层）", "卷积窗口：最近 3 个 token 的 qkv", "状态矩阵 S：48 × 128 × 128，double"], size=14, dash=True)
arrow([(bx+40, y), (bx+40, ay+40+60+60+60+16)], color=C[6][1], dash=True)

# 全注意力子面板
fx, fy2, fw, fh = 1062, 392, 395, 535
out.append(f'<rect x="{fx}" y="{fy2}" width="{fw}" height="{fh}" rx="10" fill="#FFFFFF" fill-opacity="0.7" stroke="{C[4][1]}" stroke-width="1.6"/>')
text(fx+fw/2, fy2+24, "全注意力层（每 4 层 1 个，16 层）", 16, weight="bold", color=C[4][1])
bx = fx+15; bw = fw-30
y = fy2+40
box(bx, y, 175, 46, 3, ["q_proj", "→ 24 头 × (256 q + 256 门控)"], size=12); box(bx+185, y, 180, 46, 3, ["k_proj、v_proj", "→ 4 组 × 256"], size=13)
y += 60; arrow([(bx+87, y-14), (bx+87, y)]); arrow([(bx+275, y-14), (bx+275, y)])
box(bx, y, 175, 46, 3, ["q_norm（RMSNorm）", "按头，256 维"], size=13); box(bx+185, y, 180, 46, 3, ["k_norm（RMSNorm）", "按头，256 维"], size=13)
y += 60; arrow([(bx+87, y-14), (bx+87, y)]); arrow([(bx+275, y-14), (bx+275, y)])
box(bx, y, bw, 46, 3, ["RoPE 旋转位置编码", "每头只转前 64 维，θ = 10⁷"], size=14)
y += 60; arrow([(bx+bw/2, y-14), (bx+bw/2, y)])
box(bx, y, bw, 58, 4, ["因果注意力（分组查询）", "softmax(q·kᵀ/√256)·v，查询头 h 用第 h/6 组键值"], size=14)
y += 72; arrow([(bx+bw/2, y-14), (bx+bw/2, y)])
box(bx, y, bw, 46, 3, ["× sigmoid(门控)", "输出门控，逐元素"], size=14)
y += 60; arrow([(bx+bw/2, y-14), (bx+bw/2, y)])
box(bx, y, bw, 40, 3, ["o_proj  6144 → 5120"], size=14)
y += 54
box(bx, y, bw, 80, 6, ["KV 缓存（每层）", "过去每个 token 的 k（RoPE 之后）和 v", "每个 token 2 × 4 × 256 个数"], size=14, dash=True)
arrow([(bx+bw-40, y), (bx+bw-40, fy2+40+180+16)], color=C[6][1], dash=True)
# 混合器到全注意力的连线

# 两个面板的输出回到骨架的 + 号
arrow([(ax+aw/2, ay+ah), (ax+aw/2, 942)], color=C[4][1], head=False)
arrow([(fx+fw/2, fy2+fh), (fx+fw/2, 942)], color=C[4][1], head=False)
arrow([(fx+fw/2, 942), (628, 942), (628, yy+304), (cx+14, yy+304)], color=C[4][1])
label(1056, 960, "token 混合器的输出 [n, 5120]，加回残差", 13, color=C[4][1])
tx0, ty0 = 655, 985
out.append(f'<rect x="{tx0}" y="{ty0}" width="800" height="210" rx="10" fill="#FFFFFF" stroke="#999" stroke-width="1.2"/>')
text(tx0+14, ty0+24, "术语", 15, anchor="start", weight="bold", color="#333")
terms = [
 ("残差", "把一步的输出加回它的输入，信息可以绕过这一步直接往后传。"),
 ("RMSNorm", "向量除以自己的均方根，再逐元素乘权重；“1+w 形式”乘的是 1+w。"),
 ("RoPE", "按 token 的位置把 q、k 的成对分量转一个角度，注意力由此知道先后。"),
 ("分组查询注意力", "几个查询头共用一组键和值（这里 6 个共用 1 组），缓存小 6 倍。"),
 ("门控 delta 规则", "线性注意力的核心：每个头一个固定大小的状态矩阵 S，每个 token 先让旧记忆衰减，"),
 ("", "再把“预测误差”写进去；不保留每个 token 的 k、v，所以缓存大小不随长度增长。"),
 ("SwiGLU", "前馈层的写法：silu(gate(x)) 与 up(x) 逐元素相乘，再由 down 投影回 5120 维。"),
 ("logits", "词表里每个 token 的分数，softmax 之后就是下一个 token 的概率。"),
]
for i, (k, v) in enumerate(terms):
    yy_ = ty0 + 48 + i * 21
    if k: text(tx0+14, yy_, k, 13, anchor="start", weight="bold", color="#333")
    text(tx0+130, yy_, v, 13, anchor="start", color="#444")
# ---------- 第 5 章：层之后 ----------
arrow([(925, ly+lh_), (925, ly+lh_+26)])
yb = ly+lh_+28
box(700, yb, 450, 52, 3, ["最终 RMSNorm", "norm.weight，1+w 形式"])
arrow([(925, yb+52), (925, yb+80)])
box(700, yb+82, 450, 52, 5, ["输出层 lm_head", "lm_head.weight [248320, 5120]（不与嵌入表共用）"])
arrow([(925, yb+134), (925, yb+162)])
box(700, yb+164, 450, 46, 5, ["logits [n, 248320]", "词表里每个 token 一个分数，只用最后一行"], fill="#FFFFFF")

# ---------- 第 1 章：权重 ----------
box(20, 196, 315, 150, 1, ["权重文件", "18 个 safetensors 分片，55.6 GB", "mmap 只读映射，合并成一张表", "按名字取 BF16 张量", "1199 个张量"], size=16)
for yy_, lab in [(279, "嵌入表"), (470, "每层的权重"), (1342, "norm、lm_head")]:
    pass
arrow([(335, 260), (698, 279)], color=C[1][1], dash=True)
arrow([(335, 300), (378, 470)], color=C[1][1], dash=True)
arrow([(335, 320), (345, 330), (345, yb+108), (698, yb+108)], color=C[1][1], dash=True)
label(345, 1420, "", 12)
text(178, 370, "虚线：权重从这里来", 13, color=C[1][1])
text(178, 388, "（所有矩阵都按 BF16 原样读取）", 13, color=C[1][1])

# 左栏：config
box(20, 420, 315, 150, 4, ["配置 config.json", "hidden 5120，64 层", "全注意力 24 个查询头、4 组键值", "线性注意力 16 个键头、48 个值头", "前馈中间维 17408"], size=15)
text(178, 594, "第 3 章讲解各字段，第 4 章读入并检查", 13, color=C[4][1])
text(178, 612, "（只接受 Qwen3.8 这一种配置）", 13, color=C[4][1])

# ---------- 第 6 章：生成循环 ----------
gy = 1510
panel(20, gy, 1470, 250, 6, "生成循环：每次取一个 token，接到后面再算", 0.35, tx=230)
by = gy+60
box(45, by, 175, 70, 6, ["温度", "logits ÷ T（默认 1.0）"])
box(250, by, 190, 70, 6, ["top-k", "只留分数前 k 名（默认 20）"])
box(470, by, 190, 70, 6, ["top-p", "累计概率到 p 为止（默认 0.95）"])
box(690, by, 200, 70, 6, ["softmax 重新归一", "得到保留 token 的概率"])
box(920, by, 230, 70, 6, ["逆变换采样", "随机数 splitmix64，可复现"])
box(1180, by, 280, 70, 6, ["是结束符吗？", "<|im_end|> 或 <|endoftext|>，或已满 max_new"])
for a, b in [(220, 248), (440, 468), (660, 688), (890, 918), (1150, 1178)]:
    arrow([(a, by+35), (b, by+35)])
arrow([(925, yb+210), (925, gy-22), (132, gy-22), (132, by-2)])
label(140, gy-30, "最后一行 logits", 13, anchor="start")
box(45, by+100, 395, 60, 5, ["贪心解码（argmax）", "不采样，直接取分数最大的 token（同分取 id 小的）"], size=14)
label(452, by+126, "← 第 5 章先做的最朴素版本：没有缓存，", 13, anchor="start", color=C[5][1])
label(452, by+146, "    每一步都从头算全部 token；第 6 章用它当标准答案", 13, anchor="start", color=C[5][1])
# 否：回到模型
arrow([(1380, by+70), (1380, by+106)], color=C[6][1])
label(1388, by+92, "否", 14, anchor="start", color=C[6][1])
box(1250, by+108, 210, 50, 6, ["新 token 接在后面", "只算这 1 个，其余靠缓存"], size=14)
arrow([(1460, by+133), (1505, by+133), (1505, 279), (1152, 279)], color=C[6][1])
label(1500, 240, "下一步的输入", 13, anchor="end", color=C[6][1])
label(1495, 900, "", 12)
# 是：输出


# ---------- 输出 ----------
oy = gy+282
box(370, oy, 300, 70, 2, ["解码", "token id → 字节 → UTF-8 文本"])
box(710, oy, 260, 70, 2, ["回答（文本）", "边生成边输出"], fill="#FFFFFF")
arrow([(1215, by+70), (1215, gy+268), (520, gy+268), (520, oy-2)], color=C[6][1])
label(1223, by+95, "是：停止", 14, anchor="start", color=C[6][1])
arrow([(670, oy+35), (708, oy+35)])
box(1010, oy, 470, 70, 8, ["SSE 流式发回客户端", "每生成一个 token 发一个事件；客户端断开就取消"])
arrow([(970, oy+35), (1008, oy+35)])

# ---------- 图例 ----------
ly2 = oy+105
text(20, ly2, "图例", 17, anchor="start", weight="bold")
x = 70
for ch in range(1, 7):
    f, s, name = C[ch]
    pass
cols = [(70, ly2), (560, ly2), (1050, ly2), (70, ly2+35), (560, ly2+35), (1050, ly2+35), (70, ly2+70)]
for ch, (x, y) in zip([1, 2, 3, 4, 5, 6, 8], cols):
    f, s, name = C[ch]
    out.append(f'<rect x="{x}" y="{y-15}" width="26" height="18" rx="4" fill="{f}" stroke="{s}" stroke-width="1.6"/>')
    text(x+36, y, name, 14, anchor="start")
text(20, ly2+110, "大框是“组装”，小框是“零件”：比如解码层（橙色大框）由第 3 章的算子（黄色）拼成，第 4 章新增的是两种注意力的核心和残差加法。虚线框是第 6 章加的缓存。", 13, anchor="start", color="#444")
text(20, ly2+135, "第 7 章把图里的全部计算（嵌入、64 层、输出层和缓存）搬到一张 GPU 上：步骤与上图一一对应，结果与 CPU 版逐位相同。", 13, anchor="start", color="#444")
text(20, ly2+160, "第 8 章让很多请求同时走这条路径：每一步把各请求的新 token 拼成一批，一次前向一起算（连续批处理），每个请求的结果与单独算逐位相同。", 13, anchor="start", color="#444")

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="PingFang SC, Hiragino Sans GB, Noto Sans CJK SC, Source Han Sans SC, Microsoft YaHei, WenQuanYi Zen Hei, sans-serif">
<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs>
<rect width="{W}" height="{H}" fill="#FFFFFF"/>
''' + "\n".join(out) + "\n</svg>\n"
open(sys.argv[1] if len(sys.argv) > 1 else "book/images/qwen38-architecture.svg", "w", encoding="utf-8").write(svg)
