"""可视化小工具：骨架绘制、文字、概率条。采集端与实时端共用。

中文支持说明
------------
OpenCV 的 putText 只认 ASCII，中文会渲染成 ????。
这里对非 ASCII 文字改用 Pillow 渲染，并把渲染结果缓存成位图，
因此每帧只是贴图，不影响帧率。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import config as C

# ---------- 中文字体 ----------
# 粗体与常规分开：中文靠描边加粗会发糊，改用真正的粗体字体更清晰
_FONT_NORMAL = [
    "C:/Windows/Fonts/msyh.ttc",     # 微软雅黑
    "C:/Windows/Fonts/simhei.ttf",   # 黑体
    "C:/Windows/Fonts/deng.ttf",     # 等线
    "C:/Windows/Fonts/simsun.ttc",   # 宋体
]
_FONT_BOLD = [
    "C:/Windows/Fonts/msyhbd.ttc",   # 微软雅黑 Bold
    "C:/Windows/Fonts/simhei.ttf",   # 黑体（本身较粗）
] + _FONT_NORMAL

_font_cache: dict[tuple[int, bool], object] = {}
_glyph_cache: dict[tuple, tuple] = {}
_PIL_OK = True

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    _PIL_OK = False


def _get_font(px: int, bold: bool = False):
    """按像素高度取字体，找不到任何中文字体时返回 None。"""
    key = (px, bold)
    if key in _font_cache:
        return _font_cache[key]
    font = None
    if _PIL_OK:
        for p in (_FONT_BOLD if bold else _FONT_NORMAL):
            if Path(p).exists():
                try:
                    font = ImageFont.truetype(p, px)
                    break
                except Exception:
                    continue
    _font_cache[key] = font
    return font


def _render_cn(text: str, px: int, color, bold: bool = False):
    """把一行中文渲染成 BGRA 位图，返回 (bgra, w, h, left, top)。"""
    key = (text, px, tuple(color), bold)
    if key in _glyph_cache:
        return _glyph_cache[key]

    font = _get_font(px, bold)
    if font is None:                      # 没有中文字体，退回占位，至少不崩
        return None

    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    bbox = probe.textbbox((0, 0), text, font=font, anchor="ls")
    left, top, right, bottom = [int(v) for v in bbox]
    w = max(right - left, 1)
    h = max(bottom - top, 1)

    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).text((-left, -top), text, font=font, fill=255, anchor="ls")
    m = np.asarray(mask, dtype=np.uint8)

    bgra = np.zeros((h, w, 4), dtype=np.uint8)
    bgra[..., 0] = color[0]
    bgra[..., 1] = color[1]
    bgra[..., 2] = color[2]
    bgra[..., 3] = m
    out = (bgra, w, h, left, top)
    _glyph_cache[key] = out
    return out


def put_text(img, text, org, scale=0.6, color=(255, 255, 255), thick=1):
    """在 img 上写字。org 是文字左下角（基线）坐标，与 cv2.putText 语义一致。"""
    text = str(text)
    if not text:
        return

    # ASCII 走 OpenCV 快速路径
    if text.isascii():
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                    cv2.LINE_AA)
        return

    px = max(12, int(round(30 * scale)))
    glyph = _render_cn(text, px, color, bold=(thick >= 2))
    if glyph is None:
        cv2.putText(img, "?", org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                    cv2.LINE_AA)
        return

    bgra, w, h, left, top = glyph
    x = int(org[0]) + left
    y = int(org[1]) + top

    # 裁剪到画面内，避免越界崩溃
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    sx0, sy0 = x0 - x, y0 - y

    roi = img[y0:y1, x0:x1]
    patch = bgra[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
    alpha = (patch[..., 3:4].astype(np.float32) / 255.0)
    roi[:] = (patch[..., :3].astype(np.float32) * alpha
              + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)


def text_size(text: str, scale: float = 0.6) -> tuple[int, int]:
    """估算文字像素宽高，用于排版（比如右对齐）。"""
    text = str(text)
    if text.isascii():
        w, h = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0]
        return w, h
    px = max(12, int(round(30 * scale)))
    g = _render_cn(text, px, (255, 255, 255), bold=False)
    return (g[1], g[2]) if g else (len(text) * px, px)


def put_text_right(img, text, right_x, org_y, scale=0.6, color=(255, 255, 255),
                   thick=1):
    """右对齐写字，right_x 是文字右边界。"""
    w, _ = text_size(text, scale)
    put_text(img, text, (int(right_x - w), org_y), scale, color, thick)


def draw_skeleton(img, kps, conf, line_color=(0, 220, 255), dot_color=(60, 60, 255),
                  scale: float = 1.0, thick: int = 2, radius: int = 4,
                  offset=(0, 0)):
    """在 img 上画骨架。scale/offset 用于缩放和平移到显示面板坐标。"""
    k = np.asarray(kps, dtype=np.float32) * scale
    k[:, 0] += offset[0]
    k[:, 1] += offset[1]
    t = max(1, int(round(thick * max(scale, 0.6))))
    r = max(2, int(round(radius * max(scale, 0.6))))
    for i, j in C.SKELETON:
        if conf[i] >= C.KP_CONF and conf[j] >= C.KP_CONF:
            cv2.line(img, tuple(np.int32(k[i])), tuple(np.int32(k[j])),
                     line_color, t, cv2.LINE_AA)
    for i in range(C.NUM_KP):
        if conf[i] >= C.KP_CONF:
            cv2.circle(img, tuple(np.int32(k[i])), r, dot_color, -1, cv2.LINE_AA)


# ---------------------------------------------------------------- 纯骨骼面板
_PANEL_BG = (28, 26, 22)          # 深暖灰，比纯黑柔和，投影不发灰
_SKEL_MAIN = (0, 232, 238)        # 黄（BGR）：主骨架线
_SKEL_JOINT = (110, 250, 120)     # 黄绿：关节点
_TEXT = (240, 240, 240)
_TEXT_DIM = (170, 175, 180)

# 左画面里标坐标的关键点：只标核心几个，全标会糊成一团
COORD_KP = (0, 5, 6, 11, 12, 13, 14, 15, 16)


def blank_panel(w: int, h: int, bg=_PANEL_BG):
    return np.full((max(int(h), 1), max(int(w), 1), 3), bg, dtype=np.uint8)


def draw_skeleton_glow(img, kps, conf, scale: float = 1.0, offset=(0, 0),
                       main=_SKEL_MAIN, joint=_SKEL_JOINT, glow: bool = True):
    """纯骨骼画法：三层描边由粗到细，远看像发光，缩到小图也不断线。"""
    k = np.asarray(kps, dtype=np.float32) * scale
    k[:, 0] += offset[0]
    k[:, 1] += offset[1]
    ok = np.asarray(conf) >= C.KP_CONF
    base = max(1, int(round(2.2 * scale)))

    if glow:
        dim = tuple(int(c * 0.30) for c in main)
        mid = tuple(int(c * 0.62) for c in main)
        layers = [(dim, base * 4), (mid, base * 2), (main, base)]
    else:
        layers = [(main, base)]

    for color, th in layers:
        for i, j in C.SKELETON:
            if ok[i] and ok[j]:
                cv2.line(img, tuple(np.int32(k[i])), tuple(np.int32(k[j])),
                         color, th, cv2.LINE_AA)

    r = max(2, int(round(3.0 * scale)))
    for i in range(C.NUM_KP):
        if not ok[i]:
            continue
        p = tuple(np.int32(k[i]))
        if glow:
            cv2.circle(img, p, r + max(2, int(round(3 * scale))),
                       tuple(int(c * 0.45) for c in joint), -1, cv2.LINE_AA)
        cv2.circle(img, p, r, joint, -1, cv2.LINE_AA)


def fit(frame, w: int, h: int):
    """等比缩放 + 居中黑边，避免拉伸变形。"""
    fh, fw = frame.shape[:2]
    if fw == w and fh == h:
        return frame
    s = min(w / fw, h / fh)
    nw, nh = max(int(fw * s), 1), max(int(fh * s), 1)
    out = blank_panel(w, h, bg=(0, 0, 0))
    out[:] = (0, 0, 0)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (nw, nh), interpolation=interp)
    x, y = (w - nw) // 2, (h - nh) // 2
    out[y:y + nh, x:x + nw] = resized
    return out


def hstack(left, right, gap: int = 8, bg=(14, 14, 16)):
    """左右拼屏，中间留一条分隔缝。"""
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + gap + right.shape[1]
    canvas = np.full((h, w, 3), bg, dtype=np.uint8)
    canvas[:left.shape[0], :left.shape[1]] = left
    canvas[:right.shape[0], left.shape[1] + gap:] = right
    return canvas


def draw_coords(img, kps, conf, scale: float = 1.0, offset=(0, 0),
                idxs=COORD_KP, color=(70, 235, 250), small=0.42):
    """在关节旁标出像素坐标。

    左右两侧的关键点朝各自外侧写，避免正对镜头时两髋/两膝的标签叠在一起。
    """
    k = np.asarray(kps, dtype=np.float32) * scale
    k[:, 0] += offset[0]
    k[:, 1] += offset[1]
    left_side = {C.L_SHO, C.L_ELB, C.L_WRI, C.L_HIP, C.L_KNE, C.L_ANK}
    slot = {True: 0, False: 0}               # 左右两侧各自错行计数
    for i in idxs:
        if i >= len(conf) or conf[i] < C.KP_CONF:
            continue
        x, y = int(k[i][0]), int(k[i][1])
        txt = f"({x},{y})"
        tw, th = text_size(txt, small)
        is_left = i in left_side
        tx = x - tw - 8 if is_left else x + 8
        # 同侧标签轮流错开三档，避免手腕/髋这类高度相近的点叠在一起
        ty = y - 6 - (slot[is_left] % 3) * (th + 3)
        slot[is_left] += 1
        tx = min(max(tx, 2), img.shape[1] - tw - 2)
        ty = min(max(ty, th + 2), img.shape[0] - 4)
        put_text(img, txt, (tx, ty), small, (25, 25, 30), 3)
        put_text(img, txt, (tx, ty), small, color, 1)


def draw_joint_indices(img, kps, conf, scale: float = 1.0, offset=(0, 0),
                       idxs=(0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
                       color=_SKEL_MAIN, small=0.38):
    """在骨骼面板的关节旁标出关键点序号，和左侧清单对得上。"""
    k = np.asarray(kps, dtype=np.float32) * scale
    k[:, 0] += offset[0]
    k[:, 1] += offset[1]
    r = max(3, int(round(3 * scale)))
    for i in idxs:
        if i >= len(conf) or conf[i] < C.KP_CONF:
            continue
        x, y = int(k[i][0]), int(k[i][1])
        put_text(img, str(i), (x + r + 3, y - r - 1), small, (20, 20, 24), 3)
        put_text(img, str(i), (x + r + 3, y - r - 1), small, color, 1)


def draw_label_box(img, x: int, y: int, text: str, scale: float = 0.5,
                   fg=(255, 255, 255), bg=(58, 62, 190), pad: int = 4):
    """在 (x, y) 左上角处贴一个带底色的小标签，比如 person 0.85。"""
    tw, th = text_size(text, scale)
    x, y = max(int(x), 0), max(int(y), 0)
    cv2.rectangle(img, (x, y), (x + tw + pad * 2, y + th + pad * 2), bg, -1)
    put_text(img, text, (x + pad, y + th + pad - 2), scale, fg, 1)


def draw_kp_list(img, kps, conf, x0: int, y0: int, w: int,
                 line_h: int = 19, scale: float = 0.46,
                 title_scale: float = 0.62):
    """右面板左侧的关键点坐标清单：序号 + 像素坐标，只列可见点。"""
    # 半透明底，免得文字压着骨架看不清
    rows = int(sum(1 for c in conf if c >= C.KP_CONF))
    box_h = 34 + rows * line_h + 10
    ov = img.copy()
    cv2.rectangle(ov, (x0 - 8, y0 - 8), (x0 + w, y0 + box_h), (10, 10, 12), -1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)

    put_text(img, "Keypoints:", (x0, y0 + 14), title_scale, _TEXT, 2)
    y = y0 + 40
    for i in range(C.NUM_KP):
        if conf[i] < C.KP_CONF:
            continue
        px, py = kps[i]
        put_text(img, f"{i:>2}", (x0, y), scale, _SKEL_MAIN, 1)
        put_text(img, f"({px:.0f},{py:.0f})", (x0 + 26, y), scale, _TEXT, 1)
        y += line_h
    return y


def panel_title(img, text, x0: int, w: int, top: int = 0, height: int = 36,
                color=_TEXT, bg=(40, 40, 46), scale: float = 0.6):
    """面板顶部标题条（居中）。"""
    cv2.rectangle(img, (x0, top), (x0 + w, top + height), bg, -1)
    cv2.line(img, (x0, top + height), (x0 + w, top + height), (70, 70, 78), 1)
    tw, _ = text_size(text, scale)
    put_text(img, text, (x0 + max((w - tw) // 2, 6), top + height - 10),
             scale, color, 2)


def draw_prob_bars(img, probs, origin=(16, 96), width=170, height=16, gap=8):
    """在画面左上角画三类行为的概率条。"""
    x, y = origin
    for i, name in enumerate(C.LABEL_NAMES):
        p = float(probs[i]) if probs is not None else 0.0
        yy = y + i * (height + gap)
        cv2.rectangle(img, (x, yy), (x + width, yy + height), (60, 60, 60), -1)
        cv2.rectangle(img, (x, yy), (x + int(width * p), yy + height),
                      (90, 200, 120), -1)
        put_text(img, f"{C.LABEL_ZH[name]} {p*100:4.0f}%", (x + width + 8, yy + 13),
                 scale=0.5, color=(230, 230, 230))
