"""渲染四类版软件界面图（直接调用 realtime/viz 里的真实绘制函数）。

手机不在线时用合成人形画面代替实拍，保证版式与实际运行时完全一致。
输出：reports/doc/fig8_界面_站立.png、fig8_界面_蹲下.png
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

import config as C
from realtime import MAX_TOTAL_W, build_split, draw_bottom_bar

OUT = Path("reports/doc")
OUT.mkdir(parents=True, exist_ok=True)
W, H = 1280, 720


def backdrop(kind="room"):
    """造一张类室内背景，别让左屏是一片纯灰。"""
    img = np.zeros((H, W, 3), np.uint8)
    # 墙面渐变
    for y in range(H):
        v = int(78 + 26 * (y / H))
        img[y, :] = (v, v, v + 3)
    # 地面
    cv2.rectangle(img, (0, int(H * 0.93)), (W, H), (62, 64, 68), -1)
    # 墙板缝隙
    for x in range(0, W, 160):
        cv2.line(img, (x, 0), (x, int(H * 0.93)), (86, 88, 92), 2)
    # 左右墙脚线
    cv2.line(img, (0, int(H * 0.93)), (W, int(H * 0.93)), (98, 100, 104), 3)
    # 右边摆个柜子增加层次
    cv2.rectangle(img, (int(W * 0.86), int(H * 0.42)), (W, int(H * 0.93)),
                  (66, 68, 74), -1)
    cv2.rectangle(img, (int(W * 0.86), int(H * 0.42)), (W, int(H * 0.46)),
                  (54, 56, 62), -1)
    noise = np.random.default_rng(7).normal(0, 3.2, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def standing():
    """站立：镜头正对，占画面 2/3 高。"""
    kps = np.array([
        (640, 96), (620, 90), (660, 90), (602, 104), (678, 104),
        (556, 176), (724, 176), (534, 292), (746, 292),
        (528, 408), (752, 408),
        (582, 398), (698, 398),
        (578, 528), (702, 528),
        (572, 656), (708, 656)], np.float32)
    conf = np.full(17, 0.93, np.float32)
    conf[[1, 2, 3, 4, 7, 8]] = 0.86
    box = (508.0, 62.0, 772.0, 672.0)
    return kps, conf, box, 0.94, np.array([0.91, 0.04, 0.03, 0.02])


def squatting():
    """蹲下：髋明显下沉，膝盖外张且高于髋。"""
    kps = np.array([
        (640, 258), (604, 252), (676, 252), (588, 266), (692, 266),
        (566, 340), (714, 340), (520, 430), (760, 430),
        (500, 520), (780, 520),
        (600, 500), (680, 500),
        (540, 470), (740, 470),
        (556, 650), (724, 650)], np.float32)
    conf = np.full(17, 0.92, np.float32)
    conf[[3, 4, 9, 10]] = 0.71
    box = (492.0, 238.0, 788.0, 668.0)
    return kps, conf, box, 0.92, np.array([0.05, 0.09, 0.03, 0.83])


def draw_figure(frame, kps, cloth=(96, 100, 112), skin=(118, 128, 150)):
    """按关键点画一个真人轮廓，让骨架落在身体上（不是悬空）。"""
    k = np.asarray(kps, np.float32)

    def mid(i, j):
        return (k[i] + k[j]) / 2

    # 头
    head = (k[0] + mid(3, 4)) / 2
    r = int(np.linalg.norm(k[5] - k[6]) * 0.22)
    cv2.circle(frame, tuple(np.int32(head + [0, -r * 0.55])), r, skin, -1,
               cv2.LINE_AA)
    # 脖子
    cv2.line(frame, tuple(np.int32(head + [0, r * 0.4])), tuple(np.int32(mid(5, 6))),
             skin, int(r * 0.8), cv2.LINE_AA)

    # 躯干（梯形：肩宽 -> 髋宽）
    sh_l, sh_r, hp_l, hp_r = k[5], k[6], k[11], k[12]
    torso = np.array([sh_l, sh_r, hp_r, hp_l], np.int32)
    cv2.fillPoly(frame, [torso], cloth, cv2.LINE_AA)

    # 四肢
    def limb(a, b, w, c):
        cv2.line(frame, tuple(np.int32(k[a])), tuple(np.int32(k[b])), c, w,
                 cv2.LINE_AA)

    for a, b, c in ((5, 7, 7), (7, 9, 8), (6, 8, 7), (8, 10, 8)):
        limb(a, b, int(r * 0.85), skin if c == 8 else cloth)
    for a, b in ((11, 13), (13, 15), (12, 14), (14, 16)):
        limb(a, b, int(r * 1.15), cloth)
    # 髋部
    cv2.line(frame, tuple(np.int32(hp_l)), tuple(np.int32(hp_r)), cloth,
             int(r * 1.1), cv2.LINE_AA)
    return frame


def render(make, name, proba=None):
    frame = backdrop()
    kps, conf, box, score, pr = make()
    draw_figure(frame, kps)
    view = build_split(frame, kps, conf, box, score, 0, MAX_TOTAL_W)
    names = C.LABEL_NAMES[int(np.argmax(pr))]
    view = draw_bottom_bar(view, names, pr, C.WINDOW_SEC + 0.6,
                           {"age": 0.018, "infer": 0.082, "src_fps": 26}, None)
    p = OUT / name
    cv2.imwrite(str(p), view)
    print("  ->", p, view.shape)


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    print("渲染界面图：")
    render(standing, "fig8_界面_站立.png")
    render(squatting, "fig8_界面_蹲下.png")
    print("完成")
