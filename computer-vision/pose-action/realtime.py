"""实时判定：摄像头 + YOLOv8-Pose + 时间窗分类，边看边出结果。

画面版式（默认 --view split）
-----------------------------
    左：① 实时画面      原视频 + 骨架 + 关节像素坐标
    右：② SKELETON PANEL  纯骨骼 + 关键点坐标清单（无原图）

左边看得清"人"，右边看得清"算法看到的东西"。演示时两块对照着讲最直观：
屏幕上明明是坐着，右边膝盖折起来一眼就能指出证据。

用法
----
    python realtime.py                 # 双屏版（默认）
    python realtime.py --view single   # 旧版单画面
    python realtime.py --imgsz 320     # 卡顿时降精度换帧率
"""
from __future__ import annotations

import argparse
import os
import time
from collections import Counter, deque

import cv2
import joblib
import numpy as np
import torch
from ultralytics import YOLO

import config as C
from features import frame_features, normalize_keypoints, window_features
from video_source import SourceReader, describe
from viz import (blank_panel, draw_coords, draw_joint_indices, draw_kp_list,
                 draw_label_box, draw_prob_bars, draw_skeleton,
                 draw_skeleton_glow, fit, hstack, panel_title, put_text,
                 put_text_right, text_size)

WINDOW_NAME = "Pose Action - Realtime"
BG = (16, 16, 18)
BAR_H = 132         # 底部信息栏高度（四类概率条，每行 18+10 间距）
GAP = 8             # 两屏之间的缝
MAX_TOTAL_W = 1720  # 整窗宽度上限，超出就自动缩面板


# ---------------------------------------------------------------- 画面合成
def panel_size(frame_w: int, frame_h: int, target_h: int, max_total_w: int):
    """算单个面板尺寸：保持原画面比例，并限制整窗不要超出屏幕。

    target_h<=0 表示自动：竖屏（手机竖放）用更高的面板，否则人只有一小条。
    """
    if target_h <= 0:
        target_h = 720 if frame_h > frame_w else 480
    ph = min(target_h, frame_h)
    pw = int(round(frame_w * ph / frame_h))
    if 2 * pw + GAP > max_total_w:
        pw = (max_total_w - GAP) // 2
        ph = int(round(pw * frame_h / frame_w))
    return max(pw, 160), max(ph, 90)


def build_split(frame, kps, conf, bbox, score, target_h, max_total_w,
                show_coords=True):
    """左边原视频+骨架+坐标，右边纯骨骼面板。返回合成图（不含底部信息栏）。"""
    fh, fw = frame.shape[:2]
    pw, ph = panel_size(fw, fh, target_h, max_total_w)
    s = ph / fh                       # 原图 -> 面板 的缩放比
    offx = (pw - int(round(fw * s))) // 2
    offy = (ph - int(round(fh * s))) // 2

    # ---------- 左：实时画面 ----------
    left = fit(frame, pw, ph)
    if kps is not None:
        draw_skeleton(left, kps, conf, line_color=(0, 210, 255),
                      dot_color=(40, 60, 255), scale=s, thick=1, radius=3,
                      offset=(offx, offy))
        if show_coords:
            draw_coords(left, kps, conf, scale=s, offset=(offx, offy))
    if bbox is not None:
        x1, y1, x2, y2 = [int(v * s) for v in bbox]
        x1 += offx; x2 += offx; y1 += offy; y2 += offy
        cv2.rectangle(left, (x1, y1), (x2, y2), (120, 200, 120), 1)
        draw_label_box(left, x1 + 1, y1 + 1,
                       f"person {score:.2f}" if score is not None else "person",
                       scale=0.48)
    panel_title(left, "① 实时画面", 0, pw, 0, 30)

    # ---------- 右：纯骨骼面板 ----------
    right = blank_panel(pw, ph)
    if kps is not None and conf is not None:
        draw_skeleton_glow(right, kps, conf, scale=s, offset=(offx, offy))
        draw_joint_indices(right, kps, conf, scale=s, offset=(offx, offy))
    panel_title(right, "② SKELETON PANEL", 0, pw, 0, 30)

    if kps is not None and conf is not None:
        list_w = max(int(pw * 0.40), 150)
        draw_kp_list(right, kps, conf, 14, 44, list_w,
                     line_h=max(int(ph / 26), 15),
                     scale=max(ph / 1150, 0.36))
    else:
        hint = "未检测到人体"
        tw, _ = text_size(hint, 0.7)
        put_text(right, hint, ((pw - tw) // 2, ph // 2), 0.7, (120, 120, 235), 2)

    canvas = hstack(left, right, GAP, bg=BG)
    cv2.line(canvas, (pw + GAP // 2, 0), (pw + GAP // 2, ph), (70, 70, 78), 1)
    return canvas


def draw_bottom_bar(canvas, name, proba, span, frm_info, warn_text=None):
    """底部信息栏：行为判定 + 概率条 + 运行状态。"""
    h, w = canvas.shape[:2]
    out = np.full((h + BAR_H, w, 3), BG, dtype=np.uint8)
    out[:h] = canvas
    cv2.line(out, (0, h), (w, h), (70, 70, 78), 1)
    y0 = h

    if span < C.WINDOW_SEC * 0.95:
        put_text(out, f"缓冲中 {span:.1f}/{C.WINDOW_SEC:.1f}s", (18, y0 + 44),
                 0.75, (190, 190, 120), 2)
        put_text(out, "正在收集骨骼数据…", (18, y0 + 74), 0.5, (150, 150, 150), 1)
    else:
        put_text(out, C.LABEL_ZH.get(name, name), (18, y0 + 56), 1.25,
                 (110, 250, 160), 3)
        draw_prob_bars(out, proba, origin=(190, y0 + 14),
                       width=max(int(w * 0.17), 140), height=18, gap=10)

    edge = w - 18
    put_text_right(out, f"画面延迟 {frm_info['age']*1000:.0f} ms", edge, y0 + 30,
                   0.52, (205, 205, 205))
    put_text_right(out, f"推理 {frm_info['infer']*1000:.0f} ms "
                        f"({1/max(frm_info['infer'], 1e-6):.0f} FPS)",
                   edge, y0 + 54, 0.52, (205, 205, 205))
    put_text_right(out, f"手机流 {frm_info['src_fps']:.0f} FPS | q 退出", edge,
                   y0 + 78, 0.48, (150, 150, 150))

    if warn_text:
        tw, _ = text_size(warn_text, 0.55)
        put_text(out, warn_text, ((w - tw) // 2, y0 + 74), 0.55, (60, 170, 255), 2)
    return out


def draw_single(frame, name, proba, span, frm_info, warn_text=None):
    """旧版单画面：骨架直接压在视频上。"""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 92), (0, 0, 0), -1)
    if span < C.WINDOW_SEC * 0.95:
        put_text(frame, f"缓冲中 {span:.1f}/{C.WINDOW_SEC:.1f}s", (16, 34), 0.75,
                 (190, 190, 120), 2)
    else:
        put_text(frame, C.LABEL_ZH.get(name, name), (16, 40), 1.1, (90, 255, 160), 3)
        draw_prob_bars(frame, proba, origin=(16, 52))
    if warn_text:
        put_text(frame, warn_text, (16, h - 40), 0.6, (60, 160, 255), 2)
    put_text_right(frame, f"延迟 {frm_info['age']*1000:.0f}ms  "
                          f"{1/max(frm_info['infer'], 1e-6):.0f} FPS",
                   w - 16, 30, 0.55, (200, 200, 200))
    put_text_right(frame, "q = quit", w - 16, 56, 0.5, (150, 150, 150))
    return frame


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=C.SOURCE,
                    help="摄像头编号(0/1) 或手机地址；默认读 camera_source.txt")
    ap.add_argument("--camera", type=int, default=None,
                    help="旧写法，等价于 --source N")
    ap.add_argument("--model", default=C.MODEL_NAME)
    ap.add_argument("--clf", default=str(C.MODEL_DIR / "pose_best.pkl"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--imgsz", type=int, default=C.IMGSZ,
                    help="推理尺寸，越小越快；320 约快一倍，精度略降")
    ap.add_argument("--smooth", type=int, default=5, help="多数投票的历史长度")
    ap.add_argument("--view", choices=("split", "single"), default="split",
                    help="split=左实时右骨骼（默认），single=旧版单画面")
    ap.add_argument("--disp-h", type=int, default=0,
                    help="每个面板的显示高度；0=自动（竖屏 720，横屏 480）")
    ap.add_argument("--no-coords", action="store_true", help="不显示关节坐标")
    ap.add_argument("--rotate", type=int, default=C.ROTATE,
                    help="画面旋转 0/90/180/270，手机竖放躺倒时用")
    args = ap.parse_args()
    if args.camera is not None:
        args.source = str(args.camera)

    if C.CPU_THREADS:
        torch.set_num_threads(C.CPU_THREADS)
    else:
        torch.set_num_threads(max(1, (os.cpu_count() or 4)))

    clf = joblib.load(args.clf)
    pose = YOLO(args.model)

    reader = SourceReader(args.source, args.width, args.height,
                          rotate=args.rotate)
    if not reader.is_open:
        raise RuntimeError(f"打不开视频源 {describe(args.source)}")

    # 按时间缓冲，而不是按帧数
    feat_buf: deque[tuple[float, np.ndarray]] = deque()
    norm_buf: deque[tuple[float, np.ndarray]] = deque()
    hist = deque(maxlen=args.smooth)
    proba = np.zeros(len(C.LABEL_NAMES))
    prev_center, prev_t = None, time.time()
    t_start = time.time()
    last_pred_t = -1e9
    warn_missing = False
    max_total_w = MAX_TOTAL_W
    infer_s = 0.0

    print(f"[实时] 视频源：{describe(args.source)}")
    print(f"[实时] 缓冲 {C.WINDOW_SEC}s 后开始输出；窗口内按 q 退出")
    print("[实时] 画面：左=实时原图，右=纯骨骼面板")

    bad = 0
    while True:
        ok, frame = reader.read()
        if not ok:
            bad += 1
            if bad > 150:
                print("[实时] 视频源中断，退出")
                break
            continue
        bad = 0
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        t0 = time.time()
        res = pose.predict(frame, imgsz=args.imgsz, conf=C.CONF, iou=C.IOU,
                           device=C.DEVICE, verbose=False)[0]
        dt = max(t0 - prev_t, 1e-3)
        prev_t = t0
        infer_s = time.time() - t0
        now = time.time() - t_start

        feat = norm_vec = None
        kps = conf_kp = None
        bbox = None
        score = None
        detected = (res.boxes is not None and len(res.boxes) > 0
                    and res.keypoints is not None)

        if detected:
            boxes = res.boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            pid = int(np.argmax(areas))

            bbox = tuple(float(v) for v in boxes[pid])
            kps = res.keypoints.xy[pid].cpu().numpy()
            conf_kp = res.keypoints.conf[pid].cpu().numpy()
            if res.boxes.conf is not None:
                score = float(res.boxes.conf[pid])

            feat, center = frame_features(kps, conf_kp, bbox,
                                          prev_center=prev_center, dt=dt,
                                          img_w=w, img_h=h)
            if feat is not None:
                norm_vec = normalize_keypoints(kps, conf_kp, bbox)
                prev_center = center

        warn_missing = detected and feat is None

        # 检测失败时沿用上一帧数据，避免窗口被清空导致判定中断
        if feat is None and feat_buf:
            feat_buf.append((now, feat_buf[-1][1]))
            norm_buf.append((now, norm_buf[-1][1]))
        elif feat is not None:
            feat_buf.append((now, feat))
            norm_buf.append((now, norm_vec))

        # 滑出窗口的老帧丢掉
        while feat_buf and now - feat_buf[0][0] > C.WINDOW_SEC:
            feat_buf.popleft()
            norm_buf.popleft()

        # 每 STRIDE_SEC 秒推理一次
        if len(feat_buf) >= C.MIN_FRAMES and now - last_pred_t >= C.STRIDE_SEC:
            X = np.stack([f for _, f in feat_buf])
            N = np.stack([n for _, n in norm_buf])
            try:
                proba = clf.predict_proba(window_features(X, N).reshape(1, -1))[0]
                hist.append(int(np.argmax(proba)))
            except Exception as e:      # 特征维度对不上等
                proba = np.zeros(len(C.LABEL_NAMES))
                print(f"[实时] 预测失败：{e}")
            last_pred_t = now

        span = now - feat_buf[0][0] if feat_buf else 0.0
        name = C.LABEL_NAMES[Counter(hist).most_common(1)[0][0]] if hist else "-"
        frm_info = {"age": reader.frame_age, "infer": infer_s,
                    "src_fps": reader.fps}
        warn_text = "腿部关键点不足，请后退露出全身" if warn_missing else None

        if args.view == "split":
            view = build_split(frame, kps, conf_kp, bbox, score, args.disp_h,
                               max_total_w, show_coords=not args.no_coords)
            view = draw_bottom_bar(view, name, proba, span, frm_info, warn_text)
        else:
            if kps is not None:
                draw_skeleton(frame, kps, conf_kp)
                if not args.no_coords:
                    draw_coords(frame, kps, conf_kp)
            view = draw_single(frame, name, proba, span, frm_info, warn_text)

        cv2.imshow(WINDOW_NAME, view)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break

    reader.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
