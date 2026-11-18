"""抓一帧真实画面，渲染成实时窗口的样子并存图（不用开窗口就能看版式）。

用途
----
    1. 调版式时反复看效果，不必每改一句就开一次摄像头
    2. 写报告/PPT 时直接导出「左实时右骨骼」的对照图

用法
----
    python snapshot.py                    # 抓 20 帧，存 reports/view_snapshot.png
    python snapshot.py --frames 60 --out reports/fig_split.png
    python snapshot.py --source 0         # 临时改用电脑摄像头
"""
from __future__ import annotations

import argparse
import time
from collections import Counter, deque

import cv2
import joblib
import numpy as np
import torch
from ultralytics import YOLO

import config as C
from features import frame_features, normalize_keypoints, window_features
from realtime import MAX_TOTAL_W, build_split, draw_bottom_bar
from video_source import SourceReader, describe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=C.SOURCE)
    ap.add_argument("--model", default=C.MODEL_NAME)
    ap.add_argument("--clf", default=str(C.MODEL_DIR / "pose_best.pkl"))
    ap.add_argument("--imgsz", type=int, default=C.IMGSZ)
    ap.add_argument("--frames", type=int, default=20, help="抓多少帧再存图")
    ap.add_argument("--every", type=int, default=10, help="每隔几帧存一张")
    ap.add_argument("--out", default=str(C.REPORT_DIR / "view_snapshot.png"))
    ap.add_argument("--disp-h", type=int, default=0, help="0=自动")
    ap.add_argument("--rotate", type=int, default=C.ROTATE,
                    help="画面旋转 0/90/180/270，手机竖放躺倒时用")
    args = ap.parse_args()

    torch.set_num_threads(max(1, (C.CPU_THREADS or (torch.get_num_threads()))))
    clf = joblib.load(args.clf)
    pose = YOLO(args.model)
    reader = SourceReader(args.source, 1280, 720, rotate=args.rotate)
    if not reader.is_open:
        raise RuntimeError(f"打不开视频源 {describe(args.source)}")

    feat_buf, norm_buf, hist = deque(), deque(), deque(maxlen=5)
    proba = np.zeros(len(C.LABEL_NAMES))
    prev_center, prev_t = None, time.time()
    t_start = time.time()
    last_pred_t = -1e9
    saved, labels = [], []

    for n in range(args.frames):
        ok, frame = reader.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        t0 = time.time()
        res = pose.predict(frame, imgsz=args.imgsz, conf=C.CONF, iou=C.IOU,
                           device=C.DEVICE, verbose=False)[0]
        dt = max(t0 - prev_t, 1e-3)
        prev_t = t0
        now = time.time() - t_start

        kps = conf_kp = bbox = score = None
        feat = norm = None
        if res.boxes is not None and len(res.boxes) > 0 and res.keypoints is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            pid = int(np.argmax((boxes[:, 2] - boxes[:, 0]) *
                                (boxes[:, 3] - boxes[:, 1])))
            bbox = tuple(float(v) for v in boxes[pid])
            kps = res.keypoints.xy[pid].cpu().numpy()
            conf_kp = res.keypoints.conf[pid].cpu().numpy()
            score = float(res.boxes.conf[pid]) if res.boxes.conf is not None else None
            feat, center = frame_features(kps, conf_kp, bbox, prev_center=prev_center,
                                          dt=dt, img_w=w, img_h=h)
            if feat is not None:
                norm = normalize_keypoints(kps, conf_kp, bbox)
                prev_center = center

        if feat is not None:
            feat_buf.append((now, feat))
            norm_buf.append((now, norm))
        while feat_buf and now - feat_buf[0][0] > C.WINDOW_SEC:
            feat_buf.popleft()
            norm_buf.popleft()

        if len(feat_buf) >= C.MIN_FRAMES and now - last_pred_t >= C.STRIDE_SEC:
            X = np.stack([f for _, f in feat_buf])
            N = np.stack([n_ for _, n_ in norm_buf])
            try:
                proba = clf.predict_proba(window_features(X, N).reshape(1, -1))[0]
                hist.append(int(np.argmax(proba)))
            except Exception as e:
                print("预测失败:", e)
            last_pred_t = now
            labels.append(C.LABEL_NAMES[int(np.argmax(proba))])

        if n % args.every == 0 or n == args.frames - 1:
            span = now - feat_buf[0][0] if feat_buf else 0.0
            name = C.LABEL_NAMES[Counter(hist).most_common(1)[0][0]] if hist else "-"
            view = build_split(frame, kps, conf_kp, bbox, score, args.disp_h,
                               MAX_TOTAL_W)
            view = draw_bottom_bar(view, name, proba, span,
                                   {"age": reader.frame_age,
                                    "infer": time.time() - t0,
                                    "src_fps": reader.fps},
                                   "腿部关键点不足，请后退露出全身"
                                   if (kps is not None and feat is None) else None)
            out = args.out.replace(".png", f"_{n:03d}.png")
            cv2.imwrite(out, view)
            saved.append(out)
            print(f"[存图] {out}  尺寸 {view.shape[1]}x{view.shape[0]}")

    print(f"[统计] 抓帧 {reader.fps:.1f} FPS，重连 {reader.reconnects} 次")
    if labels:
        print("[统计] 预测序列:", " ".join(labels))
    reader.release()
    if saved:
        print("[完成] 共", len(saved), "张")


if __name__ == "__main__":
    main()
