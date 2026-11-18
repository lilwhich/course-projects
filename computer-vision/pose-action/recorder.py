"""采集端：摄像头 + YOLOv8-Pose + 实时骨架可视化 + 逐帧落盘 CSV。

用法
----
    python recorder.py                 # 默认摄像头 0
    python recorder.py --camera 1      # 指定摄像头
    python recorder.py --model yolov8s-pose.pt

操作
----
    1 / 2 / 3   切换录制标签：站立 / 坐下 / 行走
    空格         开始 / 结束一段录制（每段一个独立 CSV）
    s           截图保存当前画面（带骨架）
    q / ESC     退出

采集建议
--------
每段录 15~30 秒，每类至少 5 段，并覆盖不同距离、角度、光照，
否则模型会靠"你在画面里的位置"作弊而不是靠姿态。
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import config as C
from features import frame_features
from video_source import SourceReader, describe
from viz import draw_skeleton, put_text

WINDOW_NAME = "YOLOv8-Pose Recorder"


def build_header() -> list[str]:
    h = ["frame", "time", "label", "x1", "y1", "x2", "y2", "img_w", "img_h"]
    for i, name in enumerate(C.KP_NAMES):
        h += [f"{name}_x", f"{name}_y", f"{name}_c"]
    h += list(C.FRAME_FEATURE_NAMES)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=C.SOURCE,
                    help="摄像头编号(0/1) 或手机地址；默认读 camera_source.txt")
    ap.add_argument("--camera", type=int, default=None,
                    help="旧写法，等价于 --source N")
    ap.add_argument("--model", default=C.MODEL_NAME)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()
    if args.camera is not None:
        args.source = str(args.camera)

    model = YOLO(args.model)
    torch.set_num_threads(C.CPU_THREADS or max(1, (os.cpu_count() or 4)))
    reader = SourceReader(args.source, args.width, args.height)
    if not reader.is_open:
        raise RuntimeError(f"打不开视频源 {describe(args.source)}，"
                           f"检查摄像头是否被占用，或先跑「手机摄像头.bat」")

    header = build_header()
    label = "stand"
    recording = False
    writer = None
    fh = None
    cur_path = None
    frame_idx = 0
    rec_frames = 0
    rec_start = 0.0
    fps_q = deque(maxlen=30)
    prev_center = None
    prev_t = time.time()
    last_frame_gray = None

    print("[采集] 1=站立 2=坐下 3=行走 | 空格=开始/停止 | s=截图 | q=退出")
    print(f"[采集] 视频源：{describe(args.source)}")

    bad = 0
    while True:
        ok, frame = reader.read()
        if not ok:
            bad += 1
            if bad > 150:
                print("[警告] 视频源中断，退出")
                break
            continue
        bad = 0

        frame = cv2.flip(frame, 1)          # 镜像，符合照镜子的直觉
        h, w = frame.shape[:2]

        t0 = time.time()
        res = model.predict(frame, imgsz=C.IMGSZ, conf=C.CONF, iou=C.IOU,
                            device=C.DEVICE, verbose=False)[0]
        dt = max(t0 - prev_t, 1e-3)
        prev_t = t0

        feat = None
        center = None
        kps = conf_kp = None
        bbox = (np.nan,) * 4

        if res.boxes is not None and len(res.boxes) > 0 and res.keypoints is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            keep = np.argsort(areas)[::-1][:C.MAX_PERSONS]   # 只跟最近/最大的那个人

            for pid in keep:
                bbox = tuple(float(v) for v in boxes[pid])
                kps = res.keypoints.xy[pid].cpu().numpy()
                conf_kp = res.keypoints.conf[pid].cpu().numpy()

                feat, center = frame_features(
                    kps, conf_kp, bbox, prev_center=prev_center,
                    dt=dt, img_w=w, img_h=h)
                if center is not None:
                    prev_center = center

                draw_skeleton(frame, kps, conf_kp)
                cv2.rectangle(frame,
                              (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])), (120, 200, 120), 2)

        # ---------- 落盘 ----------
        if recording and writer is not None:
            row = [frame_idx, f"{time.time():.3f}", label,
                   *[f"{v:.1f}" if np.isfinite(v) else "" for v in bbox], w, h]
            if kps is not None:
                for i in range(C.NUM_KP):
                    row += [f"{kps[i][0]:.1f}", f"{kps[i][1]:.1f}", f"{conf_kp[i]:.3f}"]
            else:
                row += [""] * (C.NUM_KP * 3)

            row += [f"{v:.4f}" if (feat is not None and np.isfinite(v)) else ""
                    for v in (feat if feat is not None else np.full(len(C.FRAME_FEATURE_NAMES), np.nan))]
            writer.writerow(row)
            fh.flush()
            rec_frames += 1

        # ---------- HUD ----------
        elapsed = time.time() - t0
        fps_q.append(1.0 / max(elapsed, 1e-6))
        fps = float(np.mean(fps_q))

        cv2.rectangle(frame, (0, 0), (w, 62), (0, 0, 0), -1)
        color = (80, 80, 255) if recording else (200, 200, 200)
        state = "REC" if recording else "IDLE"
        put_text(frame, f"{state}  label={label}({C.LABEL_ZH[label]})", (12, 26),
                 scale=0.75, color=color, thick=2)
        if recording:
            put_text(frame, f"frames={rec_frames}  {time.time()-rec_start:.1f}s", (12, 52),
                     scale=0.6, color=(200, 255, 200))
        put_text(frame, f"fps={fps:.1f}", (w - 130, 26), scale=0.65)
        if kps is None:
            put_text(frame, "未检测到人体", (w // 2 - 80, h // 2), scale=0.9,
                     color=(80, 80, 255), thick=2)
        elif feat is None:
            put_text(frame, "腿部关键点不足，请后退露出全身（髋/膝需可见）",
                     (16, h - 40), scale=0.6, color=(60, 160, 255), thick=2)
        put_text(frame, "1 stand | 2 sit | 3 walk | SPACE rec | s shot | q quit",
                 (12, h - 14), scale=0.55, color=(170, 170, 170))

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key in C.HOTKEY_LABEL:
            label = C.HOTKEY_LABEL[key]
            print(f"[标签] -> {label} ({C.LABEL_ZH[label]})")

        elif key == ord(" "):
            if not recording:
                cur_path = C.RAW_DIR / f"{label}_{datetime.now():%Y%m%d_%H%M%S}.csv"
                fh = open(cur_path, "w", newline="", encoding="utf-8-sig")
                writer = csv.writer(fh)
                writer.writerow(header)
                recording = True
                rec_frames = 0
                rec_start = time.time()
                prev_center = None
                print(f"[录制] 开始 -> {cur_path.name}")
            else:
                recording = False
                dur = time.time() - rec_start
                if fh is not None:
                    fh.close()
                    fh = None
                writer = None
                print(f"[录制] 结束：{rec_frames} 帧 / {dur:.1f}s -> {cur_path}")

        elif key == ord("s"):
            shot = C.REPORT_DIR / f"shot_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            cv2.imwrite(str(shot), frame)
            print(f"[截图] {shot}")

        elif key in (ord("q"), 27):
            break

        frame_idx += 1

    if fh is not None:
        fh.close()
    reader.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
