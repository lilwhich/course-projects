"""快速采集：不用键盘，按命令行参数直接录一段，录完自动报"体检结果"。

为什么单独做一个
----------------
recorder.py 要按键操作，适合自己一个人慢慢采；演示/赶时间时更希望
「一条命令 = 一段数据」：先倒数 3 秒给你站好，然后录固定时长，
结束时告诉你这段有没有采到全身（膝盖可见率、有效帧比例）。

用法
----
    python quick_record.py --label stand --sec 15
    python quick_record.py --label sit  --sec 15 --count 2
    python quick_record.py --label walk --sec 20 --preview

产出
----
    data/raw/<label>_<时间戳>.csv      与 recorder.py 完全同格式，train.py 直接能读
    reports/rec_<label>_<时间戳>.jpg   录到一半时的画面快照（带骨架），用于回头看机位
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
from recorder import build_header
from video_source import SourceReader, describe
from viz import draw_skeleton, draw_coords, fit, put_text

WINDOW = "Quick Record - 看构图用，不用按键"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, choices=C.LABEL_NAMES)
    ap.add_argument("--sec", type=float, default=15.0, help="每段录多少秒")
    ap.add_argument("--count", type=int, default=1, help="连录几段")
    ap.add_argument("--gap", type=float, default=4.0, help="两段之间休息几秒")
    ap.add_argument("--source", default=C.SOURCE)
    ap.add_argument("--model", default=C.MODEL_NAME)
    ap.add_argument("--imgsz", type=int, default=C.IMGSZ)
    ap.add_argument("--preview", action="store_true", help="开个小窗看构图")
    ap.add_argument("--warmup", type=float, default=3.0, help="开始前倒数几秒")
    ap.add_argument("--rotate", type=int, default=C.ROTATE,
                    help="画面旋转 0/90/180/270，手机竖放躺倒时用")
    args = ap.parse_args()

    zh = C.LABEL_ZH[args.label]
    torch.set_num_threads(C.CPU_THREADS or max(1, (os.cpu_count() or 4)))
    model = YOLO(args.model)
    reader = SourceReader(args.source, 1280, 720, rotate=args.rotate)
    if not reader.is_open:
        raise RuntimeError(f"打不开视频源 {describe(args.source)}")

    header = build_header()
    prev_center, prev_t = None, time.time()
    print(f"[快速采集] 动作：{zh}（{args.label}） 每段 {args.sec:.0f}s × {args.count} 段")

    for seg in range(args.count):
        # ---------- 倒数 ----------
        t_end = time.time() + args.warmup
        while time.time() < t_end:
            ok, frame = reader.read()
            if ok and args.preview:
                show = cv2.flip(frame, 1)
                left = time.time() - t_end
                h, w = show.shape[:2]
                cv2.rectangle(show, (0, 0), (w, 70), (0, 0, 0), -1)
                put_text(show, f"准备：{zh}   {max(left,0):.1f}s 后开始",
                         (14, 44), 0.9, (200, 220, 120), 2)
                cv2.imshow(WINDOW, fit(show, 900, 506))
                cv2.waitKey(1)
        print(f"[第 {seg+1}/{args.count} 段] 开始录制 {args.sec:.0f}s …")

        path = C.RAW_DIR / f"{args.label}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        fh = open(path, "w", newline="", encoding="utf-8-sig")
        writer = csv.writer(fh)
        writer.writerow(header)

        n = n_feat = n_knee = 0
        shot = None
        t0 = time.time()
        while time.time() - t0 < args.sec:
            ok, frame = reader.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            ts = time.time()
            res = model.predict(frame, imgsz=args.imgsz, conf=C.CONF, iou=C.IOU,
                                device=C.DEVICE, verbose=False)[0]
            dt = max(ts - prev_t, 1e-3)
            prev_t = ts

            feat = None
            kps = conf_kp = None
            bbox = (np.nan,) * 4
            if res.boxes is not None and len(res.boxes) > 0 \
                    and res.keypoints is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                pid = int(np.argmax((boxes[:, 2] - boxes[:, 0]) *
                                    (boxes[:, 3] - boxes[:, 1])))
                bbox = tuple(float(v) for v in boxes[pid])
                kps = res.keypoints.xy[pid].cpu().numpy()
                conf_kp = res.keypoints.conf[pid].cpu().numpy()
                feat, center = frame_features(kps, conf_kp, bbox,
                                              prev_center=prev_center, dt=dt,
                                              img_w=w, img_h=h)
                if center is not None:
                    prev_center = center
                if min(conf_kp[C.L_KNE], conf_kp[C.R_KNE]) >= C.KP_CONF:
                    n_knee += 1

            row = [n, f"{time.time():.3f}", args.label,
                   *[f"{v:.1f}" if np.isfinite(v) else "" for v in bbox], w, h]
            if kps is not None:
                for i in range(C.NUM_KP):
                    row += [f"{kps[i][0]:.1f}", f"{kps[i][1]:.1f}",
                            f"{conf_kp[i]:.3f}"]
            else:
                row += [""] * (C.NUM_KP * 3)
            row += [f"{v:.4f}" if (feat is not None and np.isfinite(v)) else ""
                    for v in (feat if feat is not None
                              else np.full(len(C.FRAME_FEATURE_NAMES), np.nan))]
            writer.writerow(row)
            n += 1
            if feat is not None:
                n_feat += 1

            if args.preview:
                show = frame.copy()
                if kps is not None:
                    draw_skeleton(show, kps, conf_kp)
                    draw_coords(show, kps, conf_kp)
                left = args.sec - (time.time() - t0)
                cv2.rectangle(show, (0, 0), (w, 66), (0, 0, 0), -1)
                put_text(show, f"REC {zh}  {left:.0f}s  有效帧 {n_feat}/{n}",
                         (14, 42), 0.85, (80, 80, 255), 2)
                if n == 6:
                    shot = C.REPORT_DIR / f"rec_{args.label}_{datetime.now():%H%M%S}.jpg"
                    cv2.imwrite(str(shot), show)
                cv2.imshow(WINDOW, fit(show, 900, 506))
                cv2.waitKey(1)

        fh.close()
        rate = n_feat / max(n, 1) * 100
        knee = n_knee / max(n, 1) * 100
        print(f"[第 {seg+1} 段] 完成 {n} 帧 / {rate:.0f}% 有效 / 膝盖可见 {knee:.0f}%"
              f" -> {path.name}")
        if rate < 60:
            print("    ⚠ 有效帧偏低：人再退远一些，让髋和膝都进画面")
        if knee < 70:
            print("    ⚠ 膝盖常丢：手机放低一点、镜头略上仰，或人往后退")
        if shot:
            print(f"    快照 {shot}")
        if seg + 1 < args.count:
            print(f"[休息] {args.gap:.0f}s 后录下一段…")
            time.sleep(args.gap)

    reader.release()
    cv2.destroyAllWindows()
    print("[快速采集] 全部完成。下一步：python train.py")


if __name__ == "__main__":
    main()
