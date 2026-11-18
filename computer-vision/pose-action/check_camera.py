"""机位自检：确认摄像头能不能拍到全身。

为什么需要这一步
----------------
站立和坐下在骨骼上的差别全在**下半身**（髋、膝、踝）。
如果摄像头只拍到上半身，这两类在特征上完全一样，任何分类器都分不出来。
这是物理限制，不是调参能解决的。

所以采集数据前必须先跑这个工具，确认三件事：
  1. 髋、膝、脚踝的关键点都能稳定检出
  2. 人没有被画面上下边缘裁掉
  3. 人在画面里的高度占比在 55%~92% 之间（太小关键点会抖）

用法
----
    双击「机位自检.bat」，或 python check_camera.py

按 s 保存截图，按 q 退出。
"""
from __future__ import annotations

import argparse
import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import config as C
from video_source import SourceReader, describe
from viz import draw_skeleton, put_text, put_text_right, text_size

WINDOW = "Camera Check - q to quit"

GOOD = (80, 220, 120)      # 绿
WARN = (60, 175, 255)      # 橙
BAD = (70, 70, 235)        # 红
GREY = (170, 170, 170)
WHITE = (245, 245, 245)


def diagnose(conf_kp, bbox, img_w: int, img_h: int):
    """根据关键点置信度与人体框给出机位结论。

    返回 (level, title, advice)
        level: "none" | "bad" | "warn" | "good"
    """
    if conf_kp is None or bbox is None:
        return "none", "未检测到人体", "请站到画面中央，正对摄像头"

    x1, y1, x2, y2 = bbox
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    ratio = bh / float(img_h)

    hip_ok = min(conf_kp[C.L_HIP], conf_kp[C.R_HIP]) >= C.KP_CONF
    knee_n = int(conf_kp[C.L_KNE] >= C.KP_CONF) + int(conf_kp[C.R_KNE] >= C.KP_CONF)
    ankle_n = int(conf_kp[C.L_ANK] >= C.KP_CONF) + int(conf_kp[C.R_ANK] >= C.KP_CONF)
    clipped_top = y1 <= img_h * 0.02
    clipped_bot = y2 >= img_h * 0.98

    # 优先级从"致命"到"可接受"
    if not hip_ok:
        return "bad", "看不到髋部", "请靠近摄像头，或让身体完整进入画面"
    if knee_n == 0:
        return ("bad", "看不到膝盖",
                "站/坐无法区分！请后退 1~1.5 米，或把摄像头放低")
    if clipped_top and clipped_bot:
        return "bad", "人超出画面上下边界", "请后退，让人完整进入画面"
    if ratio > 1.0:
        return "warn", "人占满画面", "请后退半步，留出头顶和脚下余量"
    if ratio < 0.42:
        return "warn", "人在画面中偏小", "请往前站一点，否则关键点抖动会很大"
    if ankle_n < 2:
        return ("warn", "看得到膝但看不到脚踝",
                "可以采集，但建议再退半步，让脚踝也进入画面")
    if ratio > 0.95:
        return "warn", "人几乎顶到画面边缘", "请后退一点点"
    return "good", "机位合格", "髋、膝、脚踝全部可见，可以开始采集"


def draw_target_zone(img, color=GREY):
    """画一个建议的人体站位框，直观提示人该多大。"""
    h, w = img.shape[:2]
    top, bot = int(h * 0.19), int(h * 0.97)
    left, right = int(w * 0.30), int(w * 0.74)
    cv2.rectangle(img, (left, top), (right, bot), color, 1, cv2.LINE_AA)
    tip = "把全身放进这个框"
    tw = len(tip) * 20
    cv2.rectangle(img, (left, top - 26), (left + tw + 14, top - 2), (18, 18, 18), -1)
    put_text(img, tip, (left + 7, top - 7), 0.52, color, 1)


def draw_overlay(frame, kps, conf_kp, bbox, ok_hist, fps):
    """在画面上画骨架、状态面板与结论条，返回 (level, title, advice)。"""
    h, w = frame.shape[:2]

    if kps is not None:
        draw_skeleton(frame, kps, conf_kp)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])), (120, 200, 120), 2)
        # 髋线 / 膝线，方便肉眼判断拍到哪了
        for idx, color, name in ((C.L_HIP, (200, 200, 90), "髋线"),
                                 (C.L_KNE, (200, 140, 90), "膝线")):
            if conf_kp[idx] >= C.KP_CONF:
                y = int(kps[idx][1])
                cv2.line(frame, (0, y), (w, y), color, 1, cv2.LINE_AA)
                cv2.rectangle(frame, (w - 74, y - 24), (w - 4, y - 2), (18, 18, 18), -1)
                put_text(frame, name, (w - 68, y - 7), 0.5, color, 1)

    level, title, advice = diagnose(conf_kp, bbox, w, h)
    ok_hist.append(1 if level == "good" else 0)
    color = {"good": GOOD, "warn": WARN, "bad": BAD, "none": GREY}[level]

    draw_target_zone(frame)

    # ---------- 顶部结论条 ----------
    cv2.rectangle(frame, (0, 0), (w, 104), (18, 18, 18), -1)
    cv2.rectangle(frame, (0, 0), (10, 104), color, -1)          # 左侧色块
    cv2.rectangle(frame, (0, 102), (w, 104), color, -1)
    put_text(frame, title, (26, 50), 1.35, color, 2)
    put_text(frame, advice, (26, 88), 0.75, WHITE, 1)

    # ---------- 左侧状态面板 ----------
    px, py = 20, 148
    panel_h = 3 * 36 + 52
    cv2.rectangle(frame, (px - 10, py - 32), (px + 336, py + panel_h), (18, 18, 18), -1)

    if conf_kp is not None:
        rows = [
            ("髋 HIP", int(conf_kp[C.L_HIP] >= C.KP_CONF)
             + int(conf_kp[C.R_HIP] >= C.KP_CONF), 2),
            ("膝 KNEE", int(conf_kp[C.L_KNE] >= C.KP_CONF)
             + int(conf_kp[C.R_KNE] >= C.KP_CONF), 2),
            ("踝 ANKLE", int(conf_kp[C.L_ANK] >= C.KP_CONF)
             + int(conf_kp[C.R_ANK] >= C.KP_CONF), 2),
        ]
        for i, (name, got, need) in enumerate(rows):
            yy = py + i * 36
            dot = GOOD if got >= need else (WARN if got > 0 else BAD)
            cv2.circle(frame, (px + 9, yy - 7), 7, dot, -1, cv2.LINE_AA)
            put_text(frame, name, (px + 26, yy), 0.66, WHITE, 1)
            put_text(frame, f"{got}/{need}", (px + 168, yy), 0.66, dot, 1)

        ratio = (bbox[3] - bbox[1]) / float(h)
        bar_y = py + 3 * 36 + 12
        put_text(frame, f"人体高度 {ratio*100:.0f}%", (px + 4, bar_y), 0.62, WHITE, 1)
        cv2.rectangle(frame, (px + 168, bar_y - 15), (px + 326, bar_y - 1),
                      (60, 60, 60), -1)
        fill = int(np.clip((ratio - 0.3) / 0.7, 0, 1) * 158)
        cv2.rectangle(frame, (px + 168, bar_y - 15), (px + 168 + fill, bar_y - 1),
                      GOOD if 0.55 <= ratio <= 0.92 else WARN, -1)
    else:
        put_text(frame, "画面中没有人", (px + 4, py + 8), 0.7, WARN, 1)

    # ---------- 右下角 ----------
    rate = (sum(ok_hist) / len(ok_hist) * 100) if ok_hist else 0.0
    rate_txt = f"合格率 {rate:.0f}%"
    rw, rh = text_size(rate_txt, 0.72)
    cv2.rectangle(frame, (w - rw - 30, h - 48 - rh - 6), (w - 12, h - 48 + 8),
                  (18, 18, 18), -1)
    put_text_right(frame, rate_txt, w - 18, h - 48, 0.72,
                   GOOD if rate >= 75 else WARN, 1)
    cv2.rectangle(frame, (w - 76, h - 40), (w - 12, h - 8), (18, 18, 18), -1)
    put_text_right(frame, f"{fps:.0f} fps", w - 18, h - 16, 0.58, GREY, 1)

    # ---------- 左下角按键提示 ----------
    tip = "s 保存截图    q 退出"
    tw, th = text_size(tip, 0.6)
    cv2.rectangle(frame, (14, h - 16 - th - 8), (14 + tw + 14, h - 4), (18, 18, 18), -1)
    put_text(frame, tip, (21, h - 16), 0.6, GREY, 1)
    return level, title, advice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=C.SOURCE,
                    help="摄像头编号(0/1) 或手机地址；默认读 camera_source.txt")
    ap.add_argument("--camera", type=int, default=None,
                    help="旧写法，等价于 --source N")
    ap.add_argument("--model", default=C.MODEL_NAME)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="跑到指定帧数自动退出，0=一直跑（自测用）")
    args = ap.parse_args()
    if args.camera is not None:
        args.source = str(args.camera)

    torch.set_num_threads(C.CPU_THREADS or max(1, os.cpu_count() or 4))
    pose = YOLO(args.model)

    reader = SourceReader(args.source, args.width, args.height)
    if not reader.is_open:
        print(f"[错误] 打不开视频源：{describe(args.source)}")
        print("       本机摄像头换个编号试试：--source 1")
        print("       手机摄像头先跑一遍「手机摄像头.bat」")
        return 1

    print("=" * 58)
    print(" 机位自检")
    print(f" 视频源：{describe(args.source)}")
    print(" 请退到 1.5~2.5 米，让全身进入画面，站定 2 秒")
    print(" s = 保存截图    q = 退出")
    print("=" * 58)

    ok_hist: deque[int] = deque(maxlen=45)
    last_save = 0.0
    verdicts: dict[str, int] = {}
    snaps: list[str] = []
    last_level: str | None = None
    frame_i = 0
    t_fps = time.time()
    fps = 0.0
    bad = 0

    while True:
        ok, frame = reader.read()
        if not ok:
            bad = bad + 1
            if bad > 150:            # 网络流断线时先给重连机会
                print("[错误] 视频源中断，退出。")
                break
            continue
        bad = 0
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frame_i += 1
        if frame_i % 5 == 0:
            fps = 5.0 / max(time.time() - t_fps, 1e-6)
            t_fps = time.time()

        res = pose.predict(frame, imgsz=C.IMGSZ, conf=C.CONF, iou=C.IOU,
                           device=C.DEVICE, verbose=False)[0]

        conf_kp = bbox = kps = None
        if res.boxes is not None and len(res.boxes) > 0 and res.keypoints is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            pid = int(np.argmax(areas))
            bbox = tuple(float(v) for v in boxes[pid])
            kps = res.keypoints.xy[pid].cpu().numpy()
            conf_kp = res.keypoints.conf[pid].cpu().numpy()

        level, title, advice = draw_overlay(frame, kps, conf_kp, bbox, ok_hist, fps)
        verdicts[level] = verdicts.get(level, 0) + 1

        # 状态一变就自动存一张，省得事后靠回忆描述现场
        if level != last_level and level != "none" and len(snaps) < 6:
            last_level = level
            out = C.REPORT_DIR / f"机位_{level}_{datetime.now():%H%M%S}.png"
            cv2.imwrite(str(out), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            snaps.append(out.name)
            print(f"[自动截图] {out.name}  <- {title}")

        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if args.max_frames and frame_i >= args.max_frames:
            break
        if key == ord("s") and time.time() - last_save > 0.6:
            last_save = time.time()
            out = C.REPORT_DIR / f"机位截图_{datetime.now():%Y%m%d_%H%M%S}.png"
            cv2.imwrite(str(out), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            print(f"[截图] {out.name}  结论: {title}")

    reader.release()
    cv2.destroyAllWindows()

    # ---------- 退出总结 ----------
    total = sum(verdicts.values())
    has_person = total - verdicts.get("none", 0)
    rate = (verdicts.get("good", 0) / has_person * 100) if has_person else 0.0

    names = {"good": "合格", "warn": "勉强", "bad": "不合格", "none": "无人"}
    detail = "  ".join(f"{names[k]}{verdicts.get(k, 0)}" for k in ("good", "warn", "bad", "none"))

    print()
    print("=" * 58)
    print(f" 自检总结：共 {total} 帧（{detail}）")
    if has_person:
        print(f" 有人出现的 {has_person} 帧中，机位合格 {rate:.0f}%")
    else:
        print(" 全程没有检测到人——手机可能没对着你站的位置。")

    if not has_person:
        print(" 结论：没看到人，无法判断机位。")
        print(" 请把手机转向你要站的那片空地，人站进画面后重跑本工具。")
    elif rate >= 75:
        print(" 结论：机位可用，可以开始采集数据。")
        print(" 下一步：双击「②采集数据.bat」")
    elif verdicts.get("warn", 0) > verdicts.get("bad", 0):
        print(" 结论：勉强可用，但关键点不够稳，建议按上面提示调整后再采。")
    else:
        print(" 结论：机位不合格，站/坐无法区分。")
        print(" 处理办法（按顺序试）：")
        print("   1. 人往后退 1~1.5 米，直到膝盖进画面")
        print("   2. 把摄像头/手机放到更低处（椅面、地面），镜头略微朝上")
        print("   3. 换外接 USB 摄像头，摆在 1 米高、正对全身")
    if snaps:
        print(f" 已自动保存 {len(snaps)} 张现场截图到 reports 目录，可用来对照。")
    print("=" * 58)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
