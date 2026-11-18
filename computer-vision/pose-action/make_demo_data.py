"""生成合成骨骼数据，用来在没有摄像头/没采集前验证整条链路。

这不是给最终训练用的，只做两件事：
  1. 验证 采集格式 -> 特征 -> 训练 -> 报告 全链路能跑通；
  2. 让你提前看到 report.py 出的图长什么样。

用法
----
    python make_demo_data.py                 # 每类 3 段 x 20 秒
    python make_demo_data.py --secs 30 --segs 5
    python train.py --raw data/demo
    python report.py --raw data/demo --all
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

import config as C

W, H = 1280, 720
L_THIGH, L_SHANK = 90, 90


def _torso(hip, tilt_deg=0.0, height=160):
    """按倾角算肩中点，tilt>0 表示上半身前倾。"""
    a = np.radians(tilt_deg)
    return np.array([hip[0] + height * np.sin(a), hip[1] - height * np.cos(a)])


def synth(label: str, t: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """在时刻 t 合成一帧 17 点关键点（像素坐标）。"""
    kps = np.zeros((C.NUM_KP, 2), dtype=np.float32)

    if label == "stand":
        hx = 640 + 8 * np.sin(0.5 * t)          # 轻微重心晃动
        hy = 400 + 2 * np.sin(1.3 * t)
        hip = np.array([hx, hy])
        tilt = 3.0 + 1.5 * np.sin(0.7 * t)
        sho = _torso(hip, tilt)
        knee_l, knee_r = np.array([hx - 35, hy + L_THIGH]), np.array([hx + 35, hy + L_THIGH])
        ank_l = np.array([hx - 37, hy + L_THIGH + L_SHANK + 5])
        ank_r = np.array([hx + 37, hy + L_THIGH + L_SHANK + 5])
        wri_y = sho[1] + 70 + 3 * np.sin(1.1 * t)

    elif label == "sit":
        hx = 640 + 5 * np.sin(0.4 * t)
        hy = 455 + 2 * np.sin(0.9 * t)          # 髋明显下沉
        hip = np.array([hx, hy])
        tilt = 8.0 + 2.0 * np.sin(0.6 * t)
        sho = _torso(hip, tilt, height=150)
        knee_l, knee_r = np.array([hx - 55, hy + 10]), np.array([hx + 55, hy + 10])
        ank_l = np.array([hx - 55, hy + 140])
        ank_r = np.array([hx + 55, hy + 140])
        wri_y = hy - 10                          # 手放大腿上，腕高于髋

    elif label == "squat":
        # 蹲下与坐下的关键差别：胯沉到膝盖以下，上身明显前倾，脚踝收到膝前下方
        hx = 640 + 4 * np.sin(0.5 * t)
        hy = 505 + 6 * np.sin(0.8 * t)          # 比坐下更低
        hip = np.array([hx, hy])
        tilt = 26.0 + 3.0 * np.sin(0.7 * t)     # 前倾明显大于坐下
        sho = _torso(hip, tilt, height=150)
        knee_l = np.array([hx - 70, hy - 18 + 4 * np.sin(0.9 * t)])
        knee_r = np.array([hx + 70, hy - 14 + 4 * np.sin(0.9 * t + 1.0)])
        ank_l = np.array([knee_l[0] + 26, knee_l[1] + 96])
        ank_r = np.array([knee_r[0] + 26, knee_r[1] + 96])
        wri_y = hy - 6 + 3 * np.sin(1.2 * t)    # 手臂前平举保持平衡

    else:  # walk
        p = 2 * np.pi * 1.6 * t                  # 步频 1.6Hz
        hx = 240 + 22 * t                        # 横向匀速走过画面
        hy = 405 - 6 * abs(np.sin(2 * p))        # 重心上下起伏
        hip = np.array([hx, hy])
        tilt = 6.0 + 3.0 * np.sin(p)
        sho = _torso(hip, tilt)
        sw = 26 * np.sin(p)                      # 大腿摆幅
        knee_l = np.array([hx - 30 + sw, hy + L_THIGH - 6 * abs(np.sin(p))])
        knee_r = np.array([hx + 30 - sw, hy + L_THIGH - 6 * abs(np.sin(p + np.pi))])
        ank_l = np.array([knee_l[0] + 34 * np.sin(p - 0.9), knee_l[1] + L_SHANK])
        ank_r = np.array([knee_r[0] + 34 * np.sin(p + np.pi - 0.9), knee_r[1] + L_SHANK])
        wri_y = sho[1] + 75 + 12 * np.sin(p + np.pi)

    sh = 62
    kps[C.NOSE] = sho + np.array([0, -42])
    kps[1] = sho + np.array([-14, -52]); kps[2] = sho + np.array([14, -52])
    kps[3] = sho + np.array([-30, -46]); kps[4] = sho + np.array([30, -46])
    kps[C.L_SHO] = sho + np.array([-sh, 0]); kps[C.R_SHO] = sho + np.array([sh, 0])
    kps[C.L_ELB] = sho + np.array([-sh - 8, 70]); kps[C.R_ELB] = sho + np.array([sh + 8, 70])
    kps[C.L_WRI] = np.array([sho[0] - sh - 5, wri_y])
    kps[C.R_WRI] = np.array([sho[0] + sh + 5, wri_y])
    kps[C.L_HIP] = hip + np.array([-32, 0]); kps[C.R_HIP] = hip + np.array([32, 0])
    kps[C.L_KNE], kps[C.R_KNE] = knee_l, knee_r
    kps[C.L_ANK], kps[C.R_ANK] = ank_l, ank_r

    kps += rng.normal(0, 1.5, kps.shape)        # 检测噪声
    conf = np.clip(rng.normal(0.92, 0.03, C.NUM_KP), 0.5, 0.99)
    return kps.astype(np.float32), conf.astype(np.float32)


def bbox_of(kps: np.ndarray) -> tuple[float, float, float, float]:
    x1, y1 = kps.min(axis=0)
    x2, y2 = kps.max(axis=0)
    pad_x, pad_y = (x2 - x1) * 0.18, (y2 - y1) * 0.10
    return (float(x1 - pad_x), float(y1 - pad_y), float(x2 + pad_x), float(y2 + pad_y))


def header() -> list[str]:
    h = ["frame", "time", "label", "x1", "y1", "x2", "y2", "img_w", "img_h"]
    for n in C.KP_NAMES:
        h += [f"{n}_x", f"{n}_y", f"{n}_c"]
    return h + list(C.FRAME_FEATURE_NAMES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--segs", type=int, default=3)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default=str(C.ROOT / "data" / "demo"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    n_frames = int(args.secs * args.fps)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for label in C.LABEL_NAMES:
        for seg in range(args.segs):
            path = out / f"demo_{label}_{seg}_{stamp}.csv"
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(header())
                for i in range(n_frames):
                    t = i / args.fps
                    kps, conf = synth(label, t, rng)
                    box = bbox_of(kps)
                    row = [i, f"{t:.3f}", label,
                           *[f"{v:.1f}" for v in box], W, H]
                    for k in range(C.NUM_KP):
                        row += [f"{kps[k][0]:.1f}", f"{kps[k][1]:.1f}", f"{conf[k]:.3f}"]
                    # 特征列留空，训练时会从关键点重算
                    row += [""] * len(C.FRAME_FEATURE_NAMES)
                    w.writerow(row)
            print(f"[demo] {path.name}  {n_frames} 帧")

    print(f"\n合成数据已写入 {out}")
    print("接着跑：")
    print("  python train.py --raw data/demo")
    print("  python report.py --raw data/demo --all")


if __name__ == "__main__":
    main()
