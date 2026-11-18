"""数据集构建：读采集 CSV -> 重算特征 -> 滑窗切样本。

为什么不直接用 CSV 里存好的特征列？
因为录制时偶发丢帧会让某些列出现空值，这里统一做前向填充 +
无效帧剔除，保证训练、推理、离线分析三条链路的数值口径完全一致。
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from features import frame_features, normalize_keypoints, window_features


def load_recording(path: Path, img_w: int | None = None,
                   img_h: int | None = None) -> dict:
    """读一段采集，返回逐帧特征与归一化坐标。

    返回 dict:
        feat  (T,14) float32  逐帧特征，已做前后向填充
        norm  (T,34) float32  归一化关键点
        ts    (T,)   秒
        valid (T,)   bool     该帧是否检测到人
        label str
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    if len(df) == 0:
        return {}

    label = str(df["label"].iloc[0])
    T = len(df)

    # 优先用录制时记录的真实分辨率，保证 center_x/center_y 口径一致
    if "img_w" in df.columns and df["img_w"].notna().any():
        img_w = int(df["img_w"].dropna().iloc[0])
    if "img_h" in df.columns and df["img_h"].notna().any():
        img_h = int(df["img_h"].dropna().iloc[0])
    img_w = img_w or 1280
    img_h = img_h or 720

    kp = np.full((T, C.NUM_KP, 2), np.nan, dtype=np.float32)
    cf = np.zeros((T, C.NUM_KP), dtype=np.float32)
    for i, name in enumerate(C.KP_NAMES):
        kp[:, i, 0] = df[f"{name}_x"].to_numpy(dtype=np.float32)
        kp[:, i, 1] = df[f"{name}_y"].to_numpy(dtype=np.float32)
        cf[:, i] = df[f"{name}_c"].fillna(0).to_numpy(dtype=np.float32)

    boxes = df[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
    ts = df["time"].to_numpy(dtype=np.float64)
    ts = ts - ts[0]

    feat = np.full((T, len(C.FRAME_FEATURE_NAMES)), np.nan, dtype=np.float32)
    norm = np.full((T, C.NUM_KP * 2), np.nan, dtype=np.float32)
    valid = np.zeros(T, dtype=bool)
    prev_center = None
    prev_t = 0.0

    for t in range(T):
        if not np.all(np.isfinite(boxes[t])):
            continue
        dt = max(ts[t] - prev_t, 1e-3)
        f, center = frame_features(kp[t], cf[t], tuple(boxes[t]),
                                  prev_center=prev_center, dt=dt,
                                  img_w=img_w, img_h=img_h)
        if f is None:
            continue
        feat[t] = f
        norm[t] = normalize_keypoints(kp[t], cf[t], tuple(boxes[t]))
        valid[t] = True
        if center is not None:
            prev_center = center
            prev_t = ts[t]

    # 填充：先前向再后向，端点用最近有效值
    if valid.any():
        first_valid = int(np.argmax(valid))
        feat[:first_valid] = feat[first_valid]
        feat = pd.DataFrame(feat).ffill().bfill().to_numpy(dtype=np.float32)
        norm = pd.DataFrame(norm).ffill().bfill().fillna(0).to_numpy(dtype=np.float32)
    else:
        feat = np.nan_to_num(feat)
        norm = np.nan_to_num(norm)

    return dict(feat=feat, norm=norm, ts=ts, valid=valid, label=label,
                source=path.name)


def slide_windows(rec: dict, win_sec: float = C.WINDOW_SEC,
                  stride_sec: float = C.STRIDE_SEC,
                  min_valid_ratio: float = 0.8, min_frames: int = C.MIN_FRAMES):
    """按时间切窗口，返回 (X, 有效标记)。

    按时间而不是按帧数：帧率会随机器性能变化（4 核 CPU 约 11 FPS），
    固定"30 帧"在不同机器上覆盖的行为时长完全不同，换台电脑模型就废了。
    改成"1.5 秒"，换机器、换分辨率都不用重新标定。
    """
    ts = rec["ts"]
    if len(ts) < min_frames:
        return np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=bool)

    t_end = float(ts[-1])
    X, keep = [], []
    s = 0.0
    while s + win_sec <= t_end + 1e-9:
        i0 = int(np.searchsorted(ts, s, side="left"))
        i1 = int(np.searchsorted(ts, s + win_sec, side="right"))
        if i1 - i0 >= min_frames:
            X.append(window_features(rec["feat"][i0:i1], rec["norm"][i0:i1]))
            keep.append(bool(rec["valid"][i0:i1].mean() >= min_valid_ratio))
        s += stride_sec

    if not X:
        return np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=bool)
    return np.asarray(X, dtype=np.float32), np.asarray(keep, dtype=bool)


def build_dataset(raw_dir: Path = C.RAW_DIR, win_sec: float = C.WINDOW_SEC,
                  stride_sec: float = C.STRIDE_SEC):
    """遍历所有采集文件，构建 (X, y, groups, rec_index)。

    groups = 文件名，用于按"段"分组交叉验证 —— 这点很关键：
    同一段的相邻窗口几乎一样，随机切分会让准确率虚高 10~20 个点。
    """
    Xs, ys, gs, recs, skipped = [], [], [], [], []
    files = sorted(raw_dir.glob("*.csv"))
    for p in files:
        rec = load_recording(p)
        if not rec:
            skipped.append((p.name, "空文件"))
            continue
        if rec["label"] not in C.LABEL_NAMES:
            skipped.append((p.name, f"未知标签 {rec['label']}"))
            continue
        X, keep = slide_windows(rec, win_sec=win_sec, stride_sec=stride_sec)
        if len(X) == 0:
            skipped.append((p.name, "长度不足一个窗口"))
            continue
        n_ok = int(keep.sum())
        if n_ok == 0:
            skipped.append((p.name, "没有检测到人的有效帧"))
            continue
        recs.append((p.name, rec["label"], len(rec["feat"]), n_ok))
        Xs.append(X[keep])
        ys += [C.LABEL_NAMES.index(rec["label"])] * n_ok
        gs += [p.name] * n_ok

    if not Xs:
        return None, None, None, recs, skipped

    X = np.vstack(Xs)
    y = np.array(ys, dtype=np.int64)
    g = np.array(gs)
    return X, y, g, recs, skipped


def summarize(recs) -> str:
    if not recs:
        return "（无可用数据）"
    cnt = Counter(r[1] for r in recs)
    lines = [f"共 {len(recs)} 段："]
    for name, c in cnt.items():
        lines.append(f"  {name:6s} {c} 段")
    return "\n".join(lines)
