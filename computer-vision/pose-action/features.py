"""特征工程：把 17 个像素关键点变成分类器能吃的时序特征。

设计原则
--------
1. 平移/尺度不变：以两髋中点为原点，以"肩髋距"为单位长度归一化。
   这样人走近、走远、站偏左右，特征都不变。
2. 物理可解释：每个特征都对应一个能写进报告的人体力学量
   （躯干倾角、髋膝角、高度比……），而不是让网络自己瞎学。
3. 时序信息靠统计量：走路的本质是"周期性"，所以窗口内取
   标准差、过零率、首尾差，而不是单帧分类。
"""
from __future__ import annotations

import warnings

import numpy as np

from config import (
    KP_CONF, NUM_KP,
    L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI,
    L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK, NOSE,
    FRAME_FEATURE_NAMES,
)

EPS = 1e-6


def _quiet(fn, arr, axis=0):
    """对全 NaN 的列取统计量会抛 RuntimeWarning，这里静默处理。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(arr, axis=axis)


# ---------------------------------------------------------------- 基础几何
def _vec_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def angle_at(vertex: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """以 vertex 为顶点，p1-vertex 与 p2-vertex 的夹角（度）。"""
    v1, v2 = p1 - vertex, p2 - vertex
    n1, n2 = _vec_norm(v1), _vec_norm(v2)
    if n1 < EPS or n2 < EPS:
        return np.nan
    c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _mid(kps: np.ndarray, i: int, j: int) -> np.ndarray:
    return (kps[i] + kps[j]) / 2.0


# ---------------------------------------------------------- 关键点归一化
def normalize_keypoints(kps: np.ndarray, conf: np.ndarray, bbox) -> np.ndarray:
    """返回 (34,) = 17 个点相对髋中心的归一化坐标；缺失点为 nan。"""
    out = np.full((NUM_KP, 2), np.nan, dtype=np.float32)
    hip_mid = _mid(kps, L_HIP, R_HIP)
    sho_mid = _mid(kps, L_SHO, R_SHO)

    scale = _vec_norm(sho_mid - hip_mid)
    if not np.isfinite(scale) or scale < 1e-3:
        # 肩髋不可靠时退回人体框高度
        scale = max(float(bbox[3] - bbox[1]), 1.0)

    for i in range(NUM_KP):
        if conf[i] >= KP_CONF:
            out[i] = (kps[i] - hip_mid) / scale
    return out.reshape(-1)


# -------------------------------------------------------------- 逐帧特征
def frame_features(kps: np.ndarray, conf: np.ndarray, bbox,
                   prev_center=None, dt: float = 1 / 30,
                   img_w: int = 640, img_h: int = 480):
    """输入一帧的像素关键点，输出 (14,) 特征向量与髋中心坐标。

    返回 (feat, center)；若有效关键点太少返回 (None, None)。
    """
    valid = conf >= KP_CONF
    # 双髋必须可见（髋中点是所有归一化的原点），且至少一条腿的膝盖可见。
    # 只露上半身时无法区分坐/站，这帧直接作废，由调用方提示用户后退。
    if not (valid[L_HIP] and valid[R_HIP] and (valid[L_KNE] or valid[R_KNE])):
        return None, None

    k = kps.copy()
    k[~valid] = np.nan

    x1, y1, x2, y2 = bbox
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)

    hip_mid = _mid(k, L_HIP, R_HIP)
    sho_mid = _mid(k, L_SHO, R_SHO)

    # 1) 躯干倾角：躯干向量与"竖直向上"的夹角，站立≈0，前倾/坐下会变大
    tv = sho_mid - hip_mid
    torso_angle = float(np.degrees(np.arctan2(abs(tv[0]), -tv[1]))) \
        if np.all(np.isfinite(tv)) else np.nan

    # 2) 躯干长度 / 框高：距离相机远近的代理量
    torso_len = _vec_norm(tv) / bh if np.all(np.isfinite(tv)) else np.nan

    # 3) 髋角（肩-髋-膝）与膝角（髋-膝-踝）：坐下时膝角从 ~175° 掉到 ~90°
    hip_angle_l = angle_at(k[L_HIP], k[L_SHO], k[L_KNE])
    hip_angle_r = angle_at(k[R_HIP], k[R_SHO], k[R_KNE])
    knee_angle_l = angle_at(k[L_KNE], k[L_HIP], k[L_ANK])
    knee_angle_r = angle_at(k[R_KNE], k[R_HIP], k[R_ANK])

    # 4) 髋部高度比：越接近 1 越靠画面底部（站立），坐下会升高
    hip_height_ratio = float((y2 - hip_mid[1]) / bh) if np.isfinite(hip_mid[1]) else np.nan

    # 5) 框宽高比：坐下/蹲下时身体横向展开
    bbox_aspect = bw / bh

    # 6) 两踝水平距离：并腿站 < 迈步走
    ankle_spread = float(abs(k[L_ANK][0] - k[R_ANK][0]) / bh) \
        if np.isfinite(k[L_ANK][0]) and np.isfinite(k[R_ANK][0]) else np.nan

    # 7) 手腕低于髋的幅度：行走摆臂、坐下手放腿上会变化
    wri_y = np.nanmean([k[L_WRI][1], k[R_WRI][1]])
    wrist_below_hip = float((wri_y - hip_mid[1]) / bh) if np.isfinite(wri_y) else np.nan

    # 8) 髋中心在画面中的归一化位置（用于轨迹图与速度）
    cx = float(hip_mid[0] / max(img_w, 1))
    cy = float(hip_mid[1] / max(img_h, 1))

    # 9) 帧间速度：以框高为单位，消除远近影响
    speed = np.nan
    if prev_center is not None and np.all(np.isfinite(hip_mid)):
        d = _vec_norm(hip_mid - prev_center)
        speed = float(d / bh / max(dt, 1e-3))

    # 10) 鼻到髋的垂直距离 / 框高：弯腰、坐下时会缩短
    head_hip_dy = float((hip_mid[1] - k[NOSE][1]) / bh) \
        if np.isfinite(k[NOSE][1]) and np.isfinite(hip_mid[1]) else np.nan

    # 11) 膝中点高出髋中点的幅度 / 框高
    #     蹲下时髋沉到膝盖以下（值为正），坐下时髋与膝同高或更高（值<=0）。
    #     这是把"蹲下"和"坐下"分开的关键量——两者膝角都接近 90°，
    #     但蹲下是"胯低于膝"，坐下是"胯落在椅面上、与膝齐平"。
    knee_mid_y = np.nanmean([k[L_KNE][1], k[R_KNE][1]])
    knee_above_hip = float((hip_mid[1] - knee_mid_y) / bh) \
        if np.isfinite(knee_mid_y) and np.isfinite(hip_mid[1]) else np.nan

    feat = np.array([
        torso_angle, torso_len,
        hip_angle_l, hip_angle_r, knee_angle_l, knee_angle_r,
        hip_height_ratio, bbox_aspect, ankle_spread, wrist_below_hip,
        cx, cy, speed, head_hip_dy, knee_above_hip,
    ], dtype=np.float32)

    assert feat.shape[0] == len(FRAME_FEATURE_NAMES), "特征维度与配置不符"
    return feat, hip_mid


# -------------------------------------------------------------- 窗口特征
def _zcr(series: np.ndarray) -> float:
    """去均值后的过零率，衡量周期性/抖动强度（走路时明显大于静止）。"""
    col = series.reshape(-1, 1)
    s = series - float(_quiet(np.nanmean, col)[0])
    sd = float(_quiet(np.nanstd, col)[0])
    if not np.isfinite(sd) or sd < 1e-9:
        return 0.0
    return float(np.mean(np.abs(np.diff(np.sign(s))) > 0))


def window_features(feat_win: np.ndarray, norm_win: np.ndarray) -> np.ndarray:
    """feat_win: (W,14) 帧特征; norm_win: (W,34) 归一化坐标。

    输出四组统计量拼成的定长向量：
      帧特征 mean/std/min/max/首尾差   -> 14*5 = 70
      归一化坐标 std                  -> 34   （动作幅度）
      归一化坐标 过零率                -> 34   （周期性，走 vs 站的核心）
    """
    f_mean = _quiet(np.nanmean, feat_win)
    f_std = _quiet(np.nanstd, feat_win)
    f_min = _quiet(np.nanmin, feat_win)
    f_max = _quiet(np.nanmax, feat_win)
    f_delta = feat_win[-1] - feat_win[0]

    n_std = _quiet(np.nanstd, norm_win)
    n_zcr = np.array([_zcr(norm_win[:, j]) for j in range(norm_win.shape[1])],
                     dtype=np.float32)

    out = np.concatenate([
        np.nan_to_num(f_mean, nan=0.0), np.nan_to_num(f_std, nan=0.0),
        np.nan_to_num(f_min, nan=0.0), np.nan_to_num(f_max, nan=0.0),
        np.nan_to_num(f_delta, nan=0.0),
        np.nan_to_num(n_std, nan=0.0), n_zcr,
    ]).astype(np.float32)
    return out
