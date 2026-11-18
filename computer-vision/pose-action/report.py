"""离线报告：对录好的片段回放分析，生成可写进报告的图。

用法
----
    python report.py                  # 分析 raw 里最新的一段
    python report.py --all            # 分析全部并汇总
    python report.py --file data/raw/sit_20261118_170000.csv

每段输出一张三联图：
    上  行为判定时间轴（真值 vs 预测）
    中  关键力学量曲线（躯干倾角 / 膝角 / 髋高比）——解释"为什么这么判"
    下  髋中心在画面中的轨迹——坐着几乎不动，走路拉出一条线
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

import config as C
from dataset import load_recording
from features import window_features

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = {"stand": "#4C9AFF", "sit": "#FF8A3D",
          "walk": "#37C978", "squat": "#B06AE8"}


def _smooth(y, k: int = 7):
    """滑动平均，只用于画图展示，不参与判定。"""
    y = np.asarray(y, dtype=float)
    if len(y) < k or k < 3:
        return y
    pad = k // 2
    yp = np.pad(y, pad, mode="edge")
    return np.convolve(yp, np.ones(k) / k, mode="valid")[:len(y)]


def predict_segment(clf, rec, win_sec: float = C.WINDOW_SEC,
                    stride_sec: float = C.STRIDE_SEC,
                    min_frames: int = C.MIN_FRAMES):
    """按时间滑窗预测，返回 (窗口中心时间, 预测标签索引)。"""
    ts = rec["ts"]
    if len(ts) < min_frames:
        return np.array([]), np.array([])

    t_end = float(ts[-1])
    times, preds = [], []
    s = 0.0
    while s + win_sec <= t_end + 1e-9:
        i0 = int(np.searchsorted(ts, s, side="left"))
        i1 = int(np.searchsorted(ts, s + win_sec, side="right"))
        if i1 - i0 >= min_frames and rec["valid"][i0:i1].mean() >= 0.8:
            x = window_features(rec["feat"][i0:i1], rec["norm"][i0:i1]).reshape(1, -1)
            times.append(float(ts[i0] + win_sec / 2))
            preds.append(int(np.argmax(clf.predict_proba(x)[0])))
        s += stride_sec
    return np.array(times), np.array(preds)


def plot_segment(rec, times, preds, out_path: Path):
    idx = {n: i for i, n in enumerate(C.FRAME_FEATURE_NAMES)}
    ts = rec["ts"]
    true_idx = C.LABEL_NAMES.index(rec["label"])

    fig, axes = plt.subplots(3, 1, figsize=(11, 10.5), dpi=130,
                             gridspec_kw={"height_ratios": [1.0, 1.5, 1.55]})
    fig.suptitle(f"{rec['source']}   真值={C.LABEL_ZH[rec['label']]}   "
                 f"时长={ts[-1]:.1f}s", fontsize=12)

    # 1) 行为时间轴
    ax = axes[0]
    ax.axhspan(true_idx - 0.4, true_idx + 0.4, color=COLORS[rec["label"]],
               alpha=0.20, label="真值区间")
    for k, name in enumerate(C.LABEL_NAMES):
        ax.axhline(k, color="#CCCCCC", lw=0.8, ls="--")
    if len(times):
        ax.step(times, preds, where="mid", color="#D64545", lw=1.8, label="预测")
        ax.scatter(times, preds, s=10, color="#D64545", zorder=3)
    ax.set_yticks(range(len(C.LABEL_NAMES)))
    ax.set_yticklabels([C.LABEL_ZH[n] for n in C.LABEL_NAMES])
    ax.set_ylim(-0.6, len(C.LABEL_NAMES) - 0.4)
    ax.set_xlim(0, max(ts[-1], 1))
    ax.set_title("行为判定时间轴", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    # 2) 力学量曲线
    ax = axes[1]
    ax.plot(ts, _smooth(rec["feat"][:, idx["torso_angle"]]), color="#7B61FF",
            lw=1.4, label="躯干倾角(°)")
    ax.plot(ts, _smooth(rec["feat"][:, idx["knee_angle_L"]]), color="#2FA8CC",
            lw=1.4, label="左膝角(°)")
    ax.plot(ts, _smooth(rec["feat"][:, idx["knee_angle_R"]]), color="#2FA8CC",
            lw=1.1, alpha=0.55, ls="--", label="右膝角(°)")
    ax.plot(ts, _smooth(rec["feat"][:, idx["hip_height_ratio"]] * 180), color="#8C6D31",
            lw=1.3, alpha=0.85, label="髋高比×180")
    ax.set_xlim(0, max(ts[-1], 1))
    ax.set_ylim(-10, 205)   # 固定量程，便于不同片段之间直接对比
    ax.set_title("判定依据：关节角度与髋部高度（坐姿 → 膝角骤降、髋高比上升）",
                 fontsize=10, pad=18)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4,
              fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    ax.set_xlabel("时间 (s)")

    # 3) 轨迹
    ax = axes[2]
    cx = rec["feat"][:, idx["center_x"]]
    cy = rec["feat"][:, idx["center_y"]]
    ax.plot(cx, cy, color="#BBBBBB", lw=0.8, alpha=0.8, zorder=1)
    colors_t = plt.cm.viridis(np.linspace(0, 1, len(cx)))
    ax.scatter(cx, cy, c=colors_t, s=7, zorder=2)
    # 起点/终点标记，替代色条，避免挤压坐标区宽度
    ax.scatter([cx[0]], [cy[0]], s=90, marker="o", facecolor="none",
               edgecolor="#2E7D32", lw=2, zorder=3)
    ax.annotate("起点", (cx[0], cy[0]), textcoords="offset points",
                xytext=(6, -18), fontsize=8, color="#2E7D32")
    ax.scatter([cx[-1]], [cy[-1]], s=110, marker=">", color="#C62828", zorder=3)
    ax.annotate("终点", (cx[-1], cy[-1]), textcoords="offset points",
                xytext=(8, -12), fontsize=8, color="#C62828")

    # 等比例显示，否则纵向的微小起伏会被自动放大成"抖动假象"
    dx = np.nanmax(cx) - np.nanmin(cx)
    dy = np.nanmax(cy) - np.nanmin(cy)
    span = max(dx, dy, 0.04) * 1.45
    ax.set_xlim(np.nanmean(cx) - span / 2, np.nanmean(cx) + span / 2)
    ax.set_ylim(np.nanmean(cy) + span / 2, np.nanmean(cy) - span / 2)  # y 轴向下
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("画面横向归一化位置")
    ax.set_ylabel("画面纵向归一化位置")
    ax.set_title(f"髋中心轨迹（横向 {dx*100:.1f}% × 纵向 {dy*100:.1f}% 画面）",
                 fontsize=10)
    ax.grid(alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--clf", default=str(C.MODEL_DIR / "pose_best.pkl"))
    ap.add_argument("--raw", default=str(C.RAW_DIR), help="采集数据目录")
    args = ap.parse_args()

    if not Path(args.clf).exists():
        print(f"找不到模型 {args.clf}，先跑 train.py")
        return
    clf = joblib.load(args.clf)

    if args.file:
        files = [Path(args.file)]
    else:
        files = sorted(Path(args.raw).glob("*.csv"))
        if not args.all and files:
            files = [files[-1]]

    if not files:
        print("raw 目录里没有采集文件")
        return

    y_true, y_pred = [], []
    for p in files:
        rec = load_recording(p)
        if not rec:
            print(f"[跳过] {p.name} 空文件")
            continue
        times, preds = predict_segment(clf, rec)
        if len(preds) == 0:
            print(f"[跳过] {p.name} 没有有效窗口")
            continue

        t_idx = C.LABEL_NAMES.index(rec["label"])
        acc = accuracy_score([t_idx] * len(preds), preds)
        print(f"{p.name:38s} 真值={rec['label']:5s} 窗口={len(preds):4d} 段内准确率={acc:.3f}")

        y_true += [t_idx] * len(preds)
        y_pred += list(preds)

        out = C.REPORT_DIR / f"report_{p.stem}.png"
        plot_segment(rec, times, preds, out)
        print(f"    -> {out}")

    if len(files) > 1 and y_true:
        print("\n" + "=" * 56)
        print(f"总体窗口准确率 {accuracy_score(y_true, y_pred):.4f}"
              f"  （{len(y_true)} 个窗口）")
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(C.LABEL_NAMES))))
        print("混淆矩阵（行=真值，列=预测）：")
        print("            " + "  ".join(f"{n:>6s}" for n in C.LABEL_NAMES))
        for i, n in enumerate(C.LABEL_NAMES):
            print(f"  {n:>8s}  " + "  ".join(f"{v:6d}" for v in cm[i]))


if __name__ == "__main__":
    main()
