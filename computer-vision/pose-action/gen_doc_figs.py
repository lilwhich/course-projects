"""生成讲解文档用的示意图（全部矢量化绘至 PNG，白底，适合 Word 排版）。

跑法：.venv/Scripts/python.exe gen_doc_figs.py
输出：reports/doc/*.png
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

import config as C

OUT = Path("reports/doc")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

NAVY = "#1f4e79"
BLUE = "#2e86c1"
LBLUE = "#d6e6f5"
ORANGE = "#e67e22"
GREEN = "#27ae60"
RED = "#c0392b"
GRAY = "#7f8c8d"
LGRAY = "#eceff1"
INK = "#1c2833"


def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white",
                pad_inches=0.16)
    plt.close(fig)
    print("  ->", p, f"{p.stat().st_size/1024:.0f} KB")


# --------------------------------------------------------------- 人体坐标
# 归一化人体：画布 0~100 宽，0~160 高，头顶 160、脚踝 8
KP = {
    "nose": (50.0, 143.5), "left_eye": (46.6, 152.0), "right_eye": (53.4, 152.0),
    "left_ear": (41.8, 148.5), "right_ear": (58.2, 148.5),
    "left_shoulder": (38.0, 126.0), "right_shoulder": (62.0, 126.0),
    "left_elbow": (31.0, 104.0), "right_elbow": (69.0, 104.0),
    "left_wrist": (27.0, 82.0), "right_wrist": (73.0, 82.0),
    "left_hip": (42.5, 86.0), "right_hip": (57.5, 86.0),
    "left_knee": (41.5, 47.0), "right_knee": (58.5, 47.0),
    "left_ankle": (40.5, 8.0), "right_ankle": (59.5, 8.0),
}
NAME_ZH = ["鼻", "左眼", "右眼", "左耳", "右耳", "左肩", "右肩", "左肘",
           "右肘", "左腕", "右腕", "左髋", "右髋", "左膝", "右膝", "左踝", "右踝"]
IDX = {n: i for i, n in enumerate(C.KP_NAMES)}


def P(name):
    return KP[name]


def body(ax, kp, scale_x=1.0, cx=50.0, dy=0.0, color=LGRAY, edge=GRAY):
    """画一个简笔人体（躯干+头+四肢，用粗线）。"""
    def q(name):
        x, y = kp[name]
        return ((x - cx) * scale_x + cx, y + dy)

    torso = np.array([q("left_shoulder"), q("left_hip"), q("right_hip"),
                      q("right_shoulder"), q("left_shoulder")])
    ax.plot(torso[:, 0], torso[:, 1], color=edge, lw=1.4, zorder=2)
    ax.fill(torso[:, 0], torso[:, 1], color=color, zorder=1)
    hx, hy = q("nose")
    ax.add_patch(Circle((hx, hy + 8.5), 8.5, facecolor=color, edgecolor=edge,
                        lw=1.4, zorder=1))
    for a, b in [("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
                 ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
                 ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
                 ("right_hip", "right_knee"), ("right_knee", "right_ankle")]:
        (x1, y1), (x2, y2) = q(a), q(b)
        ax.plot([x1, x2], [y1, y2], color=edge, lw=11, solid_capstyle="round",
                zorder=1, alpha=.9)
        ax.plot([x1, x2], [y1, y2], color=color, lw=8.4, solid_capstyle="round",
                zorder=2)


# ============================================================ 图1 关键点拓扑
# 每个点的中文标签偏移（头部 5 点太密，只标序号，名字在右侧表里查）
LABEL_OFF = {
    "left_shoulder": (-7.5, 3.0), "right_shoulder": (7.5, 3.0),
    "left_elbow": (-7.5, 0.0), "right_elbow": (7.5, 0.0),
    "left_wrist": (-7.5, -1.0), "right_wrist": (7.5, -1.0),
    "left_hip": (-7.5, 2.0), "right_hip": (7.5, 2.0),
    "left_knee": (-7.5, 0.0), "right_knee": (7.5, 0.0),
    "left_ankle": (-7.5, 0.0), "right_ankle": (7.5, 0.0),
}
HEAD = {"nose", "left_eye", "right_eye", "left_ear", "right_ear"}


def fig_skeleton():
    fig, ax = plt.subplots(figsize=(12.6, 6.2))
    ax.set_xlim(-16, 142)
    ax.set_ylim(-6, 168)
    ax.set_aspect("equal")
    ax.axis("off")

    # 骨架连线
    for i, j in C.SKELETON:
        a, b = P(C.KP_NAMES[i]), P(C.KP_NAMES[j])
        ax.plot([a[0], b[0]], [a[1], b[1]], color=BLUE, lw=3.0,
                solid_capstyle="round", zorder=2, alpha=.85)

    for i, nm in enumerate(C.KP_NAMES):
        x, y = P(nm)
        ax.add_patch(Circle((x, y), 3.1, facecolor="white", edgecolor=NAVY,
                            lw=2.1, zorder=4))
        ax.text(x, y, str(i), ha="center", va="center", fontsize=8.2,
                color=NAVY, zorder=5, fontweight="bold")
        if nm in HEAD:
            continue
        dx, dy = LABEL_OFF[nm]
        ax.text(x + dx, y + dy, NAME_ZH[i], ha="right" if dx < 0 else "left",
                va="center", fontsize=9.4, color=INK, zorder=5)

    # 右侧对照表（单列，避免英文名与下一列撞车）
    ax.text(92, 158, "COCO-17 关键点编号", fontsize=12.5, color=NAVY,
            fontweight="bold")
    ax.plot([92, 140], [154.5, 154.5], color=NAVY, lw=1.4)
    for i, nm in enumerate(C.KP_NAMES):
        y = 147.5 - i * 8.0
        ax.add_patch(Circle((94, y), 2.5, facecolor=LBLUE, edgecolor=BLUE,
                            lw=1.1, zorder=3))
        ax.text(94, y, str(i), ha="center", va="center", fontsize=7.4, color=NAVY)
        ax.text(99, y, NAME_ZH[i], ha="left", va="center", fontsize=9.2, color=INK)
        ax.text(110, y, nm, ha="left", va="center", fontsize=8.0, color=GRAY)
    save(fig, "fig1_关键点拓扑.png")


# ========================================================== 图2 判定几何量
def _angle(ax, vertex, p1, p2, r, text, label_xy, color=ORANGE, fs=10.5):
    """在 vertex 处画 p1-vertex-p2 的角度弧，文字放在 label_xy。"""
    v1 = np.array(p1, float) - vertex
    v2 = np.array(p2, float) - vertex
    a1 = float(np.degrees(np.arctan2(v1[1], v1[0])))
    a2 = float(np.degrees(np.arctan2(v2[1], v2[0])))
    if a2 - a1 > 180:
        a2 -= 360
    elif a1 - a2 > 180:
        a1 -= 360
    if a1 > a2:
        a1, a2 = a2, a1
    ax.add_patch(Arc(vertex, r * 2, r * 2, theta1=a1, theta2=a2,
                     color=color, lw=2.0, zorder=6))
    if text:
        ax.text(label_xy[0], label_xy[1], text, color=color, fontsize=fs,
                ha="center", va="center", fontweight="bold", zorder=9)


def side_body(ax, sho, hip, knee, ank, elb, wri, color=GRAY, demo=None):
    """画侧视简笔人体：头 + 躯干 + 一条腿 + 一条手臂。"""
    sho, hip = np.array(sho, float), np.array(hip, float)
    knee, ank = np.array(knee, float), np.array(ank, float)
    elb, wri = np.array(elb, float), np.array(wri, float)

    # 躯干（肩髋连线的粗带）
    t = sho - hip
    d = t / max(np.linalg.norm(t), 1e-6)
    n = np.array([-d[1], d[0]]) * 8.0
    quad = [sho + n, sho - n, hip - n, hip + n]
    ax.add_patch(plt.Polygon(quad, closed=True, facecolor=LGRAY,
                             edgecolor=GRAY, lw=1.3, zorder=2))
    # 手臂
    ax.plot([sho[0], elb[0], wri[0]], [sho[1], elb[1], wri[1]],
            color=GRAY, lw=8.0, solid_capstyle="round", zorder=2)
    ax.plot([sho[0], elb[0], wri[0]], [sho[1], elb[1], wri[1]],
            color="white", lw=4.2, solid_capstyle="round", alpha=.55, zorder=3)
    # 腿
    ax.plot([hip[0], knee[0], ank[0]], [hip[1], knee[1], ank[1]],
            color=GRAY, lw=12.0, solid_capstyle="round", zorder=2)
    ax.plot([hip[0], knee[0], ank[0]], [hip[1], knee[1], ank[1]],
            color="#fdfefe", lw=7.0, solid_capstyle="round", alpha=.5, zorder=3)
    # 脚
    ax.plot([ank[0] - 2, ank[0] + 9], [ank[1], ank[1]], color=GRAY, lw=5.0,
            solid_capstyle="round", zorder=2)
    # 头（紧贴肩部）
    hc = sho + d * 19.0
    ax.add_patch(Circle(hc, 10.5, facecolor=LGRAY, edgecolor=GRAY, lw=1.4,
                        zorder=2))

    if demo is None:
        demo = {}
    if "shoulder" in demo:
        ax.add_patch(Circle(sho, 2.4, facecolor=demo["shoulder"],
                            edgecolor="white", lw=1.0, zorder=8))
    if "hip" in demo:
        ax.add_patch(Circle(hip, 2.4, facecolor=demo["hip"],
                            edgecolor="white", lw=1.0, zorder=8))
    if "knee" in demo:
        ax.add_patch(Circle(knee, 2.4, facecolor=demo["knee"],
                            edgecolor="white", lw=1.0, zorder=8))
    if "ankle" in demo:
        ax.add_patch(Circle(ank, 2.4, facecolor=demo["ankle"],
                            edgecolor="white", lw=1.0, zorder=8))
    return hc


def fig_geometry():
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.9))
    fig.subplots_adjust(wspace=.04, bottom=.16)

    # 三组侧视姿态：脚踝都落在地面 y=8
    poses = [
        dict(name="① 站立 stand", color=GREEN,
             sho=(50, 140), hip=(50, 92), knee=(50, 50), ank=(50, 8),
             elb=(50, 112), wri=(50, 84),
             knee_t="膝角 ≈180°", knee_xy=(76, 36),
             hip_t="髋角 ≈180°", hip_xy=(26, 86),
             torso_t="躯干角 ≈0°", torso_xy=(30, 152),
             note="膝角→180°\n髋高比 0.50\n膝高于髋 −0.42"),
        dict(name="② 坐下 sit", color=ORANGE,
             sho=(24, 100), hip=(24, 54), knee=(66, 54), ank=(66, 8),
             elb=(30, 78), wri=(46, 64),
             knee_t="膝角 ≈90°", knee_xy=(44, 36),
             hip_t="髋角 ≈95°", hip_xy=(22, 26),
             torso_t="躯干角 ≈8°", torso_xy=(44, 116),
             note="膝角→90°\n髋高比 0.23\n膝高于髋 +0.00"),
        dict(name="③ 蹲下 squat", color="#8e44ad",
             sho=(41, 96), hip=(22, 52), knee=(62, 58), ank=(64, 8),
             elb=(42, 72), wri=(54, 58),
             knee_t="膝角 ≈82°", knee_xy=(44, 32),
             hip_t="髋角 ≈55°", hip_xy=(18, 22),
             torso_t="躯干角 ≈24°", torso_xy=(62, 104),
             note="膝角→82°\n髋高比 0.15\n膝高于髋 +0.07"),
    ]

    for ax, po in zip(axes, poses):
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 176)
        ax.set_aspect("equal")
        ax.axis("off")
        demo = {"shoulder": po["color"], "hip": po["color"],
                "knee": po["color"], "ankle": po["color"]}
        side_body(ax, po["sho"], po["hip"], po["knee"], po["ank"],
                  po["elb"], po["wri"], demo=demo)

        _angle(ax, np.array(po["knee"], float), po["hip"], po["ank"], 13,
               po["knee_t"], po["knee_xy"], ORANGE)
        _angle(ax, np.array(po["hip"], float), po["sho"], po["knee"], 13,
               po["hip_t"], po["hip_xy"], BLUE)

        # 躯干角：竖直参考线 + 躯干轴线
        sha, hia = np.array(po["sho"], float), np.array(po["hip"], float)
        ax.plot([hia[0], hia[0]], [hia[1], sha[1] + 8], ls=(0, (4, 3)),
                color=GRAY, lw=1.2, zorder=1)
        ax.plot([hia[0], sha[0]], [hia[1], sha[1]], color=RED, lw=2.4, zorder=7)
        ax.text(po["torso_xy"][0], po["torso_xy"][1], po["torso_t"],
                color=RED, fontsize=10.2, ha="center", va="center",
                fontweight="bold", zorder=9)

        ax.plot([2, 98], [8, 8], color="#b3b6b7", lw=1.6, zorder=0)
        ax.set_title(po["name"], fontsize=13, color=po["color"],
                     fontweight="bold", pad=2)
        ax.text(50, -9, po["note"], ha="center", va="top", fontsize=9.4,
                color=INK, linespacing=1.6)

    fig.text(.5, -.055,
             "躯干角＝肩髋连线与竖直方向夹角（红）　髋角＝肩-髋-膝（蓝）　膝角＝髋-膝-踝（橙）",
             ha="center", fontsize=9.6, color=GRAY)
    fig.text(.5, -.105,
             "注意：坐下与蹲下的膝角都接近 90°，光看膝角分不开；"
             "真正把两者分开的是髋高比（0.23 vs 0.15）与「膝高于髋」(−0.42 / 0.00 / +0.07)",
             ha="center", fontsize=9.4, color=RED)
    save(fig, "fig2_判定几何量.png")


# ========================================================== 图3 处理流水线
def _box(ax, x, y, w, h, title, sub, color=NAVY, fill=LBLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.6",
                                facecolor=fill, edgecolor=color, lw=1.8, zorder=2))
    ax.text(x + w / 2, y + h * .62, title, ha="center", va="center",
            fontsize=11.5, color=color, fontweight="bold", zorder=3)
    ax.text(x + w / 2, y + h * .26, sub, ha="center", va="center",
            fontsize=8.8, color=INK, zorder=3)


def _arrow(ax, x1, y1, x2, y2, color=NAVY, text=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=17, color=color, lw=1.9,
                                 zorder=1, shrinkA=0, shrinkB=0))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 2.6, text, ha="center",
                va="bottom", fontsize=9.0, color=INK, zorder=4,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none"))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12.2, 4.4))
    ax.set_xlim(0, 122)
    ax.set_ylim(0, 46)
    ax.axis("off")

    _box(ax, 1, 17, 17, 15, "① 采集", "手机 1920×1080\nHTTP-MJPEG 流", BLUE)
    _box(ax, 25, 17, 18, 15, "② 检测", "YOLOv8n-Pose\nimgsz 480 / conf 0.4")
    _box(ax, 50, 17, 17, 15, "③ 关键点", "17 点 x,y,c\n置信度过滤 0.3")
    _box(ax, 74, 17, 17, 15, "④ 特征", "逐帧 15 维\n几何量", GREEN)
    _box(ax, 98, 17, 21, 15, "⑤ 时序判定", "1.5s 窗口 143 维\n随机森林 → 四类", ORANGE)

    for x1, x2, t in [(18.6, 24.4, "帧"), (43.6, 49.4, "bbox+17点"),
                      (67.6, 73.4, "坐标"), (91.6, 97.4, "窗口")]:
        _arrow(ax, x1, 24.5, x2, 24.5, text=t)

    ax.text(61, 38.5, "PoseAction 处理流水线：从手机画面到行为判定",
            ha="center", fontsize=14, color=NAVY, fontweight="bold")
    ax.text(61, 6.5,
            "全流程单帧耗时 ≈ 80 ms（含检测 45 ms + 特征与判定 5 ms + 绘制 30 ms）　"
            "输出帧率 ≈ 12 FPS　画面延迟 < 30 ms",
            ha="center", fontsize=9.6, color=GRAY)
    save(fig, "fig3_处理流水线.png")


# ============================================================ 图4 滑窗时序
def fig_window():
    fig = plt.figure(figsize=(12.2, 4.2))
    ax = fig.add_axes([0.05, 0.05, 0.93, 0.86])
    ax.set_xlim(-0.2, 5.9)
    ax.set_ylim(-2.0, 1.55)
    ax.axis("off")

    # 时间轴
    ax.annotate("", xy=(5.65, -1.25), xytext=(-0.1, -1.25),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ax.text(5.72, -1.25, "时间 (s)", fontsize=9.5, color=INK, va="center")
    for t in np.arange(0, 5.6, 0.1):
        ax.plot([t, t], [-1.25, -1.14], color=GRAY, lw=.7)
    for t in np.arange(0, 5.6, 0.5):
        ax.plot([t, t], [-1.25, -1.09], color=GRAY, lw=1.1)
        ax.text(t, -1.40, f"{t:.1f}", fontsize=8, color=GRAY, ha="center")

    wins = [(0.0, 1.5, "#cfe3f6", "第 1 次判定"),
            (0.6, 2.1, "#a9cbe8", ""),
            (1.2, 2.7, "#7fb2dc", "第 2 次判定"),
            (2.4, 3.9, "#4f93c9", "第 3 次判定"),
            (3.6, 5.1, "#3a7fb8", "第 4 次判定")]
    for k, (s0, e0, c, lab) in enumerate(wins):
        y = 0.52 - k * 0.30
        ax.add_patch(Rectangle((s0, y), e0 - s0, 0.22, facecolor=c,
                               edgecolor=NAVY, lw=1.0, alpha=.96))
        ax.text(s0 + 0.05, y + 0.11, f"{s0:.1f}~{e0:.1f}s", fontsize=7.8,
                va="center", color=INK)
        if lab:
            ax.text(e0 + 0.07, y + 0.11, f"{lab}", fontsize=9.0,
                    va="center", color=NAVY)

    # 步长标注
    ax.annotate("", xy=(0.0, -0.62), xytext=(0.6, -0.62),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.6))
    ax.plot([0.0, 0.0], [-0.62, -0.10], color=ORANGE, lw=.9, ls=":")
    ax.plot([0.6, 0.6], [-0.62, -0.10], color=ORANGE, lw=.9, ls=":")
    ax.text(0.3, -0.78, "步长 0.3s", fontsize=9.0, color=ORANGE, ha="center")

    ax.text(-0.15, 1.38,
            "① 滑窗切分：窗口固定 1.5 秒，每 0.3 秒前移一次，每个窗口出一次判定",
            fontsize=11.0, color=NAVY, fontweight="bold")
    ax.text(-0.15, -1.92,
            "② 每窗统计：15 维帧特征各取 mean/std/min/max/首尾差 = 75 维，"
            "＋34 维归一化坐标 std（幅度）＋34 维过零率（周期性）= 143 维",
            fontsize=9.6, color=INK)
    save(fig, "fig4_滑窗时序.png")


# ======================================================= 图5 三类特征对比
FEATS = ["knee_angle_L", "hip_height_ratio", "ankle_spread", "torso_angle"]
TITLES = ["左膝角 (度)", "髋高比 (髋高/框高)", "踝间距 (除以框高)", "躯干倾角 (度)"]
COLORS = {"stand": GREEN, "sit": ORANGE, "walk": BLUE,
          "squat": "#8e44ad"}


def fig_curves():
    """从采集文件的关键点重算逐帧特征再画曲线（CSV 里的特征列是空的）。"""
    import dataset as D

    series = {k: {f: [] for f in FEATS} for k in C.LABEL_NAMES}
    times = {k: [] for k in C.LABEL_NAMES}
    files = list(Path("data/demo").glob("demo_*_0_*.csv"))
    # 同一批次的文件名带相同时间戳，只取最新那批，避免新旧数据叠在一起
    stamps = sorted({p.stem.split("_")[-1] for p in files})
    latest = stamps[-1] if stamps else ""

    col = [C.FRAME_FEATURE_NAMES.index(f) for f in FEATS]
    for path in sorted(files):
        if not path.stem.endswith(latest):
            continue
        lab = path.stem.split("_")[1]
        if lab not in series:
            continue
        rec = D.load_recording(path)
        if not rec:
            continue
        ts = np.asarray(rec["ts"][:560], dtype=float)
        feat = np.asarray(rec["feat"][:560], dtype=float)
        times[lab] = ts
        for j, f in enumerate(FEATS):
            series[lab][f] = feat[:, col[j]]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 6.6))
    fig.subplots_adjust(hspace=.46, wspace=.20)
    for ax, f, title in zip(axes.flat, FEATS, TITLES):
        for lab in C.LABEL_NAMES:
            y = np.asarray(series[lab][f], dtype=float)
            if not len(y):
                continue
            x = times[lab]
            ax.plot(x, y, color=COLORS[lab], lw=1.9,
                    label=f"{C.LABEL_ZH[lab]} {lab}")
            base = np.nanmin(y[np.isfinite(y)]) if np.isfinite(y).any() else 0
            ax.fill_between(x, y, base, color=COLORS[lab], alpha=.07)
        ax.set_title(title, fontsize=11.5, color=NAVY, fontweight="bold")
        ax.set_xlabel("时间 (s)", fontsize=9)
        ax.grid(alpha=.25, ls=":")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=9)
    h, l = axes[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=9.5,
               frameon=False, bbox_to_anchor=(.5, .945))
    fig.suptitle("四类动作的逐帧特征曲线（项目自带样例数据）",
                 fontsize=13.5, color=NAVY, fontweight="bold", y=.99)
    save(fig, "fig5_四个特征曲线.png")


# ======================================================= 图6 手机连接与机位
def fig_phone():
    fig = plt.figure(figsize=(12.6, 4.9))
    ax = fig.add_axes([0.02, 0.05, 0.55, 0.86])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")
    ax.text(50, 58, "① 手机取流链路", ha="center", fontsize=13,
            color=NAVY, fontweight="bold")

    # 手机
    ax.add_patch(FancyBboxPatch((3, 26), 17, 26,
                                boxstyle="round,pad=0.5,rounding_size=2.2",
                                facecolor="#34495e", edgecolor="#2c3e50", lw=1.5))
    ax.add_patch(Rectangle((5, 29), 13, 20, facecolor="#aed6f1", edgecolor="none"))
    ax.text(11.5, 39.5, "手机", ha="center", fontsize=10, color="#1b4f72",
            fontweight="bold")
    ax.text(11.5, 34.5, "IP Webcam\n开始服务", ha="center", fontsize=8.2,
            color="#1b4f72", linespacing=1.5)

    # WiFi
    ax.add_patch(Circle((36, 39), 7.0, facecolor=LBLUE, edgecolor=BLUE, lw=1.6))
    ax.text(36, 40.6, "Wi-Fi", ha="center", fontsize=9.4, color=NAVY,
            fontweight="bold")
    ax.text(36, 36.4, "同一局域网", ha="center", fontsize=7.6, color=GRAY)

    # 电脑
    ax.add_patch(FancyBboxPatch((53, 26), 22, 26,
                                boxstyle="round,pad=0.5,rounding_size=1.8",
                                facecolor="#eaf2f8", edgecolor=NAVY, lw=1.8))
    ax.add_patch(Rectangle((55, 34), 18, 14, facecolor="white", edgecolor=NAVY, lw=1))
    ax.text(64, 41, "OpenCV\nVideoCapture", ha="center", fontsize=7.4,
            color=NAVY, linespacing=1.4)
    ax.plot([60.5, 67.5], [36.4, 36.4], color=BLUE, lw=1.8)
    ax.plot([60.5, 63], [34.9, 36.4], color=BLUE, lw=1.8)
    ax.plot([67.5, 65], [34.9, 36.4], color=BLUE, lw=1.8)
    ax.text(64, 30.5, "后台线程死抓\n只留最新帧", ha="center", fontsize=7.6,
            color=INK, linespacing=1.4)
    ax.add_patch(Rectangle((57, 27), 14, 2.0, facecolor=NAVY, edgecolor="none"))

    _arrow(ax, 20.6, 39, 28.4, 39, color=BLUE)
    _arrow(ax, 43.6, 39, 52.4, 39, color=BLUE)
    ax.text(46.5, 41.2, "HTTP", ha="center", fontsize=8, color=GRAY)

    ax.text(50, 19.5, "http://手机IP:8080/video", ha="center", fontsize=10.5,
            color=RED, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#fdedec",
                      edgecolor=RED, lw=1.2))
    ax.text(50, 13.2, "写入 camera_source.txt 后，所有脚本自动改用手机",
            ha="center", fontsize=8.8, color=GRAY)
    ax.text(50, 8.6, "竖屏需旋转 90°：写在 camera_rotate.txt，无需改 App",
            ha="center", fontsize=8.8, color=GRAY)
    ax.text(50, 3.8, "手机端建议：插电、关自动锁屏、关美颜、锁定曝光",
            ha="center", fontsize=8.8, color=GRAY)

    # ---------------- 右：机位侧视 ----------------
    ax2 = fig.add_axes([0.575, 0.05, 0.42, 0.86])
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 62)
    ax2.axis("off")
    ax2.text(50, 61, "② 机位摆放（侧视）", ha="center", fontsize=13,
             color=NAVY, fontweight="bold")

    GROUND = 4.0
    ax2.plot([2, 98], [GROUND, GROUND], color="#95a5a6", lw=2.2)

    # 小桌 + 手机
    ax2.add_patch(Rectangle((6, GROUND), 15, 3.0, facecolor="#bdc3c7",
                            edgecolor="#7f8c8d"))
    ax2.add_patch(FancyBboxPatch((10, 7.0), 7, 12,
                                 boxstyle="round,pad=0.3,rounding_size=1.2",
                                 facecolor="#34495e", edgecolor="#2c3e50"))
    ax2.add_patch(Rectangle((11.2, 8.4), 4.6, 9.2, facecolor="#aed6f1"))
    ax2.text(13.5, 22.4, "手机", ha="center", fontsize=9.2, color=NAVY,
             fontweight="bold")

    # 高度标注
    ax2.annotate("", xy=(4.2, 7.0), xytext=(4.2, GROUND),
                 arrowprops=dict(arrowstyle="<->", color=RED, lw=1.2))
    ax2.text(3.4, 5.5, "0.6~1 m", ha="right", va="center", fontsize=8.6,
             color=RED)

    # 人（站在 x=70，身高占画面大半）
    px = 70.0
    ax2.add_patch(Circle((px, 46.0), 4.6, facecolor="#d5dbdb", edgecolor=NAVY,
                         lw=1.6, zorder=3))
    ax2.plot([px, px], [41.4, 24.0], color=NAVY, lw=5.0,
             solid_capstyle="round", zorder=3)
    ax2.plot([px - 5.2, px, px + 5.2], [GROUND, 24.0, GROUND], color=NAVY,
             lw=4.4, solid_capstyle="round", zorder=3)
    ax2.plot([px - 7.5, px - 3.0, px, px + 3.0, px + 7.5],
             [24.0, 27.5, 39.0, 27.5, 24.0], color=NAVY, lw=3.2,
             solid_capstyle="round", zorder=3)
    ax2.text(px + 6.5, 46.0, "人", fontsize=9.4, color=NAVY, fontweight="bold",
             va="center")

    # 视野扇形
    ax2.add_patch(plt.Polygon([[13.5, 13.0], [px + 4, GROUND + 1],
                               [px + 4, 51]], closed=True, facecolor=BLUE,
                              alpha=.09, edgecolor="none", zorder=1))
    for ex, ey in ((px + 4, GROUND + 1), (px + 4, 51)):
        ax2.plot([13.5, ex], [13.0, ey], color=BLUE, lw=1.1, ls=(0, (5, 3)),
                 alpha=.85, zorder=2)

    # 距离标注
    ax2.annotate("", xy=(13.5, 52.0), xytext=(px - 1, 52.0),
                 arrowprops=dict(arrowstyle="<->", color=RED, lw=1.3))
    ax2.text(41, 53.6, "2~2.5 m，全身入画（膝盖以下必须可见）", ha="center",
             fontsize=8.6, color=RED)
    ax2.text(px + 5.5, 2.0, "距 2~2.5 m", ha="left", fontsize=8.6, color=RED,
             va="center")
    save(fig, "fig6_手机连接与机位.png")


# ======================================================== 图7 判定结果示意
def fig_decision():
    fig, ax = plt.subplots(figsize=(11.4, 4.0))
    ax.set_xlim(0, 112)
    ax.set_ylim(0, 40)
    ax.axis("off")
    ax.text(56, 35.5, "底部信息栏：一次判定的输出", ha="center", fontsize=13.5,
            color=NAVY, fontweight="bold")

    TOP, BOT = 26.0, 9.5
    ax.add_patch(Rectangle((2, BOT), 108, TOP - BOT, facecolor="#2c3e50",
                           edgecolor="none"))

    # 判定结果块
    ax.add_patch(Rectangle((4.5, BOT + 1.5), 22, TOP - BOT - 3.0,
                           facecolor="#1f618d", edgecolor="none"))
    ax.text(15.5, 20.6, "站立 stand", ha="center", fontsize=12.0, color="white",
            fontweight="bold")
    ax.text(15.5, 15.4, "行为判定", ha="center", fontsize=8.6, color="#aed6f1")

    # 四条概率条
    bars = [("站立", .71, GREEN), ("坐下", .12, ORANGE),
            ("行走", .05, BLUE), ("蹲下", .12, "#8e44ad")]
    bw, bh, gap = 34.0, 2.5, 1.2
    for i, (nm, p, c) in enumerate(bars):
        y = TOP - 1.6 - bh - i * (bh + gap)
        ax.add_patch(Rectangle((33, y), bw, bh, facecolor="#566573",
                               edgecolor="none"))
        ax.add_patch(Rectangle((33, y), bw * p, bh, facecolor=c, edgecolor="none"))
        ax.text(31.5, y + bh / 2, nm, ha="right", va="center", fontsize=9.0,
                color="white")
        ax.text(68.0, y + bh / 2, f"{p*100:.0f}%", ha="left", va="center",
                fontsize=9.0, color="white")

    # 右侧运行指标
    for i, (lab, val) in enumerate([("画面延迟", "18 ms"), ("推理耗时", "82 ms"),
                                    ("手机流", "26 FPS"), ("有效窗口", "236")]):
        x = 78.5 + i * 8.2
        ax.text(x, 21.6, lab, ha="center", fontsize=8.0, color="#aed6f1")
        ax.text(x, 15.4, val, ha="center", fontsize=10.4, color="white",
                fontweight="bold")

    ax.text(56, 4.0,
            "概率来自随机森林 300 棵树的投票平均；取最大者为判定结果，"
            "低于 0.5 时底栏转为黄色提示“动作不确定”",
            ha="center", fontsize=9.6, color=GRAY)
    save(fig, "fig7_判定输出.png")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    print("生成讲解插图：")
    fig_skeleton()
    fig_geometry()
    fig_pipeline()
    fig_window()
    fig_curves()
    fig_phone()
    fig_decision()
    print("完成")
