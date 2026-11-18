"""训练：滑窗特征 -> 分类器，带"按段分组"交叉验证。

用法
----
    python train.py
    python train.py --window 30 --stride 5

为什么要按段分组做交叉验证
--------------------------
同一段录制里相邻两秒的画面几乎一样，如果随机划分训练/测试集，
测试集里的窗口在训练集里几乎都能找到"双胞胎"，准确率会虚高 10~20 个点。
按文件名分组，模型必须靠"没见过的录制片段"来预测，才是真实水平。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import config as C
from dataset import build_dataset, summarize


def make_rf():
    return RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1)


def make_mlp():
    return Pipeline([
        ("sc", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3,
                              max_iter=1200, early_stopping=True,
                              random_state=42)),
    ])


def cross_validate(factory, X, y, groups, n_splits):
    gkf = GroupKFold(n_splits=n_splits)
    accs, y_true, y_pred = [], [], []
    for tr, te in gkf.split(X, y, groups):
        m = factory()
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        accs.append(accuracy_score(y[te], p))
        y_true += list(y[te])
        y_pred += list(p)
    return accs, np.array(y_true), np.array(y_pred)


def plot_confusion(cm, path, title):
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    names = [C.LABEL_ZH[n] for n in C.LABEL_NAMES]
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-sec", type=float, default=C.WINDOW_SEC,
                    help="判定窗口时长（秒）")
    ap.add_argument("--stride-sec", type=float, default=C.STRIDE_SEC,
                    help="滑窗步长（秒）")
    ap.add_argument("--splits", type=int, default=0, help="0=自动")
    ap.add_argument("--raw", default=str(C.RAW_DIR), help="采集数据目录")
    args = ap.parse_args()

    X, y, g, recs, skipped = build_dataset(Path(args.raw), args.window_sec, args.stride_sec)

    print("=" * 60)
    print("数据概况")
    print("=" * 60)
    print(summarize(recs))
    for name, why in skipped:
        print(f"  [跳过] {name}: {why}")

    if X is None:
        print("\n没有可用数据。先跑 recorder.py 采集几段（每类至少 2 段）。")
        return

    uniq = np.unique(g)
    per_class = {n: len(set(g[y == i])) for i, n in enumerate(C.LABEL_NAMES)}
    print(f"\n窗口样本 {X.shape[0]} 个 × {X.shape[1]} 维，来自 {len(uniq)} 段")
    print(f"每类段数：{per_class}")
    if min(per_class.values()) < 2:
        print("⚠ 有类别少于 2 段，交叉验证结果不可信，建议每类至少 5 段。")

    n_splits = args.splits or min(5, len(uniq))
    if n_splits < 2:
        print("⚠ 段数不足 2，无法做分组交叉验证，仅做全量训练。")
        n_splits = 0

    results = {}
    if n_splits >= 2:
        candidates = [("RandomForest", make_rf)]
        # MLP 在样本太少时容易不收敛，样本够才加入对比
        if X.shape[0] >= 90 and min(np.bincount(y, minlength=len(C.LABEL_NAMES))) >= 20:
            candidates.append(("MLP", make_mlp))
        else:
            print("\n（窗口样本偏少，跳过 MLP 对比，只用随机森林）")

        for name, factory in candidates:
            t0 = time.time()
            accs, y_true, y_pred = cross_validate(factory, X, y, g, n_splits)
            print("\n" + "-" * 60)
            print(f"{name}  {n_splits} 折分组交叉验证")
            print("-" * 60)
            print("每折准确率：" + "  ".join(f"{a:.3f}" for a in accs))
            print(f"平均 {np.mean(accs):.4f} ± {np.std(accs):.4f}")
            print(classification_report(y_true, y_pred,
                                        labels=list(range(len(C.LABEL_NAMES))),
                                        target_names=C.LABEL_NAMES,
                                        zero_division=0))
            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(C.LABEL_NAMES))))
            plot_confusion(cm, C.REPORT_DIR / f"confusion_{name}.png",
                           f"{name}  mean acc={np.mean(accs):.3f}")
            results[name] = dict(mean=float(np.mean(accs)), std=float(np.std(accs)),
                                 folds=[float(a) for a in accs], secs=time.time() - t0)

    # ---------- 全量重训并保存 ----------
    best = max(results, key=lambda k: results[k]["mean"]) if results else "RandomForest"
    print(f"\n用于部署的模型：{best}")

    to_fit = [("RandomForest", make_rf(), "pose_rf.pkl")]
    if "MLP" in results:
        to_fit.append(("MLP", make_mlp(), "pose_mlp.pkl"))

    models = {}
    for name, m, fname in to_fit:
        m.fit(X, y)
        joblib.dump(m, C.MODEL_DIR / fname)
        models[name] = m

    joblib.dump(models[best], C.MODEL_DIR / "pose_best.pkl")

    if best == "RandomForest":
        rf = models["RandomForest"]
        n_single = len(C.FRAME_FEATURE_NAMES)
        # 前 n_single 维是窗口内"均值"，单独看它最能代表该特征本身的贡献
        imp = sorted(zip(C.FRAME_FEATURE_NAMES, rf.feature_importances_[:n_single]),
                     key=lambda t: -t[1])
        print("\n单帧特征重要性（窗口均值统计部分）：")
        for k, v in imp[:10]:
            print(f"  {k:20s} {v:.4f}")

    meta = dict(
        labels=C.LABEL_NAMES, window_sec=args.window_sec, stride_sec=args.stride_sec,
        feat_dim=int(X.shape[1]), n_windows=int(X.shape[0]),
        n_recordings=len(uniq), per_class_recordings=per_class,
        cv=results, best=best, trained_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        frame_feature_names=C.FRAME_FEATURE_NAMES,
    )
    (C.MODEL_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(f"\n模型已保存到 {C.MODEL_DIR}")
    print(f"混淆矩阵图见 {C.REPORT_DIR}")


if __name__ == "__main__":
    main()
