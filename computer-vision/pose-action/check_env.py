"""环境自检：换电脑后第一时间跑这个，确认能不能干活。

用法
----
双击「环境自检.bat」，或
    .venv\\Scripts\\python.exe check_env.py

检查项
------
Python 版本 / 关键库版本 / 摄像头可用性 / 模型与数据的就位情况
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OK = "[ OK ]"
BAD = "[FAIL]"
WARN = "[WARN]"


def main():
    print("=" * 58)
    print(" PoseAction 环境自检")
    print("=" * 58)

    # ---------- Python ----------
    major, minor = sys.version_info[:2]
    tag = OK if (3, 9) <= (major, minor) <= (3, 12) else WARN
    print(f"{tag} Python {sys.version.split()[0]}  ({platform.machine()})")
    if (major, minor) != (3, 11):
        print("       建议用 Python 3.11，其他版本部分依赖可能没有现成轮子。")

    # ---------- 关键库 ----------
    libs = [
        ("numpy", "numpy"), ("cv2", "opencv-python"), ("pandas", "pandas"),
        ("sklearn", "scikit-learn"), ("torch", "torch"),
        ("ultralytics", "ultralytics"), ("matplotlib", "matplotlib"),
        ("openpyxl", "openpyxl"), ("joblib", "joblib"),
    ]
    missing = []
    for mod, pkg in libs:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            print(f"{OK} {pkg:<16} {ver}")
        except Exception as e:
            missing.append(pkg)
            print(f"{BAD} {pkg:<16} 未安装 ({type(e).__name__})")

    if missing:
        print("\n依赖不全，请双击「⓪一键安装环境.bat」重装。")
        print("缺失：" + ", ".join(missing))
        return pause()

    # ---------- 模型文件 ----------
    weights = ROOT / "yolov8n-pose.pt"
    if weights.exists():
        print(f"{OK} 检测权重 yolov8n-pose.pt  {weights.stat().st_size/1024/1024:.1f} MB")
    else:
        print(f"{WARN} 检测权重缺失，首次运行会自动联网下载（约 6.6MB）")

    trained = ROOT / "models" / "pose_best.pkl"
    if trained.exists():
        print(f"{OK} 训练好的分类器 models/pose_best.pkl")
    else:
        print(f"{WARN} 还没有训练好的分类器，先采集再跑第 2 步")

    # ---------- 摄像头 ----------
    found = []
    try:
        import cv2
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
        found = []
        for idx in (0, 1):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ok, fr = cap.read()
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                if ok:
                    found.append((idx, w, h))
        if found:
            for idx, w, h in found:
                print(f"{OK} 摄像头 {idx} 可用，当前分辨率 {w}x{h}")
        else:
            print(f"{BAD} 没找到可用摄像头。检查是否被其他程序占用（微信/QQ/浏览器视频）。")
    except Exception as e:
        print(f"{BAD} 摄像头检测异常：{e}")

    # ---------- 数据 ----------
    raw = ROOT / "data" / "raw"
    clips = sorted(raw.glob("*.csv")) if raw.exists() else []
    print(f"{OK if clips else WARN} 已采集片段 {len(clips)} 个  ({raw})")
    if clips:
        from collections import Counter
        try:
            import pandas as pd
            cnt = Counter()
            for p in clips:
                try:
                    cnt[str(pd.read_csv(p, encoding="utf-8-sig", usecols=["label"],
                                        nrows=1)["label"].iloc[0])] += 1
                except Exception:
                    cnt["读取失败"] += 1
            print("       " + "  ".join(f"{k}:{v}段" for k, v in cnt.items()))
            per = [cnt.get(k, 0) for k in C.LABEL_NAMES]
            if per and min(per) < 2:
                print("       注意：每类至少 2 段才能训练，建议 5 段。")
        except Exception:
            pass
    else:
        print("       还没有数据，双击「①采集数据.bat」开始录。")

    # ---------- 结论 ----------
    print("\n" + "=" * 58)
    if not missing and found:
        print(" 结论：环境正常，可以按 ①→②→③ 的顺序开始。")
    elif missing:
        print(" 结论：依赖不全，先跑「⓪一键安装环境.bat」。")
    else:
        print(" 结论：软件正常，但没检测到摄像头，接上摄像头再试。")
    print("=" * 58)
    pause()


def pause():
    try:
        input("\n按回车键退出...")
    except EOFError:
        pass


if __name__ == "__main__":
    main()
