"""建身份库：把 data/faces/<姓名>/*.jpg 变成 data/gallery.npz。

用法：
    py -3.11 -m tools.enroll              # 全量重建
    py -3.11 -m tools.enroll --list       # 只看现有身份库
    py -3.11 -m tools.enroll --test a.jpg # 拿一张图试一下会认成谁

目录约定（每个人一个文件夹，5 张以上不同角度最稳）：
    data/faces/张三/1.jpg
    data/faces/张三/2.jpg
    data/faces/李四/1.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.face import ACCEPT_SIM, FaceEngine, cosine  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FACE_DIR = ROOT / "data" / "faces"
GALLERY = ROOT / "data" / "gallery.npz"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _one_face(eng: FaceEngine, img: np.ndarray):
    """取画面里最大的一张脸，顺带返回相似度候选。"""
    faces = eng.analyze(img, max_faces=3)
    if not faces:
        return None
    faces.sort(key=lambda f: f["dim"], reverse=True)
    return faces[0]


def enroll(eng: FaceEngine) -> int:
    if not FACE_DIR.exists():
        print(f"没有找到 {FACE_DIR}")
        print("先建目录：data/faces/张三/1.jpg  每人 5 张以上不同角度")
        return 0

    names: list[str] = []
    vectors: list[np.ndarray] = []
    print(f"{'姓名':<12}{'照片':>5}{'可用':>6}{'平均自相似':>12}")
    print("-" * 38)

    for person_dir in sorted(p for p in FACE_DIR.iterdir() if p.is_dir()):
        files = [f for f in sorted(person_dir.iterdir()) if f.suffix.lower() in EXTS]
        embs = []
        for f in files:
            img = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            face = _one_face(eng, img)
            if face is None or face["emb"] is None:
                continue
            embs.append(face["emb"])
        if not embs:
            print(f"{person_dir.name:<12}{len(files):>5}{0:>6}   跳过（没检出人脸）")
            continue
        mat = np.stack(embs)
        mean = mat.mean(axis=0)
        mean /= (np.linalg.norm(mean) or 1.0)
        self_sim = float(np.mean([cosine(mean, e) for e in embs]))
        names.append(person_dir.name)
        vectors.append(mean)
        print(f"{person_dir.name:<12}{len(files):>5}{len(embs):>6}{self_sim:>12.3f}")

    if not names:
        print("\n没有可用的照片。检查 data/faces 目录结构。")
        return 0

    GALLERY.parent.mkdir(parents=True, exist_ok=True)
    np.savez(GALLERY, names=np.array(names), embs=np.stack(vectors).astype(np.float32))
    print(f"\n已写入 {GALLERY}  共 {len(names)} 人")
    print(f"判定阈值 相似度 >= {ACCEPT_SIM} 才算认出")
    return len(names)


def test_one(eng: FaceEngine, path: str) -> None:
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"读不了这张图：{path}")
        return
    faces = eng.analyze(img, max_faces=3)
    if not faces:
        print("这张图里没检出人脸")
        return
    for f in faces:
        verdict = "认出 " + f["name"] if f["name"] else "陌生人"
        print(f"  框 {f['box']}  相似度 {f['sim']:.3f}  ->  {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--test", default="")
    args = ap.parse_args()

    eng = FaceEngine()
    print("引擎", eng.describe(), "\n")

    if args.list:
        g = eng.gal
        if not g.size:
            print("身份库是空的，跑一次 py -3.11 -m tools.enroll")
        else:
            for i, n in enumerate(g.names):
                print(f"  {i + 1:>2}. {n}")
        return

    if args.test:
        test_one(eng, args.test)
        return

    enroll(eng)
    print("\n从机要重启才会重新加载身份库（它是启动时读的）。")


if __name__ == "__main__":
    main()
