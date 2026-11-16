"""人脸检测 + 特征提取 + 身份比对。

检测：优先 YuNet（``models/yunet.onnx``，CPU 单帧 5~10ms，输出 5 点关键点）；
模型缺失时退化为 OpenCV 自带的 Haar 级联 —— 精度低一档，但离线可用，不影响跑通。

识别：SFace，112x112 输入、输出 128 维，余弦相似度比对。

为什么不用 insightface / dlib：insightface 要拉 onnxruntime + 编译轮子，
dlib 在 Windows 上装一次半小时。OpenCV 这两件事都能干，就不引外敌。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
YUNET = MODEL_DIR / "yunet.onnx"
SFACE = MODEL_DIR / "face_recognition_sface_2021dec.onnx"

ACCEPT_SIM = 0.40    # 高于此值才算认出来
GREY_SIM = 0.34      # 介于两者之间：疑是，但不够格，按陌生人处理


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if not na or not nb:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def crop_face(frame: np.ndarray, box: tuple[int, int, int, int],
              margin: float = 0.22) -> np.ndarray | None:
    """按人脸框外扩一圈裁出来，多留点额头和下巴，SFace 更稳。"""
    h, w = frame.shape[:2]
    x, y, bw, bh = box
    cx, cy = x + bw / 2.0, y + bh / 2.0
    side = max(bw, bh) * (1.0 + margin * 2)
    x0 = max(0, int(cx - side / 2))
    y0 = max(0, int(cy - side / 2))
    x1 = min(w, int(cx + side / 2))
    y1 = min(h, int(cy + side / 2))
    if x1 - x0 < 24 or y1 - y0 < 24:
        return None
    return frame[y0:y1, x0:x1]


class Detector:
    """返回 [(x, y, w, h, score), ...]，坐标是原图像素。"""

    def __init__(self, score: float = 0.62):
        self.mode = "none"
        self._yunet = None
        self._haar = None
        self._size = (0, 0)

        if YUNET.exists() and YUNET.stat().st_size > 100_000:
            try:
                self._yunet = cv2.FaceDetectorYN.create(
                    str(YUNET), "", (320, 320), score, 0.3, 5000)
                self.mode = "yunet"
            except cv2.error:
                self._yunet = None

        if self._yunet is None:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(path)
            self.mode = "haar" if not self._haar.empty() else "none"
        self.score = score

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        if self.mode == "yunet":
            h, w = frame.shape[:2]
            if self._size != (w, h):
                self._yunet.setInputSize((w, h))
                self._size = (w, h)
            _ret, faces = self._yunet.detect(frame)
            if faces is None:
                return []
            out = []
            for f in faces:
                x, y, bw, bh = (int(v) for v in f[:4])
                out.append((max(0, x), max(0, y), int(bw), int(bh), float(f[-1])))
            return out

        if self.mode == "haar":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            found = self._haar.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
            return [(int(x), int(y), int(w), int(h), 1.0) for x, y, w, h in found]

        return []


class Embedder:
    def __init__(self, path: Path = SFACE):
        self.ok = SFACE.exists() and SFACE.stat().st_size > 1_000_000
        self.net = cv2.dnn.readNetFromONNX(str(path)) if self.ok else None
        self.dim = 128

    def embed(self, face_bgr: np.ndarray) -> np.ndarray | None:
        if self.net is None or face_bgr is None or face_bgr.size == 0:
            return None
        img = cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
        # SFace 的预处理：(x - 127.5) / 128，BGR 顺序直接进网络
        blob = cv2.dnn.blobFromImage(img, 1.0 / 128.0, (112, 112),
                                     (127.5, 127.5, 127.5), swapRB=False)
        self.net.setInput(blob)
        v = self.net.forward().flatten().astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n else None


class Gallery:
    """身份库：names[i] 对应 embs[i]。"""

    def __init__(self, path: Path | None = None):
        self.path = path or (Path(__file__).resolve().parent.parent / "data" / "gallery.npz")
        self.names: list[str] = []
        self.embs: np.ndarray | None = None
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                d = np.load(self.path, allow_pickle=True)
                self.names = [str(n) for n in d["names"]]
                self.embs = d["embs"].astype(np.float32)
                return
            except (OSError, KeyError, ValueError):
                pass
        self.names, self.embs = [], None

    @property
    def size(self) -> int:
        return len(self.names)

    def match(self, emb: np.ndarray) -> tuple[str | None, float]:
        """返回 (姓名, 相似度)。低于 GREY_SIM 时姓名为 None。"""
        if self.embs is None or not len(self.names):
            return None, 0.0
        v = emb / (np.linalg.norm(emb) or 1.0)
        sims = self.embs @ v
        i = int(np.argmax(sims))
        sim = float(sims[i])
        return (self.names[i] if sim >= GREY_SIM else None), sim


class FaceEngine:
    """三个部件捏一起，给识别线程直接用。"""

    def __init__(self) -> None:
        self.det = Detector()
        self.emb = Embedder()
        self.gal = Gallery()

    def describe(self) -> dict:
        return {
            "detector": self.det.mode,
            "embedder": "sface" if self.emb.ok else "none",
            "gallery": self.gal.size,
            "accept_sim": ACCEPT_SIM,
            "grey_sim": GREY_SIM,
        }

    def analyze(self, frame: np.ndarray, max_faces: int = 4) -> list[dict]:
        """对一帧做检测 + 提特征 + 比对，返回每个人脸的结论。"""
        faces = []
        h, w = frame.shape[:2]
        area = float(w * h) or 1.0
        for box in self.det.detect(frame)[:max_faces]:
            x, y, bw, bh, score = box
            crop = crop_face(frame, (x, y, bw, bh))
            emb = self.emb.embed(crop) if crop is not None else None
            name, sim = self.gal.match(emb) if emb is not None else (None, 0.0)
            faces.append({
                "box": (x, y, bw, bh),
                "score": round(score, 3),
                "dim": round(bw * bh / area, 4),
                "name": name if (name and sim >= ACCEPT_SIM) else None,
                "sim": round(sim, 3),
                "emb": emb,
                "crop": crop,
            })
        return faces


class Tracker:
    """极简 IoU 跟踪。目的不是跟得准，是让同一个人在一串帧里只算一次。"""

    def __init__(self, iou_thr: float = 0.3, ttl_s: float = 2.0, confirm_hits: int = 3):
        self.iou_thr = iou_thr
        self.ttl_s = ttl_s
        self.confirm_hits = confirm_hits
        self.tracks: dict[int, dict] = {}
        self._next = 1

    @staticmethod
    def _iou(a, b) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        union = aw * ah + bw * bh - inter
        return inter / union if union else 0.0

    def update(self, faces: list[dict], now: float) -> list[dict]:
        """把这一帧的人脸并进轨迹，返回"刚被确认"的轨迹列表。"""
        for tid in [t for t, tr in self.tracks.items() if now - tr["seen"] > self.ttl_s]:
            del self.tracks[tid]

        confirmed_now = []
        used = set()
        for f in faces:
            best, best_iou = None, 0.0
            for tid, tr in self.tracks.items():
                if tid in used:
                    continue
                v = self._iou(tr["box"], f["box"])
                if v > best_iou:
                    best, best_iou = tid, v
            if best is not None and best_iou >= self.iou_thr:
                tr = self.tracks[best]
                used.add(best)
            else:
                tr = {"id": self._next, "box": f["box"], "votes": Counter(),
                      "hits": 0, "seen": now, "confirmed": False, "last_emit": 0.0,
                      "name": None, "crop": None}
                self.tracks[self._next] = tr
                self._next += 1

            label = f["name"] or "unknown"
            tr["votes"][label] += 1
            tr["box"] = f["box"]
            tr["seen"] = now
            tr["hits"] += 1
            if f["crop"] is not None and (label != "unknown" or tr["crop"] is None):
                tr["crop"] = f["crop"]
            tr["sim"] = max(tr.get("sim", 0.0), f["sim"])

            top, cnt = tr["votes"].most_common(1)[0]
            tr["name"] = top
            if not tr["confirmed"] and cnt >= self.confirm_hits and cnt / tr["hits"] >= 0.6:
                tr["confirmed"] = True
                confirmed_now.append(tr)
        return confirmed_now
