"""从机识别线程：检测 → 跟踪 → 投票确认 → 上报事件。

它跑在 camera_server 进程里，和推流共用同一路摄像头 —— 一个设备只开一次，
Windows 上重复打开同一个摄像头必然报错。

判定链路的四道闸：
  1. 人脸框太小（dim < MIN_DIM）→ 远处误检，丢
  2. 单帧不算数，同一轨迹累计 3 帧同一个名字才确认（Traffic 级去抖）
  3. 同一身份同一机位 20 秒内只报一次（事件级去抖）
  4. 主电脑暂时连不上 → 写本地队列，连上补传，靠 uuid 幂等去重
"""

from __future__ import annotations

import base64
import json
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import requests

from common import netcfg
from common.face import FaceEngine, Tracker

QUEUE = Path(__file__).resolve().parent.parent / "data" / "queue.jsonl"

COOLDOWN_S = 20.0     # 同一身份同一机位的上报间隔
MIN_DIM = 0.015       # 人脸框面积占画面比例下限
TARGET_FPS = 5.0      # 识别帧率，够用又不占 CPU

RECOG: dict = {
    "engine": {},
    "role": "camera",
    "master": None,
    "frames": 0,
    "faces": 0,
    "events": 0,
    "fps": 0.0,
    "last": None,
    "queue": 0,
}


def _session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False          # 别让系统代理吃掉局域网请求
    return s


def _find_master() -> tuple[str, int] | None:
    found = netcfg.discover(timeout=1.5) or netcfg.load_cache()
    if not found:
        return None
    ip, port, _name = found
    # 地址是本机也照样试：主电脑和从机跑在同一台机器上（本机联调）是合法情形。
    # 真的没有服务在听，连接会立刻被拒，几毫秒就返回，不会拖慢节奏。
    return ip, port


def _queue_size() -> int:
    if not QUEUE.exists():
        return 0
    try:
        return sum(1 for line in QUEUE.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


def _spool(ev: dict) -> None:
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    RECOG["queue"] = _queue_size()


def _flush(ses: requests.Session, master: tuple[str, int]) -> None:
    """把断网期间攒下的事件补传，成功后清空队列。"""
    if not QUEUE.exists():
        return
    lines = [ln for ln in QUEUE.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return
    left = []
    for ln in lines:
        try:
            r = ses.post(f"http://{master[0]}:{master[1]}/event",
                         json=json.loads(ln), timeout=4)
            if not r.ok:
                left.append(ln)
        except requests.RequestException:
            left.append(ln)
    QUEUE.write_text("".join(x + "\n" for x in left), encoding="utf-8")
    RECOG["queue"] = len(left)
    if len(left) < len(lines):
        print(f"  [补传] {len(lines) - len(left)} 条事件已送达主电脑，剩 {len(left)} 条",
              flush=True)


def _encode_snap(crop: np.ndarray | None) -> str | None:
    if crop is None or crop.size == 0:
        return None
    side = 160
    h, w = crop.shape[:2]
    s = side / max(h, w)
    small = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return base64.b64encode(buf.tobytes()).decode() if ok else None


def _loop(state: dict, cfg: dict) -> None:
    engine = FaceEngine()
    tracker = Tracker()
    ses = _session()
    role = cfg.get("role", "camera")
    kind = "IN" if role == "in" else "OUT"
    RECOG["engine"] = engine.describe()
    RECOG["role"] = role

    if engine.det.mode == "none" or not engine.emb.ok:
        print("  警告 识别引擎未就绪（缺模型？），只推流不统计", flush=True)

    last_master = 0.0
    master = None
    cooldown: dict[str, float] = {}
    stamp_win: list[float] = []

    print(f"  识别已启动  机位={role}  方向={kind}  "
          f"检测={engine.det.mode}  身份库={engine.gal.size} 人", flush=True)

    while True:
        frame = state.get("frame")
        if frame is None or not state.get("cam_ok"):
            time.sleep(0.25)
            continue

        t0 = time.time()
        now = time.time()
        faces = [f for f in engine.analyze(frame) if f["dim"] >= MIN_DIM]
        RECOG["faces"] = len(faces)

        for tr in tracker.update(faces, now):
            person = tr["name"] or "unknown"
            if now - cooldown.get(person, 0.0) < COOLDOWN_S:
                continue
            cooldown[person] = now

            ev = {
                "uuid": uuid.uuid4().hex,
                "cam": role,
                "kind": kind,
                "person": person,
                "sim": round(float(tr.get("sim", 0.0)), 3),
                "ts": now,
                "dim": round(faces[0]["dim"] if faces else 0.0, 4),
                "snap_b64": _encode_snap(tr.get("crop")),
            }
            RECOG["events"] += 1
            RECOG["last"] = {"person": person, "kind": kind, "ts": now,
                             "sim": ev["sim"], "track": tr["id"]}

            if now - last_master > 15:
                master = _find_master()
                last_master = now
                RECOG["master"] = master
                if master:
                    _flush(ses, master)

            sent = False
            if master:
                try:
                    r = ses.post(f"http://{master[0]}:{master[1]}/event", json=ev, timeout=4)
                    sent = r.ok
                    if sent:
                        body = r.json()
                        print(f"  [{kind}] {person} sim={ev['sim']} "
                              f"→ {body.get('action')} 在室 {body.get('inside')}", flush=True)
                        if body.get("action") == "duplicate":
                            sent = True
                except requests.RequestException:
                    sent = False
            if not sent:
                _spool(ev)
                print(f"  [{kind}] {person} 主电脑暂不可达，已入本地队列", flush=True)

        # 只把坐标和标签交给推流端：画框与 JPEG 编码由 camera_server 按推流帧率做，
        # 这样识别 5fps 也不会让画面上出现"每 200ms 卡一下"的观感
        state["det_faces"] = [
            {"box": f["box"],
             "label": f"{f['name'] or 'unknown'} {f['sim']:.2f}",
             "color": (90, 220, 120) if f["name"] else (90, 160, 240)}
            for f in faces
        ]

        RECOG["frames"] += 1
        stamp_win.append(time.time())
        stamp_win = [t for t in stamp_win if now - t <= 3.0]
        RECOG["fps"] = round(len(stamp_win) / 3.0, 1)

        gap = 1.0 / TARGET_FPS - (time.time() - t0)
        if gap > 0:
            time.sleep(gap)


def start(state: dict, cfg: dict) -> None:
    threading.Thread(target=_loop, args=(state, cfg), daemon=True,
                     name="recog").start()
