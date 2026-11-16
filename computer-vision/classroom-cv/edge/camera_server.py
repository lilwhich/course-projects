"""从机摄像头服务：把本机摄像头变成可通过 HTTP 访问的网络摄像头。

启动后会自动找到主电脑并注册自己，主电脑页面上直接就能看到画面。

用法：
    py -3.11 -m edge.camera_server                  # 默认摄像头 0
    py -3.11 -m edge.camera_server --cam 1          # 第二路摄像头
    py -3.11 -m edge.camera_server --source a.mp4   # 用视频文件顶替摄像头
    py -3.11 -m edge.camera_server --no-register    # 只在本机开服务，不注册
"""

from __future__ import annotations

import argparse
import socket
import threading
import time

import cv2
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from common import netcfg

VERSION = "0.5.0"      # 主电脑靠它判断从机跑的是不是最新代码

app = FastAPI(title="Classroom CV Edge Camera")
# 主电脑的面板是另一个源（另一个 IP:端口），要读 /recog.json 就得放行跨源
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

STATE = {
    "jpeg": None,
    "frame": None,     # 原始帧，给识别线程用（不重复开摄像头）
    "overlay": None,   # 识别线程画好的调试帧
    "det_faces": [],   # 最新的检测结果坐标，推流端实时叠框用
    "cam_ok": False,
    "w": 0,
    "h": 0,
    "fps": 0.0,
    "frames": 0,
    "source": "",
    "started": time.time(),
}
_CFG = {"cam": 0, "source": "", "width": 1280, "height": 720, "role": "camera"}


def _open_capture(cam_index: int, source: str):
    if source:
        cap = cv2.VideoCapture(source)
        return cap, f"file:{source}"
    for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):  # DSHOW 在 Windows 上打开更快
        cap = cv2.VideoCapture(cam_index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, _CFG["width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _CFG["height"])
            return cap, f"cam{backend}:{cam_index}"
        cap.release()
    return None, "none"


def _placeholder(reason: str) -> np.ndarray:
    """摄像头打不开时也推流，这样能先把网络链路的锅和硬件的锅分开。"""
    img = np.full((_CFG["height"] // 2, _CFG["width"] // 2, 3), 24, np.uint8)
    cv2.putText(img, "NO CAMERA", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (90, 150, 240), 3)
    cv2.putText(img, reason[:38], (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1)
    cv2.putText(img, time.strftime("%H:%M:%S"), (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    return img


def _reader():
    cap, tag = _open_capture(_CFG["cam"], _CFG["source"])
    STATE["source"] = tag
    STATE["cam_ok"] = bool(cap and cap.isOpened())
    if not STATE["cam_ok"]:
        print(f"  警告 摄像头打不开（{tag}），已降级为占位画面，网络链路仍可测", flush=True)

    window: list[float] = []
    while True:
        if STATE["cam_ok"]:
            ok, frame = cap.read()
            if not ok:
                print("  警告 读帧失败，转为占位画面", flush=True)
                STATE["cam_ok"] = False
                continue
            STATE["h"], STATE["w"] = frame.shape[:2]
        else:
            frame = _placeholder(f"{_CFG['source'] or 'cam' + str(_CFG['cam'])} unavailable")
            STATE["h"], STATE["w"] = frame.shape[:2]
            time.sleep(1 / 15)

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            STATE["jpeg"] = buf.tobytes()
            STATE["frame"] = frame
            STATE["frames"] += 1
            now = time.time()
            window.append(now)
            window = [t for t in window if now - t <= 2.0]
            STATE["fps"] = round(len(window) / 2.0, 1)


def _mjpeg():
    while True:
        jpg = STATE["jpeg"]
        if jpg is None:
            time.sleep(0.03)
            continue
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
            + str(len(jpg)).encode()
            + b"\r\n\r\n"
            + jpg
            + b"\r\n"
        )
        time.sleep(0.02)


@app.get("/health")
def health():
    return {
        "ok": True,
        "role": "edge-camera",
        "version": VERSION,
        "name": socket.gethostname(),
        "cam_ok": STATE["cam_ok"],
        "width": STATE["w"],
        "height": STATE["h"],
        "fps": STATE["fps"],
        "frames": STATE["frames"],
        "source": STATE["source"],
        "port": _CFG.get("port", netcfg.CAM_PORT),
        "ts": time.time(),
    }


@app.get("/snapshot")
def snapshot():
    if STATE["jpeg"] is None:
        return Response(status_code=503, content=b"frame not ready")
    return Response(content=STATE["jpeg"], media_type="image/jpeg")


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/overlay")
def overlay():
    """带检测框的单帧，截图和肉眼核对用。"""
    frame = STATE.get("frame")
    if frame is None:
        return Response(status_code=503, content=b"frame not ready")
    ok, buf = cv2.imencode(".jpg", _annotate(frame), [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return Response(status_code=503, content=b"encode failed")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/recog.json")
def recog_json():
    from edge import recognize
    return {"ok": True, "name": socket.gethostname(), "version": VERSION,
            **recognize.RECOG, "cam_ok": STATE["cam_ok"], "ts": time.time()}


def _annotate(frame: np.ndarray) -> np.ndarray:
    """把最新的检测结果画到当前帧上。

    识别只跑 5fps，但画框走推流帧率 —— 框的位置由识别结果更新，
    画面本身是新的，所以看起来是"实时"，不是"每 200ms 卡一下"。
    """
    vis = frame.copy()
    faces = STATE.get("det_faces") or []
    for f in faces:
        x, y, w, h = f["box"]
        color = f.get("color") or (90, 220, 120)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        label = f.get("label") or ""
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (x, max(0, y - th - 8)), (x + tw + 8, y), color, -1)
            cv2.putText(vis, label, (x + 4, max(14, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 24, 30), 2)
    role = {"in": "IN", "out": "OUT"}.get(_CFG.get("role", ""), "CAM")
    hud = f"{role}  faces={len(faces)}  {_CFG.get('role', '')}"
    cv2.putText(vis, hud, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 220, 255), 2)
    return vis


def _mjpeg_overlay():
    last_key = None
    while True:
        frame = STATE.get("frame")
        if frame is None:
            time.sleep(0.05)
            continue
        vis = _annotate(frame)
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if not ok:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
               + str(buf.size).encode() + b"\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.055)


@app.get("/overlay_feed")
def overlay_feed():
    """带识别框的实时流，供给主电脑的双画面页直接用。"""
    return StreamingResponse(_mjpeg_overlay(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/", response_class=HTMLResponse)
def index():
    me = STATE
    status = "正常" if me["cam_ok"] else "占位画面"
    return f"""<!doctype html><meta charset="utf-8"><title>从机摄像头 {socket.gethostname()}</title>
<style>body{{margin:0;background:#14161b;color:#e6e8eb;font:14px/1.6 system-ui,sans-serif;padding:24px}}
img{{max-width:100%;border-radius:10px;border:1px solid #2a3040}}
.k{{color:#8b93a1}}</style>
<h2 style="font-weight:500">{socket.gethostname()} · 摄像头</h2>
<p class="k">v{VERSION} · {status} · {me['w']}x{me['h']} · {me['fps']} fps · 已推 {me['frames']} 帧 · 来源 {me['source']}</p>
<img src="/video_feed"></p>
<p class="k">这个地址给主电脑用：http://{netcfg.local_ipv4s()[0] if netcfg.local_ipv4s() else '127.0.0.1'}:{_CFG.get('port', netcfg.CAM_PORT)}/video_feed</p>"""


def _try_register(port: int) -> bool:
    master = netcfg.discover() or netcfg.load_cache()
    if not master:
        return False
    ip, mport, _ = master
    ips = [i for i in netcfg.local_ipv4s() if not netcfg.is_virtual(i)]
    payload = {
        "name": socket.gethostname(),
        "role": _CFG.get("role", "camera"),
        "version": VERSION,
        "ip": ips[0] if ips else "127.0.0.1",
        "port": port,
        "cam_ok": STATE["cam_ok"],
        "width": STATE["w"],
        "height": STATE["h"],
        "fps": STATE["fps"],
    }
    s = requests.Session()
    s.trust_env = False
    try:
        r = s.post(f"http://{ip}:{mport}/register", json=payload, timeout=3)
        return bool(r.ok)
    except requests.RequestException:
        return False


def _register_loop(port: int) -> None:
    """持续心跳注册。

    主电脑可能比从机晚启动，也可能中途被关掉再打开 —— 它的注册表在内存里，
    一重启就空了。所以这里不"注册成功就退出"，而是每 20 秒报一次到。
    """
    was_ok = None
    while True:
        ok = _try_register(port)
        if ok != was_ok:
            if ok:
                print("  已注册到主电脑，主电脑页面能看到本机画面", flush=True)
            else:
                print("  暂未连上主电脑，每 20 秒自动重试（主电脑启动后无需重启本机）",
                      flush=True)
            was_ok = ok
        time.sleep(20)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--source", default="")
    ap.add_argument("--port", type=int, default=netcfg.CAM_PORT)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--no-stats", action="store_true",
                    help="只推流，不跑人脸识别统计")
    ap.add_argument("--role", default="camera", choices=["in", "out", "camera"],
                    help="in=入口, out=出口")
    args = ap.parse_args()

    _CFG.update(cam=args.cam, source=args.source, width=args.width, height=args.height,
                port=args.port, role=args.role)

    threading.Thread(target=_reader, daemon=True).start()
    time.sleep(1.5)

    ips = [i for i in netcfg.local_ipv4s() if not netcfg.is_virtual(i)]
    print("=" * 58)
    print(f"  从机摄像头服务  {socket.gethostname()}   v{VERSION}")
    print(f"  机位    {args.role}    端口 {args.port}")
    print(f"  摄像头  {'正常 ' + str(STATE['w']) + 'x' + str(STATE['h']) if STATE['cam_ok'] else '未就绪/占位'}")
    for ip in ips:
        print(f"  画面    http://{ip}:{args.port}/")
        print(f"  单帧    http://{ip}:{args.port}/snapshot")
    print("=" * 58, flush=True)

    if not args.no_register:
        threading.Thread(target=_register_loop, args=(args.port,), daemon=True).start()

    if not args.no_stats:
        try:
            from edge import recognize
            recognize.start(STATE, _CFG)
        except Exception as e:      # 识别坏了不能拖累推流
            print(f"  警告 识别线程启动失败：{type(e).__name__}: {e}", flush=True)

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
