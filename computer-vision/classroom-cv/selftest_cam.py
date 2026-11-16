"""摄像头链路自检：本机起主服务 + 从机摄像头服务，再让主服务自己去拉流。

这一趟跑通，说明「注册 -> 发现 -> 拉流 -> 解码」整条链路没问题，
之后再换成两台机器，只剩网络是否互通这一个变量。
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common import netcfg  # noqa: E402

CAM_PORT = netcfg.CAM_PORT


def _s() -> requests.Session:
    s = requests.Session()
    s.trust_env = False
    return s


def _start(args, tag):
    return subprocess.Popen(
        [sys.executable, "-m", *args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def main() -> int:
    print("[1/4] 启动主服务")
    master = _start(["master.app"], "master")
    time.sleep(4)
    if master.poll() is not None:
        print(master.stdout.read() if master.stdout else "主服务启动失败")
        return 1

    print("[2/4] 启动从机摄像头服务")
    cam = _start(["edge.camera_server", "--no-register", "--port", str(CAM_PORT)], "cam")
    time.sleep(5)
    if cam.poll() is not None:
        print(cam.stdout.read() if cam.stdout else "摄像头服务启动失败")
        master.terminate()
        return 1

    ok = True
    try:
        print("[3/4] 读 /health")
        r = _s().get(f"http://127.0.0.1:{CAM_PORT}/health", timeout=5)
        h = r.json()
        print(f"     cam_ok={h['cam_ok']} {h['width']}x{h['height']} fps={h['fps']} 来源={h['source']}")
        if not h["cam_ok"]:
            print("     注意：本机没摄像头，已用占位画面，链路仍然有效")

        print("[4/4] 主服务拉流测试")
        r = _s().get(f"http://127.0.0.1:{netcfg.HTTP_PORT}/camtest",
                     params={"ip": "127.0.0.1", "port": CAM_PORT, "seconds": 5, "max_frames": 30},
                     timeout=20)
        d = r.json()
        if d.get("ok"):
            print(f"     通过 {d['frames']} 帧 · {d['fps']} fps · 首帧 {d['first_frame_ms']}ms · {d['resolution']}")
            print(f"     截图 {d['saved']}")
        else:
            print(f"     失败 {d.get('error')}")
            ok = False

        print("     注册接口验证")
        reg = _s().post(f"http://127.0.0.1:{netcfg.HTTP_PORT}/register", json={
            "name": "selftest-machine", "ip": "127.0.0.1", "port": CAM_PORT,
            "cam_ok": h["cam_ok"], "width": h["width"], "height": h["height"], "fps": h["fps"],
        }, timeout=5)
        print(f"     register -> {reg.json()}")
        print("     从 /cameras.json 读回：")
        print(f"     {_s().get(f'http://127.0.0.1:{netcfg.HTTP_PORT}/cameras.json', timeout=5).text[:300]}")
    finally:
        for p in (cam, master):
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

    print("=" * 58)
    print("  链路自检通过" if ok else "  链路自检未通过，看上面输出")
    print("=" * 58)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
