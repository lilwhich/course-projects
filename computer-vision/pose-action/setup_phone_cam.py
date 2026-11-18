"""手机无线摄像头 —— 连接向导。

用法
----
    双击「手机摄像头.bat」，或 python setup_phone_cam.py

它会：
    1. 告诉你手机端该怎么装、怎么开
    2. 自动扫描局域网找手机（也可以手动填地址）
    3. 测试能否出帧，成功后保存到 camera_source.txt
    4. 之后 ①机位自检 / ②采集 / ③训练 / ④实时 都会自动用这个源
"""
from __future__ import annotations

import sys

import cv2

import config as C
import video_source as V

BAR = "─" * 62


def rule(ch: str = "═") -> None:
    print(ch * 62)


def finish(url: str, w: int, h: int, frame) -> int:
    V.save_source(url)
    shot = C.REPORT_DIR / "手机摄像头测试.jpg"
    try:
        cv2.imwrite(str(shot), frame)
    except Exception:
        shot = None

    print()
    rule()
    print(f"  [成功] 已连接 {url}")
    print(f"         分辨率 {w} × {h}")
    if shot:
        print(f"         测试画面已存：{shot.name}（在 reports 目录）")
    rule()
    print()
    print(" 已写入 camera_source.txt。")
    print(" 之后 ①机位自检 / ②采集数据 / ④实时识别 都会自动用手机摄像头，")
    print(" 不需要每次再填地址。")
    print()

    if h < 480:
        print(f" [提示] 分辨率偏低（{h}p），关键点会发飘。")
        print("        在手机 App 的设置里把分辨率调到 720p 或以上。")
    elif h >= 1080:
        print(f" [提示] 分辨率 {h}p 偏高，WiFi 带宽吃紧时容易卡。")
        print("        如果画面卡顿，在手机 App 里降到 720p 更稳。")
    print()
    print(" 下一步：双击「①机位自检.bat」")
    print("         把手机放到大约 1 米高、镜头略朝上，或者干脆放地上仰拍，")
    print("         人要退到能拍到膝盖以下的位置。")
    return 0


def try_urls(urls: list[str]) -> int:
    for url in urls:
        print(f"   测试 {url} ...", end="", flush=True)
        ok, w, h, fr = V.probe(url, reads=4)
        print(" 成功" if ok else " 失败")
        if ok:
            return finish(url, w, h, fr)
    return -1


def main() -> int:
    rule()
    print("  手机无线摄像头 —— 连接向导")
    rule()
    print("""
 手机端准备（只需做一次）：

   Android：应用商店搜「IP Webcam」→ 安装 → 打开
            滑到最底部，点「Start server」
            屏幕上会显示一行地址，形如 192.168.1.7:8080

   iPhone ：App Store 搜「IP Camera Lite」→ 打开 → 点「Turn on IP Camera Server」
            点底部「Connect?」，抄 MJPEG 那一行地址
            该 App 默认账号密码都是 admin，地址形如
              http://admin:admin@192.168.1.7:8080/video
            把 IP 和端口填进来也行，程序会自动补上 admin:admin
            例：admin:admin@192.168.1.7:8080

   ⚠ 手机和这台电脑必须连同一个 WiFi。
     手机用流量、或路由器开了「AP 隔离」，都会连不上。
     ★ 最稳的做法：手机开热点，电脑连手机的热点。
""")
    rule()

    cur = V.load_source()
    print(f" 当前视频源：{V.describe(cur)}")
    print()
    print(" 直接回车  = 自动扫描局域网找手机（约 3 秒）")
    print(" 填地址    = 手动指定，例如 192.168.1.7:8080")
    print("             iPhone 若提示要密码，填 admin:admin@192.168.1.7:8080")
    print(" 填 0      = 改回用电脑自带摄像头")
    print()

    try:
        ans = input(" > ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return 0

    # ---------- 改回本机摄像头 ----------
    if ans == "0":
        V.save_source("0")
        print()
        print(" 已改回电脑自带摄像头。")
        return 0

    # ---------- 手动地址 ----------
    if ans:
        url = V.normalize(ans)
        print()
        print(f" 正在测试 {url}（最多等 10 秒）…")
        rc = try_urls([url])
        if rc >= 0:
            return rc
        print()
        print(" [失败] 这个地址读不到画面。常见原因：")
        print("   1. 端口或路径不对（IP Webcam 是 /video，不是 /）")
        print("   2. 手机 App 还没点「启动服务器 / Start server」")
        print("   3. 手机上显示的是 127.0.0.1 之类，那是手机自己，要用局域网 IP")
        for line in V.advice_text(url):
            print(line)
        return 1

    # ---------- 自动扫描 ----------
    ip = V.local_ip()
    print()
    print(f" 本机局域网地址 {ip}，正在扫描同网段…")
    t0 = __import__("time").time()
    cands = V.discover()
    print(f" 扫描完成，用时 {__import__('time').time() - t0:.1f}s，"
          f"发现 {len(cands)} 个候选设备")

    if cands:
        print()
        rc = try_urls(cands)
        if rc >= 0:
            return rc

    # ---------- 扫不到，给排查清单 ----------
    print()
    rule()
    print("  [没找到] 请按顺序检查：")
    print("    1. 手机上的 App 是否已经点了「启动 / Start server」")
    print("    2. 手机连的是不是同一个 WiFi（不是手机流量）")
    print("    3. 路由器是否开了「AP 隔离 / 访客网络」，开了要关掉")
    print("    4. Windows 防火墙首次会弹窗，要允许 python 访问「专用网络」")
    print()
    print("  也可以手动填手机上显示的那行地址，例如 192.168.1.7:8080")
    rule()
    try:
        ans = input(" 手动输入地址（直接回车退出）> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return 0
    if not ans:
        return 0

    print()
    print(f" 正在测试 {V.normalize(ans)} …")
    if try_urls([V.normalize(ans)]) >= 0:
        return 0
    for line in V.advice_text(ans):
        print(line)
    return 1


if __name__ == "__main__":
    sys.exit(main())
