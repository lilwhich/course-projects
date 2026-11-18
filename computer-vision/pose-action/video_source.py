"""统一的视频源：本地摄像头 / 手机无线摄像头 / 视频文件。

为什么需要这一层
----------------
笔记本屏幕摄像头的机位通常太高、视角太窄，拍不到膝盖以下，
而站和坐在骨骼上的差别全在下半身——这是物理限制，调代码没用。

手机可以随手放到地面、椅子上、书架下层，镜头朝上仰拍全身，
是最省事的解法。手机端只需要一个能在局域网里提供 MJPEG 流的 App，
PC 端不装任何驱动，OpenCV 直接读 http 地址即可。

支持的地址写法
--------------
    0                        本机第一个摄像头
    1                        本机第二个摄像头
    192.168.1.7:8080         只写 IP 和端口（自动补 http:// 和 /video）
    http://192.168.1.7:8080/video    完整地址
    rtsp://...               RTSP 流
    D:/videos/walk.mp4       本地视频文件
"""
from __future__ import annotations

import re
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
SOURCE_FILE = ROOT / "camera_source.txt"

# 常见的 MJPEG 端点与端口
COMMON_PATHS = ("/video", "/videofeed", "/mjpeg", "/stream", "/shot.jpg")
COMMON_PORTS = (8080, 4747, 8081, 8000, 5000)


# ---------------------------------------------------------------- 地址规范化
def normalize(raw) -> str:
    """把用户随手输入的东西变成 cv2 能吃的地址。"""
    s = str(raw).strip().strip('"').strip("'").strip()
    if not s:
        return "0"
    if s.isdigit():
        return s
    if re.match(r"^(https?|rtsp)://", s, re.I):
        return s
    if Path(s).exists():
        return str(Path(s).resolve())
    # 只给了 IP 或 IP:端口，或带了账号密码（iPhone 的 IP Camera Lite 默认 admin/admin）
    m = re.match(r"^(?:([^:@/\s]+):([^@/\s]*)@)?"
                 r"(\d{1,3}(?:\.\d{1,3}){3})(?::(\d+))?(/.*)?$", s)
    if m:
        user, pwd, host, port, path = m.groups()
        auth = f"{user}:{pwd}@" if user else ""
        port = port or "8080"
        return f"http://{auth}{host}:{port}{path or '/video'}"
    return s


def is_network(src: str) -> bool:
    return bool(re.match(r"^(https?|rtsp)://", str(src).strip(), re.I))


def describe(src: str) -> str:
    """给用户看的来源描述。"""
    src = normalize(src)
    if src.isdigit():
        return f"本机摄像头 {src}"
    if is_network(src):
        return f"网络摄像头 {src}"
    return f"视频文件 {Path(src).name}"


# ---------------------------------------------------------------- 取流
def open_stream(source, width: int = 0, height: int = 0):
    """打开视频源。width/height 只对本地摄像头有效，网络流由手机决定。"""
    src = normalize(source)
    if src.isdigit():
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
        cap = cv2.VideoCapture(int(src), backend)
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 只留一帧，别攒
        except Exception:
            pass
        return cap
    cap = cv2.VideoCapture(src)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def rotate_frame(frame, deg: int):
    """按角度摆正画面：手机竖持时传感器仍给横图，需要转 90/270。"""
    deg = int(deg) % 360
    if deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


class SourceReader:
    """读帧封装：后台线程持续抓帧，主线程永远拿到**最新**那一帧。

    为什么要单独开线程（手机无线流的关键）
    --------------------------------------
    MJPEG over HTTP 是一条持续推送的流，OpenCV 内部有解码缓冲。如果主线程
    一边读帧一边跑 YOLO（一帧 80~130ms），读的比推的慢，缓冲就越积越多：
    画面看起来"越来越滞后"，跑一分钟能落后好几秒，而且不会自己恢复。

    后台线程只管一件事——不停地 grab，只把最新一帧留在内存里，旧的直接丢。
    主线程要帧时拿到的永远是最新画面，缓冲堆积从根上被掐断。
    顺带还能报出「这一帧是多久以前抓到的」，即真实画面延迟。

    threaded=False 时退回原来的同步读法（本地摄像头、离线视频文件够用）。
    """

    def __init__(self, source, width: int = 0, height: int = 0,
                 max_fail: int = 10, verbose: bool = True, threaded: bool = True,
                 rotate: int = 0):
        self.source = normalize(source)
        self.width, self.height = width, height
        self.max_fail = max_fail
        self.verbose = verbose
        self.network = is_network(self.source)
        self.rotate = int(rotate) % 360
        self.fail = 0
        self.reconnects = 0
        self.cap = open_stream(self.source, width, height)

        # 后台抓帧用
        self.threaded = bool(threaded)
        self._frame = None          # 最新一帧
        self._seq = 0               # 帧序号，单调递增
        self._t = 0.0               # 该帧被抓到的时刻
        self._last_seq = -1
        self._fps_mark = (time.time(), 0)
        self._fps_val = 0.0
        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._thread = None
        if self.threaded and self.is_open:
            self._start()

    # ------------------------------------------------------------ 后台抓帧
    @property
    def is_open(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def _start(self):
        self._thread = threading.Thread(target=self._loop, name="frame-grabber",
                                        daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            cap = self.cap
            if cap is None:
                time.sleep(0.05)
                continue
            try:
                ok, frame = cap.read()
            except Exception:
                ok, frame = False, None

            if ok and frame is not None and getattr(frame, "size", 0):
                if self.rotate:
                    frame = rotate_frame(frame, self.rotate)
                with self._cond:
                    self._frame = frame
                    self._seq += 1
                    self._t = time.time()
                    self._cond.notify_all()
                self.fail = 0
                continue

            self.fail += 1
            # 只有网络流才重连：本地摄像头读失败多半是真拔了，重连没意义
            if self.network and self.fail >= self.max_fail:
                if self.verbose:
                    print(f"[重连] 连续 {self.fail} 帧读不到，正在重新连接…")
                self._reopen()
                self.reconnects += 1
                self.fail = 0
            else:
                time.sleep(0.01)

    def _reopen(self):
        old = self.cap
        self.cap = None
        if old is not None:
            try:
                old.release()
            except Exception:
                pass
        time.sleep(0.6)
        if self._stop.is_set():
            return
        self.cap = open_stream(self.source, self.width, self.height)

    # ------------------------------------------------------------ 取帧
    def read(self, wait: float = 0.3):
        """返回 (ok, frame)。threaded 模式下返回最新帧的副本。"""
        if not self.threaded:
            return self._read_sync()

        with self._cond:
            if self._frame is None:
                self._cond.wait(max(wait, 1.5))      # 等第一帧
            elif self._seq == self._last_seq:
                self._cond.wait(wait)                # 等下一帧，最多 wait 秒
            if self._frame is None:
                return False, None
            self._last_seq = self._seq
            return True, self._frame.copy()

    def _read_sync(self):
        ok, frame = (False, None)
        if self.cap is not None:
            try:
                ok, frame = self.cap.read()
            except Exception:
                ok, frame = False, None
        if ok and frame is not None and getattr(frame, "size", 0):
            self.fail = 0
            return True, (rotate_frame(frame, self.rotate) if self.rotate else frame)
        self.fail += 1
        if self.network and self.fail >= self.max_fail:
            if self.verbose:
                print(f"[重连] 连续 {self.fail} 帧读不到，正在重新连接…")
            self._reopen()
            self.reconnects += 1
            self.fail = 0
        return False, None

    # ------------------------------------------------------------ 状态
    @property
    def frame_age(self) -> float:
        """当前最新帧是几秒前抓到的——即画面相对真实的滞后。"""
        if not self._t:
            return 0.0
        return max(0.0, time.time() - self._t)

    @property
    def fps(self) -> float:
        """后台线程的抓帧帧率（近 1 秒）。"""
        now = time.time()
        with self._cond:
            seq = self._seq
        if now - self._fps_mark[0] < 1.0:
            return self._fps_val
        self._fps_val = (seq - self._fps_mark[1]) / max(now - self._fps_mark[0], 1e-6)
        self._fps_mark = (now, seq)
        return self._fps_val

    def release(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None


# ---------------------------------------------------------------- 探测
def local_ip() -> str:
    """取本机在局域网里的地址，用来推断网段。"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.4)
        s.connect(("223.5.5.5", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def ip_prefix24(ip: str) -> str:
    """取 IPv4 前三段，用来判断是不是同一个 /24 网段。"""
    parts = str(ip).strip().split(".")
    return ".".join(parts[:3]) if len(parts) == 4 else ""


def host_of(src: str) -> str:
    """从地址里抠出 IP（去掉协议、账号密码、端口、路径）。"""
    s = normalize(src)
    s = re.sub(r"^[a-z]+://", "", s, flags=re.I)
    s = s.split("@")[-1]          # 去掉 user:pass@
    s = s.split("/")[0]           # 去掉路径
    s = s.split(":")[0]           # 去掉端口（IPv6 另说，这里只管 IPv4）
    return s


def same_subnet(src: str, me: str | None = None) -> bool:
    """手机和本机是否在同一 /24 网段——不在的话扫描和直连都必然失败。"""
    me = me or local_ip()
    host = host_of(src)
    if not host or host.startswith("127.") or host == "":
        return True               # 本机摄像头 / 视频文件，不参与判断
    a, b = ip_prefix24(host), ip_prefix24(me)
    return bool(a) and a == b


def advice_text(src: str, me: str | None = None) -> list[str]:
    """连不上时，按实际网段给出针对性建议，而不是泛泛的排查清单。"""
    me = me or local_ip()
    host = host_of(src)
    lines: list[str] = []
    if not same_subnet(src, me):
        lines += [
            "",
            f"  [定位] 本机 {me}  ←→  手机 {host}",
            "         两者不在同一网段，属于校园网/公共 WiFi 的「设备隔离」。",
            "         这种网络下任何两台设备都互相看不见，不是程序问题，",
            "         手动填地址也一样连不上。",
            "",
            "  [解决] 手机开热点，电脑连手机的热点：",
            "         1. 手机：设置 → 个人热点 → 打开",
            "         2. 电脑：右下角 WiFi → 连这个热点（会断开校园网，正常）",
            "         3. 手机 App 里那行地址会变成 192.168.43.x 之类",
            "         4. 再跑一次本向导，直接回车自动扫描",
            "",
            "         电脑连热点后没外网不影响使用，这套程序全离线跑。",
        ]
    else:
        lines += [
            "",
            f"  [定位] 本机 {me} 与手机 {host} 在同一网段，网络是通的，",
            "         问题多半在手机 App 或防火墙上：",
            "         1. 手机 App 里是否点了「启动服务器 / Start server」",
            "         2. Windows 防火墙首次弹窗是否点了「允许」",
            "            控制面板 → Windows Defender 防火墙 →",
            "            允许应用通过防火墙 → 找到 python.exe，勾「专用」",
        ]
    return lines


def _port_open(host: str, port: int, timeout: float = 0.35) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        return s.connect_ex((host, port)) == 0
    finally:
        s.close()


def discover(tries_per_port: int = 1) -> list[str]:
    """扫描局域网，返回候选的视频地址（尚未验证能否出帧）。"""
    me = local_ip()
    if me.startswith("127."):
        return []
    base = me.rsplit(".", 1)[0]
    hosts = [f"{base}.{i}" for i in range(1, 255)]

    hits: list[tuple[str, int]] = []
    with ThreadPoolExecutor(max_workers=200) as ex:
        futs = {ex.submit(_port_open, h, p): (h, p)
                for h in hosts for p in COMMON_PORTS}
        for f in as_completed(futs):
            try:
                if f.result():
                    hits.append(futs[f])
            except Exception:
                continue

    return [f"http://{h}:{p}/video" for h, p in sorted(hits)]


def probe(url, width: int = 0, height: int = 0, reads: int = 3):
    """测试地址能否出帧。返回 (ok, 宽, 高, 最后一帧)。"""
    cap = open_stream(url, width, height)
    if not cap.isOpened():
        cap.release()
        return False, 0, 0, None
    frame = None
    for _ in range(max(reads, 1)):
        ok, fr = cap.read()
        if ok and fr is not None and fr.size:
            frame = fr
    cap.release()
    if frame is None:
        return False, 0, 0, None
    h, w = frame.shape[:2]
    return True, w, h, frame


# ---------------------------------------------------------------- 持久化
def load_source() -> str:
    """读取上次保存的视频源，默认本机摄像头 0。"""
    try:
        if SOURCE_FILE.exists():
            s = SOURCE_FILE.read_text(encoding="utf-8").strip()
            if s:
                return s
    except Exception:
        pass
    return "0"


def save_source(src: str) -> None:
    SOURCE_FILE.write_text(normalize(src), encoding="utf-8")
