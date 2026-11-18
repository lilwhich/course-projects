"""全局配置：路径、模型参数、行为类别、COCO-17 关键点定义。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
FEAT_DIR = ROOT / "data" / "features"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

for _d in (RAW_DIR, FEAT_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------- 检测模型 ----------
MODEL_NAME = "yolov8n-pose.pt"   # n 最快，够用；想更准可换 yolov8s-pose.pt
IMGSZ = 480                      # 480 在 4 核 CPU 上约 11 FPS，640 约 7 FPS
CONF = 0.4                       # 人体框置信度阈值
IOU = 0.5
DEVICE = "cpu"                   # 有 N 卡改成 "0"
MAX_PERSONS = 1                  # 只跟画面中"最大"的那个人，避免路人干扰
CPU_THREADS = 0                  # 0=自动取 CPU 核心数，实测比 torch 默认值快 4 倍

# ---------- 行为类别 ----------
# 键盘 1/2/3/4 -> 标签
HOTKEY_LABEL = {ord("1"): "stand", ord("2"): "sit",
                ord("3"): "walk", ord("4"): "squat"}
LABEL_NAMES = ["stand", "sit", "walk", "squat"]
LABEL_ZH = {"stand": "站立", "sit": "坐下", "walk": "行走", "squat": "蹲下"}

# ---------- 时序窗口（按时间，不按帧数）----------
# 用时间而不是帧数，是为了让模型不依赖机器性能：
# 同一台机器 11 FPS、换台机器 30 FPS，1.5 秒窗口覆盖的行为完全一样。
# 1.5 秒 ≈ 2.4 个步态周期（步频 1.6Hz），足够体现走路的周期性。
WINDOW_SEC = 1.5     # 一个判定窗口的时间长度（秒）
STRIDE_SEC = 0.30    # 滑窗步长（秒）
MIN_FRAMES = 8       # 一个窗口至少要这么多帧才有效（帧率过低时自动跳过）

# ---------- 关键点可用性 ----------
KP_CONF = 0.30       # 单关键点可信阈值，低于此视为缺失
# 判定坐/站必须看到下半身，所以只要求核心点（髋、膝）齐备即可，
# 手腕、耳朵等缺失不影响判定，没必要整帧作废。
# CORE_KP 见文件末尾（依赖下面的索引常量）

# ---------- COCO-17 关键点 ----------
KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
NUM_KP = len(KP_NAMES)

NOSE = 0
L_SHO = 5; R_SHO = 6
L_ELB = 7; R_ELB = 8
L_WRI = 9; R_WRI = 10
L_HIP = 11; R_HIP = 12
L_KNE = 13; R_KNE = 14
L_ANK = 15; R_ANK = 16

# 一帧有效的条件（见 features.frame_features）：
#   双髋可见（髋中点是归一化原点）+ 至少一个膝盖可见。
# 只露上半身时无法区分坐/站，这类帧会被丢弃并提示用户后退。

# 用于可视化的骨架连线
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
]

# 逐帧特征名，顺序必须与 features.frame_features 的输出一致
FRAME_FEATURE_NAMES = [
    "torso_angle",        # 躯干与竖直方向夹角(度)
    "torso_len",          # 肩髋距 / 框高，反映离镜头远近与被遮挡
    "hip_angle_L",        # 左髋角(肩-髋-膝)
    "hip_angle_R",
    "knee_angle_L",       # 左膝角(髋-膝-踝)
    "knee_angle_R",
    "hip_height_ratio",   # 髋中点离画面底部的高度 / 框高，坐下、蹲下都会下降
    "bbox_aspect",        # 框宽/框高，坐下、蹲下时变宽
    "ankle_spread",       # 左右踝水平距离 / 框高
    "wrist_below_hip",    # 手腕低于髋的幅度 / 框高
    "center_x",           # 髋中点归一化横坐标
    "center_y",           # 髋中点归一化纵坐标
    "speed",              # 髋中点帧间速度(框高/秒)
    "head_hip_dy",        # 鼻到髋的垂直距离 / 框高
    "knee_above_hip",     # 膝中点高出髋中点的幅度 / 框高：蹲深时>0，坐与站<=0
]

# ---------- 视频源 ----------
# 默认用本机摄像头；跑过「手机摄像头.bat」后会写入 camera_source.txt，
# 之后所有脚本自动改用手机。
# 这里内联读取，不 import video_source，避免底层模块依赖 cv2。
_SOURCE_FILE = ROOT / "camera_source.txt"


def _load_source() -> str:
    try:
        if _SOURCE_FILE.exists():
            s = _SOURCE_FILE.read_text(encoding="utf-8").strip()
            if s:
                return s
    except Exception:
        pass
    return "0"


SOURCE = _load_source()

# ---------- 画面旋转 ----------
# 手机竖着放时，传感器仍按横屏出图，画面会躺倒 90°。
# 这里记录一个默认旋转角（0/90/180/270），由 camera_rotate.txt 决定，
# 各脚本的 --rotate 参数可以临时覆盖，不用改 App 设置。
_ROTATE_FILE = ROOT / "camera_rotate.txt"


def _load_rotate() -> int:
    try:
        if _ROTATE_FILE.exists():
            v = _ROTATE_FILE.read_text(encoding="utf-8").strip()
            if v:
                return int(v) % 360
    except Exception:
        pass
    return 0


ROTATE = _load_rotate()
