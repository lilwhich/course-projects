# 教室在室人数统计 · classroom-cv

> 课程设计 · 两台电脑 + 网络摄像头的主从分布式系统

摄像头挂在从机上，识别与统计跑在主机上。从机开机即自动找到主机注册，不用手填 IP；主机汇总事件、算出在室名单与停留时长，用网页看板展示，可导出 CSV。

人脸检测与识别只用 OpenCV 自带的 YuNet + SFace，不引入 insightface / dlib —— 后者在 Windows 上需要 MSVC 编译工具链，前者 `pip install opencv-python` 之后直接可用。

---

## 架构

```
从机 edge（端口 8767）                     主机 master（端口 8765）
┌──────────────────────────┐              ┌──────────────────────────┐
│ 摄像头（只打开一次）      │              │ FastAPI 汇总服务          │
│   ├→ MJPEG 推流 ─────────┼── HTTP ────→ │   ├→ 双画面实时预览       │
│   └→ 检测 → 跟踪 → 投票  │              │   ├→ SQLite 事件账本      │
│        ↓                 │   上报事件    │   ├→ Web 统计看板         │
│      身份 + 时间戳 ──────┼── POST ────→ │   └→ CSV 导出             │
└──────────────────────────┘              └──────────────────────────┘
          ↑                                            ↑
          └──── UDP 8766 广播发现，从机免配 IP ─────────┘
```

| 端口 | 协议 | 用途 |
|---|---|---|
| 8765 | HTTP | 主机汇总服务、看板、预览、导出 |
| 8766 | UDP | 发现广播，主机应答自己的地址 |
| 8767 | HTTP | 从机摄像头服务（MJPEG 流 + 事件接口） |

---

## 快速开始

### 0. 环境

Python 3.11（3.9 以上可用，脚本优先找 `py -3.11`）。不需要显卡。

### 1. 下载人脸模型（必需）

模型不进版本库，首次使用需手动放到 `models/` 目录：

```powershell
New-Item -ItemType Directory -Force models | Out-Null

# 人脸检测 YuNet（232 KB）
curl.exe -L -o models/yunet.onnx `
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

# 人脸识别 SFace（38.7 MB）
curl.exe -L -o models/face_recognition_sface_2021dec.onnx `
  https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

校验（两个值都必须对上）：

| 文件 | 大小 | SHA-256 |
|---|---|---|
| `models/yunet.onnx` | 232,589 B | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| `models/face_recognition_sface_2021dec.onnx` | 38,696,353 B | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |

```powershell
Get-FileHash models/*.onnx -Algorithm SHA256 | Format-List
```

> 两个模型均来自 [OpenCV Zoo](https://github.com/opencv/opencv_zoo)，许可证见该仓库。
> 缺 YuNet 时会退化到 OpenCV 自带的 Haar 级联：能跑通，但精度低一档。

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

Windows 上直接双击 `setup_edge.bat`，它会自检 Python、装依赖、再启动摄像头服务。

### 3. 建身份库（可选）

不建也能跑，只是所有人都计为「陌生人」，只统计人数、不区分是谁。

建库的目录约定是每人一个文件夹、每个文件夹 5 张以上不同角度的照片：

```
data/faces/张三/1.jpg
data/faces/张三/2.jpg
data/faces/李四/1.jpg
```

然后生成特征库：

```bash
py -3.11 -m tools.enroll            # 全量重建 data/gallery.npz
py -3.11 -m tools.enroll --list     # 查看现有身份
py -3.11 -m tools.enroll --test 1.jpg   # 拿一张图试试会认成谁
```

### 4. 启动

| 场景 | 操作 |
|---|---|
| 单机演示（一台电脑同时当主机和从机） | 双击 `run_all.bat` |
| 主机 | 双击 `run_master.bat` |
| 从机 | 双击 `setup_edge.bat`（首次会自动装依赖） |
| 只想测连通性 | 双击 `run_probe.bat`，或 `py -3.11 -m edge.probe` |
| 环境自检 | 双击 `check_env.bat` |

主机启动后打开 <http://127.0.0.1:8765> 看双画面，<http://127.0.0.1:8765/board> 看统计看板。

从机默认用摄像头 0，多路或换设备：

```bash
py -3.11 -m edge.camera_server --cam 1
py -3.11 -m edge.camera_server --source test.mp4    # 用视频文件顶替摄像头
py -3.11 -m edge.camera_server --no-register        # 只开服务，不注册到主机
```

---

## 几个关键设计

### 1. 一个摄像头只打开一次

Windows 上重复打开同一个摄像头设备会报错。推流和识别都需要画面，所以 `camera_server` 只用一个读取线程，识别线程拿同一份帧。

### 2. 四道过滤

单帧识别结果不可信，直接上报会产生大量假事件。链路里串了四道过滤：

| 闸 | 规则 | 挡住什么 |
|---|---|---|
| 1 | 人脸框小于 `MIN_DIM` 直接丢弃 | 远处模糊的误检 |
| 2 | 同一轨迹累计 3 帧识别为同一名字才确认 | 单帧抖动 / 瞬时误判 |
| 3 | 同一身份、同一机位 20 秒内只报一次 | 站在镜头前反复触发 |
| 4 | 主机连不上时写本地队列，恢复后补传 | 网络抖动导致丢事件 |

### 3. uuid 幂等：两台机器各自识别也能合并

每条事件以 uuid 为主键。从机断网补传、重复上报都不会重复记账——这是「主从各自识别」能安全汇总的前提。少了这个，补传一次人数就翻倍。

### 4. 不存「在室表」

在室名单不是一个字段，而是由 `status='OPEN'` 的会话实时算出来的。少一张表就少一份可能与事件流水不一致的数据。

### 5. 陌生人用先进先出配对

没有身份时，无法用「谁」把人串起来。剩下的合理假设只有一条：最早进去的那个先出来。配对质量不如有身份时，但不会凭空造出一堆永不结束的会话。

### 6. 不用 insightface / dlib

这两个库精度更好，但在 Windows 上安装需要处理编译器和依赖链。OpenCV 自带 YuNet（检测）+ SFace（128 维特征，余弦相似度比对），`pip install` 后直接可用，CPU 上可实时。

### 7. 从机免配 IP

主机在 UDP 8766 上应答，从机广播一次即可获知主机地址，并把结果缓存 24 小时。换教室、换网段不用改配置文件。

---

## 目录结构

```
classroom-cv/
├── common/
│   ├── face.py        检测 + 特征提取 + 身份比对 + 轨迹跟踪
│   ├── ledger.py      SQLite 事件账本：流水、进出配对、停留时长
│   └── netcfg.py      网络自发现，两端共用
├── edge/
│   ├── camera_server.py   摄像头服务：MJPEG 推流 + 注册 + 快照
│   ├── recognize.py       识别线程：检测→跟踪→投票→上报
│   ├── probe.py           连通性探针
│   └── list_cams.py       列出本机可用摄像头
├── master/
│   ├── app.py         FastAPI 主服务
│   ├── stats.py       统计接口 + CSV 导出
│   └── board.html     统计看板
├── tools/
│   ├── enroll.py      建身份库
│   └── simulate.py    灌模拟事件，不用摄像头也能调看板
├── models/            ← 自行下载（已 gitignore）
├── data/              ← 运行时生成（已 gitignore）
├── setup_edge.bat     从机一键安装 + 启动
├── run_all.bat        单机同时起主机与摄像头
└── check_env.bat      环境自检
```

---

## 已知局限

- **无活体检测**：2D 特征比对，照片对着镜头理论上能骗过。只适合「统计在室人数」这类低风险场景，不能用于门禁。
- **遮挡会断轨**：跟踪依赖帧间位置连续性，人从镜头前走过、被桌椅上的人挡住，轨迹会断开，表现为一次多余的进出。
- **陌生人无法跨机位关联**：不知道身份，就没法判断「入口进去的那个人」和「出口出来的那个人」是不是同一个，只能按先进先出配。
- **明文 HTTP**：局域网内通信不加密，只适合可信内网。
- **仅在 Windows 上验证**：批处理启动脚本、摄像头打开方式都是 Windows 路径，Linux 需自己改。
- **光照敏感**：逆光、侧光下人脸特征会漂移，识别率下降。代码里没有做光照归一化。

---

## 隐私

本仓库**不包含任何人脸图片、姓名、人脸特征库或运行截图**。`data/` 与 `models/` 均在 `.gitignore` 中，身份库需要使用者按第 3 步自行建立。

人脸属于敏感个人信息。实际部署时，采集前应取得被采集者同意、明确告知用途与保存期限，并限制原始抓拍图的留存时间。本项目只提供技术实现，合规责任在使用方。
