# 课程项目合集

大学在读期间课程实践与实验项目的代码、结果图归档。

本仓库按学科分目录，每个项目自带 README、依赖清单和结果图。

## 项目索引

| 学科 | 项目 | 关键技术 | 目录 |
|---|---|---|---|
| 计算机视觉 / 视觉识别 | 多张图片合成全景图 | Python · OpenCV · SIFT · RANSAC · 柱面投影 · 接缝搜索 | [`computer-vision/panorama-stitching`](computer-vision/panorama-stitching) |
| 计算机视觉 / 人脸识别 | 教室在室人数统计（主从分布式） | Python · OpenCV · YuNet · SFace · FastAPI · SQLite · UDP 发现 | [`computer-vision/classroom-cv`](computer-vision/classroom-cv) |
| 计算机视觉 / 行为识别 | 骨骼关键点人体行为识别（站/坐/走/蹲） | Python · YOLOv8-Pose · ultralytics · 时序特征 · 滑窗 · sklearn | [`computer-vision/pose-action`](computer-vision/pose-action) |
| 语音识别 / 语音处理 | 录音转文字（离线字幕生成） | Python · faster-whisper · PyAV · VAD · ModelScope · SRT | [`speech-processing/audio-transcription`](speech-processing/audio-transcription) |

## 目录规范

```
<学科目录>/
└── <项目名>/
    ├── README.md          项目说明：背景、原理、用法、局限
    ├── requirements.txt   依赖
    ├── 源码               *.py，或按模块分包（如 common/ · edge/ · master/）
    └── assets/            结果图与中间产物（可选）
        ├── source/        输入素材
        ├── result/        最终结果 + 数值报告
        └── process/       中间过程图
```

体积大的第三方模型、运行时数据、原始素材不进版本库，由各项目自己的 `.gitignore` 排除，README 里写下载方式。

## 环境约定

各项目独立建虚拟环境，互不干扰：

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## 新增项目

```bash
# 1. 生成骨架（目录 + README + .gitignore + requirements.txt）
python _template/new_project.py --subject computer-vision --name my-demo --title "项目标题"

# 2. 放源码、填 README

# 3. 提交前检查
python scripts/check_repo.py --staged
```

检查脚本的检查项：个人敏感词（真实姓名等，按本地词表比对）、内网 IP、本机绝对路径、邮箱、手机号、班级编号、私有密钥、超过 5 MB 的文件、缺 README 或缺依赖清单。完整流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可

代码采用 [MIT 许可](LICENSE)。

第三方内容不适用本仓库的 MIT 条款：`computer-vision/classroom-cv` 的人脸模型来自 [OpenCV Zoo](https://github.com/opencv/opencv_zoo)，`speech-processing/audio-transcription` 使用的 faster-whisper 及其 Whisper 权重版权归各自作者，遵循其原始许可。

---

<sub>素材与结果图为降采样或裁切后的版本，原始高分辨率素材与第三方模型未入库（体积原因）。</sub>
