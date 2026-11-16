# 录音转文字

一个本地离线运行的语音转文字小工具，基于 [faster-whisper](https://github.com/SYSTRAN/faster-whisper)。

把录音（mp3 / m4a / wav / flac / mp4 等）转成文字，同时生成带时间轴的 `.srt` 字幕。
不上传云端、不需要 API Key，除第一次下载模型外全程可断网使用。

## 特点

- **完全离线** —— 音频不出本机，隐私敏感场景可用
- **零外部依赖** —— 不需要单独安装 ffmpeg，靠 PyAV 直接解码
- **双份输出** —— 同名 `.txt`（纯文字）+ 同名 `.srt`（时间轴字幕）
- **提示词纠错** —— `提示词.txt` 里写人名、班级、专业术语，明显减少同音字错认
- **批量处理** —— 支持多文件、整个文件夹、通配符
- **国内下载** —— 模型默认走 ModelScope（魔搭），实测 10 MB/s 左右

## 环境要求

- Python 3.9+（推荐 3.11）
- 无需显卡，CPU 即可运行

```bash
pip install -r requirements.txt
```

> `av` 必须小于 19。av 19 删掉了 faster-whisper 依赖的 `metadata_errors` 参数，
> 装最新版会报 `TypeError: open() got an unexpected keyword argument 'metadata_errors'`。

## 快速开始

```bash
# 转单个文件
python 录音转文字.py 录音.m4a

# 一次转多个
python 录音转文字.py 第一段.mp3 第二段.wav

# 整个文件夹批量
python 录音转文字.py D:\我的录音

# 更高精度（更慢）
python 录音转文字.py 会议.m4a --model medium

# 现场录 60 秒麦克风再转写（需 pip install sounddevice numpy）
python 录音转文字.py --mic 60

# 不带参数运行，会提示粘贴路径
python 录音转文字.py
```

Windows 上也可以双击 `启动.bat`，把音频文件拖进窗口回车即可。

输出默认写在音频文件同目录：

```
录音.m4a  ->  录音.txt  +  录音.srt
```

## 提示词（重要）

同目录下的 `提示词.txt` 会在启动时自动读取，作为识别的上下文提示。
把常用人名、班级、课程名、专业术语写进去，同音字错认会大幅减少。

例如里面写了「张三」，就不会再被识别成「张珊」。

```text
以下是普通话录音，可能是自我介绍、课堂讲授或会议讨论，涉及张三、机器学习、深度学习、Python、图像识别、语音识别等词。
```

一行中文即可，改完保存下次运行就生效。命令行传了 `--prompt` 时以命令行为准。

## 模型

首次运行会自动下载到 `models/modelscope-faster-whisper-small`（约 461 MB），之后完全离线。

| 模型 | 体积 | 说明 |
|---|---|---|
| tiny | ~75 MB | 最快，效果一般，适合试水 |
| base | ~145 MB | 较快，能听懂大意 |
| small | ~461 MB | **默认**，日常会议 / 网课够用 |
| medium | ~1.5 GB | 精度明显更好，CPU 上较慢 |
| large-v3 | ~3 GB | 精度最好，建议有 NVIDIA 显卡 |

模型目录不进版本库（见 `.gitignore`），换机器重新下载一次即可。
也可以用 `--model` 直接指向一个已有的本地模型文件夹，实现 U 盘离线搬运。

## 参数速查

| 参数 | 说明 | 默认 |
|---|---|---|
| `--model` | tiny / base / small / medium / large-v3，或本地模型目录 | small |
| `--language` | `zh` / `en` / `auto` | zh |
| `--device` | `cpu` / `cuda` | cpu |
| `--compute-type` | cpu 默认 int8，cuda 默认 float16 | 自动 |
| `--prompt` | 提示词，覆盖 `提示词.txt` | 无 |
| `--out` | 输出目录 | 音频同目录 |
| `--no-srt` | 不生成 srt | 关闭 |
| `--no-vad` | 关闭静音检测 | 关闭 |
| `--model-dir` | 模型缓存目录 | 脚本旁 `models` |
| `--download-source` | `modelscope` / `huggingface` | modelscope |
| `--mic N` | 录 N 秒麦克风再转写 | 0 |

## 识别不准时

1. **先加提示词** —— 人名、术语的同音字最有效，成本最低
2. **换更大模型** —— `--model medium` 或 `large-v3`
3. **确认音频本身清晰** —— 距离远、环境嘈杂的录音上限就在那
4. **极短音频本来就容易错** —— 4~5 秒的片段上下文太少，长音频准确率明显更高

## 项目结构

```
录音转文字/
├── 录音转文字.py      主程序
├── 提示词.txt         识别提示词，可自行编辑
├── 启动.bat           Windows 双击启动
├── 安装依赖.bat       Windows 一键装依赖
├── requirements.txt
├── 使用说明.txt
└── models/            模型缓存（自动下载，已被 git 忽略）
```

## 许可

MIT
