# -*- coding: utf-8 -*-
"""
录音 / 音频 转文字工具（离线运行，基于 faster-whisper）

特点：
  - 不需要联网上传，也不需要 API Key（只有第一次会下载模型文件）
  - 不需要系统安装 ffmpeg（faster-whisper 自带 PyAV 解码 mp3/m4a/wav/flac 等）
  - 自动输出两个文件：同名 .txt（纯文字）+ 同名 .srt（带时间轴字幕）
  - 模型默认从 ModelScope（魔搭）下载，国内实测约 10 MB/s，比 hf-mirror 快得多
  - 同目录下的 提示词.txt 会被自动读取，用来纠正人名、术语的同音字

环境准备（只需做一次）：
  pip install faster-whisper "av<19"
  注意 av 必须小于 19：av 19 删掉了 faster-whisper 依赖的 metadata_errors 参数，
  装最新 av 会报 TypeError: open() got an unexpected keyword argument 'metadata_errors'

常见用法：
  python 录音转文字.py 录音.mp3
  python 录音转文字.py 录音.mp3 会议录音.wav            # 一次处理多个
  python 录音转文字.py D:\\我的录音                      # 整个文件夹批量处理
  python 录音转文字.py 录音.m4a --model medium          # 更高精度（更慢）
  python 录音转文字.py 录音.mp3 --prompt "以下是普通话会议，涉及Python、机器学习"
  python 录音转文字.py --mic 60                        # 现场录 60 秒麦克风再转写
  python 录音转文字.py                                  # 不带参数会提示输入路径

模型大小（中文效果 / 速度 的权衡）：
  tiny   最快，效果一般，只适合试水
  base   较快，能听懂大意
  small  默认，日常会议/网课够用（约 500MB 内存，CPU 上 1 小时音频约 10 分钟）
  medium 精度明显更好，CPU 上较慢
  large-v3  精度最好，建议有 NVIDIA 显卡再用
"""

import argparse
import glob
import os
import sys
import time

# ---------------- 可处理的音频 / 视频后缀 ----------------
AUDIO_EXTS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus",
    ".wma", ".amr", ".3gp", ".mp4", ".mov", ".mkv", ".avi", ".webm",
}

# ---------------- 模型获取相关 ----------------
# 国内访问 huggingface.co 基本会超时，hf-mirror 镜像有时也会掉到几十 KB/s，
# 所以默认从 ModelScope（魔搭）下载同款 CTranslate2 模型，实测 10 MB/s 左右。
MS_REPOS = {
    "tiny": "pengzhendong/faster-whisper-tiny",
    "base": "pengzhendong/faster-whisper-base",
    "small": "pengzhendong/faster-whisper-small",
    "medium": "pengzhendong/faster-whisper-medium",
    "large-v3": "pengzhendong/faster-whisper-large-v3",
}

# 这 4 个文件是加载模型必需的（preprocessor_config.json 有就下，没有也能跑）
MS_MODEL_FILES = [
    "model.bin", "config.json", "tokenizer.json",
    "vocabulary.txt", "preprocessor_config.json",
]


def fix_console_encoding():
    """Windows 控制台默认可能是 GBK，强制用 UTF-8 输出，避免中文乱码。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def fmt_srt_time(seconds: float) -> str:
    """把秒数转成 SRT 字幕格式：00:00:01,250"""
    if seconds < 0:
        seconds = 0.0
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    millis = int(round((secs - int(secs)) * 1000))
    if millis == 1000:            # 四舍五入进位
        millis = 0
        secs += 1
    return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d},{millis:03d}"


def collect_input_files(paths):
    """把用户给的路径展开成音频文件列表（支持文件夹与通配符）。"""
    result = []
    for p in paths:
        # 允许直接写 *.mp3 这种通配
        expanded = glob.glob(p)
        if not expanded:
            expanded = [p]
        for item in expanded:
            if os.path.isdir(item):
                for name in sorted(os.listdir(item)):
                    full = os.path.join(item, name)
                    if os.path.isfile(full) and os.path.splitext(name)[1].lower() in AUDIO_EXTS:
                        result.append(full)
            elif os.path.isfile(item):
                result.append(item)
            else:
                print(f"[跳过] 路径不存在：{item}")
    return result


def download_from_modelscope(model_size, dest_dir):
    """从 ModelScope 拉取 CTranslate2 格式的 faster-whisper 模型（国内实测约 10 MB/s）。

    返回模型目录；失败返回 None。已下载完整的大文件会跳过，中断后重跑可续用。
    """
    import json
    import urllib.request

    repo = MS_REPOS[model_size]
    os.makedirs(dest_dir, exist_ok=True)
    api = (f"https://www.modelscope.cn/api/v1/models/{repo}"
           "/repo/files?Revision=master&Recursive=true")
    meta = json.loads(urllib.request.urlopen(api, timeout=20).read().decode("utf-8"))
    file_list = meta["Data"]["Files"]

    for item in file_list:
        name = item["Path"]
        if name not in MS_MODEL_FILES:
            continue
        target = os.path.join(dest_dir, name)
        expect = item["Size"]
        if os.path.exists(target) and os.path.getsize(target) == expect:
            print(f"[模型] 已有 {name}，跳过")
            continue

        url = f"https://www.modelscope.cn/models/{repo}/resolve/master/{name}"
        print(f"[模型] 下载 {name}（{expect / 1048576:.1f} MB）")
        tmp = target + ".part"
        got, last_pct = 0, 0
        with urllib.request.urlopen(url, timeout=30) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(262144)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                pct = int(got * 100 / expect)
                if pct >= last_pct + 10:
                    last_pct = pct
                    print(f"[模型]   {min(pct, 100)}%")
        os.replace(tmp, target)

    if os.path.exists(os.path.join(dest_dir, "model.bin")):
        return dest_dir
    return None


def load_model(model_name, device, compute_type, model_dir, hf_endpoint, download_source):
    """获取并加载模型。第一次运行会自动下载到 model_dir，之后完全离线可用。"""
    # 走 HuggingFace 下载时才用得上，必须在真正 import huggingface_hub 之前设置
    if hf_endpoint and hf_endpoint.lower() != "none":
        os.environ["HF_ENDPOINT"] = hf_endpoint

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[错误] 还没安装 faster-whisper，请先在命令行执行：")
        print("       pip install faster-whisper \"av<19\"")
        sys.exit(1)

    print(f"[加载] 模型 {model_name} / 设备 {device} / 计算类型 {compute_type}")

    # 情况一：用户直接给了一个本地模型文件夹
    if os.path.isdir(model_name):
        model_path = model_name
        print(f"[加载] 使用本地模型目录：{os.path.abspath(model_path)}")

    # 情况二：给的是 tiny/base/small/medium/large-v3，优先从 ModelScope 下
    elif download_source == "modelscope" and model_name in MS_REPOS:
        dest = os.path.join(model_dir, f"modelscope-faster-whisper-{model_name}")
        if not os.path.exists(os.path.join(dest, "model.bin")):
            print(f"[加载] 本地还没有这个模型，开始下载到 {os.path.abspath(dest)}")
            dest = download_from_modelscope(model_name, dest)
        if dest and os.path.exists(os.path.join(dest, "model.bin")):
            model_path = dest
            print(f"[加载] 使用本地模型目录：{os.path.abspath(model_path)}")
        else:
            print("[加载] ModelScope 获取失败，回退到 HuggingFace 方式")
            model_path = model_name
    else:
        model_path = model_name

    try:
        return WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type,
            download_root=model_dir,
        )
    except Exception as e:
        print(f"\n[错误] 模型加载失败：{type(e).__name__}: {e}")
        print("可能的原因与处理：")
        print("  1) 网络下不了模型 -> 加参数 --download-source modelscope（默认就是它）")
        print("  2) 模型文件损坏 -> 删掉 models 目录重新跑一次")
        print("  3) 完全离线 -> 把已下好的模型文件夹路径直接传给 --model")
        sys.exit(1)


def record_mic(seconds, out_wav):
    """从麦克风录制指定秒数的音频，保存为 16k 单声道 wav。"""
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        print("[错误] 麦克风录音需要额外依赖，请执行：")
        print("       pip install sounddevice numpy")
        sys.exit(1)

    print(f"[录音] 开始录制 {seconds} 秒，请说话……")
    data = sd.rec(int(seconds * 16000), samplerate=16000, channels=1, dtype="int16")
    sd.wait()
    print("[录音] 录制结束")

    import wave
    with wave.open(out_wav, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(data.tobytes())
    return out_wav


def transcribe_one(model, audio_path, language, prompt, vad_filter):
    """对单个音频做转写，返回 (段落列表, 纯文本, 语言信息)。"""
    t0 = time.time()
    segments, info = model.transcribe(
        audio_path,
        language=language,          # None = 自动识别语种
        initial_prompt=prompt,      # 给模型一点上下文/术语提示，可提高专有名词准确率
        vad_filter=vad_filter,      # 静音检测，自动跳过长时间无声段落
        vad_parameters=dict(min_silence_duration_ms=500),
        beam_size=5,
        condition_on_previous_text=False,  # 关掉可减少"越听越错"的累积偏航
    )

    print(f"[识别] 检测到的语言：{info.language}（置信度 {info.language_probability:.2f}），"
          f"音频时长 {info.duration:.1f} 秒")

    collected = []
    for seg in segments:            # 这是生成器，边解码边产出，所以边打印边收集
        collected.append((seg.start, seg.end, seg.text.strip()))
        print(f"  [{fmt_srt_time(seg.start)} -> {fmt_srt_time(seg.end)}] {seg.text.strip()}")

    plain = "\n".join(s[2] for s in collected)
    print(f"[完成] {os.path.basename(audio_path)} 用时 {time.time() - t0:.1f} 秒，"
          f"共 {len(collected)} 段")
    return collected, plain, info


def save_outputs(audio_path, segments, plain, out_dir=None, make_srt=True):
    """写出 .txt（纯文字）与 .srt（时间轴字幕）。"""
    base = os.path.splitext(os.path.basename(audio_path))[0]
    folder = out_dir or os.path.dirname(os.path.abspath(audio_path))

    txt_path = os.path.join(folder, base + ".txt")
    # utf-8-sig：记事本、Excel、Word 打开都不会乱码
    with open(txt_path, "w", encoding="utf-8-sig") as f:
        f.write(plain + "\n")

    srt_path = None
    if make_srt:
        srt_path = os.path.join(folder, base + ".srt")
        with open(srt_path, "w", encoding="utf-8-sig") as f:
            for i, (start, end, text) in enumerate(segments, 1):
                f.write(f"{i}\n{fmt_srt_time(start)} --> {fmt_srt_time(end)}\n{text}\n\n")

    print(f"[输出] 文字：{txt_path}")
    if srt_path:
        print(f"[输出] 字幕：{srt_path}")
    return txt_path, srt_path


def main():
    fix_console_encoding()

    parser = argparse.ArgumentParser(
        description="录音/音频转文字（离线 faster-whisper）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="*", help="音频文件、文件夹或通配符，可写多个")
    parser.add_argument("--model", default="small",
                        help="模型：tiny/base/small/medium/large-v3，也可直接写本地模型文件夹路径")
    parser.add_argument("--language", default="zh",
                        help="语种代码，中文 zh、英文 en；写 auto 表示自动识别")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="有 NVIDIA 显卡可写 cuda")
    parser.add_argument("--compute-type", default=None,
                        help="cpu 默认 int8，cuda 默认 float16")
    parser.add_argument("--prompt", default=None,
                        help="术语/场景提示，例如“以下是关于机器学习的普通话讲座”")
    parser.add_argument("--out", default=None, help="输出目录，默认与音频同目录")
    parser.add_argument("--no-srt", action="store_true", help="不生成 srt 字幕")
    parser.add_argument("--no-vad", action="store_true", help="关闭静音检测")
    parser.add_argument("--model-dir", default=None,
                        help="模型下载/缓存目录，默认是脚本旁边的 models 文件夹")
    parser.add_argument("--download-source", default="modelscope",
                        choices=["modelscope", "huggingface"],
                        help="模型下载源：modelscope（国内快，默认）/ huggingface")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com",
                        help="走 huggingface 时的镜像地址；写 none 则用官方站")
    parser.add_argument("--mic", type=int, default=0, metavar="秒",
                        help="录制麦克风指定秒数后再转写")
    args = parser.parse_args()

    if args.compute_type is None:
        args.compute_type = "float16" if args.device == "cuda" else "int8"

    # 如果脚本旁边有“提示词.txt”，就把它当作默认提示词：
    # 常用的人名、班级、专业名词只写一次，之后每次运行自动带上，不用每次敲 --prompt
    if args.prompt is None:
        hint_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "提示词.txt")
        if os.path.exists(hint_file):
            try:
                with open(hint_file, encoding="utf-8") as f:
                    args.prompt = f.read().strip() or None
            except Exception:
                args.prompt = None
            if args.prompt:
                print(f"[提示] 已使用 提示词.txt：{args.prompt}")

    # 模型目录默认固定在“脚本所在目录\models”，这样无论从哪个工作目录启动
    # （IDLE 里按 F5、双击运行、命令行切换到别的目录），都复用同一份已下好的模型
    if args.model_dir is None:
        args.model_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "models"
        )

    # ---- 先录音（如果用了 --mic）----
    inputs = args.inputs
    if args.mic > 0:
        wav = os.path.abspath("mic_recording.wav")
        record_mic(args.mic, wav)
        inputs = [wav] + inputs

    if not inputs:
        # 在 IDLE / PyCharm 里直接按运行是没有命令行参数的，这里用交互输入兜底
        print("可以直接把音频文件拖进来，或者手动输入路径。")
        try:
            typed = input("音频文件路径（直接回车退出）: ").strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt):
            typed = ""
        if not typed:
            parser.print_help()
            print("\n提示：也可以带参数运行，例如 python 录音转文字.py D:\\录音.mp3")
            sys.exit(0)
        inputs = [typed]

    files = collect_input_files(inputs)
    if not files:
        print("[错误] 没有找到可处理的音频文件。")
        sys.exit(1)

    print(f"[任务] 共 {len(files)} 个文件待处理")
    language = None if args.language.lower() == "auto" else args.language

    model = load_model(args.model, args.device, args.compute_type, args.model_dir,
                       args.hf_endpoint, args.download_source)

    for idx, audio in enumerate(files, 1):
        print(f"\n===== ({idx}/{len(files)}) {audio} =====")
        segments, plain, _info = transcribe_one(
            model, audio, language, args.prompt, vad_filter=not args.no_vad
        )
        if not segments:
            print("[警告] 没识别到任何语音内容（可能是纯音乐/空白录音）。")
            continue
        save_outputs(audio, segments, plain, out_dir=args.out, make_srt=not args.no_srt)

    print("\n全部处理完毕。")


if __name__ == "__main__":
    main()
