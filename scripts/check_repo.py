#!/usr/bin/env python3
"""入库前检查：个人信息、内网地址、超大文件、文档缺失。

用法：
    python scripts/check_repo.py                # 检查全部受版本控制的文件
    python scripts/check_repo.py --staged       # 只检查已暂存的文件
    python scripts/check_repo.py 路径 [路径...]   # 检查指定文件或目录
    python scripts/check_repo.py --all          # 附带项目结构检查（默认已启用）

退出码：0 无阻断问题，1 发现问题。

个人敏感词（真实姓名、学号、班级）放在 scripts/sensitive_words.txt，
一行一个，该文件已在 .gitignore 中排除，不要提交。格式见
sensitive_words.example.txt。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_BYTES = 5 * 1024 * 1024
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".idea", ".vscode"}
BINARY_EXT = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".ico",
    ".zip", ".gz", ".7z", ".rar", ".pdf", ".onnx", ".pt", ".pth", ".bin",
}

# 行级规则：规则名 → 正则。新增规则直接往这里加。
RULES: list[tuple[str, re.Pattern[str]]] = [
    ("内网 IP", re.compile(
        r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
        r"|192\.168\.\d{1,3}\.\d{1,3}"
        r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b")),
    ("本机绝对路径", re.compile(
        r"[A-Za-z]:\\{1,2}Users\\{1,2}[^\\\s\"'<>]+"
        r"|/home/[a-z_][a-z0-9_-]*/"
        r"|/Users/[A-Za-z0-9._-]+/")),
    ("邮箱地址", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("手机号", re.compile(r"\b1[3-9]\d{9}\b")),
    ("班级编号", re.compile(r"\d{4}\s*[A-Z]\s*班")),
    ("私有密钥", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}")),
]

# 允许出现的邮箱：仓库署名用的 GitHub noreply 地址
EMAIL_ALLOW = re.compile(r"(noreply@github\.com|@users\.noreply\.github\.com)$", re.I)

# 扫描时跳过的文件自身
SELF_SKIP = {"scripts/check_repo.py"}

# 行内豁免标记：某行确认无需处理时，在行尾注释里写该标记，整行跳过所有规则
IGNORE_MARK = "check:ignore"


def run_git(args: list[str]) -> list[str]:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False,
        )
    except FileNotFoundError:
        print("[!] 未找到 git，改用目录遍历")
        return []
    if out.returncode != 0:
        print(f"[!] git {' '.join(args)} 失败：{out.stderr.strip()[:200]}")
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def collect_files(mode: str, extra: list[str]) -> list[Path]:
    if extra:
        found: list[Path] = []
        for item in extra:
            p = (REPO_ROOT / item).resolve() if not Path(item).is_absolute() else Path(item)
            if p.is_dir():
                for sub in p.rglob("*"):
                    if sub.is_file() and not any(part in SKIP_DIRS for part in sub.parts):
                        found.append(sub)
            elif p.is_file():
                found.append(p)
        return sorted(set(found))

    if mode == "staged":
        names = run_git(["diff", "--cached", "--name-only", "--diff-filter=ACMR"])
    else:
        names = run_git(["ls-files"])

    if names:
        return [REPO_ROOT / n for n in names]

    # 没有 git 输出时退回遍历
    return sorted(
        p for p in REPO_ROOT.rglob("*")
        if p.is_file() and not any(part in SKIP_DIRS for part in p.relative_to(REPO_ROOT).parts)
    )


def read_text(path: Path) -> str | None:
    if path.suffix.lower() in BINARY_EXT:
        return None
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:8192]:
        return None
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def mask(value: str) -> str:
    v = value.strip()
    if len(v) <= 4:
        return "*" * len(v)
    return f"{v[:2]}{'*' * min(6, len(v) - 4)}{v[-2:]}"


def load_words() -> list[str]:
    f = REPO_ROOT / "scripts" / "sensitive_words.txt"
    if not f.exists():
        return []
    words = []
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words.append(line)
    return words


def check_size(path: Path, rel: str) -> tuple[str, str, str] | None:
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size > MAX_BYTES:
        return (rel, "", f"体积 {size / 1048576:.1f} MB，超过 {MAX_BYTES / 1048576:.0f} MB 上限；"
                         "改用 .gitignore 排除并在 README 写下载方式")
    return None


def check_structure() -> list[tuple[str, str, str]]:
    """检查二级目录（学科/项目）是否具备 README 与依赖说明。"""
    issues: list[tuple[str, str, str]] = []
    reserved = {"_template", "scripts", ".git", ".github"}
    for subject in sorted(REPO_ROOT.iterdir()):
        if not subject.is_dir() or subject.name.startswith(".") or subject.name in reserved:
            continue
        for proj in sorted(subject.iterdir()):
            if not proj.is_dir() or proj.name.startswith("."):
                continue
            rel = f"{subject.name}/{proj.name}"
            if not (proj / "README.md").exists():
                issues.append((rel, "", "缺少 README.md"))
            has_req = (proj / "requirements.txt").exists()
            has_py = any(proj.rglob("*.py"))
            if has_py and not has_req:
                issues.append((rel, "", "有 Python 源码但缺少 requirements.txt"))
    return issues


def main() -> None:
    ap = argparse.ArgumentParser(description="入库前检查")
    ap.add_argument("paths", nargs="*", help="指定文件或目录，省略则用 git 文件列表")
    ap.add_argument("--staged", action="store_true", help="只检查已暂存的文件")
    ap.add_argument("--no-structure", action="store_true", help="跳过项目结构检查")
    args = ap.parse_args()

    files = collect_files("staged" if args.staged else "all", args.paths)
    words = load_words()

    hits: list[tuple[str, int, str, str]] = []   # (文件, 行号, 规则, 片段)
    scanned = 0

    for path in files:
        try:
            rel = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            rel = path.as_posix()
        if rel in SELF_SKIP or not path.exists():
            continue
        scanned += 1

        big = check_size(path, rel)
        if big:
            hits.append(big)

        text = read_text(path)
        if text is None:
            continue

        for lineno, line in enumerate(text.splitlines(), 1):
            if IGNORE_MARK in line:
                continue
            for name, pat in RULES:
                for m in pat.finditer(line):
                    value = m.group(0)
                    if name == "邮箱地址" and EMAIL_ALLOW.search(value):
                        continue
                    hits.append((rel, lineno, name, mask(value)))
            for word in words:
                if word in line:
                    hits.append((rel, lineno, "个人敏感词", f'"{word}"'))

    struct = [] if args.no_structure else check_structure()

    label = "已暂存文件" if args.staged else ("指定路径" if args.paths else "受版本控制的文件")
    print(f"扫描 {scanned} 个{label}")
    if words:
        print(f"已加载个人敏感词 {len(words)} 条")
    else:
        print("未找到 scripts/sensitive_words.txt，仅按通用规则检查")
    print()

    if hits:
        current = None
        for rel, lineno, rule, snippet in hits:
            if rel != current:
                print(f"  {rel}")
                current = rel
            loc = f"行 {lineno}" if lineno else "文件"
            print(f"    [{rule}] {loc}: {snippet}")
        print()

    if struct:
        print("  项目结构")
        for rel, _lineno, msg in struct:
            print(f"    [{rel}] {msg}")
        print()

    if not hits and not struct:
        print("[ok] 未发现问题")
        return

    total = len(hits) + len(struct)
    print(f"[x] 共 {total} 处需要处理")
    if hits:
        print("    处理方式：改写为通用示例，或在行尾加 check:ignore 注释说明无需处理")
    sys.exit(1)


if __name__ == "__main__":
    main()
