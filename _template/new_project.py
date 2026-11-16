#!/usr/bin/env python3
"""按模板生成一个新的课程项目目录。

用法：
    python _template/new_project.py --subject computer-vision --name my-demo --title "项目标题"

生成：
    <subject>/<name>/
        README.md            由 _template/README.md 生成，标题已填入
        .gitignore           由 _template/project.gitignore 生成
        requirements.txt     由 _template/requirements.txt 生成

生成后按 README 里的注释逐节填写，再跑 scripts/check_repo.py 检查一遍。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = REPO_ROOT / "_template"

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def fail(msg: str) -> None:
    print(f"[x] {msg}")
    sys.exit(1)


def copy_template(src: Path, dst: Path, transform=None) -> None:
    if not src.exists():
        fail(f"模板文件缺失：{src}")
    text = src.read_text(encoding="utf-8")
    if transform:
        text = transform(text)
    dst.write_text(text, encoding="utf-8", newline="\n")
    print(f"    + {dst.relative_to(REPO_ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="生成新的课程项目目录骨架")
    ap.add_argument("--subject", required=True, help="学科目录，如 computer-vision")
    ap.add_argument("--name", required=True, help="项目目录名，小写字母数字与连字符，如 my-demo")
    ap.add_argument("--title", required=True, help="README 一级标题显示名")
    ap.add_argument("--course", default="课程设计", help="课程名，默认「课程设计」")
    args = ap.parse_args()

    for label, value in (("--subject", args.subject), ("--name", args.name)):
        if not SLUG_RE.match(value):
            fail(f"{label} 只允许小写字母、数字、连字符，且以字母数字开头：{value!r}")

    target = REPO_ROOT / args.subject / args.name
    if target.exists():
        fail(f"目标目录已存在：{target}")

    target.mkdir(parents=True)
    print(f"[ok] 创建目录 {target.relative_to(REPO_ROOT)}")

    def fill_readme(text: str) -> str:
        text = text.replace("# <项目名>", f"# {args.title}", 1)
        text = text.replace("> 课程：<课程名 / 课程设计>", f"> 课程：{args.course}", 1)
        return text

    copy_template(TEMPLATE_DIR / "README.md", target / "README.md", fill_readme)
    copy_template(TEMPLATE_DIR / "project.gitignore", target / ".gitignore")
    copy_template(TEMPLATE_DIR / "requirements.txt", target / "requirements.txt")

    print()
    print("下一步：")
    print(f"  1. 把源码放进 {target.relative_to(REPO_ROOT)}/")
    print(f"  2. 填完 README.md 里所有 <占位> 与注释块，注释块写完要删掉")
    print(f"  3. 在根 README.md 的项目索引表补一行")
    print(f"  4. python scripts/check_repo.py --staged 检查后再提交")


if __name__ == "__main__":
    main()
