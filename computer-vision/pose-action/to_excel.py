"""把采集的 CSV 汇总成一个 Excel，方便直接翻数据 / 贴进报告附录。

用法
----
    python to_excel.py                       # data/raw -> reports/dataset.xlsx
    python to_excel.py --raw data/demo --out reports/demo.xlsx

结构：每个采集一段一个 sheet（超长会截断到 MAX_ROWS 行），
外加一个「汇总」sheet 列出每段的标签、时长、帧数、检测率。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import config as C

MAX_ROWS = 20000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(C.RAW_DIR))
    ap.add_argument("--out", default=str(C.REPORT_DIR / "dataset.xlsx"))
    args = ap.parse_args()

    files = sorted(Path(args.raw).glob("*.csv"))
    if not files:
        print(f"{args.raw} 里没有 CSV")
        return

    summary, sheets = [], {}
    for p in files:
        df = pd.read_csv(p, encoding="utf-8-sig")
        if df.empty:
            continue
        label = str(df["label"].iloc[0])
        dur = float(df["time"].iloc[-1] - df["time"].iloc[0])
        det = float(df["x1"].notna().mean())
        summary.append(dict(file=p.name, label=label, frames=len(df),
                            seconds=round(dur, 2), detect_rate=round(det, 3)))
        sheets[p.stem[:28]] = df.head(MAX_ROWS)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        pd.DataFrame(summary).to_excel(xw, sheet_name="汇总", index=False)
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)

    print(f"已写入 {out}")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
