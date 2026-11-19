"""Aggregate results/<method>_s<seed>/metrics.json into mean +- std per condition."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


def aggregate(root: Path = Path("results")) -> pd.DataFrame:
    rows = []
    for f in sorted(root.glob("*_s*/metrics.json")):
        m = re.match(r"(.+)_s(\d+)$", f.parent.name)
        if not m:
            continue
        for cond, v in json.loads(f.read_text()).items():
            rows.append({"method": m[1], "seed": int(m[2]), "condition": cond, **v})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, metric: str = "wer") -> pd.DataFrame:
    g = df.groupby(["method", "condition"])[metric]
    s = pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=1).fillna(0.0), "n_seeds": g.count()})
    return s.reset_index()


def to_markdown(df: pd.DataFrame, metric: str = "wer") -> str:
    s = summarize(df, metric)
    s["cell"] = [
        f"{100 * m:.1f} ± {100 * sd:.1f}" if n > 1 else f"{100 * m:.1f}"
        for m, sd, n in zip(s["mean"], s["std"], s["n_seeds"], strict=True)
    ]
    piv = s.pivot(index="condition", columns="method", values="cell").fillna("-")
    seeds = s.groupby("method")["n_seeds"].max()
    piv.columns = [f"{c} (n={seeds[c]})" for c in piv.columns]
    return f"### {metric.upper()} % (mean ± std over seeds)\n\n" + piv.to_markdown() + "\n"


def main() -> None:
    df = aggregate()
    if df.empty:
        raise SystemExit("no results found")
    md = to_markdown(df, "wer") + "\n" + to_markdown(df, "cer")
    Path("results/summary.md").write_text(md)
    df.to_csv("results/all_results.csv", index=False)
    print(md)


if __name__ == "__main__":
    main()
