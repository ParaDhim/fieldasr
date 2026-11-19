"""Error analysis: S/D/I decomposition, rule-based failure taxonomy, listening sheet."""

from __future__ import annotations

import html
import os
import re
from pathlib import Path

import jiwer
import pandas as pd

from fieldasr.eval.metrics import normalize

ENGLISHISH = re.compile(r"[qx]|tion\b|ing\b")
TAG_ORDER = [
    "empty_output",
    "hallucination_loop",
    "numeral",
    "word_segmentation",
    "proper_noun",
    "foreign_token",
    "orthographic_near_miss",
    "noise_or_channel",
    "speaker_or_unclear",
]


def sdi(ref: str, hyp: str) -> tuple[int, int, int, int]:
    """(substitutions, deletions, insertions, ref_words)."""
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0, 0, 0, 0
    if not h:
        n = len(r.split())
        return 0, n, 0, n
    o = jiwer.process_words(r, h)
    return o.substitutions, o.deletions, o.insertions, len(r.split())


def _loops(hyp: str) -> bool:
    w = normalize(hyp).split()
    if len(w) < 6:
        return False
    tri = [" ".join(w[i : i + 3]) for i in range(len(w) - 2)]
    return max(tri.count(t) for t in set(tri)) >= 3


def tag_item(ref: str, raw_ref: str, hyp: str, condition: str) -> list[str]:
    r, h = normalize(ref), normalize(hyp)
    rw, hw = r.split(), h.split()
    tags = []
    if not h:
        tags.append("empty_output")
    if h and (_loops(hyp) or len(hw) > 2 * len(rw) + 3):
        tags.append("hallucination_loop")
    if re.search(r"\d", r) and not set(re.findall(r"\d+", r)) <= set(re.findall(r"\d+", h)):
        tags.append("numeral")
    if h and r.replace(" ", "") == h.replace(" ", "") and r != h:
        tags.append("word_segmentation")
    caps = [normalize(w) for i, w in enumerate(raw_ref.split()) if i > 0 and w[:1].isupper()]
    if any(c and c not in hw for c in caps):
        tags.append("proper_noun")
    if any(ENGLISHISH.search(w) for w in rw if w not in hw):
        tags.append("foreign_token")
    if h and r != h and jiwer.cer(r, h) < 0.15 and "word_segmentation" not in tags:
        tags.append("orthographic_near_miss")
    if condition != "clean":
        tags.append("noise_or_channel")
    if not tags:
        tags.append("speaker_or_unclear")
    return tags


def primary(tags: list[str]) -> str:
    return min(tags, key=TAG_ORDER.index)


def load_hyps(run_dir: Path, manifest_qc: Path) -> pd.DataFrame:
    h = pd.read_csv(run_dir / "hyps.csv", keep_default_na=False)
    m = pd.read_csv(manifest_qc, keep_default_na=False)
    m = m[m["split"] == "test"][["file_name", "raw_text", "duration"]]
    h["file_name"] = h.path.map(lambda p: Path(p).name)
    return h.merge(m, on="file_name", how="left")


def decomposition(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond, g in df.groupby("condition", sort=False):
        s = d = i = n = 0
        for r, h in zip(g.text, g.hyp, strict=True):
            a, b, c, k = sdi(r, h)
            s, d, i, n = s + a, d + b, i + c, n + k
        rows.append(
            {
                "condition": cond,
                "sub%": 100 * s / n,
                "del%": 100 * d / n,
                "ins%": 100 * i / n,
                "wer%": 100 * (s + d + i) / n,
            }
        )
    return pd.DataFrame(rows).set_index("condition")


def tag_failures(df: pd.DataFrame, thresh: float = 0.5) -> pd.DataFrame:
    f = df[df.item_wer >= thresh].copy()
    all_tags = [
        tag_item(r, rr, h, c)
        for r, rr, h, c in zip(f.text, f.raw_text, f.hyp, f.condition, strict=True)
    ]
    f["tags"] = [",".join(t) for t in all_tags]
    f["primary_tag"] = [primary(t) for t in all_tags]
    return f


def sample_failures(f: pd.DataFrame, n: int = 100, seed: int = 0) -> pd.DataFrame:
    """Stratified: 30 clean + remaining spread over field conditions."""
    clean = f[f.condition == "clean"]
    other = f[~f.condition.isin(["clean", "domain_terms"])]
    k_clean = min(30, len(clean))
    pick = [clean.sample(k_clean, random_state=seed)]
    conds = other.condition.unique()
    per = max((n - k_clean) // max(len(conds), 1), 1)
    for c in conds:
        g = other[other.condition == c]
        pick.append(g.sample(min(per, len(g)), random_state=seed))
    out = pd.concat(pick)
    return out.sample(frac=1, random_state=seed).head(n).reset_index(drop=True)


def listening_sheet(sel: pd.DataFrame, out_html: Path) -> None:
    rows = []
    for i, r in sel.iterrows():
        src = os.path.relpath(r.path, out_html.parent)
        rows.append(
            f"<tr><td>{i}</td><td>{html.escape(r.condition)}</td>"
            f"<td><audio controls preload='none' src='{html.escape(src)}'></audio></td>"
            f"<td>{html.escape(r.text)}</td><td>{html.escape(r.hyp)}</td>"
            f"<td>{r.item_wer:.2f}</td><td>{r.primary_tag}</td><td><input size=14></td></tr>"
        )
    out_html.write_text(
        "<!doctype html><meta charset=utf-8><title>Listening sheet</title>"
        "<style>body{font:14px system-ui;margin:16px}td,th{border:1px solid #ccc;padding:4px;"
        "vertical-align:top}table{border-collapse:collapse}</style>"
        "<h2>FieldASR failure listening sheet</h2><p>Auto tag = rule-based guess; use the last "
        "column for your own label.</p><table><tr><th>#</th><th>condition</th><th>audio</th>"
        "<th>reference</th><th>model output</th><th>WER</th><th>auto tag</th><th>your tag</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def main() -> None:
    run = Path("results/whisper_lora_s0")
    out = Path("results/error_analysis")
    out.mkdir(parents=True, exist_ok=True)
    md = []
    for name in [
        "whisper_zeroshot",
        "whisper_full",
        "whisper_lora",
        "whisper_full_aug",
        "w2v2_xlsr",
    ]:
        d = Path("results") / f"{name}_s0"
        if not (d / "hyps.csv").exists():
            continue
        dec = decomposition(load_hyps(d, Path("data/processed/manifest_qc.csv")))
        md.append(
            f"### {name}_s0: error decomposition (% of reference words)\n\n{dec.round(1).to_markdown()}\n"
        )
    (out / "wer_decomposition.md").write_text("\n".join(md))

    df = load_hyps(run, Path("data/processed/manifest_qc.csv"))
    f = tag_failures(df)
    tab = f.groupby(["condition", "primary_tag"]).size().unstack(fill_value=0)
    tab["n_failures"] = tab.sum(axis=1)
    tot = df.groupby("condition").size()
    tab["failure_rate%"] = (100 * tab["n_failures"] / tot).round(1)
    (out / "taxonomy_by_condition.md").write_text(
        f"Primary tag among failures (item WER >= 0.5), {run.name}\n\n{tab.to_markdown()}\n\n"
        f"Overall (primary tag): {f.primary_tag.value_counts().to_dict()}\n\n"
        "Non-exclusive prevalence (an item can carry several tags; 'proper_noun' fires whenever any "
        "capitalised mid-sentence word is missed, so it is inflated on long sentences):\n\n"
        + f.tags.str.split(",").explode().value_counts().to_frame("n_failures").to_markdown()
        + "\n"
    )
    sel = sample_failures(f)
    sel[
        ["condition", "file_name", "path", "text", "hyp", "item_wer", "tags", "primary_tag"]
    ].to_csv(out / "failures_100.csv", index=False)
    listening_sheet(sel, out / "listening_sheet.html")
    print("\n".join(md)[:3000])
    print(tab.to_string())
    print(f.primary_tag.value_counts().to_string())


if __name__ == "__main__":
    main()
