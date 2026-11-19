"""Run a transcriber over every eval condition; write per-condition metrics + hypotheses."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import jiwer
import pandas as pd

from fieldasr.eval.metrics import corpus_metrics, normalize

Transcribe = Callable[[list[str]], list[str]]


def run_eval(
    transcribe: Transcribe,
    manifest: Path,
    out_dir: Path,
    limit: int | None = None,
) -> dict:
    df = pd.read_csv(manifest, keep_default_na=False)
    metrics: dict[str, dict] = {}
    parts = []
    for cond, g in df.groupby("condition", sort=False):
        g = g.head(limit) if limit else g
        hyps = transcribe(g.path.tolist())
        g = g.assign(hyp=hyps)
        g["item_wer"] = [
            jiwer.wer(normalize(r), normalize(h)) if normalize(r) else float("nan")
            for r, h in zip(g.text, g.hyp, strict=True)
        ]
        metrics[str(cond)] = corpus_metrics(g.text.tolist(), g.hyp.tolist())
        if cond == "clean":
            for gen, gg in g.groupby("gender"):
                metrics[f"clean/gender={gen}"] = corpus_metrics(gg.text.tolist(), gg.hyp.tolist())
        if cond == "domain_terms":
            for tag in ("numerals", "dates", "health", "school"):
                gt = g[g.domain.str.contains(tag)]
                if len(gt):
                    metrics[f"domain_terms/{tag}"] = corpus_metrics(
                        gt.text.tolist(), gt.hyp.tolist()
                    )
        parts.append(g)
        print(f"[eval] {cond}: {metrics[str(cond)]}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(parts).to_csv(out_dir / "hyps.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
