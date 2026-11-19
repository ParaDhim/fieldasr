"""Text normalisation + WER/CER."""

from __future__ import annotations

import re
import unicodedata

import jiwer

_KEEP = re.compile(r"[^\w\s'-]")


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = _KEEP.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def corpus_metrics(refs: list[str], hyps: list[str]) -> dict[str, float]:
    r = [normalize(x) for x in refs]
    h = [normalize(x) for x in hyps]
    keep = [i for i, x in enumerate(r) if x]
    r, h = [r[i] for i in keep], [h[i] for i in keep]
    if not r:
        return {"wer": float("nan"), "cer": float("nan"), "n": 0}
    return {"wer": jiwer.wer(r, h), "cer": jiwer.cer(r, h), "n": len(r)}
