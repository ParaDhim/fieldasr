"""Scan dataset transcripts with the scrubber; write counts + flagged examples for manual audit."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from fieldasr.privacy.scrub import scrub


def main(
    manifest: str = "data/processed/manifest_qc.csv", out: str = "results/privacy_report.json"
) -> None:
    df = pd.read_csv(manifest, keep_default_na=False)
    total: Counter[str] = Counter()
    flagged = []
    for split, text in zip(df["split"], df["raw_text"], strict=True):
        r = scrub(str(text))
        total.update(r.counts)
        if r.counts:
            flagged.append({"split": split, "before": text, "after": r.text, "counts": r.counts})
    rep = {
        "n_transcripts": len(df),
        "items_flagged": len(flagged),
        "entity_counts": dict(total),
        "examples": flagged[:25],
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print({k: v for k, v in rep.items() if k != "examples"})
    for e in flagged[:10]:
        print(" -", e["counts"], "|", e["after"][:140])


if __name__ == "__main__":
    main()
