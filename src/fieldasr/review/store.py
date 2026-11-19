"""Review store: queue of clips, append-only corrections log, PII-scrubbed export."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from fieldasr.privacy.scrub import scrub


class ReviewStore:
    """Queue CSV needs columns: item_id, path, hyp (model draft), optional condition."""

    def __init__(self, queue_csv: Path, log_path: Path):
        self.queue = pd.read_csv(queue_csv, keep_default_na=False)
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # -- corrections log (append-only; the latest entry per item wins) -------------------
    def _entries(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        if self.log_path.exists():
            for ln in self.log_path.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    e = json.loads(ln)
                    out[e["item_id"]] = e
        return out

    def save(
        self, item_id: str, text: str, bad_audio: bool, reviewer: str = "", note: str = ""
    ) -> None:
        e = {
            "item_id": item_id,
            "text": text.strip(),
            "bad_audio": bool(bad_audio),
            "reviewer": reviewer,
            "note": note,
            "ts": time.time(),
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    def done_ids(self) -> set[str]:
        return set(self._entries())

    def next_pending(self, after: int = -1) -> int | None:
        done = self.done_ids()
        for i in range(after + 1, len(self.queue)):
            if str(self.queue.iloc[i].item_id) not in done:
                return i
        for i in range(0, after + 1):
            if str(self.queue.iloc[i].item_id) not in done:
                return i
        return None

    def progress(self) -> tuple[int, int]:
        return len(self.done_ids() & set(self.queue.item_id.astype(str))), len(self.queue)

    # -- export ---------------------------------------------------------------------------
    def export(self, out_csv: Path) -> dict:
        """Write corrected data (path,text,...). Transcripts are PII-scrubbed; bad-audio and empty
        items are excluded from training data and reported separately."""
        ent = self._entries()
        q = self.queue.assign(item_id=self.queue.item_id.astype(str)).set_index("item_id")
        rows, n_bad, n_pii, n_changed = [], 0, 0, 0
        for iid, e in ent.items():
            if iid not in q.index:
                continue
            if e["bad_audio"] or not e["text"]:
                n_bad += 1
                continue
            r = scrub(e["text"])
            n_pii += int(bool(r.counts))
            n_changed += int(e["text"] != q.loc[iid, "hyp"])
            rows.append(
                {
                    "item_id": iid,
                    "path": q.loc[iid, "path"],
                    "text": r.text,
                    "pii_redacted": bool(r.counts),
                    "model_draft": q.loc[iid, "hyp"],
                }
            )
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            rows, columns=["item_id", "path", "text", "pii_redacted", "model_draft"]
        ).to_csv(out_csv, index=False)
        return {
            "exported": len(rows),
            "flagged_bad_audio_or_empty": n_bad,
            "with_pii_redacted": n_pii,
            "changed_vs_model_draft": n_changed,
        }
