"""Build the review queue: dev clips under random field conditions, ranked by model failure.

Writes data/review/queue.csv (what a human reviewer sees: path + model draft) and the *simulated*
reviewer outputs (oracle = FLEURS reference) used for the controlled retraining experiment:
  corrections_targeted.csv  - the K worst failures
  corrections_random.csv    - K random clips (control for "any extra data helps")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from fieldasr.asr.whisper import WhisperTranscriber
from fieldasr.data import augment as A
from fieldasr.eval.metrics import corpus_metrics
from fieldasr.train.common import load_split

K = 60
SEED = 7


def main(base: str = "results/whisper_lora_s0/model", out: Path = Path("data/review")) -> None:
    out.mkdir(parents=True, exist_ok=True)
    wav_dir = Path("data/dev_field")
    wav_dir.mkdir(parents=True, exist_ok=True)
    dev = load_split("data/processed/manifest_qc.csv", "dev")
    pool = [sf.read(p, dtype="float32")[0] for p in dev.path.sample(30, random_state=SEED)]
    paths = []
    for i, r in dev.iterrows():
        x, sr = sf.read(r.path, dtype="float32")
        y = A.random_field_aug(x, np.random.default_rng([SEED, int(i)]), pool)  # type: ignore[call-overload]
        p = wav_dir / Path(r.path).name
        sf.write(p, y, sr)
        paths.append(str(p))
    dev = dev.assign(path=paths)
    hyp = WhisperTranscriber.from_pretrained(base)(dev.path.tolist())
    dev["hyp"] = hyp
    dev["item_wer"] = [corpus_metrics([t], [h])["wer"] for t, h in zip(dev.text, hyp, strict=True)]
    dev = dev.sort_values("item_wer", ascending=False).reset_index(drop=True)
    dev["item_id"] = range(len(dev))
    dev[["item_id", "path", "hyp"]].head(K).to_csv(out / "queue.csv", index=False)
    cols = {"path": "path", "raw_text": "text"}
    dev.head(K)[list(cols)].rename(columns=cols).to_csv(
        out / "corrections_targeted.csv", index=False
    )
    dev.sample(K, random_state=SEED)[list(cols)].rename(columns=cols).to_csv(
        out / "corrections_random.csv", index=False
    )
    print(f"dev_field WER (base): {corpus_metrics(dev.text.tolist(), hyp)}")
    print(
        f"worst-{K} mean item WER: {dev.item_wer.head(K).mean():.2f}; overall mean {dev.item_wer.mean():.2f}"
    )


if __name__ == "__main__":
    main()
