"""CPU benchmark: CT2 FP32 vs INT8 -- WER/CER change, latency, real-time factor."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import soundfile as sf

from fieldasr.eval.metrics import corpus_metrics

CONDS = ["clean", "babble_snr10", "amr_nb_8k"]


def run(model_dir: Path, df: pd.DataFrame, threads: int) -> dict:
    from faster_whisper import WhisperModel

    m = WhisperModel(str(model_dir), device="cpu", compute_type="default", cpu_threads=threads)
    res: dict = {}
    for cond, g in df.groupby("condition", sort=False):
        hyps, lat, dur = [], [], []
        for p in g.path:
            d = sf.info(p).duration
            t0 = time.perf_counter()
            segs, _ = m.transcribe(
                p, language="sw", beam_size=1, vad_filter=False, condition_on_previous_text=False
            )
            hyps.append(" ".join(s.text.strip() for s in segs))
            lat.append(time.perf_counter() - t0)
            dur.append(d)
        met = corpus_metrics(g.text.tolist(), hyps)
        res[str(cond)] = {
            **met,
            "latency_mean_s": sum(lat) / len(lat),
            "latency_p95_s": float(pd.Series(lat).quantile(0.95)),
            "rtf": sum(lat) / sum(dur),
            "audio_s": sum(dur),
        }
        print(f"[bench] {model_dir.name} {cond}: {res[str(cond)]}", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct2-dir", required=True)
    ap.add_argument("--eval-manifest", default="data/evalsets/manifest.csv")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="results/deploy/bench.json")
    a = ap.parse_args()
    df = pd.read_csv(a.eval_manifest, keep_default_na=False)
    df = df[df.condition.isin(CONDS)]
    df = df.groupby("condition", sort=False).head(a.n)
    out = {q: run(Path(a.ct2_dir) / q, df, a.threads) for q in ("float32", "int8")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    rows = []
    for c in CONDS:
        f, i = out["float32"][c], out["int8"][c]
        rows.append(
            f"| {c} | {f['wer']:.3f} | {i['wer']:.3f} | {i['wer'] - f['wer']:+.3f} | "
            f"{f['latency_mean_s']:.2f} | {i['latency_mean_s']:.2f} | {f['rtf']:.3f} | {i['rtf']:.3f} |"
        )
    md = (
        "| condition | WER fp32 | WER int8 | dWER | lat fp32 (s) | lat int8 (s) | RTF fp32 | RTF int8 |\n"
        "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n"
    )
    Path(a.out).with_suffix(".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
