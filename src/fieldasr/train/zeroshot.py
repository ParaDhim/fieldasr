"""Zero-shot baseline: pretrained Whisper, no fine-tuning."""

from __future__ import annotations

from pathlib import Path

from fieldasr.asr.whisper import WhisperTranscriber
from fieldasr.eval.harness import run_eval
from fieldasr.train.common import Tracker, log_eval, parse_args


def main() -> None:
    cfg, a = parse_args("zero-shot eval")
    out = Path(a.out_root) / f"{cfg['name']}_s0"
    tracker = Tracker(out.name, cfg)
    metrics = run_eval(
        WhisperTranscriber.from_pretrained(cfg["model"]),
        Path(a.eval_manifest),
        out,
        limit=a.eval_limit,
    )
    log_eval(tracker, metrics)
    tracker.end()


if __name__ == "__main__":
    main()
