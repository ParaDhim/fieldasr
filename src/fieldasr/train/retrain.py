"""Small retraining round: continue fine-tuning a base model on corrections (+ train replay).

    python -m fieldasr.train.retrain --config configs/retrain_targeted.yaml --seed 0

Config keys: name, base, corrections (csv with path,text or empty), replay, oversample, epochs, lr.
The corrections CSV can be a real export from the review app (data/review/corrected.csv).
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    get_linear_schedule_with_warmup,
)

from fieldasr.asr.whisper import WhisperTranscriber
from fieldasr.eval.harness import run_eval
from fieldasr.train.common import Tracker, load_split, log_eval, parse_args, seed_everything
from fieldasr.train.whisper_ft import TrainDS, collate


def main() -> None:
    cfg, a = parse_args("retrain on corrections")
    run = f"{cfg['name']}_s{a.seed}"
    out = Path(a.out_root) / run
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(a.seed)
    tracker = Tracker(run, {**cfg, "seed": a.seed})
    proc = WhisperProcessor.from_pretrained(cfg["base"])
    proc.tokenizer.set_prefix_tokens(language="swahili", task="transcribe")

    parts = [
        load_split(a.train_manifest, "train").sample(cfg["replay"], random_state=a.seed)[
            ["path", "raw_text"]
        ]
    ]
    n_corr = 0
    if cfg.get("corrections"):
        c = pd.read_csv(cfg["corrections"], keep_default_na=False).rename(
            columns={"text": "raw_text"}
        )
        c = c[c.raw_text.str.strip() != ""]
        if "pii_redacted" in c and not cfg.get("keep_redacted", False):
            c = c[~c.pii_redacted.astype(bool)]
        n_corr = len(c)
        parts += [c[["path", "raw_text"]]] * cfg.get("oversample", 3)
    df = pd.concat(parts, ignore_index=True)
    print(
        f"retrain set: {len(df)} rows ({n_corr} corrections x{cfg.get('oversample', 3)} + replay)",
        flush=True,
    )

    model = WhisperForConditionalGeneration.from_pretrained(cfg["base"]).cuda()  # type: ignore[call-arg]
    model.generation_config.forced_decoder_ids = None
    start = model.config.decoder_start_token_id
    dl = DataLoader(
        TrainDS(df, proc, False, a.seed),
        batch_size=16,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(a.seed),
        num_workers=8,
        collate_fn=lambda b: collate(b, start),
    )
    total = len(dl) * cfg["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, max(1, int(0.1 * total)), total)
    model.train()
    step, t0 = 0, time.time()
    for _ in range(cfg["epochs"]):
        for batch in dl:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)  # noqa: E702
            step += 1
            if step % 10 == 0:
                print(
                    f"step {step}/{total} loss {loss.item():.4f} {time.time() - t0:.0f}s",
                    flush=True,
                )
    tr = WhisperTranscriber(model, proc)
    metrics = run_eval(tr, Path(a.eval_manifest), out, limit=a.eval_limit)
    log_eval(tracker, metrics)
    tracker.end()


if __name__ == "__main__":
    main()
