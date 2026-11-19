"""Fine-tune Whisper (full or LoRA, optional field augmentation) and evaluate on all conditions."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    get_linear_schedule_with_warmup,
)

from fieldasr.asr.whisper import WhisperTranscriber
from fieldasr.data import augment as A
from fieldasr.eval.harness import run_eval
from fieldasr.eval.metrics import corpus_metrics
from fieldasr.train.common import (
    Tracker,
    load_split,
    log_eval,
    parse_args,
    seed_everything,
)


class TrainDS(Dataset):
    def __init__(self, df, processor, augment: bool, seed: int):
        self.df, self.proc, self.augment, self.seed = df, processor, augment, seed
        self.paths = df.path.tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        r = self.df.iloc[i]
        x, sr = sf.read(r.path, dtype="float32")
        if self.augment:
            rng = np.random.default_rng()
            pool = [
                sf.read(self.paths[j], dtype="float32")[0]
                for j in rng.choice(len(self.paths), 3, replace=False)
            ]
            x = A.random_field_aug(x, rng, pool)
        feats = self.proc.feature_extractor(
            x, sampling_rate=sr, return_tensors="np"
        ).input_features[0]
        labels = self.proc.tokenizer(r.raw_text).input_ids[:440]
        return {"feats": feats, "labels": labels}


def collate(batch: list[dict], start_id: int) -> dict:
    feats = torch.tensor(np.stack([b["feats"] for b in batch]))
    labs = [b["labels"][1:] if b["labels"][0] == start_id else b["labels"] for b in batch]
    m = max(map(len, labs))
    lab = torch.full((len(labs), m), -100, dtype=torch.long)
    for i, x in enumerate(labs):
        lab[i, : len(x)] = torch.tensor(x)
    return {"input_features": feats, "labels": lab}


def main() -> None:
    cfg, a = parse_args("whisper fine-tune")
    run = f"{cfg['name']}_s{a.seed}"
    out = Path(a.out_root) / run
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(a.seed)
    tracker = Tracker(run, {**cfg, "seed": a.seed})

    proc = WhisperProcessor.from_pretrained(cfg["model"])
    proc.tokenizer.set_prefix_tokens(language="swahili", task="transcribe")
    model_dir = out / "model"

    if not model_dir.exists():
        model = WhisperForConditionalGeneration.from_pretrained(cfg["model"]).cuda()  # type: ignore[call-arg]
        model.config.forced_decoder_ids = None
        model.generation_config.forced_decoder_ids = None
        if cfg["method"] == "lora":
            from peft import LoraConfig, get_peft_model

            model = get_peft_model(
                model,
                LoraConfig(
                    r=cfg.get("lora_r", 32),
                    lora_alpha=cfg.get("lora_alpha", 64),
                    lora_dropout=0.05,
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
                ),
            )
            model.print_trainable_parameters()
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tracker.log({"trainable_params": float(n_train)})

        train_df = load_split(a.train_manifest, "train")
        dev_df = load_split(a.train_manifest, "dev").head(a.eval_limit or 120)
        ds = TrainDS(train_df, proc, cfg.get("augment", False), a.seed)
        start_id = model.config.decoder_start_token_id if hasattr(model, "config") else 50258
        g = torch.Generator().manual_seed(a.seed)
        dl = DataLoader(
            ds,
            batch_size=cfg["batch_size"],
            shuffle=True,
            drop_last=True,
            generator=g,
            num_workers=12,
            collate_fn=lambda b: collate(b, start_id),
            persistent_workers=True,
        )
        steps_per_epoch = len(dl)
        total = a.max_steps or steps_per_epoch * cfg["epochs"]
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=0.01
        )
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
        dev_tr = WhisperTranscriber(model, proc)
        model.train()
        step, t0, done = 0, time.time(), False
        while not done:
            for batch in dl:
                batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)  # noqa: E702
                step += 1
                if step % 20 == 0:
                    print(
                        f"step {step}/{total} loss {loss.item():.4f} {time.time() - t0:.0f}s",
                        flush=True,
                    )
                    tracker.log({"train_loss": loss.item()}, step=step)
                if step % steps_per_epoch == 0 or step == total:
                    m = corpus_metrics(dev_df.text.tolist(), dev_tr(dev_df.path.tolist()))
                    print(f"[dev] step {step} WER {m['wer']:.4f} CER {m['cer']:.4f}", flush=True)
                    tracker.log({"dev_wer": m["wer"], "dev_cer": m["cer"]}, step=step)
                if step >= total:
                    done = True
                    break
        if cfg["method"] == "lora":
            model = model.merge_and_unload()
        model.save_pretrained(model_dir)
        proc.save_pretrained(model_dir)
        (out / "train_info.json").write_text(
            json.dumps(
                {
                    "steps": step,
                    "train_seconds": time.time() - t0,
                    "trainable_params": n_train,
                    **cfg,
                }
            )
        )
        del model, dev_tr, opt
        torch.cuda.empty_cache()

    tr = WhisperTranscriber.from_pretrained(str(model_dir))
    metrics = run_eval(tr, Path(a.eval_manifest), out, limit=a.eval_limit)
    log_eval(tracker, metrics)
    tracker.end()


if __name__ == "__main__":
    main()
