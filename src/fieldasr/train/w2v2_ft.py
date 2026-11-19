"""wav2vec2 / XLS-R CTC baseline (character vocab)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2ForCTC, get_linear_schedule_with_warmup

from fieldasr.eval.harness import run_eval
from fieldasr.eval.metrics import corpus_metrics, normalize
from fieldasr.train.common import Tracker, load_split, log_eval, parse_args, seed_everything


def _wave(path: str) -> np.ndarray:
    x, _ = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return (x - x.mean()) / (x.std() + 1e-7)


class DS(Dataset):
    def __init__(self, df, vocab: dict[str, int]):
        self.paths, self.vocab = df.path.tolist(), vocab
        self.text = [normalize(t).replace(" ", "|") for t in df.raw_text]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        return _wave(self.paths[i]), [self.vocab[c] for c in self.text[i] if c in self.vocab]


def collate(b):
    m = max(len(x) for x, _ in b)
    wav = torch.zeros(len(b), m)
    att = torch.zeros(len(b), m, dtype=torch.long)
    ml = max(len(y) for _, y in b)
    lab = torch.full((len(b), ml), -100, dtype=torch.long)
    for i, (x, y) in enumerate(b):
        wav[i, : len(x)] = torch.from_numpy(x)
        att[i, : len(x)] = 1
        lab[i, : len(y)] = torch.tensor(y)
    return wav, att, lab


class CTCTranscriber:
    def __init__(self, model, inv: dict[int, str], bs: int = 16):
        self.model, self.inv, self.bs = model, inv, bs

    @torch.no_grad()
    def __call__(self, paths: list[str]) -> list[str]:
        was = self.model.training
        self.model.eval()
        out: list[str] = []
        dl = DataLoader(DS_infer(paths), batch_size=self.bs, num_workers=8, collate_fn=collate_x)
        for wav, att in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.model(wav.cuda(), attention_mask=att.cuda()).logits
            for ids in logits.argmax(-1).cpu().numpy():
                prev, s = -1, []
                for t in ids:
                    if t != prev and t != 0:
                        s.append(self.inv[int(t)])
                    prev = t
                out.append("".join(s).replace("|", " ").strip())
        self.model.train(was)
        return out


class DS_infer(Dataset):  # noqa: N801
    def __init__(self, paths):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        return _wave(self.paths[i])[: 30 * 16000]


def collate_x(b):
    m = max(len(x) for x in b)
    wav, att = torch.zeros(len(b), m), torch.zeros(len(b), m, dtype=torch.long)
    for i, x in enumerate(b):
        wav[i, : len(x)] = torch.from_numpy(x)
        att[i, : len(x)] = 1
    return wav, att


def main() -> None:
    cfg, a = parse_args("wav2vec2 CTC")
    run = f"{cfg['name']}_s{a.seed}"
    out = Path(a.out_root) / run
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(a.seed)
    tracker = Tracker(run, {**cfg, "seed": a.seed})
    train_df = load_split(a.train_manifest, "train")
    dev_df = load_split(a.train_manifest, "dev").head(a.eval_limit or 120)
    chars = sorted({c for t in train_df.raw_text for c in normalize(t).replace(" ", "|")})
    vocab = {"<pad>": 0, **{c: i + 1 for i, c in enumerate(chars)}}
    inv = {i: c for c, i in vocab.items()}
    model = Wav2Vec2ForCTC.from_pretrained(  # type: ignore[call-arg]
        cfg["model"],
        vocab_size=len(vocab),
        pad_token_id=0,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        mask_time_prob=0.05,
        layerdrop=0.05,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
    ).cuda()
    model.freeze_feature_encoder()
    dl = DataLoader(
        DS(train_df, vocab),
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=12,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(a.seed),
    )
    spe = len(dl)
    total = a.max_steps or spe * cfg["epochs"]
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=0.0
    )
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    tr = CTCTranscriber(model, inv)
    model.train()
    step, t0, done = 0, time.time(), False
    while not done:
        for wav, att, lab in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(wav.cuda(), attention_mask=att.cuda(), labels=lab.cuda()).loss
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
            if step % (spe * 2) == 0 or step == total:
                m = corpus_metrics(dev_df.text.tolist(), tr(dev_df.path.tolist()))
                print(f"[dev] step {step} WER {m['wer']:.4f} CER {m['cer']:.4f}", flush=True)
                tracker.log({"dev_wer": m["wer"], "dev_cer": m["cer"]}, step=step)
            if step >= total:
                done = True
                break
    model.save_pretrained(out / "model")
    (out / "vocab.json").write_text(json.dumps(vocab))
    (out / "train_info.json").write_text(
        json.dumps({"steps": step, "train_seconds": time.time() - t0, **cfg})
    )
    metrics = run_eval(tr, Path(a.eval_manifest), out, limit=a.eval_limit)
    log_eval(tracker, metrics)
    tracker.end()


if __name__ == "__main__":
    main()
