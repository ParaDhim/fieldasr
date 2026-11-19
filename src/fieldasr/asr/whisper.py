"""Batched Whisper transcription (HF)."""

from __future__ import annotations

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class _Feats(Dataset):
    def __init__(self, paths: list[str], processor: WhisperProcessor):
        self.paths, self.fe = paths, processor.feature_extractor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> np.ndarray:
        x, sr = sf.read(self.paths[i], dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        return self.fe(x[: 30 * sr], sampling_rate=sr, return_tensors="np").input_features[0]


class WhisperTranscriber:
    def __init__(self, model, processor: WhisperProcessor, batch_size: int = 32, device="cuda"):
        self.model = getattr(model, "get_base_model", lambda: model)()
        self.processor, self.bs, self.device = processor, batch_size, device

    @classmethod
    def from_pretrained(cls, path: str, **kw) -> WhisperTranscriber:
        proc = WhisperProcessor.from_pretrained(path)
        model = WhisperForConditionalGeneration.from_pretrained(path).to(kw.get("device", "cuda"))
        return cls(model, proc, **kw)

    @torch.no_grad()
    def __call__(self, paths: list[str]) -> list[str]:
        was_training = self.model.training
        self.model.eval()
        dl = DataLoader(_Feats(paths, self.processor), batch_size=self.bs, num_workers=8)
        out: list[str] = []
        for feats in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ids = self.model.generate(
                    input_features=feats.to(self.device),
                    language="sw",
                    task="transcribe",
                    max_new_tokens=200,
                    num_beams=1,
                )
            out += self.processor.batch_decode(ids, skip_special_tokens=True)
        self.model.train(was_training)
        return [o.strip() for o in out]
