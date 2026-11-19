"""CPU inference with a CTranslate2 (faster-whisper) model. Shared by the CLI and the API."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

from fieldasr.privacy.scrub import scrub


@dataclass
class Result:
    file: str
    text: str
    audio_seconds: float
    seconds: float
    rtf: float
    pii_redacted: dict[str, int]


class Transcriber:
    def __init__(self, model_dir: str | Path, threads: int = 4, language: str = "sw"):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            str(model_dir), device="cpu", compute_type="default", cpu_threads=threads
        )
        self.language = language

    def __call__(self, path: str | Path, redact: bool = True) -> Result:
        t0 = time.perf_counter()
        segs, info = self.model.transcribe(
            str(path),
            language=self.language,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        text = " ".join(s.text.strip() for s in segs).strip()
        dt = time.perf_counter() - t0
        counts: dict[str, int] = {}
        if redact:
            r = scrub(text)
            text, counts = r.text, r.counts
        return Result(
            Path(path).name, text, info.duration, dt, dt / max(info.duration, 1e-9), counts
        )

    @staticmethod
    def to_dict(r: Result) -> dict:
        return asdict(r)
