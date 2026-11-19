"""Small offline-capable transcription API.

    MODEL_DIR=/models/int8 uvicorn fieldasr.deploy.api:app --host 0.0.0.0 --port 8000

Privacy defaults: uploads are written to a temp file that is deleted immediately after decoding;
transcripts and audio are never logged or stored; PII scrubbing is on unless ?redact=false.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from fieldasr.deploy.infer import Result, Transcriber

MAX_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "25")) * 1024 * 1024
app = FastAPI(title="FieldASR", version="0.1.0")
_model: Transcriber | None = None


def get_transcriber() -> Callable[..., Result]:
    global _model
    if _model is None:
        d = os.environ.get("MODEL_DIR", "models/int8")
        if not Path(d, "model.bin").exists():
            raise HTTPException(503, f"model not found at {d} (set MODEL_DIR)")
        _model = Transcriber(d, threads=int(os.environ.get("CPU_THREADS", "4")))
    return _model


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_dir": os.environ.get("MODEL_DIR", "models/int8")}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    redact: bool = True,
    tr: Callable[..., Result] = Depends(get_transcriber),
) -> dict:
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"file larger than {MAX_BYTES // 2**20} MB")
    if not data:
        raise HTTPException(400, "empty upload")
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "a.wav").suffix or ".wav") as f:
        f.write(data)
        f.flush()
        try:
            r = tr(f.name, redact=redact)
        except Exception as e:  # undecodable audio etc.
            raise HTTPException(422, f"could not transcribe: {type(e).__name__}") from e
    out = Transcriber.to_dict(r)
    out["file"] = file.filename
    return out
