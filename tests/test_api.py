import io
import wave

from fastapi.testclient import TestClient

from fieldasr.deploy import api
from fieldasr.deploy.infer import Result
from fieldasr.privacy.scrub import scrub


def fake_tr(path, redact=True):
    text = "Bwana Hassan simu 0712345678"
    counts = {}
    if redact:
        r = scrub(text)
        text, counts = r.text, r.counts
    return Result("x", text, 1.0, 0.1, 0.1, counts)


def wav_bytes():
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)  # noqa: E702
        w.writeframes(b"\x00\x00" * 16000)
    return b.getvalue()


def client():
    api.app.dependency_overrides[api.get_transcriber] = lambda: fake_tr
    return TestClient(api.app)


def test_health_and_redaction():
    c = client()
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/transcribe", files={"file": ("a.wav", wav_bytes(), "audio/wav")}).json()
    assert "[NAME]" in r["text"] and "[PHONE]" in r["text"] and "0712" not in r["text"]
    raw = c.post(
        "/transcribe?redact=false", files={"file": ("a.wav", wav_bytes(), "audio/wav")}
    ).json()
    assert "0712345678" in raw["text"]


def test_rejects_empty_and_oversize(monkeypatch):
    c = client()
    assert c.post("/transcribe", files={"file": ("a.wav", b"", "audio/wav")}).status_code == 400
    monkeypatch.setattr(api, "MAX_BYTES", 10)
    assert (
        c.post("/transcribe", files={"file": ("a.wav", wav_bytes(), "audio/wav")}).status_code
        == 413
    )
