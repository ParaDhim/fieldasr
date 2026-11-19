import numpy as np
import pandas as pd
import pytest

from fieldasr.data.qc import analyze_array, flag_row
from fieldasr.data.splits import assert_clean, check_leakage

SR = 16000


def tone(sec=3.0, amp=0.3):
    t = np.arange(int(SR * sec)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_clipping_flagged():
    x = np.clip(tone(amp=3.0), -1, 1)
    assert "clipping" in flag_row(analyze_array(x, SR), "hi")


def test_silence_and_snr_flagged():
    x = np.zeros(SR * 3, dtype=np.float32)
    f = flag_row(analyze_array(x, SR), "hi")
    assert "mostly_silence" in f


def test_bad_duration_and_empty_transcript():
    f = flag_row(analyze_array(tone(0.2), SR), "  ")
    assert {"too_short", "empty_transcript"} <= set(f)
    assert "too_long" in flag_row(analyze_array(tone(31), SR), "hi")


def test_clean_clip_passes():
    x = tone()
    assert "clipping" not in flag_row(analyze_array(x.astype(np.float32), SR), "habari")


def test_leakage_detection():
    df = pd.DataFrame({"split": ["train", "test"], "text": ["a b", "a b"], "audio_md5": ["x", "y"]})
    rep = check_leakage(df)
    assert rep["text_overlap:test-train"] == 1
    with pytest.raises(ValueError):
        assert_clean(rep)
