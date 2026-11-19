import numpy as np
import pytest

from fieldasr.data import augment as A
from fieldasr.data.domain import domain_tags

RNG = lambda: np.random.default_rng(0)  # noqa: E731


def speechlike(n=32000):
    t = np.arange(n) / A.SR
    return (0.3 * np.sin(2 * np.pi * 300 * t) * (1 + np.sin(2 * np.pi * 3 * t))).astype(np.float32)


@pytest.mark.parametrize("snr", [0, 10, 20])
def test_mix_at_snr_is_accurate(snr):
    x = speechlike()
    n = A.pink_noise(len(x), RNG())
    y = A.mix_at_snr(x, n, snr)
    got = 20 * np.log10(A.rms(x) / A.rms(y - x))
    assert abs(got - snr) < 0.2


def test_reverb_and_cheap_mic_preserve_length_and_level():
    x = speechlike()
    for y in (A.reverb(x, 0.6, RNG()), A.cheap_mic(x, RNG())):
        assert len(y) == len(x) and np.isfinite(y).all()


def test_deterministic():
    x = speechlike()
    assert np.array_equal(A.reverb(x, 0.5, RNG()), A.reverb(x, 0.5, RNG()))


@pytest.mark.skipif(not A.have_ffmpeg(), reason="ffmpeg missing")
@pytest.mark.parametrize("codec", ["amr_nb", "mp3_8k"])
def test_codec_roundtrip_bandlimits(codec):
    x = speechlike() + 0.1 * np.sin(2 * np.pi * 6000 * np.arange(32000) / A.SR).astype(np.float32)
    y = A.codec_roundtrip(x, codec)
    assert len(y) == len(x)
    spec = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(len(y), 1 / A.SR)
    assert spec[(f > 4500)].max() < 0.05 * spec.max()  # nothing above 4 kHz survives


def test_domain_tags():
    assert "numerals" in domain_tags("saa 4 asubuhi")
    assert "health" in domain_tags("Hospitali kubwa")
    assert "school" in domain_tags("walimu na wanafunzi")
    assert "dates" in domain_tags("mwezi Machi")
    assert domain_tags("habari yako") == []
