import json

from fieldasr.aggregate import aggregate, summarize
from fieldasr.eval.metrics import corpus_metrics, normalize


def test_normalize():
    assert normalize("Habari,  Yako!") == "habari yako"
    assert normalize("tindikali-za-amino 4") == "tindikali-za-amino 4"


def test_wer_cer():
    m = corpus_metrics(["a b c d"], ["a b x d"])
    assert abs(m["wer"] - 0.25) < 1e-9 and m["n"] == 1
    assert corpus_metrics([""], ["x"])["n"] == 0


def test_aggregate(tmp_path):
    for s, w in [(0, 0.2), (1, 0.4)]:
        d = tmp_path / f"m_s{s}"
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps({"clean": {"wer": w, "cer": w / 2, "n": 5}}))
    s = summarize(aggregate(tmp_path))
    assert abs(s.iloc[0]["mean"] - 0.3) < 1e-9 and s.iloc[0]["n_seeds"] == 2
