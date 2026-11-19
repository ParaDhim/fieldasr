import pandas as pd

from fieldasr.review.store import ReviewStore


def make(tmp_path):
    q = pd.DataFrame(
        {
            "item_id": ["a", "b", "c"],
            "path": ["a.wav", "b.wav", "c.wav"],
            "hyp": ["habari", "asante", "karibu"],
            "condition": ["clean"] * 3,
        }
    )
    q.to_csv(tmp_path / "q.csv", index=False)
    return ReviewStore(tmp_path / "q.csv", tmp_path / "log.jsonl")


def test_flow_and_export(tmp_path):
    s = make(tmp_path)
    assert s.next_pending(-1) == 0
    s.save("a", "habari yako", False)
    s.save("b", "???", True)  # bad audio -> excluded
    s.save("c", "Bwana Juma alifika", False)  # PII -> redacted
    s.save("a", "habari za asubuhi", False)  # latest entry wins
    assert s.next_pending(-1) is None and s.progress() == (3, 3)
    stats = s.export(tmp_path / "out.csv")
    out = pd.read_csv(tmp_path / "out.csv")
    assert stats["exported"] == 2 and stats["flagged_bad_audio_or_empty"] == 1
    assert out.set_index("item_id").loc["a", "text"] == "habari za asubuhi"
    assert "[NAME]" in out.set_index("item_id").loc["c", "text"] and "Juma" not in " ".join(
        out.text
    )


def test_resume_after_restart(tmp_path):
    s = make(tmp_path)
    s.save("a", "x", False)
    s2 = ReviewStore(tmp_path / "q.csv", tmp_path / "log.jsonl")
    assert s2.next_pending(-1) == 1
