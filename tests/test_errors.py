from fieldasr.analysis.errors import primary, sdi, tag_item


def test_sdi():
    assert sdi("a b c d", "a x c") == (1, 1, 0, 4)
    assert sdi("a b", "a b z z") == (0, 0, 2, 2)
    assert sdi("a b", "") == (0, 2, 0, 2)


def test_tags():
    assert "empty_output" in tag_item("habari yako", "Habari yako", "", "clean")
    assert "hallucination_loop" in tag_item(
        "saa nne", "Saa nne", "asante asante asante " * 5, "clean"
    )
    assert "numeral" in tag_item("saa 4 asubuhi", "Saa 4 asubuhi", "saa nne asubuhi", "clean")
    assert "word_segmentation" in tag_item("kwa ajili ya", "kwa ajili ya", "kwaajili ya", "clean")
    assert "proper_noun" in tag_item(
        "alienda makka jana", "Alienda Makka jana", "alienda maka jana", "clean"
    )
    assert "noise_or_channel" in tag_item("a b c", "a b c", "a x c", "pink_snr0")
    assert primary(["noise_or_channel", "numeral"]) == "numeral"
