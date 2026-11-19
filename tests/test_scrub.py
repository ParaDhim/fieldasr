import pytest

from fieldasr.privacy.scrub import scrub


@pytest.mark.parametrize(
    "inp,tag",
    [
        ("Nipigie 0712 345 678 kesho", "[PHONE]"),
        ("namba yangu +255 754 123 456", "[PHONE]"),
        ("barua pepe juma.k@example.co.tz tafadhali", "[EMAIL]"),
        ("kitambulisho 19850412123456 kinahitajika", "[ID]"),
        ("gari T123ABC liliegeshwa", "[ID]"),
        ("Bwana Hassan Mwinyi alifika", "[NAME]"),
        ("Dkt. Amina alitoa dawa", "[NAME]"),
        ("Fatuma alienda shule", "[NAME]"),
    ],
)
def test_scrubs(inp, tag):
    r = scrub(inp)
    assert tag in r.text, r


def test_leaves_ordinary_swahili_and_short_numbers():
    s = "Hosteli iliporomoka katika Makka takriban saa 4 asubuhi ya mwaka 2012"
    r = scrub(s)
    assert r.text == s and r.counts == {}


def test_counts_and_idempotent():
    r = scrub("Mama Neema, simu 0712345678")
    assert r.counts["PHONE"] == 1 and r.counts["NAME"] >= 1
    assert scrub(r.text).text == r.text
