"""Rule-based PII scrubbing for transcripts: phones, emails, URLs, ID-like strings, names.

Deliberately conservative and transparent (regex + gazetteer + honorific heuristics), not a
trained NER model. Recall for *names* is limited to the gazetteer and honorific patterns; see
docs/data_handling.md for what this does and does not guarantee.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
URL = re.compile(r"\b(?:https?://|www\.)\S+", re.I)
# +255 712 345 678 / 0712-345-678 / 254712345678 / (07) 12 345 678 style
PHONE = re.compile(
    r"(?<!\w)(?:\+?(?:255|254|256|250)[\s.-]?|0)[67]\d{1,2}[\s.-]?\d{3}[\s.-]?\d{3}(?!\d)"
)
# ID-like: 7+ digit runs, or mixed letter+digit tokens with >=6 chars (national ID, plates, MRNs)
LONG_DIGITS = re.compile(r"(?<![\w])\d[\d\s-]{6,}\d(?![\w])")
MIXED_ID = re.compile(r"\b(?=[A-Za-z0-9-]*\d)(?=[A-Za-z0-9-]*[A-Za-z])[A-Za-z0-9-]{6,}\b")
HONORIFIC = re.compile(
    r"\b(?:bwana|bw\.?|bi|bibi|mama|baba|mzee|mheshimiwa|dkt\.?|dr\.?|mwalimu|sheikh|mchungaji|"
    r"mr\.?|mrs\.?|ms\.?)\s+([A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)?)",
)
# Small gazetteer of common East-African given names (extend via load_names()).
GIVEN_NAMES = {
    "juma",
    "amina",
    "fatuma",
    "hassan",
    "hamisi",
    "ali",
    "omar",
    "halima",
    "zainab",
    "rehema",
    "neema",
    "baraka",
    "mwajuma",
    "salma",
    "rashid",
    "said",
    "juliana",
    "grace",
    "joseph",
    "john",
    "mary",
    "peter",
    "paulo",
    "emmanuel",
    "elizabeth",
    "wanjiru",
    "otieno",
    "kamau",
    "njeri",
    "atieno",
    "mwangi",
    "kipchoge",
    "achieng",
    "musa",
    "issa",
    "khadija",
    "mariam",
    "hawa",
    "shabani",
    "athumani",
    "mwanaidi",
    "upendo",
    "furaha",
    "zawadi",
    "tatu",
}


@dataclass
class ScrubResult:
    text: str
    counts: dict[str, int]


def load_names(path: str) -> set[str]:
    with open(path, encoding="utf-8") as f:
        return {ln.strip().lower() for ln in f if ln.strip()}


def scrub(text: str, names: set[str] | None = None) -> ScrubResult:
    names = GIVEN_NAMES if names is None else names
    counts = {"EMAIL": 0, "URL": 0, "PHONE": 0, "ID": 0, "NAME": 0}

    def sub(pat: re.Pattern[str], tag: str, s: str) -> str:
        s, n = pat.subn(f"[{tag}]", s)
        counts[tag] += n
        return s

    s = sub(EMAIL, "EMAIL", text)
    s = sub(URL, "URL", s)
    s = sub(PHONE, "PHONE", s)
    s = sub(LONG_DIGITS, "ID", s)
    s = sub(MIXED_ID, "ID", s)

    def honorific(m: re.Match[str]) -> str:
        counts["NAME"] += 1
        return m.group(0).replace(m.group(1), "[NAME]")

    s = HONORIFIC.sub(honorific, s)

    def gaz(m: re.Match[str]) -> str:
        if m.group(0).lower() in names and m.group(0)[0].isupper():
            counts["NAME"] += 1
            return "[NAME]"
        return m.group(0)

    s = re.sub(r"\b[\w'-]+\b", gaz, s)
    return ScrubResult(s, {k: v for k, v in counts.items() if v})
