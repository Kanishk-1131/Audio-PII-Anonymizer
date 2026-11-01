"""Script-aware text normalisation and a coarse cross-script phonetic key.

Every token coming out of the ASR goes through :func:`normalize` before it is
compared with any lexicon entry, so lexicon files only need to list a word
once (nukta / chandrabindu / case / punctuation variants are folded here).

:func:`phonetic_key` maps a word written in Devanagari, Tamil or Latin script
to the same rough Latin skeleton, so that "Karthik", "கார்த்திக்" and
"कार्तिक" all compare equal. It is used for name matching only.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)

# Devanagari
_NUKTA = "\u093c"
_CHANDRABINDU = "\u0901"
_ANUSVARA = "\u0902"
_VIRAMA_DEVA = "\u094d"

# Characters that may be glued to a token by the ASR and must be ignored.
_EXTRA_STRIP = "\u0964\u0965"  # danda, double danda


def script_of(text: str) -> str:
    """Return the dominant script of ``text``: 'deva', 'taml', 'knda', 'latn' or 'other'."""
    counts = {"deva": 0, "taml": 0, "knda": 0, "latn": 0, "other": 0}
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F or 0xA8E0 <= cp <= 0xA8FF:
            counts["deva"] += 1
        elif 0x0B80 <= cp <= 0x0BFF:
            counts["taml"] += 1
        elif 0x0C80 <= cp <= 0x0CFF:
            counts["knda"] += 1
        elif ch.isascii() and ch.isalpha():
            counts["latn"] += 1
        elif ch.isalpha():
            counts["other"] += 1
    best = max(counts, key=counts.get)
    return best if counts[best] else "other"


def _strip_edges(text: str) -> str:
    """Remove punctuation / symbols / spaces from both ends of a token.

    ``str.strip`` with a regex ``\\W`` would be wrong for Indic scripts because
    combining vowel signs (matras) are not alphanumeric in Python, so we strip
    by Unicode category instead.
    """
    def is_junk(ch: str) -> bool:
        if ch in _EXTRA_STRIP:
            return True
        cat = unicodedata.category(ch)
        return cat[0] in ("P", "S", "Z", "C")

    start, end = 0, len(text)
    while start < end and is_junk(text[start]):
        start += 1
    while end > start and is_junk(text[end - 1]):
        end -= 1
    return text[start:end]


@lru_cache(maxsize=65536)
def normalize(text: str) -> str:
    """Canonical form used for every lexicon comparison."""
    if not text:
        return ""
    t = unicodedata.normalize("NFD", text)
    t = t.translate(_ZERO_WIDTH)
    t = t.replace(_NUKTA, "")
    t = unicodedata.normalize("NFC", t)
    t = t.replace(_CHANDRABINDU, _ANUSVARA)
    # Visarga used as a lazy "ह" in छः -> treat छः == छह is handled in lexicon;
    # here we only fold typographic variants.
    t = t.replace("\u2019", "'").replace("\u2018", "'")
    t = t.lower()
    t = _strip_edges(t)
    # Remove apostrophes that are not between letters (e.g. "'priya'").
    t = t.strip("'")
    return t


def ends_sentence(raw: str) -> bool:
    """True if the raw ASR token carries sentence / clause punctuation."""
    raw = raw.rstrip()
    return bool(raw) and raw[-1] in ".?!,;:\u0964\u0965"


_SPLIT_RE = re.compile(r"[\s\-\u2010-\u2015,/_+]+")


def split_compound(raw: str) -> list[str]:
    """Split one ASR word into sub-words ("98450-12345" -> two parts).

    A leading '+' (as in "+91") is dropped; the digits are kept.
    """
    parts = [p for p in _SPLIT_RE.split(raw) if p and _strip_edges(p)]
    return parts or [raw]


def native_digits_to_ascii(text: str) -> str:
    """Convert Devanagari / Tamil / other Unicode decimal digits to ASCII."""
    out = []
    for ch in text:
        if ch.isdigit() and not ch.isascii():
            try:
                out.append(str(unicodedata.digit(ch)))
                continue
            except (TypeError, ValueError):
                pass
        out.append(ch)
    return "".join(out)


# --------------------------------------------------------------------------
# Phonetic key (names)
# --------------------------------------------------------------------------

def _translit_to_latin(text: str, script: str) -> str:
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    except Exception:  # pragma: no cover - dependency missing
        return text
    if script == "deva":
        # Make schwa deletion explicit: a word that ends in a bare consonant
        # is pronounced without the final 'a' (राजन = rajan, not rajana).
        if text and (0x0915 <= ord(text[-1]) <= 0x0939 or 0x0958 <= ord(text[-1]) <= 0x095F):
            text = text + _VIRAMA_DEVA
        return transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    if script == "taml":
        # Replace alveolar ன (U+0BA9) with dental ந (U+0BA8) so ITRANS maps it to 'n'
        text = text.replace("\u0ba9", "\u0ba8")
        return transliterate(text, sanscript.TAMIL, sanscript.ITRANS)
    if script == "knda":
        return transliterate(text, sanscript.KANNADA, sanscript.ITRANS)
    return text


_DIGRAPHS = [
    ("ksh", "ks"), ("kh", "k"), ("gh", "k"), ("chh", "s"), ("ch", "s"),
    ("jh", "s"), ("th", "t"), ("dh", "t"), ("ph", "p"), ("bh", "p"),
    ("sh", "s"), ("zh", "l"), ("aa", "a"), ("ee", "i"), ("ii", "i"),
    ("oo", "u"), ("uu", "u"), ("ou", "u"), ("au", "u"), ("ai", "e"),
    ("ey", "e"), ("ay", "e"),
]
_SINGLE = str.maketrans({
    "g": "k", "q": "k", "c": "s", "j": "s", "z": "s", "x": "s",
    "d": "t", "b": "p", "f": "p", "w": "v", "o": "u", "h": "",
})


@lru_cache(maxsize=65536)
def phonetic_key(word: str) -> str:
    """Coarse, script-independent sound skeleton of a word.

    Deliberately lossy: voiced/unvoiced, aspirated/plain, retroflex/dental,
    vowel length and doubled consonants are all merged, because Tamil script
    does not distinguish most of them and ASR output is inconsistent.
    """
    w = normalize(word)
    if not w:
        return ""
    sc = script_of(w)
    if sc in ("deva", "taml", "knda"):
        w = _translit_to_latin(w, sc)
        # ITRANS markers for anusvara / chandrabindu / nasals -> n
        w = w.replace(".N", "n").replace("M", "n").replace("~N", "n").replace("~n", "n")
        w = w.replace(".h", "").replace("^", "")
    w = w.lower()
    w = re.sub(r"[^a-z]", "", w)
    for a, b in _DIGRAPHS:
        w = w.replace(a, b)
    w = w.translate(_SINGLE)
    # 'y' between vowels / at end behaves like 'i' in romanisations
    w = re.sub(r"(?<=[aeiu])y(?=$)", "i", w)
    w = re.sub(r"iy(?=[aeiu])", "i", w)  # priya / pria
    w = re.sub(r"(.)\1+", r"\1", w)  # collapse doubles
    return w
