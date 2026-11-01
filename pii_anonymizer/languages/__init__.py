"""Language-pack loader.

A language pack is a YAML file in this directory (``hi.yaml``, ``ta.yaml``,
...). Packs can ``include`` other packs (every Indian pack includes ``en``
because callers code-switch into English). Adding a language therefore means
adding ONE data file - no Python changes. See README "Adding a language".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from ..textnorm import normalize, phonetic_key

LANG_DIR = Path(__file__).resolve().parent

# Kinds of number words
DIGITS = "digits"      # value is a digit string ("9", "45", "560034")
TENS = "tens"          # value is the tens digit, combines with a following unit
HUNDRED = "hundred"
REPEAT = "repeat"      # value is the repeat count (double=2, triple=3)


@dataclass(frozen=True)
class NumWord:
    kind: str
    value: str
    weak: bool = False      # only a digit when it touches another digit
    is_numeral: bool = False  # written with digits by the ASR ("98450")


@dataclass
class LanguagePack:
    code: str
    name: str
    whisper_language: str
    number_words: dict[str, NumWord] = field(default_factory=dict)
    mobile_keywords: set[str] = field(default_factory=set)
    pincode_keywords: set[str] = field(default_factory=set)
    fillers: set[str] = field(default_factory=set)
    cues: list[tuple[str, ...]] = field(default_factory=list)
    weak_cues: list[tuple[str, ...]] = field(default_factory=list)
    gazetteer_cues: list[tuple[str, ...]] = field(default_factory=list)
    frames: list[tuple[tuple[str, ...], tuple[str, ...]]] = field(default_factory=list)
    weak_frames: list[tuple[tuple[str, ...], tuple[str, ...]]] = field(default_factory=list)
    skip_after_cue: set[str] = field(default_factory=set)
    terminators: set[str] = field(default_factory=set)
    honorifics_after: set[str] = field(default_factory=set)
    honorifics_before: set[str] = field(default_factory=set)
    name_suffixes: list[str] = field(default_factory=list)
    stopwords: set[str] = field(default_factory=set)
    stopword_keys: set[str] = field(default_factory=set)
    gazetteer: set[str] = field(default_factory=set)           # phonetic keys
    gazetteer_ambiguous: set[str] = field(default_factory=set)  # phonetic keys

    @property
    def breakers(self) -> set[str]:
        """Words that terminate a digit run (keywords of either PII type)."""
        return self.mobile_keywords | self.pincode_keywords


def available_languages() -> list[str]:
    return sorted(p.stem for p in LANG_DIR.glob("*.yaml"))


def _phrase(text: str) -> tuple[str, ...]:
    return tuple(normalize(w) for w in str(text).split() if normalize(w))


def _words(values) -> set[str]:
    out: set[str] = set()
    for v in values or []:
        n = normalize(str(v))
        if n:
            out.add(n)
    return out


def _read_yaml(code: str) -> dict:
    path = LANG_DIR / f"{code}.yaml"
    if not path.exists():
        raise ValueError(
            f"Unknown language '{code}'. Available: {', '.join(available_languages())}"
        )
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if data.get("code") != code:
        raise ValueError(f"{path.name}: 'code' field must be '{code}'")
    return data


def _load_gazetteer() -> tuple[set[str], set[str]]:
    strong: set[str] = set()
    ambiguous: set[str] = set()
    for name_file in sorted(LANG_DIR.glob("gazetteer*.txt")):
        for line in name_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            amb = line.startswith("*")
            key = phonetic_key(line.lstrip("*").strip())
            if len(key) < 3:
                continue
            (ambiguous if amb else strong).add(key)
    strong -= ambiguous
    return strong, ambiguous


class LexiconError(ValueError):
    pass


def _add_num(table: dict[str, NumWord], word: str, nw: NumWord, src: str) -> None:
    key = normalize(word)
    if not key:
        return
    old = table.get(key)
    if old is not None and (old.kind, old.value) != (nw.kind, nw.value):
        raise LexiconError(
            f"{src}: '{word}' is defined twice with different meanings "
            f"({old.kind}={old.value} and {nw.kind}={nw.value})"
        )
    if old is not None and old.weak and not nw.weak:
        table[key] = nw  # strong wins over weak
    elif old is None:
        table[key] = nw


def _merge_numbers(table: dict[str, NumWord], numbers: dict, src: str) -> None:
    for value, words in (numbers.get("digits") or {}).items():
        v = str(value)
        if not (v.isdigit() and len(v) == 1):
            raise LexiconError(f"{src}: digits key must be 0-9, got {value!r}")
        for w in words:
            _add_num(table, w, NumWord(DIGITS, v), src)
    for value, words in (numbers.get("weak_digits") or {}).items():
        v = str(value)
        for w in words:
            _add_num(table, w, NumWord(DIGITS, v, weak=True), src)
    for value, words in (numbers.get("compounds") or {}).items():
        v = str(value)
        if not v.isdigit():
            raise LexiconError(f"{src}: compounds key must be digits, got {value!r}")
        for w in words:
            _add_num(table, w, NumWord(DIGITS, v), src)
    for value, words in (numbers.get("tens") or {}).items():
        v = str(value)
        if not (v.isdigit() and len(v) == 1 and v != "0"):
            raise LexiconError(f"{src}: tens key must be 1-9, got {value!r}")
        for w in words:
            _add_num(table, w, NumWord(TENS, v), src)
    for w in numbers.get("hundred") or []:
        _add_num(table, w, NumWord(HUNDRED, "100"), src)
    for value, words in (numbers.get("repeat") or {}).items():
        for w in words:
            _add_num(table, w, NumWord(REPEAT, str(value)), src)


@lru_cache(maxsize=None)
def load_language(code: str) -> LanguagePack:
    code = code.strip().lower()
    seen: list[str] = []

    def collect(c: str) -> None:
        if c in seen:
            return
        data = _read_yaml(c)
        for inc in data.get("include") or []:
            collect(str(inc))
        seen.append(c)

    collect(code)
    main = _read_yaml(code)
    pack = LanguagePack(
        code=code,
        name=str(main.get("name", code)),
        whisper_language=str(main.get("whisper_language", code)),
    )
    for c in seen:  # included packs first, the requested pack last
        data = _read_yaml(c)
        src = f"{c}.yaml"
        _merge_numbers(pack.number_words, data.get("numbers") or {}, src)
        kw = data.get("keywords") or {}
        pack.mobile_keywords |= _words(kw.get("mobile"))
        pack.pincode_keywords |= _words(kw.get("pincode"))
        pack.fillers |= _words(data.get("fillers"))
        nm = data.get("names") or {}
        pack.cues += [_phrase(p) for p in nm.get("cues") or [] if _phrase(p)]
        pack.weak_cues += [_phrase(p) for p in nm.get("weak_cues") or [] if _phrase(p)]
        pack.gazetteer_cues += [_phrase(p) for p in nm.get("gazetteer_cues") or [] if _phrase(p)]
        for a, b in nm.get("frames") or []:
            pack.frames.append((_phrase(a), _phrase(b)))
        for a, b in nm.get("weak_frames") or []:
            pack.weak_frames.append((_phrase(a), _phrase(b)))
        pack.skip_after_cue |= _words(nm.get("skip_after_cue"))
        pack.terminators |= _words(nm.get("terminators"))
        pack.honorifics_after |= _words(nm.get("honorifics_after"))
        pack.honorifics_before |= _words(nm.get("honorifics_before"))
        pack.name_suffixes += [normalize(s) for s in nm.get("name_suffixes") or [] if normalize(s)]
        pack.stopwords |= _words(nm.get("stopwords"))

    # Longest cue first so "mera naam hai" wins over "mera naam".
    for lst in (pack.cues, pack.weak_cues, pack.gazetteer_cues):
        lst[:] = sorted(set(lst), key=lambda p: (-len(p), p))
    pack.name_suffixes = sorted(set(pack.name_suffixes), key=len, reverse=True)
    # Every keyword / filler / honorific / number word is also a non-name.
    pack.stopwords |= (
        pack.mobile_keywords | pack.pincode_keywords | pack.fillers
        | pack.honorifics_after | pack.honorifics_before | pack.terminators
        | set(pack.number_words)
    )
    pack.stopword_keys = {phonetic_key(w) for w in pack.stopwords if len(phonetic_key(w)) >= 2}
    pack.gazetteer, pack.gazetteer_ambiguous = _load_gazetteer()
    # A gazetteer name that sounds like a stop-word of this language is unsafe
    # to use on its own.
    clash = pack.gazetteer & pack.stopword_keys
    pack.gazetteer -= clash
    pack.gazetteer_ambiguous |= clash
    return pack
