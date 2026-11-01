"""Turn ASR words into normalised tokens with their own timings."""

from __future__ import annotations

from .textnorm import ends_sentence, native_digits_to_ascii, normalize, split_compound
from .types import Token, Word

_QUANTITY_CHARS = set("₹$€£%:")


def tokenize(words: list[Word]) -> list[Token]:
    tokens: list[Token] = []
    for wi, w in enumerate(words):
        raw_word = native_digits_to_ascii(w.text.strip())
        if not raw_word:
            continue
        parts = split_compound(raw_word)
        quantity = any(c in _QUANTITY_CHARS for c in raw_word)
        dur = max(0.0, w.end - w.start)
        total = sum(max(1, len(p)) for p in parts)
        t = w.start
        for pi, p in enumerate(parts):
            share = dur * max(1, len(p)) / total if total else 0.0
            p_start, p_end = t, t + share
            t = p_end
            norm = normalize(p)
            if not norm:
                continue
            tokens.append(Token(
                raw=p,
                norm=norm,
                start=p_start,
                end=p_end if pi < len(parts) - 1 else w.end,
                word_index=wi,
                sentence_end=ends_sentence(p) or (pi == len(parts) - 1 and ends_sentence(raw_word)),
                quantity=quantity,
            ))
    return tokens
