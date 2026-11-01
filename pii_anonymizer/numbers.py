"""Mobile-number and PIN-code detection on a token stream.

Algorithm
---------
1. Every token is looked up in the language pack's number lexicon (digits,
   compounds such as "चौरानवे"=94, tens such as "ninety", "hundred", and
   repeaters "double"/"triple") or parsed as a numeral ("98450").
2. Consecutive number tokens are chained into *runs*. A run may bridge a
   filler word ("uh", "और", "சரி") or ONE unknown word (an ASR error in the
   middle of a number) as long as the pause is short; a PII keyword
   ("pin", "number") always ends a run.
3. Each run is expanded to a digit string (double 6 -> 66, ninety eight -> 98,
   five hundred sixty -> 560).
4. The run is classified with its length and nearby keywords:
   10 digits (or 0/91/+91 prefixed) -> mobile, 6 digits -> PIN, other
   lengths -> the nearest keyword decides. Short runs that repeat part of an
   already-found number (agent read-back) inherit that number's type.
5. If a mobile/PIN run is *short* of digits, the neighbouring word on each
   side is added to the beep as a safety margin (a missed digit is a leak).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .languages import DIGITS, HUNDRED, REPEAT, TENS, LanguagePack, NumWord
from .types import Detection, Token

_NUMERAL_RE = re.compile(r"^\d+$")


@dataclass
class NumberConfig:
    max_digit_gap: float = 2.5        # s of silence allowed inside a number
    max_bridge_tokens: int = 2        # filler/unknown tokens allowed between digits
    max_unknown_bridge: int = 1       # of which unknown (non-filler) words
    max_bridge_gap: float = 1.5       # s spanned by a bridge
    context_tokens_back: int = 14     # keyword search window (tokens)
    context_seconds_back: float = 12.0
    context_tokens_fwd: int = 4
    min_digits_with_context: int = 3
    min_digits_without_context: int = 5
    deficit_extend_max_gap: float = 0.6  # s, neighbour added when digits are missing


@dataclass
class NumberRun:
    members: list[int]                    # token indices (incl. bridged words)
    num_members: list[int]                # token indices that are number words
    digits: str = ""
    type: str | None = None
    reason: str = ""
    extra_before: list[int] = field(default_factory=list)
    extra_after: list[int] = field(default_factory=list)


def lookup(tok: Token, pack: LanguagePack) -> NumWord | None:
    if tok.quantity:
        return None
    if _NUMERAL_RE.match(tok.norm):
        return NumWord(DIGITS, tok.norm, is_numeral=True)
    return pack.number_words.get(tok.norm)


def expand_digits(words: list[NumWord]) -> str:
    """Convert a sequence of number words to the digit string the caller meant."""
    out: list[str] = []
    rep = 1
    k = 0
    n = len(words)
    while k < n:
        w = words[k]
        if w.kind == REPEAT:
            rep = int(w.value)
            k += 1
            continue
        if w.kind == TENS:
            nxt = words[k + 1] if k + 1 < n else None
            if nxt is not None and nxt.kind == DIGITS and not nxt.is_numeral \
                    and len(nxt.value) == 1 and nxt.value != "0":
                s = w.value + nxt.value
                k += 2
            else:
                s = w.value + "0"
                k += 1
        elif w.kind == HUNDRED:
            k += 1
            if out and len(out[-1]) == 1:
                head = out.pop()
                rem = "00"
                nxt = words[k] if k < n else None
                if nxt is not None and nxt.kind == TENS:
                    nn = words[k + 1] if k + 1 < n else None
                    if nn is not None and nn.kind == DIGITS and not nn.is_numeral \
                            and len(nn.value) == 1 and nn.value != "0":
                        rem = nxt.value + nn.value
                        k += 2
                    else:
                        rem = nxt.value + "0"
                        k += 1
                elif nxt is not None and nxt.kind == DIGITS and not nxt.is_numeral \
                        and len(nxt.value) <= 2:
                    rem = nxt.value.zfill(2)
                    k += 1
                s = head + rem
            else:
                s = "100"
            rep_s = s
            out.append(rep_s)
            rep = 1
            continue
        else:  # DIGITS
            s = w.value
            k += 1
            # Deduplicate boundary zero when Whisper splits a hyphenated numeral
            # like "5600" + "-034" across a zero
            if w.is_numeral and out and out[-1].endswith("00") and s.startswith("0") and len(s) >= 2:
                total_len = sum(len(x) for x in out) + len(s)
                if total_len in (7, 11):
                    s = s[1:]
        if rep > 1 and s:
            if w.is_numeral and len(s) >= rep and s[:rep] == s[0] * rep:
                pass  # ASR already wrote "66" for "double 6"
            else:
                s = s[0] * rep + s[1:]
        rep = 1
        out.append(s)
    return "".join(out)


def _build_runs(tokens: list[Token], pack: LanguagePack, cfg: NumberConfig) -> list[NumberRun]:
    lex = [lookup(t, pack) for t in tokens]
    breakers = pack.breakers
    runs: list[NumberRun] = []
    i, n = 0, len(tokens)
    while i < n:
        if lex[i] is None or lex[i].kind == HUNDRED:
            i += 1
            continue
        members = [i]
        nums = [i]
        last = i
        bridge: list[int] = []
        j = i + 1
        while j < n:
            t = tokens[j]
            if lex[j] is not None:
                gap = t.start - tokens[last].end
                if gap > cfg.max_digit_gap:
                    break
                if bridge and t.start - tokens[last].end > cfg.max_bridge_gap:
                    break
                members.extend(bridge)
                bridge = []
                members.append(j)
                nums.append(j)
                last = j
                j += 1
                continue
            if t.norm in breakers:
                break
            if len(bridge) >= cfg.max_bridge_tokens:
                break
            unknown = sum(1 for b in bridge + [j] if tokens[b].norm not in pack.fillers)
            if unknown > cfg.max_unknown_bridge:
                break
            bridge.append(j)
            j += 1
        runs.append(NumberRun(members=members, num_members=nums))
        i = last + 1
    return runs


def _trim_weak_edges(run: NumberRun, tokens: list[Token], pack: LanguagePack) -> None:
    """Drop weak words ("to", "for", "no") at run edges when they are surplus."""
    def nw(idx: int) -> NumWord | None:
        return lookup(tokens[idx], pack)

    def digits_of(idxs: list[int]) -> str:
        return expand_digits([nw(x) for x in idxs])

    changed = True
    while changed and len(run.num_members) > 1:
        changed = False
        d = digits_of(run.num_members)
        for edge in (0, -1):
            idx = run.num_members[edge]
            w = nw(idx)
            if w is None or not w.weak:
                continue
            # Never strip a weak digit modified by a preceding repeater ("double R", "double oh")
            if edge == -1 and len(run.num_members) >= 2 and nw(run.num_members[-2]).kind == REPEAT:
                continue
            rest = run.num_members[1:] if edge == 0 else run.num_members[:-1]
            d2 = digits_of(rest)
            surplus = (len(d) > 10 and len(d2) >= 10) or (len(d) == 7 and len(d2) == 6)
            if surplus:
                run.num_members = rest
                if edge == 0:
                    run.members = run.members[run.members.index(rest[0]):]
                else:
                    run.members = run.members[: run.members.index(rest[-1]) + 1]
                changed = True
                break
    # A run that has no strong digit/tens word (unless REPEAT + weak digit) is not a number.
    has_strong = any(
        (nw(x) is not None and not nw(x).weak and nw(x).kind in (DIGITS, TENS))
        for x in run.num_members
    )
    has_repeated_weak = any(
        i > 0 and nw(run.num_members[i - 1]).kind == REPEAT and nw(run.num_members[i]).weak
        for i in range(len(run.num_members))
    )
    if not (has_strong or has_repeated_weak):
        run.num_members = []
    # Trailing repeaters ("... double") carry no digit
    while run.num_members and nw(run.num_members[-1]).kind == REPEAT:
        run.num_members.pop()
    while run.num_members and nw(run.num_members[0]).kind == HUNDRED:
        run.num_members.pop(0)
    if run.num_members:
        lo, hi = run.num_members[0], run.num_members[-1]
        run.members = [m for m in run.members if lo <= m <= hi]
    else:
        run.members = []


def _context(tokens: list[Token], run: NumberRun, pack: LanguagePack, cfg: NumberConfig) -> str | None:
    """Nearest PII keyword around the run: 'mobile', 'pincode' or None."""
    first, last = run.members[0], run.members[-1]
    t0 = tokens[first].start
    best: tuple[int, str] | None = None
    for dist, k in enumerate(range(first - 1, max(-1, first - 1 - cfg.context_tokens_back), -1), 1):
        if t0 - tokens[k].end > cfg.context_seconds_back:
            break
        typ = _kw_type(tokens[k].norm, pack)
        if typ:
            best = (dist, typ)
            break
    for dist, k in enumerate(range(last + 1, min(len(tokens), last + 1 + cfg.context_tokens_fwd)), 1):
        if tokens[k].start - tokens[last].end > 2.5:
            break
        typ = _kw_type(tokens[k].norm, pack)
        if typ:
            if best is None or dist < best[0]:
                best = (dist, typ)
            break
    return best[1] if best else None


def _kw_type(norm: str, pack: LanguagePack) -> str | None:
    # PIN keywords are checked first: "pin code number" means a PIN.
    if norm in pack.pincode_keywords:
        return "pincode"
    if norm in pack.mobile_keywords:
        return "mobile"
    return None


def _classify(d: str, ctx: str | None, cfg: NumberConfig) -> tuple[str | None, str]:
    L = len(d)
    if L == 10 and d[0] in "6789":
        return "mobile", "10 digits"
    if L == 11 and d[0] == "0":
        return "mobile", "0 + 10 digits"
    if L == 12 and d.startswith("91"):
        return "mobile", "91 + 10 digits"
    if L >= 8:
        return "mobile", f"{L} digits"
    if L == 7:
        if ctx == "pincode":
            return "pincode", "7 digits near pincode keyword"
        return "mobile", "7 digits"
    if L == 6:
        if ctx == "mobile":
            return "mobile", "6 digits after mobile keyword (partial number)"
        return "pincode", "6 digits"
    if ctx and L >= cfg.min_digits_with_context:
        return ctx, f"{L} digits near {ctx} keyword"
    if L >= cfg.min_digits_without_context:
        return ("mobile" if L >= 7 else "pincode"), f"{L} digits, no keyword"
    return None, ""


def detect_numbers(tokens: list[Token], pack: LanguagePack,
                   cfg: NumberConfig | None = None) -> list[Detection]:
    cfg = cfg or NumberConfig()
    runs = _build_runs(tokens, pack, cfg)
    kept: list[NumberRun] = []
    for run in runs:
        _trim_weak_edges(run, tokens, pack)
        if not run.num_members:
            continue
        run.digits = expand_digits([lookup(tokens[x], pack) for x in run.num_members])
        if not run.digits:
            continue
        ctx = _context(tokens, run, pack, cfg)
        run.type, run.reason = _classify(run.digits, ctx, cfg)
        kept.append(run)

    # Second pass: short runs that repeat part of a found number.
    found = [r for r in kept if r.type]
    for r in kept:
        if r.type or len(r.digits) < 2:
            continue
        for f in found:
            if r.digits in f.digits and len(r.digits) >= min(3, len(f.digits)):
                r.type, r.reason = f.type, f"repeats part of {f.digits}"
                break

    dets: list[Detection] = []
    for r in kept:
        if not r.type:
            continue
        target = 10 if r.type == "mobile" else 6
        if len(r.digits) < target:
            _deficit_extend(r, tokens, pack, cfg)
        dets.append(Detection(
            type=r.type, first=r.members[0], last=r.members[-1], text=r.digits,
            reason=r.reason, extra_before=r.extra_before, extra_after=r.extra_after,
        ))
    return dets


def _deficit_extend(run: NumberRun, tokens: list[Token], pack: LanguagePack,
                    cfg: NumberConfig) -> None:
    """Add one neighbouring word on each side when digits seem to be missing."""
    first, last = run.members[0], run.members[-1]
    allowed_ignore = pack.breakers | pack.stopwords | pack.fillers
    k = first - 1
    if k >= 0 and tokens[k].norm not in allowed_ignore \
            and tokens[first].start - tokens[k].end <= cfg.deficit_extend_max_gap \
            and not tokens[k].sentence_end:
        run.extra_before.append(k)
    k = last + 1
    if k < len(tokens) and tokens[k].norm not in allowed_ignore \
            and tokens[k].start - tokens[last].end <= cfg.deficit_extend_max_gap \
            and not tokens[last].sentence_end:
        run.extra_after.append(k)
