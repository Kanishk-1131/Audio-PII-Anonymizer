"""Customer-name detection on a token stream.

Names cannot be found with a fixed list - any word can be a name - so the
detector combines several signals, from strongest to weakest:

1. Cue phrases            "मेरा नाम <X>", "என் பெயர் <X>", "my name is <X>"
2. Frames                 "मैं <X> बोल रही हूं", "this is <X> speaking"
3. Honorifics             "<X> जी", "<X> சார்", "Mr <X>", "श्री <X>"
4. Gazetteer              common Indian names (phonetic match, any script)
5. Memory                 once a name is known, EVERY other mention of any
                          part of it is beeped too - this catches the agent's
                          read-back ("Dhanyavaad Priya ji", "Nandri Karthik"),
                          including mentions that come before the introduction.

Every candidate is filtered against the language's stop-word list so that
words such as "बताइए" or "என்ன" are never taken for a name.
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz.fuzz import ratio

from .languages import LanguagePack
from .numbers import lookup
from .textnorm import phonetic_key, script_of
from .types import Detection, Token


@dataclass
class NameConfig:
    max_name_tokens: int = 3
    max_gap_inside_name: float = 0.45  # s between the words of one name
    max_gap_after_cue: float = 2.0     # s between cue and name
    fuzzy_threshold: float = 85.0      # rapidfuzz ratio on phonetic keys
    min_key_len: int = 3
    use_gazetteer_alone: bool = True


def _is_wordlike(tok: Token) -> bool:
    return any(ch.isalpha() for ch in tok.norm) and not any(ch.isdigit() for ch in tok.norm)


class _Ctx:
    def __init__(self, tokens: list[Token], pack: LanguagePack, cfg: NameConfig):
        self.tokens = tokens
        self.pack = pack
        self.cfg = cfg
        self.keys = [phonetic_key(self._strip_suffix(t.norm)) for t in tokens]

    def _strip_suffix(self, norm: str) -> str:
        for suf in self.pack.name_suffixes:
            if norm.endswith(suf) and len(norm) - len(suf) >= 2:
                return norm[: -len(suf)]
        return norm

    def is_stop(self, i: int) -> bool:
        t = self.tokens[i]
        if t.norm in self.pack.stopwords:
            return True
        if not _is_wordlike(t) or lookup(t, self.pack) is not None:
            return True
        key = phonetic_key(t.norm)
        return len(key) < 2 or key in self.pack.stopword_keys

    def candidate_ok(self, i: int) -> bool:
        return 0 <= i < len(self.tokens) and not self.is_stop(i) \
            and len(self.keys[i]) >= 2

    def in_gazetteer(self, i: int, allow_ambiguous: bool = True) -> bool:
        k = self.keys[i]
        if k in self.pack.gazetteer:
            return True
        return allow_ambiguous and k in self.pack.gazetteer_ambiguous

    def match_phrase(self, i: int, phrase: tuple[str, ...]) -> bool:
        if i + len(phrase) > len(self.tokens):
            return False
        return all(self.tokens[i + k].norm == phrase[k] for k in range(len(phrase)))


def _collect_after(ctx: _Ctx, j: int, prev_end: float) -> list[int]:
    """Collect up to max_name_tokens consecutive name-like tokens from j."""
    cfg, toks = ctx.cfg, ctx.tokens
    # skip fillers like "है" in "मेरा नाम है प्रिया"
    skipped = 0
    while j < len(toks) and skipped < 2 and toks[j].norm in ctx.pack.skip_after_cue:
        prev_end = toks[j].end
        j += 1
        skipped += 1
    out: list[int] = []
    while j < len(toks) and len(out) < cfg.max_name_tokens:
        gap = toks[j].start - prev_end
        limit = cfg.max_gap_after_cue if not out else cfg.max_gap_inside_name
        if gap > limit or not ctx.candidate_ok(j):
            break
        # A 3rd name token is rare; only attach if spoken tightly and in gazetteer
        if len(out) >= 2 and (gap > 0.25 or not ctx.in_gazetteer(j)):
            break
        out.append(j)
        prev_end = toks[j].end
        if toks[j].sentence_end:
            break
        j += 1
    return out


def _followed_by_terminator(ctx: _Ctx, idxs: list[int]) -> bool:
    last = idxs[-1]
    nxt = last + 1
    if nxt >= len(ctx.tokens):
        return False
    return ctx.tokens[nxt].norm in ctx.pack.terminators


def _detect_cues(ctx: _Ctx) -> list[tuple[list[int], str]]:
    found: list[tuple[list[int], str]] = []
    toks, pack = ctx.tokens, ctx.pack
    n = len(toks)
    for i in range(n):
        # --- strong cues
        for cue in pack.cues:
            if ctx.match_phrase(i, cue):
                idxs = _collect_after(ctx, i + len(cue), toks[i + len(cue) - 1].end)
                if idxs:
                    found.append((idxs, "cue:" + " ".join(cue)))
                break
        # --- weak cues: need gazetteer or terminator
        for cue in pack.weak_cues:
            if ctx.match_phrase(i, cue):
                idxs = _collect_after(ctx, i + len(cue), toks[i + len(cue) - 1].end)
                if idxs and (any(ctx.in_gazetteer(x) for x in idxs)
                             or _followed_by_terminator(ctx, idxs)):
                    found.append((idxs, "weak_cue:" + " ".join(cue)))
                break
        # --- gazetteer cues: only known names
        for cue in pack.gazetteer_cues:
            if ctx.match_phrase(i, cue):
                idxs = _collect_after(ctx, i + len(cue), toks[i + len(cue) - 1].end)
                # keep the leading run of gazetteer names (surname may follow)
                if idxs and ctx.in_gazetteer(idxs[0]):
                    keep = [idxs[0]] + [x for x in idxs[1:2] if ctx.in_gazetteer(x)]
                    found.append((keep, "gazetteer_cue:" + " ".join(cue)))
                break
        # --- frames  "<start> NAME <end>"
        for frames, strong in ((pack.frames, True), (pack.weak_frames, False)):
            for a, b in frames:
                if not ctx.match_phrase(i, a):
                    continue
                s = i + len(a)
                for e in range(s + 1, min(n, s + ctx.cfg.max_name_tokens + 1)):
                    if ctx.match_phrase(e, b):
                        idxs = list(range(s, e))
                        if idxs and all(ctx.candidate_ok(x) for x in idxs) and \
                                (strong or any(ctx.in_gazetteer(x) for x in idxs)):
                            found.append((idxs, "frame:" + " ".join(a) + "..." + " ".join(b)))
                        break
        # --- honorific before: "Mr Sharma", "श्री राम"
        if toks[i].norm in pack.honorifics_before:
            idxs = _collect_after(ctx, i + 1, toks[i].end)
            if idxs:
                found.append((idxs[:2], "honorific_before:" + toks[i].norm))
        # --- honorific after: "Priya ji", "Karthik sir"
        if toks[i].norm in pack.honorifics_after and i > 0:
            k = i - 1
            if ctx.candidate_ok(k) and toks[i].start - toks[k].end <= ctx.cfg.max_gap_inside_name \
                    and ctx.in_gazetteer(k):
                idxs = [k]
                if k - 1 >= 0 and ctx.candidate_ok(k - 1) and ctx.in_gazetteer(k - 1, allow_ambiguous=False):
                    idxs.insert(0, k - 1)
                found.append((idxs, "honorific_after:" + toks[i].norm))
    return found


def _matches_name(key: str, name_keys: set[str], cfg: NameConfig) -> bool:
    if len(key) < 2:
        return False
    for nk in name_keys:
        if key == nk:
            return True
        if len(nk) >= 4 and len(key) >= 3:
            if ratio(key, nk) >= cfg.fuzzy_threshold:
                return True
            # agglutinated suffix: "கார்த்திக்கு", "priyaji"
            if key.startswith(nk) and len(key) - len(nk) <= 4:
                return True
    return False


def detect_names(tokens: list[Token], pack: LanguagePack,
                 cfg: NameConfig | None = None,
                 blocked: set[int] | None = None) -> list[Detection]:
    """Return one Detection per contiguous group of name tokens."""
    cfg = cfg or NameConfig()
    blocked = blocked or set()
    ctx = _Ctx(tokens, pack, cfg)
    flagged: dict[int, str] = {}

    for idxs, reason in _detect_cues(ctx):
        for x in idxs:
            if x not in blocked:
                flagged.setdefault(x, reason)

    if cfg.use_gazetteer_alone:
        for i in range(len(tokens)):
            if i in blocked or i in flagged:
                continue
            if ctx.candidate_ok(i) and len(ctx.keys[i]) >= 4 and ctx.keys[i] in pack.gazetteer:
                flagged[i] = "gazetteer"

    # Memory: every other mention of a detected name part.
    name_keys = {ctx.keys[i] for i in flagged if len(ctx.keys[i]) >= cfg.min_key_len}
    if name_keys:
        for i in range(len(tokens)):
            if i in flagged or i in blocked:
                continue
            if not ctx.candidate_ok(i):
                continue
            if _matches_name(ctx.keys[i], name_keys, cfg):
                flagged[i] = "memory"

    # Group adjacent flagged tokens into one mention.
    dets: list[Detection] = []
    for i in sorted(flagged):
        if dets and i == dets[-1].last + 1 and \
                tokens[i].start - tokens[dets[-1].last].end <= cfg.max_gap_inside_name and \
                not tokens[dets[-1].last].sentence_end:
            d = dets[-1]
            d.last = i
            d.text = f"{d.text} {_clean(tokens[i])}"
            if flagged[i] not in d.reason:
                d.reason += "+" + flagged[i]
        else:
            dets.append(Detection("name", i, i, _clean(tokens[i]), reason=flagged[i]))
    return dets


def _clean(tok: Token) -> str:
    """Token text for the JSON report (punctuation stripped, script kept)."""
    from .textnorm import _strip_edges  # local import: private helper
    txt = _strip_edges(tok.raw)
    if script_of(txt) == "latn":
        return txt[:1].upper() + txt[1:]
    return txt
