"""Convert token-level detections into beep time spans.

Whisper word timestamps have two systematic quirks on Indian-language calls:
1. When spoken digits are emitted as a single numeral ("560034"), its DTW span
   often covers only a portion of the digit utterance.
2. Turn silences (and sometimes the final 200-500 ms syllable of a number) get
   attached to the ``start`` of the following word token.

:class:`_EnergyIndex` resolves both using the 10 ms RMS contour:
* trims dead turn-silence inside the raw token span;
* finds the true acoustic offset/onset of the neighbouring non-PII words when
  Whisper lumped a turn pause (>= 280 ms silence) into them;
* extends numbers across brief intra-number pauses (<= 420 ms) within those
  acoustic guardrails; and
* finishes any continuously-voiced syllable at the span edges so a final
  consonant or digit syllable is never clipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import frame_rms
from .types import Detection, Redaction, Token


@dataclass
class SpanConfig:
    pad_before: float = 0.08             # s added before the first PII word
    pad_after: float = 0.10              # s added after the last PII word
    snap_max: float = 0.15               # extra s allowed while a name is voiced
    number_bridge_silence: float = 0.42  # max intra-number pause bridged (s)
    number_snap_max: float = 4.5         # max s to extend across grouped digits
    turn_pause_min: float = 0.28         # silence inside one word => turn boundary
    merge_gap: float = 0.20              # spans closer than this are merged (same type)
    min_duration: float = 0.20           # never emit a beep shorter than this


_PRIORITY = {"mobile": 3, "pincode": 2, "name": 1}


def detections_to_redactions(dets: list[Detection], tokens: list[Token], *,
                             duration: float, mono: np.ndarray | None,
                             sr: int, cfg: SpanConfig | None = None) -> list[Redaction]:
    cfg = cfg or SpanConfig()
    energy = _EnergyIndex(mono, sr) if mono is not None and len(mono) else None
    reds: list[Redaction] = []
    for d in dets:
        idxs = list(range(d.first, d.last + 1)) + d.extra_before + d.extra_after
        s = min(tokens[i].start for i in idxs)
        e = max(tokens[i].end for i in idxs)
        if e < s:
            s, e = e, s

        min_idx = min(idxs)
        max_idx = max(idxs)
        if energy is not None:
            # 1. Snap raw token edges inward past any dead turn-silence
            s, e = energy.trim_outer_silence(s, e)

            # 2. Compute true acoustic boundaries of adjacent non-PII words
            if min_idx > 0:
                pt = tokens[min_idx - 1]
                prev_end = energy.true_offset(pt.start, pt.end, cfg.turn_pause_min)
                prev_guard = pt.start + 0.18
            else:
                prev_end = 0.0
                prev_guard = 0.0

            if max_idx + 1 < len(tokens):
                nt = tokens[max_idx + 1]
                next_start = energy.true_onset(nt.start, nt.end, cfg.turn_pause_min)
                next_guard = nt.end - 0.18
            else:
                next_start = duration
                next_guard = duration

            # 3. Extend across grouped digits or voiced edges
            if d.type in ("mobile", "pincode"):
                s, e = energy.extend_with_gaps(
                    s, e,
                    max_extra=cfg.number_snap_max,
                    bridge_s=cfg.number_bridge_silence,
                    floor_s=prev_end + 0.06,
                    ceil_s=next_start - 0.06,
                )
            else:
                s, e = energy.extend(s, e, cfg.snap_max)

            # 4. If an edge is still mid-syllable (e.g. Whisper placed next_tok.start
            #    inside the last digit before a short <280ms intra-turn pause), finish
            #    the continuously-voiced burst without eating the neighbour word.
            s, e = energy.finish_voiced_burst(s, e, prev_guard, next_guard)
        else:
            prev_end = tokens[min_idx - 1].end if min_idx > 0 else 0.0
            next_start = tokens[max_idx + 1].start if max_idx + 1 < len(tokens) else duration

        s -= cfg.pad_before
        e += cfg.pad_after
        if min_idx > 0 and s - prev_end > 0.25:
            s = max(s, prev_end + 0.06)
        if max_idx + 1 < len(tokens) and next_start - e > 0.25:
            e = min(e, next_start - 0.06)
        if e - s < cfg.min_duration:
            mid = (s + e) / 2.0
            s, e = mid - cfg.min_duration / 2.0, mid + cfg.min_duration / 2.0
        s = max(0.0, s)
        e = min(duration, e)
        if e > s:
            reds.append(Redaction(s, e, d.type, d.text))
    return merge_redactions(reds, cfg.merge_gap)


def merge_redactions(reds: list[Redaction], merge_gap: float) -> list[Redaction]:
    reds = sorted(reds, key=lambda r: (r.start, r.end))
    out: list[Redaction] = []
    for r in reds:
        if out:
            p = out[-1]
            if r.type == p.type and r.start - p.end <= merge_gap:
                if r.text and r.text not in p.text:
                    p.text = f"{p.text} {r.text}".strip() if p.type == "name" else p.text + r.text
                p.end = max(p.end, r.end)
                continue
            if r.start < p.end:  # overlapping spans of different types
                if r.end <= p.end:
                    if _PRIORITY[r.type] > _PRIORITY[p.type]:
                        p.type, p.text = r.type, r.text
                    continue
                cut = (r.start + p.end) / 2.0
                p.end = cut
                r = Redaction(cut, r.end, r.type, r.text)
        out.append(Redaction(r.start, r.end, r.type, r.text))
    return out


class _EnergyIndex:
    """10 ms RMS energy index for acoustic span refinement."""

    def __init__(self, mono: np.ndarray, sr: int, frame_s: float = 0.010):
        self.frame_s = frame_s
        self.rms = frame_rms(mono, sr, frame_s)
        floor = float(np.percentile(self.rms, 10))
        peak = float(np.percentile(self.rms, 98))
        self.thr = max(floor * 2.5, peak * 0.03, 1e-4)

    def _voiced(self, k: int) -> bool:
        return 0 <= k < len(self.rms) and self.rms[k] > self.thr

    def trim_outer_silence(self, s: float, e: float) -> tuple[float, float]:
        """Shrink [s, e] inward if its outer edges sit in dead silence."""
        k0 = max(0, int(s / self.frame_s))
        k1 = min(len(self.rms), int(np.ceil(e / self.frame_s)))
        voiced = [k for k in range(k0, k1) if self._voiced(k)]
        if not voiced:
            return s, e
        s2 = max(s, voiced[0] * self.frame_s)
        e2 = min(e, (voiced[-1] + 1) * self.frame_s)
        return (s2, e2) if e2 > s2 else (s, e)

    def true_onset(self, ns: float, ne: float, min_pause_s: float) -> float:
        """If a single next-word token [ns, ne] contains a turn pause (>= min_pause_s)
        before its final voiced word, return the onset after that pause."""
        k0 = max(0, int(ns / self.frame_s))
        k1 = min(len(self.rms), int(np.ceil(ne / self.frame_s)))
        min_pause = max(2, int(round(min_pause_s / self.frame_s)))
        # Locate the last voiced frame inside [k0, k1)
        v_last = -1
        for k in range(k1 - 1, k0 - 1, -1):
            if self._voiced(k):
                v_last = k
                break
        if v_last < 0:
            return ne
        # Walk backward from v_last looking for a silence valley >= min_pause
        onset_k = v_last
        silent = 0
        for k in range(v_last, k0 - 1, -1):
            if self._voiced(k):
                if silent >= min_pause:
                    return onset_k * self.frame_s
                onset_k = k
                silent = 0
            else:
                silent += 1
        if silent >= min_pause:
            return onset_k * self.frame_s
        return ns

    def true_offset(self, ps: float, pe: float, min_pause_s: float) -> float:
        """If a single prev-word token [ps, pe] contains a turn pause (>= min_pause_s)
        after its initial voiced word, return the offset before that pause."""
        k0 = max(0, int(ps / self.frame_s))
        k1 = min(len(self.rms), int(np.ceil(pe / self.frame_s)))
        min_pause = max(2, int(round(min_pause_s / self.frame_s)))
        v_first = -1
        for k in range(k0, k1):
            if self._voiced(k):
                v_first = k
                break
        if v_first < 0:
            return ps
        offset_k = v_first + 1
        silent = 0
        for k in range(v_first, k1):
            if self._voiced(k):
                if silent >= min_pause:
                    return offset_k * self.frame_s
                offset_k = k + 1
                silent = 0
            else:
                silent += 1
        if silent >= min_pause:
            return offset_k * self.frame_s
        return pe

    def extend(self, s: float, e: float, max_extra: float) -> tuple[float, float]:
        steps = int(round(max_extra / self.frame_s))
        k = int(s / self.frame_s)
        moved = 0
        while moved < steps and self._voiced(k - 1):
            k -= 1
            moved += 1
        s2 = min(s, k * self.frame_s)
        k = int(np.ceil(e / self.frame_s))
        moved = 0
        while moved < steps and self._voiced(k):
            k += 1
            moved += 1
        e2 = max(e, k * self.frame_s)
        return s2, e2

    def extend_with_gaps(self, s: float, e: float, *, max_extra: float,
                         bridge_s: float, floor_s: float,
                         ceil_s: float) -> tuple[float, float]:
        """Extend across voiced speech while bridging brief intra-number pauses,
        bounded strictly by [floor_s, ceil_s] so adjacent words are untouched."""
        max_steps = int(round(max_extra / self.frame_s))
        max_bridge = max(1, int(round(bridge_s / self.frame_s)))
        floor_k = max(0, int(np.ceil(floor_s / self.frame_s)))
        ceil_k = min(len(self.rms), int(ceil_s / self.frame_s))

        k = int(s / self.frame_s)
        best_start_k = k
        silent_run = 0
        moved = 0
        while moved < max_steps and k - 1 >= floor_k:
            k -= 1
            moved += 1
            if self._voiced(k):
                best_start_k = k
                silent_run = 0
            else:
                silent_run += 1
                if silent_run > max_bridge:
                    break
        s2 = min(s, best_start_k * self.frame_s)

        k = int(np.ceil(e / self.frame_s))
        best_end_k = k
        silent_run = 0
        moved = 0
        while moved < max_steps and k < ceil_k:
            if self._voiced(k):
                best_end_k = k + 1
                silent_run = 0
            else:
                silent_run += 1
                if silent_run > max_bridge:
                    break
            k += 1
            moved += 1
        e2 = max(e, best_end_k * self.frame_s)
        return s2, e2

    def finish_voiced_burst(self, s: float, e: float, floor_s: float,
                            ceil_s: float, max_burst_s: float = 0.75) -> tuple[float, float]:
        """If s or e is still inside an active voiced syllable, walk to the end
        of that syllable (stopping at >= 20 ms of unvoiced frames)."""
        max_steps = int(round(max_burst_s / self.frame_s))
        floor_k = max(0, int(np.ceil(floor_s / self.frame_s)))
        ceil_k = min(len(self.rms), int(ceil_s / self.frame_s))

        # Left edge
        k = int(s / self.frame_s)
        best_s_k = k
        if self._voiced(k) or self._voiced(k - 1):
            unvoiced = 0
            moved = 0
            while moved < max_steps and k - 1 >= floor_k:
                k -= 1
                moved += 1
                if self._voiced(k):
                    best_s_k = k
                    unvoiced = 0
                else:
                    unvoiced += 1
                    if unvoiced >= 2:
                        break
        s2 = min(s, best_s_k * self.frame_s)

        # Right edge
        k = int(np.ceil(e / self.frame_s))
        best_e_k = k
        if self._voiced(k - 1) or self._voiced(k):
            unvoiced = 0
            moved = 0
            while moved < max_steps and k < ceil_k:
                if self._voiced(k):
                    best_e_k = k + 1
                    unvoiced = 0
                else:
                    unvoiced += 1
                    if unvoiced >= 2:
                        break
                k += 1
                moved += 1
        e2 = max(e, best_e_k * self.frame_s)
        return s2, e2
