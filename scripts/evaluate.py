"""Score the program's output against ground truth.

For every ``<stem>_expected.json`` in a folder, compares it with
``<stem>_redactions.json`` (from the output folder) and ``<stem>_redacted.wav``:

* LEAK      - voiced audio inside a true PII span that is NOT under a beep (ms).
              Must be ~0: a leak means part of a name/digit is audible.
* EXTRA     - beep time outside the true PII spans (+/- 0.25 s tolerance), i.e.
              non-PII speech that was destroyed (minimal-damage rule).
* BEEPS     - number and order of beeps/types vs. the expected list.
* TEXT      - detected digit strings / names vs. expected.
* LENGTH    - output frames / rate / channels identical to the input.

Usage:
    python scripts/evaluate.py                      # samples/ vs samples/
    python scripts/evaluate.py samples outputs      # truth dir, output dir
Exit code 1 if any file leaks or has a length mismatch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pii_anonymizer.audio import frame_rms  # noqa: E402

TOL = 0.25          # s of margin around a PII span that is not counted as damage
LEAK_OK_MS = 30.0   # tolerated uncovered voiced audio per span (TTS tail noise)


def _intervals_overlap(a0, a1, spans) -> float:
    return sum(max(0.0, min(a1, e) - max(a0, s)) for s, e in spans)


def score(truth_path: Path, out_dir: Path) -> dict:
    stem = truth_path.name[: -len("_expected.json")]
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    pred_path = out_dir / f"{stem}_redactions.json"
    wav_in = truth_path.parent / f"{stem}.wav"
    wav_out = out_dir / f"{stem}_redacted.wav"
    if not pred_path.exists() or not wav_out.exists():
        return {"file": stem, "error": "no output (run the anonymizer first)"}
    pred = json.loads(pred_path.read_text(encoding="utf-8"))
    beeps = [(r["start_sec"], r["end_sec"]) for r in pred["redactions"]]

    x, sr = sf.read(str(wav_in), always_2d=True)
    mono = x.mean(axis=1)
    rms = frame_rms(mono, sr, 0.01)
    thr = max(np.percentile(rms, 98) * 0.03, 1e-4)

    leaks = []
    for t in truth["redactions"]:
        leaked = 0.0
        for k in range(int(t["start_sec"] / 0.01), int(np.ceil(t["end_sec"] / 0.01))):
            if k < len(rms) and rms[k] > thr:
                f0, f1 = k * 0.01, (k + 1) * 0.01
                if _intervals_overlap(f0, f1, beeps) < 0.005:
                    leaked += 0.01
        leaks.append(leaked * 1000)

    allowed = [(t["start_sec"] - TOL, t["end_sec"] + TOL) for t in truth["redactions"]]
    extra = 0.0
    for s, e in beeps:
        # beep time not inside any allowed window
        grid = np.arange(s, e, 0.01)
        extra += 0.01 * sum(1 for g in grid if not any(a <= g <= b for a, b in allowed))

    info_in, info_out = sf.info(str(wav_in)), sf.info(str(wav_out))
    length_ok = (info_in.frames, info_in.samplerate, info_in.channels) == \
                (info_out.frames, info_out.samplerate, info_out.channels)

    exp_types = [t["type"] for t in truth["redactions"]]
    got_types = [r["type"] for r in pred["redactions"]]
    exp_digits = [t["text"] for t in truth["redactions"] if t["type"] != "name"]
    got_digits = [r["text"] for r in pred["redactions"] if r["type"] != "name"]
    return {
        "file": stem,
        "spans": len(exp_types),
        "beeps": len(got_types),
        "order_ok": exp_types == got_types,
        "digits_ok": exp_digits == got_digits,
        "got_digits": got_digits,
        "max_leak_ms": max(leaks) if leaks else 0.0,
        "leaks_ms": leaks,
        "extra_s": extra,
        "length_ok": length_ok,
    }


def main(argv: list[str]) -> int:
    truth_dir = Path(argv[0]) if argv else ROOT_DIR / "samples"
    out_dir = Path(argv[1]) if len(argv) > 1 else truth_dir
    files = sorted(truth_dir.glob("*_expected.json"))
    if not files:
        print(f"No *_expected.json in {truth_dir}")
        return 2
    bad = False
    print(f"{'file':<28}{'spans':>6}{'beeps':>6}  {'order':<6}{'digits':<7}"
          f"{'max leak':>9}{'extra':>8}  length")
    for f in files:
        r = score(f, out_dir)
        if "error" in r:
            print(f"{r['file']:<28}{r['error']}")
            bad = True
            continue
        leak_ok = r["max_leak_ms"] <= LEAK_OK_MS
        bad |= (not leak_ok) or (not r["length_ok"])
        print(f"{r['file']:<28}{r['spans']:>6}{r['beeps']:>6}  "
              f"{'OK' if r['order_ok'] else 'DIFF':<6}{'OK' if r['digits_ok'] else 'DIFF':<7}"
              f"{r['max_leak_ms']:>7.0f}ms{r['extra_s']:>7.2f}s  "
              f"{'OK' if r['length_ok'] else 'MISMATCH'}"
              + ("" if leak_ok else "   <-- LEAK")
              + ("" if r["digits_ok"] else f"   (digits read: {r['got_digits']})"))
    print("\nLEAK = voiced PII audio not covered by a beep (must be ~0). "
          "EXTRA = non-PII audio beeped beyond a 0.25 s margin.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
