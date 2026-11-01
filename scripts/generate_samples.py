"""Generate two-speaker test calls WITH exact ground truth.

Each speaker turn is a list of segments ``(text, label)``. Every segment is
synthesised separately (Microsoft neural voices via edge-tts), trimmed and
concatenated, so the exact start/end of every PII segment is known. For each
call two files are written to ``samples/``:

    <stem>.wav             16 kHz mono 16-bit PCM recording
    <stem>_expected.json   ground-truth PII spans (same schema as the
                           program's *_redactions.json output)

Usage:  python scripts/generate_samples.py          (needs internet)

NOTE: these synthetic calls are a reproducible regression set. The brief
also asks for recordings of two real people - record those yourself and put
them next to these files (see README, "Test recordings").
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from math import gcd
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import edge_tts  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

from pii_anonymizer.audio import Audio, load_audio, save_wav  # noqa: E402

SAMPLES_DIR = ROOT_DIR / "samples"
SR = 16000
SEGMENT_GAP = 0.06   # s between segments of one turn
TRIM_THR = 0.004     # amplitude treated as silence when trimming TTS output

VOICES = {
    "hi_agent": "hi-IN-MadhurNeural",
    "hi_customer": "hi-IN-SwaraNeural",
    "ta_agent": "ta-IN-PallaviNeural",
    "ta_customer": "ta-IN-ValluvarNeural",
}

# (voice, [(text, label, expected_text)], pause_after_turn)
# label: None | "name" | "mobile" | "pincode"
P = None

SAMPLE_1_HINDI = ("hi", [
    ("hi_agent", [("नमस्ते, ग्राहक सेवा में आपका स्वागत है। अपना नाम बताइए।", P, "")], 0.6),
    ("hi_customer", [("मेरा नाम", P, ""), ("प्रिया शर्मा", "name", "Priya Sharma"), ("है।", P, "")], 0.6),
    ("hi_agent", [("धन्यवाद", P, ""), ("प्रिया", "name", "Priya"),
                  ("जी। आपका मोबाइल नंबर बताइए।", P, "")], 0.6),
    ("hi_customer", [("नौ आठ चार पाँच, शून्य एक दो, तीन चार पाँच।", "mobile", "9845012345")], 0.6),
    ("hi_agent", [("ठीक है। और आपका पिन कोड?", P, "")], 0.6),
    ("hi_customer", [("पाँच छह शून्य शून्य तीन चार।", "pincode", "560034")], 0.6),
    ("hi_agent", [("धन्यवाद। आपका ऑर्डर कल डिलीवर हो जाएगा।", P, "")], 0.4),
])

SAMPLE_2_TAMIL = ("ta", [
    ("ta_agent", [("வணக்கம். உங்க பெயர் என்ன?", P, "")], 0.6),
    ("ta_customer", [("என் பெயர்", P, ""), ("கார்த்திக் ராஜன்.", "name", "Karthik Rajan")], 0.6),
    ("ta_agent", [("நன்றி", P, ""), ("கார்த்திக்.", "name", "Karthik"),
                  ("உங்க mobile number சொல்லுங்க.", P, "")], 0.6),
    ("ta_customer", [("ஒன்பது ஒன்பது நாலு ஒன்னு, double ஆறு, ரெண்டு பூஜ்யம் மூணு ஏழு.",
                      "mobile", "9941662037")], 0.6),
    ("ta_agent", [("சரி. உங்க pin code?", P, "")], 0.6),
    ("ta_customer", [("six zero zero zero four two.", "pincode", "600042")], 0.6),
    ("ta_agent", [("நன்றி. நாளைக்கு delivery ஆகும்.", P, "")], 0.4),
])

SAMPLE_3_HINDI_CODEMIX = ("hi", [
    ("hi_agent", [("नमस्ते, आपका नाम बताइए।", P, "")], 0.6),
    ("hi_customer", [("मैं", P, ""), ("राहुल वर्मा", "name", "Rahul Verma"),
                     ("बोल रहा हूँ।", P, "")], 0.6),
    ("hi_agent", [("धन्यवाद", P, ""), ("राहुल", "name", "Rahul"),
                  ("जी। आपका मोबाइल नंबर बताइए।", P, "")], 0.6),
    ("hi_customer", [("मेरा मोबाइल नंबर है", P, ""),
                     ("nine eight one zero, double two, तीन चार पाँच छह।", "mobile", "9810223456")], 0.6),
    ("hi_agent", [("मैं दोहराता हूँ,", P, ""),
                  ("नौ आठ एक शून्य, दो दो, तीन चार पाँच छह,", "mobile", "9810223456"),
                  ("सही है?", P, "")], 0.6),
    ("hi_customer", [("हाँ जी, सही है।", P, "")], 0.6),
    ("hi_agent", [("और पिन कोड क्या है?", P, "")], 0.6),
    ("hi_customer", [("एक एक शून्य शून्य शून्य एक।", "pincode", "110001")], 0.6),
    ("hi_agent", [("शुक्रिया", P, ""), ("राहुल", "name", "Rahul"),
                  ("जी, आपका पता अपडेट हो गया है।", P, "")], 0.4),
])

SAMPLE_4_TAMIL_COLLOQUIAL = ("ta", [
    ("ta_agent", [("வணக்கம், உங்க பெயர் சொல்லுங்க.", P, "")], 0.6),
    ("ta_customer", [("என்னோட பெயர்", P, ""), ("திவ்யா கிருஷ்ணன்.", "name", "Divya Krishnan")], 0.6),
    ("ta_agent", [("நன்றி", P, ""), ("திவ்யா", "name", "Divya"),
                  ("மேடம். உங்க மொபைல் நம்பர் என்ன?", P, "")], 0.6),
    ("ta_customer", [("எட்டு ஏழு ஆறு அஞ்சு, நாலு மூணு, triple ரெண்டு, ஒன்னு.",
                      "mobile", "8765432221")], 0.6),
    ("ta_agent", [("சரி, உங்க பின்கோடு சொல்லுங்க.", P, "")], 0.6),
    ("ta_customer", [("ஆறு நாலு ஒன்னு பூஜ்யம் பூஜ்யம் ஒன்னு.", "pincode", "641001")], 0.6),
    ("ta_agent", [("நன்றி மேடம், நாளைக்கு டெலிவரி ஆகும்.", P, "")], 0.4),
])

CALLS = {
    "sample_1_hindi": SAMPLE_1_HINDI,
    "sample_2_tamil": SAMPLE_2_TAMIL,
    "sample_3_hindi_codemix": SAMPLE_3_HINDI_CODEMIX,
    "sample_4_tamil_colloquial": SAMPLE_4_TAMIL_COLLOQUIAL,
}


async def _tts(voice: str, text: str, tmp: Path) -> np.ndarray:
    mp3 = tmp / "seg.mp3"
    await edge_tts.Communicate(text=text, voice=VOICES[voice], rate="-5%").save(str(mp3))
    aud = load_audio(mp3)
    x = aud.mono_float32().astype(np.float64)
    if aud.samplerate != SR:
        g = gcd(aud.samplerate, SR)
        x = resample_poly(x, SR // g, aud.samplerate // g)
    idx = np.where(np.abs(x) > TRIM_THR)[0]
    if idx.size:
        x = x[int(idx[0]): int(idx[-1]) + 1]
    return x.astype(np.float32)


async def build_call(stem: str, lang: str, turns) -> None:
    pieces: list[np.ndarray] = [np.zeros(int(0.4 * SR), np.float32)]
    t = 0.4
    truth = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for voice, segments, pause in turns:
            for si, (text, label, expected) in enumerate(segments):
                seg = await _tts(voice, text, tmp)
                if label:
                    truth.append({"start_sec": round(t, 3), "end_sec": round(t + len(seg) / SR, 3),
                                  "type": label, "text": expected})
                pieces.append(seg)
                t += len(seg) / SR
                if si < len(segments) - 1:
                    pieces.append(np.zeros(int(SEGMENT_GAP * SR), np.float32))
                    t += int(SEGMENT_GAP * SR) / SR
            pieces.append(np.zeros(int(pause * SR), np.float32))
            t += int(pause * SR) / SR
    full = np.concatenate(pieces)
    full *= 0.707 / max(1e-9, float(np.max(np.abs(full))))
    pcm = np.clip(np.round(full * 32767.0), -32768, 32767).astype(np.int16)[:, None]
    wav = SAMPLES_DIR / f"{stem}.wav"
    save_wav(Audio(pcm, SR, "PCM_16", "WAV"), wav)
    (SAMPLES_DIR / f"{stem}_expected.json").write_text(json.dumps(
        {"input": wav.name, "language": lang, "redactions": truth},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {wav.name} ({len(full) / SR:.2f}s, {len(truth)} PII spans)")


async def main() -> None:
    SAMPLES_DIR.mkdir(exist_ok=True)
    for stem, (lang, turns) in CALLS.items():
        await build_call(stem, lang, turns)


if __name__ == "__main__":
    asyncio.run(main())
