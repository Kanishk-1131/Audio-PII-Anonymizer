"""Unit and integration tests for pii_anonymizer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pii_anonymizer.audio import Audio, apply_beeps, load_audio, save_wav
from pii_anonymizer.languages import available_languages, load_language
from pii_anonymizer.pipeline import Anonymizer, PipelineConfig, detect
from pii_anonymizer.textnorm import normalize, phonetic_key
from pii_anonymizer.types import Word


def _words_from_script(turns: list[tuple[str, float]]) -> list[Word]:
    """Build synthetic timed Word objects from (sentence, pause_after) pairs."""
    words: list[Word] = []
    t = 0.5
    for sent, pause in turns:
        for raw in sent.split():
            words.append(Word(text=raw, start=t, end=t + 0.32))
            t += 0.36
        t += pause
    return words


def test_language_packs_load_without_collision():
    langs = available_languages()
    assert {"en", "hi", "ta", "kn"}.issubset(set(langs))
    for code in langs:
        pack = load_language(code)
        assert pack.code == code
        assert len(pack.number_words) >= 20


def test_cross_script_phonetic_key():
    assert phonetic_key("Priya") == phonetic_key("प्रिया")
    assert phonetic_key("Sharma") == phonetic_key("शर्मा")
    assert phonetic_key("Karthik") == phonetic_key("கார்த்திக்") == phonetic_key("कार्तिक")
    assert phonetic_key("Rajan") == phonetic_key("ராஜன்") == phonetic_key("राजन")


def test_sample_1_hindi_brief_exact():
    """Sample 1 from the brief: must produce 4 detections in exact order:
    name (Priya Sharma), name (Priya), mobile (9845012345), pincode (560034).
    """
    pack = load_language("hi")
    turns = [
        ("नमस्ते, ग्राहक सेवा में आपका स्वागत है। अपना नाम बताइए।", 0.8),
        ("मेरा नाम प्रिया शर्मा है।", 0.8),
        ("धन्यवाद प्रिया जी। आपका मोबाइल नंबर बताइए।", 0.8),
        ("नौ आठ चार पाँच, शून्य एक दो, तीन चार पाँच।", 0.8),
        ("ठीक है। और आपका पिन कोड?", 0.8),
        ("पाँच छह शून्य शून्य तीन चार।", 0.8),
        ("धन्यवाद। आपका ऑर्डर कल डिलीवर हो जाएगा।", 0.5),
    ]
    words = _words_from_script(turns)
    tokens, dets = detect(words, pack)

    assert [(d.type, d.text) for d in dets] == [
        ("name", "प्रिया शर्मा"),
        ("name", "प्रिया"),
        ("mobile", "9845012345"),
        ("pincode", "560034"),
    ]
    # Ensure context words ("मेरा", "नाम", "जी", "मोबाइल", "नंबर", "पिन", "कोड") are NOT inside any detection
    redacted_indices = {i for d in dets for i in range(d.first, d.last + 1)}
    kept_norms = {tokens[i].norm for i in range(len(tokens)) if i not in redacted_indices}
    for must_keep in ("मेरा", "नाम", "मोबाइल", "नंबर", "पिन", "कोड", "धन्यवाद"):
        assert normalize(must_keep) in kept_norms


def test_sample_2_tamil_brief_exact():
    """Sample 2 from the brief: must produce 4 detections in exact order:
    name (Karthik Rajan), name (Karthik), mobile (9941662037 incl. 'double aaru'
    and colloquial 'onnu', 'rendu', 'moonu'), pincode (600042 spoken in English).
    """
    pack = load_language("ta")
    turns = [
        ("வணக்கம். உங்க பெயர் என்ன?", 0.8),
        ("என் பெயர் கார்த்திக் ராஜன்.", 0.8),
        ("நன்றி கார்த்திக். உங்க mobile number சொல்லுங்க.", 0.8),
        ("ஒன்பது ஒன்பது நாலு ஒன்னு, double ஆறு, ரெண்டு பூஜ்யம் மூணு ஏழு.", 0.8),
        ("சரி. உங்க pin code?", 0.8),
        ("six zero zero zero four two.", 0.8),
        ("நன்றி. நாளைக்கு delivery ஆகும்.", 0.5),
    ]
    words = _words_from_script(turns)
    tokens, dets = detect(words, pack)

    assert [(d.type, d.text) for d in dets] == [
        ("name", "கார்த்திக் ராஜன்"),
        ("name", "கார்த்திக்"),
        ("mobile", "9941662037"),
        ("pincode", "600042"),
    ]
    redacted_indices = {i for d in dets for i in range(d.first, d.last + 1)}
    kept_norms = {tokens[i].norm for i in range(len(tokens)) if i not in redacted_indices}
    for must_keep in ("என்", "பெயர்", "நன்றி", "mobile", "number", "pin", "code", "delivery"):
        assert normalize(must_keep) in kept_norms


def test_mixed_digits_double_triple_compounds_and_readback():
    """Test code-mixed Hindi/English digits, triple, Hindi compound numbers
    (चौरानवे = 94), and agent partial read-back.
    """
    pack = load_language("hi")
    turns = [
        ("मेरा मोबाइल नंबर है nine eight चौरानवे triple five zero एक दो.", 1.2),
        ("अच्छा, zero एक दो?", 1.0),
    ]
    words = _words_from_script(turns)
    _, dets = detect(words, pack)
    assert [(d.type, d.text) for d in dets] == [
        ("mobile", "9894555012"),
        ("mobile", "012"),
    ]


def test_kannada_third_language_pack():
    """Verify the 3rd language pack (Kannada) works end-to-end on text."""
    pack = load_language("kn")
    turns = [
        ("ನಮಸ್ಕಾರ, ನನ್ನ ಹೆಸರು ರಮೇಶ್ ಶರ್ಮಾ.", 0.8),
        ("ಧನ್ಯವಾದ ರಮೇಶ್. ನಿಮ್ಮ ಮೊಬೈಲ್ ನಂಬರ್ ಹೇಳಿ.", 0.8),
        ("ಒಂಬತ್ತು ಎಂಟು ನಾಲ್ಕು ಐದು ಸೊನ್ನೆ ಒಂದು ಎರಡು ಮೂರು ನಾಲ್ಕು ಐದು.", 0.8),
        ("ಪಿನ್ ಕೋಡ್ ಐದು ಆರು ಸೊನ್ನೆ ಸೊನ್ನೆ ಮೂರು ನಾಲ್ಕು.", 0.6),
    ]
    words = _words_from_script(turns)
    _, dets = detect(words, pack)
    assert [(d.type, d.text) for d in dets] == [
        ("name", "ರಮೇಶ್ ಶರ್ಮಾ"),
        ("name", "ರಮೇಶ್"),
        ("mobile", "9845012345"),
        ("pincode", "560034"),
    ]


def test_audio_invariants_and_determinism(tmp_path: Path):
    """Verify:
    1. Output WAV has exact same frame count, sample rate, and channels.
    2. Samples outside beep spans are bit-identical to input PCM_16 WAV.
    3. Running twice yields byte-for-byte identical WAV and JSON.
    """
    sr = 16000
    dur = 6.0
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
    # Synthetic stereo speech-like signal
    rng = np.random.default_rng(42)
    sig = 0.15 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(len(t))
    stereo = np.column_stack([sig, sig * 0.9])
    pcm16 = np.clip(np.round(stereo * 32767.0), -32768, 32767).astype(np.int16)

    in_wav = tmp_path / "call_test.wav"
    sf.write(str(in_wav), pcm16, sr, subtype="PCM_16")

    words = [
        Word("मेरा", 0.4, 0.7),
        Word("नाम", 0.75, 1.0),
        Word("प्रिया", 1.2, 1.7),
        Word("है।", 2.0, 2.3),
        Word("पिन", 2.8, 3.1),
        Word("560034", 3.4, 4.4),
        Word("धन्यवाद", 4.8, 5.4),
    ]

    anon = Anonymizer(asr=None, config=PipelineConfig())
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    r1 = anon.process_file(in_wav, "hi", out_dir=out1, words=words)
    r2 = anon.process_file(in_wav, "hi", out_dir=out2, words=words)

    # Repeatable: byte-for-byte identical outputs
    assert r1.redacted_path.read_bytes() == r2.redacted_path.read_bytes()
    assert r1.json_path.read_text(encoding="utf-8") == r2.json_path.read_text(encoding="utf-8")

    # Frame count, sample rate, channels
    out_data, out_sr = sf.read(str(r1.redacted_path), dtype="int16", always_2d=True)
    assert out_sr == sr
    assert out_data.shape == pcm16.shape

    # Outside beep spans, audio is untouched (bit-identical)
    mask = np.ones(len(pcm16), dtype=bool)
    for red in r1.redactions:
        i0 = max(0, int(round(red.start * sr)) - 2)
        i1 = min(len(pcm16), int(round(red.end * sr)) + 2)
        mask[i0:i1] = False
    assert np.array_equal(out_data[mask], pcm16[mask])

    # Inside beep spans, original signal was replaced by a high-energy 1 kHz tone
    for red in r1.redactions:
        i0 = int(round((red.start + 0.02) * sr))
        i1 = int(round((red.end - 0.02) * sr))
        assert not np.array_equal(out_data[i0:i1], pcm16[i0:i1])

    # JSON schema check
    payload = json.loads(r1.json_path.read_text(encoding="utf-8"))
    assert payload["input"] == "call_test.wav"
    assert payload["language"] == "hi"
    assert [(x["type"], x["text"]) for x in payload["redactions"]] == [
        ("name", "प्रिया"),
        ("pincode", "560034"),
    ]


class _FakeASR:
    """Stands in for Whisper: returns fixed words and counts calls."""
    name = "fake"

    def __init__(self, words):
        self.words = words
        self.calls = 0

    def settings(self):
        return {"backend": "fake"}

    def transcribe(self, audio_16k, language):
        self.calls += 1
        return list(self.words)


def test_transcript_cache_skips_asr_and_is_repeatable(tmp_path: Path):
    from pii_anonymizer.cache import TranscriptCache

    sr = 16000
    sig = (0.1 * np.sin(2 * np.pi * 200 * np.arange(sr * 4) / sr)).astype(np.float32)
    wav = tmp_path / "c.wav"
    sf.write(str(wav), sig, sr, subtype="PCM_16")
    words = [Word("मेरा", 0.2, 0.5), Word("नाम", 0.5, 0.8), Word("प्रिया", 0.9, 1.3),
             Word("है।", 1.3, 1.6), Word("पिन", 2.0, 2.2), Word("560034", 2.4, 3.4)]
    asr = _FakeASR(words)
    anon = Anonymizer(asr, cache=TranscriptCache(tmp_path / "cache"))
    r1 = anon.process_file(wav, "hi", out_dir=tmp_path / "a")
    assert asr.calls == 1 and not anon.last_asr_cached
    r2 = anon.process_file(wav, "hi", out_dir=tmp_path / "b")
    assert asr.calls == 1 and anon.last_asr_cached          # served from cache
    assert r1.redacted_path.read_bytes() == r2.redacted_path.read_bytes()
    assert r1.json_path.read_text(encoding="utf-8") == r2.json_path.read_text(encoding="utf-8")
    # a different language is a different cache entry
    anon.process_file(wav, "ta", out_dir=tmp_path / "c")
    assert asr.calls == 2


def test_mp3_input_keeps_length(tmp_path: Path):
    """MP3 in -> WAV out with the same number of frames as the decoded MP3."""
    if "MP3" not in sf.available_formats():
        pytest.skip("libsndfile without MP3 support")
    sr = 16000
    sig = (0.2 * np.sin(2 * np.pi * 300 * np.arange(sr * 3) / sr)).astype(np.float32)
    mp3 = tmp_path / "call.mp3"
    sf.write(str(mp3), sig, sr, format="MP3")
    words = [Word("पिन", 0.3, 0.6), Word("560034", 0.8, 2.0)]
    res = Anonymizer(None).process_file(mp3, "hi", out_dir=tmp_path, words=words)
    decoded = load_audio(mp3)
    info = sf.info(str(res.redacted_path))
    assert info.frames == decoded.frames and info.samplerate == sr
    assert [r.type for r in res.redactions] == ["pincode"]


def test_primary_output_beeped_copy_and_local_loudness(tmp_path: Path):
    """Verify top-level anonymize() returns the beeped copy path as primary output,
    never modifies the original recording, replaces PII with a pure 1 kHz beep
    matched to each speaker's local loudness, and guards against overwriting input.
    """
    from pii_anonymizer import anonymize

    sr = 16000
    t = np.arange(sr * 6, dtype=np.float64) / sr
    # First half: quieter speaker (amp 0.12); second half: louder speaker (amp 0.30)
    env = np.where(t < 3.0, 0.12, 0.30)
    speech_like = env * np.sin(2.0 * np.pi * 240.0 * t)
    pcm16 = np.clip(np.round(speech_like * 32767.0), -32768, 32767).astype(np.int16)

    in_wav = tmp_path / "recording.wav"
    sf.write(str(in_wav), pcm16, sr, subtype="PCM_16")
    orig_bytes = in_wav.read_bytes()

    words = [
        Word("मेरा", 0.3, 0.6),
        Word("नाम", 0.65, 0.9),
        Word("प्रिया", 1.1, 1.8),
        Word("पिन", 3.3, 3.6),
        Word("560034", 3.9, 4.9),
    ]

    primary_wav = anonymize(in_wav, "hi", words=words)
    assert primary_wav == tmp_path / "recording_redacted.wav"
    assert primary_wav.exists()
    # Original input file is 100% untouched
    assert in_wav.read_bytes() == orig_bytes

    # Spectral check: inside each beep span, dominant FFT bin is 1000 Hz (240 Hz voice wiped)
    out_audio = load_audio(primary_wav).mono_float32()
    s1 = out_audio[int(1.2 * sr): int(1.7 * sr)]
    s2 = out_audio[int(4.0 * sr): int(4.8 * sr)]
    for seg in (s1, s2):
        freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)
        mag = np.abs(np.fft.rfft(seg))
        dom_hz = freqs[int(np.argmax(mag))]
        assert abs(dom_hz - 1000.0) <= 5.0

    # Local speaker loudness matching: louder speaker's beep has higher RMS than quieter speaker's beep
    rms1 = float(np.sqrt(np.mean(s1 ** 2)))
    rms2 = float(np.sqrt(np.mean(s2 ** 2)))
    assert rms2 > rms1 * 1.3

    # Guard against accidentally overwriting the original input file
    with pytest.raises(ValueError, match="preserve the original recording"):
        anonymize(in_wav, "hi", output_path=in_wav, words=words)


def test_cli_direct_wav_output_and_no_json(tmp_path: Path):
    """CLI supports -o <custom_copy.wav> directly for the primary beeped audio
    output, as well as --no-json when only the beeped recording copy is wanted.
    """
    from pii_anonymizer.cli import main

    sr = 16000
    sig = (0.15 * np.sin(2 * np.pi * 220 * np.arange(sr * 3) / sr)).astype(np.float32)
    in_wav = tmp_path / "call_in.wav"
    sf.write(str(in_wav), sig, sr, subtype="PCM_16")

    tr_json = tmp_path / "words.json"
    tr_json.write_text(
        json.dumps({"words": [{"text": "पिन", "start": 0.3, "end": 0.6},
                              {"text": "560034", "start": 0.8, "end": 2.0}]}),
        encoding="utf-8",
    )

    custom_wav = tmp_path / "beeped_only" / "my_primary_copy.wav"
    rc = main([
        str(in_wav),
        "--lang", "hi",
        "-o", str(custom_wav),
        "--no-json",
        "--transcript-in", str(tr_json),
    ])
    assert rc == 0
    assert custom_wav.exists()
    assert not (custom_wav.parent / "my_primary_copy_redactions.json").exists()
    assert sf.info(str(custom_wav)).frames == sf.info(str(in_wav)).frames


def test_web_player_byte_range_streaming():
    """Verify the built-in browser player serves HTML, /api/info, and HTTP 206
    byte-range WAV audio streaming for HTML5 <audio> playback and seeking.
    """
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from pii_anonymizer.web import _Handler

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(f"{base}/") as r:
            assert r.status == 200
            assert b"Primary Output" in r.read()

        with urllib.request.urlopen(f"{base}/api/info") as r:
            info = json.loads(r.read().decode("utf-8"))
            assert any(l["code"] == "hi" for l in info["languages"])

        req = urllib.request.Request(
            f"{base}/media/samples/sample_1_hindi_redacted.wav",
            headers={"Range": "bytes=0-43"},
        )
        with urllib.request.urlopen(req) as r:
            assert r.status == 206
            header_bytes = r.read()
            assert len(header_bytes) == 44
            assert header_bytes[:4] == b"RIFF" and header_bytes[8:12] == b"WAVE"
    finally:
        srv.shutdown()
        srv.server_close()


def test_dialogue_parser_for_custom_speech():
    from pii_anonymizer.tts import parse_dialogue

    turns = parse_dialogue(
        "Agent: नमस्ते, अपना नाम बताइए।\n"
        "Customer: मेरा नाम अमित पटेल है।\n"
        "धन्यवाद अमित जी।"
    )
    assert turns == [
        ("agent", "नमस्ते, अपना नाम बताइए।"),
        ("customer", "मेरा नाम अमित पटेल है।"),
        ("agent", "धन्यवाद अमित जी।"),
    ]
    assert parse_dialogue("मेरा नाम प्रिया है।") == [("customer", "मेरा नाम प्रिया है।")]
