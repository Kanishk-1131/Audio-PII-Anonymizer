"""Synthesize new single- or two-speaker Indian-language call recordings from text.

Uses Microsoft neural voices via ``edge-tts`` (requires internet when generating
new TTS audio; anonymizing existing WAV/MP3 files remains 100% offline).
"""

from __future__ import annotations

import asyncio
import re
import tempfile
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .audio import Audio, load_audio, save_wav

SR = 16000
TRIM_THR = 0.004

VOICE_PAIRS: dict[str, tuple[str, str]] = {
    # lang: (agent_voice, customer_voice)
    "hi": ("hi-IN-MadhurNeural", "hi-IN-SwaraNeural"),
    "ta": ("ta-IN-PallaviNeural", "ta-IN-ValluvarNeural"),
    "kn": ("kn-IN-GaganNeural", "kn-IN-SapnaNeural"),
}

_AGENT_RE = re.compile(r"^\s*(?:agent|a|एजेंट|ஏஜென்ட்)\s*[:\-]\s*(.+)$", re.IGNORECASE)
_CUST_RE = re.compile(r"^\s*(?:customer|caller|user|c|ग्राहक|வாடிக்கையாளர்)\s*[:\-]\s*(.+)$", re.IGNORECASE)


def parse_dialogue(script: str) -> list[tuple[str, str]]:
    """Parse multi-line dialogue into ``[(role, text), ...]`` where role is
    ``'agent'`` or ``'customer'``.
    """
    raw_lines = [ln.strip() for ln in script.splitlines() if ln.strip()]
    if not raw_lines:
        raise ValueError("Dialogue text is empty")

    turns: list[tuple[str, str]] = []
    for idx, ln in enumerate(raw_lines):
        m_a = _AGENT_RE.match(ln)
        if m_a:
            turns.append(("agent", m_a.group(1).strip()))
            continue
        m_c = _CUST_RE.match(ln)
        if m_c:
            turns.append(("customer", m_c.group(1).strip()))
            continue
        role = "customer" if len(raw_lines) == 1 else ("agent" if idx % 2 == 0 else "customer")
        turns.append((role, ln))
    return [(r, t) for r, t in turns if t]


async def _synth_turn(voice: str, text: str, tmp_dir: Path, idx: int) -> np.ndarray:
    import edge_tts

    mp3 = tmp_dir / f"turn_{idx}.mp3"
    await edge_tts.Communicate(text=text, voice=voice, rate="-5%").save(str(mp3))
    aud = load_audio(mp3)
    x = aud.mono_float32().astype(np.float64)
    if aud.samplerate != SR:
        g = gcd(aud.samplerate, SR)
        x = resample_poly(x, SR // g, aud.samplerate // g)
    nz = np.where(np.abs(x) > TRIM_THR)[0]
    if nz.size:
        x = x[int(nz[0]): int(nz[-1]) + 1]
    return x.astype(np.float32)


async def synthesize_dialogue_async(
    script: str,
    language: str,
    out_wav: str | Path,
    *,
    pause_sec: float = 0.55,
) -> Path:
    """Synthesize ``script`` into a 16 kHz 16-bit PCM WAV recording at ``out_wav``."""
    lang = language.lower().strip()
    if lang not in VOICE_PAIRS:
        raise ValueError(f"Unsupported TTS language '{language}'. Supported: {', '.join(VOICE_PAIRS)}")

    agent_voice, cust_voice = VOICE_PAIRS[lang]
    turns = parse_dialogue(script)
    pieces: list[np.ndarray] = [np.zeros(int(0.35 * SR), dtype=np.float32)]
    pause = np.zeros(int(pause_sec * SR), dtype=np.float32)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for idx, (role, text) in enumerate(turns):
            voice = agent_voice if role == "agent" else cust_voice
            seg = await _synth_turn(voice, text, tmp, idx)
            pieces.append(seg)
            pieces.append(pause)

    full = np.concatenate(pieces)
    peak = float(np.max(np.abs(full)))
    if peak > 1e-9:
        full = full * (0.707 / peak)
    pcm = np.clip(np.round(full * 32767.0), -32768, 32767).astype(np.int16)[:, None]
    out_path = Path(out_wav)
    save_wav(Audio(pcm, SR, "PCM_16", "WAV"), out_path)
    return out_path


def synthesize_dialogue(
    script: str,
    language: str,
    out_wav: str | Path,
    *,
    pause_sec: float = 0.55,
) -> Path:
    """Synchronous wrapper around :func:`synthesize_dialogue_async`."""
    return asyncio.run(synthesize_dialogue_async(script, language, out_wav, pause_sec=pause_sec))
