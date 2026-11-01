"""Audio I/O and beep synthesis.

Guarantees
----------
* The output has exactly the same number of frames, sample rate and channel
  count as the input (so the duration is identical, not just within 10 ms).
* For PCM/float WAV input the samples are processed in their native integer
  or float representation, so every sample outside a beep is bit-identical
  to the input.
* Inside a beep the original audio is *replaced* (not mixed), so nothing of
  the PII can be recovered from the redacted file.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

_INT_SUBTYPES = {
    "PCM_S8": np.int16, "PCM_U8": np.int16, "PCM_16": np.int16,
    "PCM_24": np.int32, "PCM_32": np.int32,
}
_FLOAT_SUBTYPES = {"FLOAT": np.float32, "DOUBLE": np.float64}


@dataclass
class Audio:
    data: np.ndarray        # shape (frames, channels), native dtype
    samplerate: int
    subtype: str            # subtype to use when writing
    source_format: str      # e.g. "WAV", "MP3"

    @property
    def frames(self) -> int:
        return self.data.shape[0]

    @property
    def channels(self) -> int:
        return self.data.shape[1]

    @property
    def duration(self) -> float:
        return self.frames / float(self.samplerate)

    def as_float(self) -> np.ndarray:
        """(frames, channels) float64 in [-1, 1]."""
        d = self.data
        if d.dtype == np.int16:
            return d.astype(np.float64) / 32768.0
        if d.dtype == np.int32:
            return d.astype(np.float64) / 2147483648.0
        return d.astype(np.float64)

    def mono_float32(self) -> np.ndarray:
        return self.as_float().mean(axis=1).astype(np.float32)

    def copy(self) -> Audio:
        """Return an independent deep copy of the recording."""
        return Audio(self.data.copy(), self.samplerate, self.subtype, self.source_format)


def load_audio(path: str | Path) -> Audio:
    path = Path(path)
    try:
        info = sf.info(str(path))
    except Exception as exc:  # libsndfile cannot read it -> try PyAV
        return _load_with_av(path, exc)
    fmt = info.format
    subtype = info.subtype
    if fmt == "WAV" and subtype in _INT_SUBTYPES:
        dtype, out_subtype = _INT_SUBTYPES[subtype], subtype
    elif fmt == "WAV" and subtype in _FLOAT_SUBTYPES:
        dtype, out_subtype = _FLOAT_SUBTYPES[subtype], subtype
    else:
        # MP3 / ADPCM / u-law / ... -> decode to float, write 16-bit PCM WAV
        dtype, out_subtype = np.float32, "PCM_16"
    data, sr = sf.read(str(path), dtype=dtype, always_2d=True)
    if data.shape[0] == 0:
        raise ValueError(f"{path.name}: audio file contains no samples")
    return Audio(np.ascontiguousarray(data), int(sr), out_subtype, fmt)


def _load_with_av(path: Path, first_error: Exception) -> Audio:
    try:
        import av  # installed with faster-whisper
    except ImportError:  # pragma: no cover
        raise first_error
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        sr = stream.codec_context.sample_rate
        chunks = []
        for frame in container.decode(stream):
            arr = frame.to_ndarray()
            if arr.dtype.kind in "iu":
                arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
            if frame.format.is_planar:
                arr = arr.T                       # (samples, channels)
            else:
                arr = arr.reshape(-1, len(frame.layout.channels))
            chunks.append(arr.astype(np.float32))
    if not chunks:
        raise ValueError(f"{path.name}: audio file contains no samples")
    data = np.concatenate(chunks, axis=0)
    return Audio(np.ascontiguousarray(data), int(sr), "PCM_16", path.suffix.upper().lstrip("."))


def save_wav(audio: Audio, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = audio.data
    if audio.subtype == "PCM_16" and data.dtype.kind == "f":
        data = np.clip(np.round(data * 32767.0), -32768, 32767).astype(np.int16)
    sf.write(str(path), data, audio.samplerate, subtype=audio.subtype, format="WAV")


def resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    """Mono float32 resampled to 16 kHz (Whisper's input rate)."""
    if sr == 16000:
        return x.astype(np.float32)
    from scipy.signal import resample_poly
    g = gcd(sr, 16000)
    y = resample_poly(x.astype(np.float64), 16000 // g, sr // g)
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Loudness / energy helpers
# ---------------------------------------------------------------------------

def frame_rms(x: np.ndarray, sr: int, frame_s: float = 0.010) -> np.ndarray:
    hop = max(1, int(round(sr * frame_s)))
    n = len(x) // hop
    if n == 0:
        return np.array([float(np.sqrt(np.mean(x ** 2)))]) if len(x) else np.zeros(1)
    f = x[: n * hop].astype(np.float64).reshape(n, hop)
    return np.sqrt(np.mean(f ** 2, axis=1))


def speech_rms(x: np.ndarray, sr: int) -> float:
    """RMS of the speech-active part of the signal (not the silences)."""
    r = frame_rms(x, sr)
    if not np.any(r > 0):
        return 0.0
    floor = np.percentile(r, 10)
    peak = np.percentile(r, 99)
    thr = max(floor * 3.0, peak * 0.05, 1e-5)
    active = r[r > thr]
    if active.size == 0:
        active = r
    return float(np.sqrt(np.mean(active ** 2)))


# ---------------------------------------------------------------------------
# Beep
# ---------------------------------------------------------------------------

def apply_beeps(audio: Audio, spans: list[tuple[float, float]], *,
                freq: float = 1000.0, level_rms: float | None = None,
                fade_s: float = 0.005) -> float:
    """Overwrite every (start, end) span with a steady sine beep, in place.

    Each beep's RMS is matched to the local active speech RMS of the speaker
    in that span (bounded smoothly around the call-wide speech RMS so loudness
    matches the surrounding voice), capped at -1 dBFS peak. Original samples
    inside each span are completely replaced (never mixed), while all samples
    outside the spans remain bit-identical.
    """
    sr = audio.samplerate
    mono = audio.mono_float32() if level_rms is None else None
    global_rms = speech_rms(mono, sr) if mono is not None else float(level_rms)
    default_amp = float(min(0.89, max(global_rms * np.sqrt(2.0), 0.02)))
    last_amp = default_amp

    for start, end in spans:
        i0 = max(0, int(round(start * sr)))
        i1 = min(audio.frames, int(round(end * sr)))
        n = i1 - i0
        if n <= 0:
            continue
        if mono is not None and global_rms > 0.0:
            local_rms = speech_rms(mono[i0:i1], sr)
            if local_rms >= 0.25 * global_rms:
                target_rms = float(np.clip(local_rms, 0.55 * global_rms, 1.60 * global_rms))
            else:
                target_rms = global_rms
            amp = float(min(0.89, max(target_rms * np.sqrt(2.0), 0.02)))
        else:
            amp = default_amp
        last_amp = amp

        t = np.arange(n, dtype=np.float64) / sr
        tone = amp * np.sin(2.0 * np.pi * freq * t)
        nf = min(int(round(fade_s * sr)), n // 4)
        if nf > 0:
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, nf))
            tone[:nf] *= ramp
            tone[-nf:] *= ramp[::-1]
        audio.data[i0:i1, :] = _to_dtype(tone, audio.data.dtype)[:, None]
    return last_amp


def create_beeped_copy(audio: Audio, spans: list[tuple[float, float]], *,
                       freq: float = 1000.0, level_rms: float | None = None,
                       fade_s: float = 0.005) -> Audio:
    """Return a new copy of ``audio`` with each (start, end) span covered by a beep."""
    redacted = audio.copy()
    apply_beeps(redacted, spans, freq=freq, level_rms=level_rms, fade_s=fade_s)
    return redacted


def open_audio_player(path: str | Path) -> None:
    """Open an audio file in the operating system's default GUI media player."""
    import os
    import subprocess
    import sys

    wav_path = str(Path(path).resolve())
    if sys.platform == "win32":
        os.startfile(wav_path)  # type: ignore[attr-defined]
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", wav_path])
        return
    subprocess.Popen(["xdg-open", wav_path])


def play_wav(path: str | Path) -> None:
    """Play a WAV file synchronously through the OS media audio output."""
    import shutil
    import subprocess
    import sys
    import time

    wav_path = str(Path(path).resolve())
    if sys.platform == "win32":
        import ctypes

        winmm = ctypes.windll.winmm
        alias = f"pii_{int(time.time() * 1000) % 1000000}"
        buf = ctypes.create_unicode_buffer(256)
        rc = winmm.mciSendStringW(f'open "{wav_path}" type waveaudio alias {alias}', buf, 256, 0)
        if rc == 0:
            try:
                if winmm.mciSendStringW(f"play {alias}", buf, 256, 0) == 0:
                    while True:
                        winmm.mciSendStringW(f"status {alias} mode", buf, 256, 0)
                        if buf.value.strip().lower() not in ("playing", "seeking"):
                            break
                        time.sleep(0.1)
                    return
            finally:
                winmm.mciSendStringW(f"close {alias}", buf, 256, 0)
        open_audio_player(wav_path)
        return
    if sys.platform == "darwin" and shutil.which("afplay"):
        subprocess.run(["afplay", wav_path], check=True)
        return
    for cmd in ("paplay", "aplay", "ffplay"):
        exe = shutil.which(cmd)
        if exe:
            args = [exe, "-nodisp", "-autoexit", wav_path] if cmd == "ffplay" else [exe, wav_path]
            subprocess.run(args, check=True)
            return
    open_audio_player(wav_path)


def _to_dtype(x: np.ndarray, dtype) -> np.ndarray:
    if dtype == np.int16:
        return np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)
    if dtype == np.int32:
        return np.clip(np.round(x * 2147483647.0), -2147483648, 2147483647).astype(np.int32)
    return x.astype(dtype)
