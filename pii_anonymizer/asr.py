"""Speech recognition backends.

The pipeline only needs ``transcribe(audio_16k, language) -> list[Word]``;
any engine that returns word-level timestamps can be plugged in by
implementing :class:`ASRBackend`. The default is faster-whisper (Whisper via
CTranslate2), which supports Hindi, Tamil, Kannada, Telugu, Marathi, Bengali
and English code-mixing out of the box.

Decoding is deterministic: beam search at temperature 0 only (no sampling
fallback), no conditioning on previous text, fixed VAD settings, timestamps
rounded to 1 ms. Together with the transcript cache this makes runs exactly
repeatable.

The model is loaded lazily - only when the first file that is not in the
transcript cache has to be transcribed.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod

import numpy as np

from .types import Word

log = logging.getLogger(__name__)

# Speed / accuracy presets (measured on an 8-core CPU, see WRITEUP.md).
PRESETS = {
    # most thorough: full 32-layer decoder with beam search
    "accurate": {"model": "large-v3", "beam_size": 5},
    # full 32-layer decoder with greedy search (~2x faster than accurate)
    "balanced": {"model": "large-v3", "beam_size": 1},
    # default: distilled 4-layer decoder with beam search (~3-4x faster, ~10s/call on CPU)
    "fast": {"model": "large-v3-turbo", "beam_size": 5},
}
DEFAULT_PRESET = "fast"


class ASRBackend(ABC):
    name = "base"

    @abstractmethod
    def transcribe(self, audio_16k: np.ndarray, language: str) -> list[Word]:
        ...

    def settings(self) -> dict:
        """Everything that influences the recognised words (cache key)."""
        return {"backend": self.name}


class FasterWhisperASR(ASRBackend):
    name = "faster-whisper"

    def __init__(self, model: str = "large-v3", device: str = "auto",
                 compute_type: str = "auto", cpu_threads: int = 0,
                 beam_size: int = 1, vad: bool = True,
                 download_root: str | None = None):
        if device == "auto":
            device = "cuda" if _cuda_available() else "cpu"
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        if cpu_threads <= 0:
            cpu_threads = os.cpu_count() or 4
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.beam_size = beam_size
        self.vad = vad
        self.download_root = download_root
        self._model = None

    # -- lazy loading ---------------------------------------------------
    @property
    def model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            t0 = time.perf_counter()
            print(f"Loading speech model '{self.model_name}' on {self.device} "
                  f"({self.compute_type}) ... ", end="", flush=True)
            self._model = WhisperModel(self.model_name, device=self.device,
                                       compute_type=self.compute_type,
                                       cpu_threads=self.cpu_threads, num_workers=1,
                                       download_root=self.download_root)
            print(f"done in {time.perf_counter() - t0:.1f}s", flush=True)
        return self._model

    def settings(self) -> dict:
        return {"backend": self.name, "model": self.model_name, "beam": self.beam_size,
                "vad": self.vad, "compute": self.compute_type, "device": self.device}

    def transcribe(self, audio_16k: np.ndarray, language: str) -> list[Word]:
        segments, _info = self.model.transcribe(
            audio_16k,
            language=language,
            task="transcribe",
            beam_size=self.beam_size,
            best_of=1,
            temperature=[0.0],
            condition_on_previous_text=False,
            word_timestamps=True,
            vad_filter=self.vad,
            vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 300,
                            "threshold": 0.35},
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
        words: list[Word] = []
        for seg in segments:
            for w in seg.words or []:
                text = (w.word or "").strip()
                if not text:
                    continue
                words.append(Word(text, round(float(w.start), 3), round(float(w.end), 3),
                                  round(float(w.probability), 4)))
        words.sort(key=lambda w: (w.start, w.end))
        return words


def _cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False
