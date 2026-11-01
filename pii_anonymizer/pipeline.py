"""End-to-end pipeline: audio -> ASR words -> PII detections -> beeped audio + JSON."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .asr import DEFAULT_PRESET, PRESETS, ASRBackend, FasterWhisperASR
from .audio import Audio, create_beeped_copy, load_audio, resample_to_16k, save_wav
from .cache import TranscriptCache, file_sha256
from .languages import LanguagePack, load_language
from .names import NameConfig, detect_names
from .numbers import NumberConfig, detect_numbers
from .spans import SpanConfig, detections_to_redactions, merge_redactions
from .tokens import tokenize
from .types import Detection, Redaction, Token, Word

log = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3"}


@dataclass
class PipelineConfig:
    numbers: NumberConfig = field(default_factory=NumberConfig)
    names: NameConfig = field(default_factory=NameConfig)
    spans: SpanConfig = field(default_factory=SpanConfig)
    beep_freq: float = 1000.0
    extra_pass_languages: list[str] = field(default_factory=list)


def detect(words: list[Word], pack: LanguagePack,
           cfg: PipelineConfig | None = None) -> tuple[list[Token], list[Detection]]:
    """Pure text step - no audio, no model. Used by the tests."""
    cfg = cfg or PipelineConfig()
    tokens = tokenize(words)
    num_dets = detect_numbers(tokens, pack, cfg.numbers)
    blocked: set[int] = set()
    for d in num_dets:
        blocked.update(range(d.first, d.last + 1))
    name_dets = detect_names(tokens, pack, cfg.names, blocked=blocked)
    dets = sorted(num_dets + name_dets, key=lambda d: (d.first, d.last))
    return tokens, dets


@dataclass
class FileResult:
    input_path: Path
    redacted_path: Path
    json_path: Path | None
    redactions: list[Redaction]
    duration: float
    samplerate: int = 0
    channels: int = 0
    transcript_path: Path | None = None

    @property
    def primary_output(self) -> Path:
        """Primary output: the copy of the recording with PII covered by beeps."""
        return self.redacted_path


class Anonymizer:
    def __init__(self, asr: ASRBackend | None, config: PipelineConfig | None = None,
                 cache: TranscriptCache | None = None):
        self.asr = asr
        self.cfg = config or PipelineConfig()
        self.cache = cache if cache is not None else TranscriptCache(enabled=False)
        self.last_asr_cached = False

    def _words_for(self, in_path: Path, audio16_fn, language: str) -> list[Word]:
        """ASR words for a file, from the cache when possible."""
        if self.asr is None:
            raise RuntimeError("No ASR backend configured and no words supplied")
        key = self.cache.key(file_sha256(in_path), self.asr.settings(), language)
        words = self.cache.get(key)
        if words is not None:
            self.last_asr_cached = True
            return words
        self.last_asr_cached = False
        words = self.asr.transcribe(audio16_fn(), language)
        self.cache.put(key, words, {"file": in_path.name, "language": language,
                                    **self.asr.settings()})
        return words

    # ------------------------------------------------------------------
    def process_file(self, in_path: str | Path, language: str,
                     out_dir: str | Path | None = None, *,
                     output_path: str | Path | None = None,
                     write_json: bool = True,
                     words: list[Word] | None = None,
                     save_transcript: bool = False) -> FileResult:
        in_path = Path(in_path)
        pack = load_language(language)
        stem = in_path.stem
        if output_path is not None:
            redacted_path = Path(output_path)
            target_dir = redacted_path.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            base_stem = (
                redacted_path.stem[: -len("_redacted")]
                if redacted_path.stem.endswith("_redacted") and len(redacted_path.stem) > len("_redacted")
                else redacted_path.stem
            )
        else:
            target_dir = Path(out_dir) if out_dir else in_path.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            redacted_path = target_dir / f"{stem}_redacted.wav"
            base_stem = stem

        if redacted_path.resolve() == in_path.resolve():
            raise ValueError(
                f"Output path '{redacted_path}' must differ from input path to preserve the original recording"
            )
        json_path = (target_dir / f"{base_stem}_redactions.json") if write_json else None

        audio = load_audio(in_path)
        mono = audio.mono_float32()
        _a16: list = []

        def audio16():
            if not _a16:
                _a16.append(resample_to_16k(mono, audio.samplerate))
            return _a16[0]

        if words is None:
            words = self._words_for(in_path, audio16, pack.whisper_language)
        tokens, dets = detect(words, pack, self.cfg)
        reds = detections_to_redactions(dets, tokens, duration=audio.duration,
                                        mono=mono, sr=audio.samplerate, cfg=self.cfg.spans)

        extra_dump = {}
        if self.cfg.extra_pass_languages and self.asr is not None:
            extra_reds = []
            for lang in self.cfg.extra_pass_languages:
                if lang == pack.whisper_language:
                    continue
                w2 = self._words_for(in_path, audio16, lang)
                t2 = tokenize(w2)
                d2 = [d for d in detect_numbers(t2, pack, self.cfg.numbers)
                      if len(d.text) >= 6]
                extra_reds += detections_to_redactions(
                    d2, t2, duration=audio.duration, mono=mono,
                    sr=audio.samplerate, cfg=self.cfg.spans)
                extra_dump[lang] = [w.to_dict() for w in w2]
            reds = _union_extra(reds, extra_reds, self.cfg.spans.merge_gap)

        # Primary output: independent copy of the recording with PII spans covered by beeps
        beeped_copy = create_beeped_copy(
            audio, [(r.start, r.end) for r in reds], freq=self.cfg.beep_freq
        )
        save_wav(beeped_copy, redacted_path)
        _check_output(in_path, redacted_path, audio)

        if json_path is not None:
            report = {
                "input": in_path.name,
                "language": pack.code,
                "redactions": [r.to_dict() for r in reds],
            }
            json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")

        transcript_path = None
        if save_transcript:
            transcript_path = target_dir / f"{base_stem}_transcript.json"
            dump = {
                "input": in_path.name,
                "language": pack.code,
                "words": [w.to_dict() for w in words],
                "detections": [
                    {"type": d.type, "text": d.text, "reason": d.reason,
                     "tokens": [tokens[i].raw for i in range(d.first, d.last + 1)],
                     "start": round(tokens[d.first].start, 3),
                     "end": round(tokens[d.last].end, 3)}
                    for d in dets
                ],
            }
            if extra_dump:
                dump["extra_pass_words"] = extra_dump
            transcript_path.write_text(json.dumps(dump, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
        return FileResult(
            in_path,
            redacted_path,
            json_path,
            reds,
            audio.duration,
            audio.samplerate,
            audio.channels,
            transcript_path,
        )


def _union_extra(primary: list[Redaction], extra: list[Redaction], gap: float) -> list[Redaction]:
    """Add extra-pass number spans that the primary pass did not already cover."""
    out = list(primary)
    for e in extra:
        covered = sum(max(0.0, min(e.end, p.end) - max(e.start, p.start)) for p in primary)
        if covered < 0.8 * (e.end - e.start):
            out.append(e)
    return merge_redactions(out, gap)


def _check_output(in_path: Path, out_path: Path, audio: Audio) -> None:
    """Hard guarantee: same frame count, rate and channels as the input."""
    import soundfile as sf
    info = sf.info(str(out_path))
    if info.frames != audio.frames or info.samplerate != audio.samplerate \
            or info.channels != audio.channels:
        raise RuntimeError(
            f"Output check failed for {in_path.name}: frames {info.frames} vs {audio.frames}, "
            f"rate {info.samplerate} vs {audio.samplerate}, channels {info.channels} vs {audio.channels}")


def list_audio_files(folder: Path, recursive: bool = False) -> list[Path]:
    it = folder.rglob("*") if recursive else folder.iterdir()
    files = [p for p in it if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
             and not p.stem.endswith("_redacted")]
    return sorted(files)


def load_words_json(path: str | Path) -> list[Word]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data["words"] if isinstance(data, dict) else data
    return [Word.from_dict(d) for d in items]


def anonymize(
    in_path: str | Path,
    language: str,
    output_path: str | Path | None = None,
    *,
    out_dir: str | Path | None = None,
    preset: str = DEFAULT_PRESET,
    write_json: bool = True,
    use_cache: bool = True,
    words: list[Word] | None = None,
    config: PipelineConfig | None = None,
) -> Path:
    """Produce a copy of ``in_path`` with all detected PII spans covered by a beep.

    Returns the ``Path`` to the beeped audio copy (the primary output).
    """
    p = PRESETS[preset]
    asr = None if words is not None else FasterWhisperASR(
        model=p["model"], beam_size=p["beam_size"]
    )
    cache = TranscriptCache(enabled=use_cache)
    res = Anonymizer(asr, config, cache=cache).process_file(
        in_path,
        language,
        out_dir=out_dir,
        output_path=output_path,
        write_json=write_json,
        words=words,
    )
    return res.primary_output
