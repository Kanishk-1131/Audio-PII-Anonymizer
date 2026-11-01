"""Command-line interface for the Audio PII Anonymizer.

Examples
--------
    python -m pii_anonymizer call_001.wav --lang hi
    python -m pii_anonymizer call_002.mp3 -l ta --out-dir outputs
    python -m pii_anonymizer recordings/ -l hi --out-dir outputs     # whole folder
    python -m pii_anonymizer call_001.wav -l hi --preset fast        # quickest
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .asr import DEFAULT_PRESET, PRESETS, FasterWhisperASR
from .audio import open_audio_player, play_wav
from .cache import TranscriptCache
from .languages import available_languages, load_language
from .pipeline import (
    AUDIO_EXTENSIONS,
    Anonymizer,
    FileResult,
    PipelineConfig,
    list_audio_files,
    load_words_json,
)


def build_parser() -> argparse.ArgumentParser:
    langs = [c for c in available_languages() if c != "en"]
    p = argparse.ArgumentParser(
        prog="pii-anonymizer",
        description="Produce a copy of an Indian-language call recording in which "
                    "customer names, mobile numbers, and PIN codes are covered by a beep.",
    )
    p.add_argument("input", nargs="?", help="Audio file (.wav/.mp3) or a folder of audio files.")
    p.add_argument("-l", "--lang", "--language", dest="language",
                   help=f"Language of the call: {', '.join(langs)}.")
    p.add_argument("-o", "--output", "--out-dir", dest="output", default=None,
                   help="Primary output .wav path (single file) or output folder "
                        "(default: <input_stem>_redacted.wav next to each input file).")
    p.add_argument("--no-json", action="store_true",
                   help="Write only the beeped audio copy (omit the companion *_redactions.json file).")
    p.add_argument("--play", action="store_true",
                   help="Play the beeped audio copy directly in the terminal after producing it.")
    p.add_argument("--open", dest="open_player", action="store_true",
                   help="Open the beeped audio copy in your OS default media player (e.g. Windows Media Player).")
    p.add_argument("-r", "--recursive", action="store_true",
                   help="With a folder input, also process sub-folders.")

    speed = p.add_argument_group("speed / accuracy")
    speed.add_argument("-p", "--preset", choices=sorted(PRESETS), default=DEFAULT_PRESET,
                       help="fast = large-v3-turbo beam 5 (default, ~10s/call on CPU); "
                            "balanced = large-v3 greedy; accurate = large-v3 beam 5.")
    speed.add_argument("-m", "--model", default=None, help="Override the Whisper model of the preset.")
    speed.add_argument("--beam-size", type=int, default=None, help="Override the beam size of the preset.")
    speed.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    speed.add_argument("--compute-type", default="auto", help="int8 / float16 / float32 (default: auto).")
    speed.add_argument("--threads", type=int, default=0, help="CPU threads (default: all cores).")
    speed.add_argument("--no-vad", action="store_true", help="Disable the voice-activity filter.")
    speed.add_argument("--no-cache", action="store_true",
                       help="Always re-run speech recognition (ignore the transcript cache).")
    speed.add_argument("--cache-dir", default=None,
                       help="Transcript cache folder (default: ~/.cache/pii_anonymizer).")

    det = p.add_argument_group("redaction")
    det.add_argument("--pad-before", type=float, default=0.08, help="Seconds of margin before each beep.")
    det.add_argument("--pad-after", type=float, default=0.10, help="Seconds of margin after each beep.")
    det.add_argument("--beep-freq", type=float, default=1000.0, help="Beep frequency in Hz.")
    det.add_argument("--dual-pass-en", action="store_true",
                     help="Extra English recognition pass to catch English digits "
                          "(slower; doubles ASR time).")

    dbg = p.add_argument_group("debugging")
    dbg.add_argument("--save-transcript", action="store_true",
                     help="Also write <name>_transcript.json (words + why each beep was placed).")
    dbg.add_argument("--transcript-in", default=None,
                     help="Use words from a saved *_transcript.json instead of running ASR.")
    dbg.add_argument("--list-languages", action="store_true", help="List language packs and exit.")
    dbg.add_argument("-v", "--verbose", action="store_true")
    return p


def _print_result(res: FileResult, elapsed: float, cached: bool, beep_freq: float) -> None:
    src = "cached transcript" if cached else "transcribed"
    meta = f"{res.duration:.2f}s"
    if res.samplerate and res.channels:
        meta += f" | {res.samplerate} Hz | {res.channels} ch"
    print(f"\n[{res.input_path.name}]  {meta} | {len(res.redactions)} beep(s) applied in {elapsed:.1f}s ({src})")
    print(f"  PRIMARY OUTPUT (beeped audio) : {res.primary_output}")
    if res.json_path is not None:
        print(f"  Companion redaction log       : {res.json_path}")
    if not res.redactions:
        print("  (no PII detected - audio copied untouched)")
    for i, r in enumerate(res.redactions, 1):
        print(f"    {i}. {r.start:7.2f}s - {r.end:7.2f}s  [BEEP {beep_freq:g} Hz]  {r.type:<8} {r.text}")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Indic text on Windows consoles
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    if args.list_languages:
        for code in available_languages():
            pack = load_language(code)
            print(f"{pack.code:<4} {pack.name}")
        return 0
    if not args.input or not args.language:
        parser.error("an input path and --lang are required (e.g. call.wav --lang hi)")

    try:
        load_language(args.language)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    target = Path(args.input)
    if not target.exists():
        print(f"error: not found: {target}", file=sys.stderr)
        return 2
    if target.is_dir():
        files = list_audio_files(target, recursive=args.recursive)
        if not files:
            print(f"error: no .wav/.mp3 files in {target}", file=sys.stderr)
            return 2
        if args.transcript_in:
            print("error: --transcript-in works with a single file only", file=sys.stderr)
            return 2
    else:
        if target.suffix.lower() not in AUDIO_EXTENSIONS:
            print(f"error: unsupported file type '{target.suffix}' (use .wav or .mp3)", file=sys.stderr)
            return 2
        files = [target]

    out_dir: Path | None = None
    output_path: Path | None = None
    if args.output:
        out_arg = Path(args.output)
        if out_arg.suffix.lower() == ".wav" and not out_arg.is_dir():
            if target.is_dir():
                print("error: -o cannot be a .wav file when input is a folder; specify an output directory",
                      file=sys.stderr)
                return 2
            output_path = out_arg
        else:
            out_dir = out_arg

    cfg = PipelineConfig()
    cfg.spans.pad_before = args.pad_before
    cfg.spans.pad_after = args.pad_after
    cfg.beep_freq = args.beep_freq
    if args.dual_pass_en:
        cfg.extra_pass_languages = ["en"]

    preset = PRESETS[args.preset]
    asr = None
    words = None
    if args.transcript_in:
        words = load_words_json(args.transcript_in)
    else:
        asr = FasterWhisperASR(model=args.model or preset["model"],
                               beam_size=args.beam_size or preset["beam_size"],
                               device=args.device, compute_type=args.compute_type,
                               cpu_threads=args.threads, vad=not args.no_vad)
    cache = TranscriptCache(args.cache_dir, enabled=not args.no_cache)
    anon = Anonymizer(asr, cfg, cache=cache)

    failures = 0
    t_all = time.perf_counter()
    for f in files:
        t0 = time.perf_counter()
        try:
            res = anon.process_file(
                f,
                args.language,
                out_dir=out_dir,
                output_path=output_path,
                write_json=not args.no_json,
                words=words,
                save_transcript=args.save_transcript,
            )
        except Exception as exc:  # keep going in folder mode
            failures += 1
            print(f"\n[{f.name}] FAILED: {exc}", file=sys.stderr)
            if args.verbose:
                raise
            continue
        _print_result(res, time.perf_counter() - t0, anon.last_asr_cached and words is None, cfg.beep_freq)
        if args.open_player:
            print(f"  Opening in Media Player: {res.primary_output}")
            open_audio_player(res.primary_output)
        if args.play:
            print(f"  Playing {res.primary_output} ({res.duration:.1f}s) ... [Ctrl+C to stop]")
            try:
                play_wav(res.primary_output)
            except KeyboardInterrupt:
                print("\n  Playback stopped.")
                break
    n = len(files)
    print(f"\nDone: {n - failures}/{n} recording(s) anonymized in {time.perf_counter() - t_all:.1f}s")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
