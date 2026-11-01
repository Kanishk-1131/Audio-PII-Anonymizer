"""CLI helper to create a new Hindi / Tamil / Kannada speech WAV from text
and optionally anonymize + play it right away.

Examples:
    .\.venv\Scripts\python.exe scripts\make_speech.py --lang hi --name my_call --anonymize --open ^
        --text "Agent: नमस्ते, अपना नाम बताइए।
Customer: मेरा नाम अमित पटेल है।
Agent: धन्यवाद अमित जी, मोबाइल नंबर और पिन कोड बताइए।
Customer: नौ आठ एक शून्य दो तीन चार पाँच छह सात, और पिन कोड चार शून्य शून्य शून्य शून्य एक।"

    .\.venv\Scripts\python.exe scripts\make_speech.py --lang ta --file my_script.txt --name tamil_test --anonymize --open
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pii_anonymizer.cli import main as cli_main  # noqa: E402
from pii_anonymizer.tts import VOICE_PAIRS, synthesize_dialogue  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    p = argparse.ArgumentParser(description="Create a new speech WAV recording from text.")
    p.add_argument("-l", "--lang", required=True, choices=sorted(VOICE_PAIRS),
                   help="Language code (hi, ta, kn).")
    p.add_argument("-n", "--name", default="custom_speech",
                   help="Output filename stem inside samples/ (default: custom_speech -> samples/custom_speech.wav).")
    p.add_argument("-t", "--text", default=None,
                   help="Dialogue text (use 'Agent: ...' and 'Customer: ...' on separate lines, or single line).")
    p.add_argument("-f", "--file", default=None,
                   help="Path to a UTF-8 .txt file containing the dialogue lines.")
    p.add_argument("-o", "--out", default=None,
                   help="Explicit output .wav path (overrides --name).")
    p.add_argument("--anonymize", action="store_true",
                   help="Immediately run PII anonymization on the generated speech.")
    p.add_argument("--open", action="store_true",
                   help="Open the resulting WAV in Windows Media Player.")
    p.add_argument("--play", action="store_true",
                   help="Play the resulting WAV in the terminal.")
    args = p.parse_args(argv)

    if args.file:
        script = Path(args.file).read_text(encoding="utf-8")
    elif args.text:
        script = args.text
    else:
        print("Type or paste your dialogue lines (Agent: ... / Customer: ...), then press Ctrl+Z + Enter:")
        script = sys.stdin.read()

    out_wav = Path(args.out) if args.out else (ROOT_DIR / "samples" / f"{Path(args.name).stem}.wav")
    print(f"Synthesizing {args.lang} speech -> {out_wav} ...")
    synthesize_dialogue(script, args.lang, out_wav)
    print(f"Created speech recording: {out_wav}")

    if args.anonymize:
        cli_args = [str(out_wav), "--lang", args.lang, "-o", str(ROOT_DIR / "outputs")]
        if args.open:
            cli_args.append("--open")
        if args.play:
            cli_args.append("--play")
        return cli_main(cli_args)

    if args.open:
        from pii_anonymizer.audio import open_audio_player
        open_audio_player(out_wav)
    if args.play:
        from pii_anonymizer.audio import play_wav
        play_wav(out_wav)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
