# Audio-PII-Anonymizer

A fast, deterministic command-line tool, local browser studio, and Python library that produces a copy of an Indian-language call recording (.wav or .mp3) in which every spoken customer name, 10-digit mobile number, and 6-digit PIN code is covered by a loudness-matched 1 kHz beep as its primary output, while leaving all non-PII audio untouched.

Features
Primary Beeped Audio Copy (<stem>_redacted.wav):
Never modifies the original recording; writes an independent copy with the exact same frame count, sample rate, and channel count (0.0 ms duration drift, verified after writing).
100% bit-identical non-PII audio for PCM/float WAV inputs.
Irreversible PII replacement: overwrites (never mixes) PII spans across all channels with a steady 1000 Hz sine tone (5 ms raised-cosine ramps to prevent clicks) matched to the active speech RMS of the speaker in that span (0.0–0.1 dB delta).
Companion Redaction Report (<stem>_redactions.json): Records exact start_sec, end_sec, type (name, mobile, pincode), and recognized text for every beep.
Built for Real Indian Call-Center Speech:
Handles Hindi (hi), Tamil (ta), and Kannada (kn), plus arbitrary code-mixing with English (en).
Recognizes grouped digits, colloquial spoken numerals (e.g., Tamil "ஒன்னு, ரெண்டு, மூணு, அஞ்சு"), repeat modifiers ("double", "triple", "डबल", "டபுள்"), 10–99 Hindi compounds, and agent read-backs of names or mobile numbers.
Uses a 10 ms RMS acoustic boundary aligner to prevent ASR timestamp compression from leaking trailing digits (0–10 ms max leak across the evaluation set).
Fast & 100% Repeatable:
Default fast preset (faster-whisper large-v3-turbo, int8, temperature=0.0) runs 2.4x faster than real-time on CPU (~9–11 s per 30 s call) with deterministic outputs.
Content-addressed SHA-256 transcript cache (~/.cache/pii_anonymizer) reduces repeat runs on unchanged audio to ~0.05 s.
1. Setup & Installation

Prerequisites: Python 3.10+ (64-bit). Everything runs locally offline once dependencies and the speech model are downloaded during setup.

Windows (One-Command Setup)
powershell
.\setup.bat
Linux / macOS (One-Command Setup)
bash
chmod +x setup.sh run.sh
./setup.sh
Manual Setup (Any OS)
powershell
python -m venv .venv
# Windows:
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\pip install -e .
# Linux / macOS:
# .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
2. Quick Start — How to Run
Option A: One-Line Runner (run.bat / run.sh)

Writes <name>_redacted.wav (primary output) and <name>_redactions.json into outputs/:

powershell
# Single Hindi or Tamil recording
.\run.bat samples\sample_1_hindi.wav hi
.\run.bat samples\sample_2_tamil.wav ta
# Produce AND automatically open the beeped copy in Windows Media Player (--open)
.\run.bat samples\sample_1_hindi.wav hi --open
# Produce AND play directly inside the terminal (--play)
.\run.bat samples\sample_1_hindi.wav hi --play
# Batch-anonymize an entire folder of .wav / .mp3 recordings
.\run.bat samples\ hi

(On Linux/macOS, replace .\run.bat with ./run.sh, e.g. ./run.sh samples/sample_1_hindi.wav hi.)

Option B: Direct Python CLI (pii-anonymizer)

By default (when -o is omitted), outputs are written next to each input file:

powershell
# Default: writes samples\sample_1_hindi_redacted.wav + samples\sample_1_hindi_redactions.json
.\.venv\Scripts\python.exe -m pii_anonymizer samples\sample_1_hindi.wav --lang hi
.\.venv\Scripts\python.exe -m pii_anonymizer samples\sample_2_tamil.wav --lang ta
# Write the primary beeped audio copy to an explicit .wav path
.\.venv\Scripts\python.exe -m pii_anonymizer samples\sample_1_hindi.wav --lang hi -o outputs\my_beeped_call.wav
# Emit ONLY the beeped audio copy (omit the companion JSON file)
.\.venv\Scripts\python.exe -m pii_anonymizer samples\sample_1_hindi.wav --lang hi -o outputs\only_audio.wav --no-json
# Process a whole folder recursively into an output directory
.\.venv\Scripts\python.exe -m pii_anonymizer recordings\ --lang hi -o outputs --recursive
# List all installed language packs
.\.venv\Scripts\python.exe -m pii_anonymizer --list-languages
Option C: Interactive Browser Audio Player & Studio
powershell
.\run_ui.bat

Opens http://127.0.0.1:8765 in your browser where you can:

Play any sample's Primary Output (Beeped Recording Copy) alongside the original recording, click red timeline markers to jump straight to each beep, and download the beeped .wav.
Upload any .wav or .mp3 recording.
Record your own voice live from your microphone and anonymize it in one click.
Type custom Agent: / Customer: dialogue text in Hindi, Tamil, or Kannada to synthesize and anonymize a new two-speaker call.
Option D: Python API
python
from pii_anonymizer import anonymize
# Produces the beeped recording copy and returns its Path as the primary output
beeped_wav = anonymize("samples/sample_1_hindi.wav", language="hi")
print("Primary output:", beeped_wav)  # samples/sample_1_hindi_redacted.wav
3. Adding Your Own Speech Recordings
Existing .wav or .mp3 file: Run .\run.bat "C:\path\to\call.mp3" hi --open (or upload via .\run_ui.bat).
Live microphone: Run .\run_ui.bat → 🎙 2. Record Your Voice (Mic) → speak → Stop & Anonymize My Voice.
Generate a 2-speaker call from text via CLI (scripts/make_speech.py, requires internet for neural TTS):
powershell
.\.venv\Scripts\python.exe scripts\make_speech.py --lang hi --name custom_call --anonymize --open --text "Agent: नमस्ते, अपना नाम बताइए।`nCustomer: मेरा नाम अमित पटेल है।`nAgent: धन्यवाद अमित जी, मोबाइल नंबर बताइए।`nCustomer: नौ आठ एक शून्य दो तीन चार पाँच छह सात।"
4. Output Specification

For an input recording sample_1_hindi.wav (or .mp3), two files are produced:

Primary Output — sample_1_hindi_redacted.wav
Copy of the input recording with identical duration (0.0 ms drift), sample rate, and channels.
Every detected PII span overwritten by a steady 1000 Hz beep matched to the speaker's active speech loudness; all other audio untouched.
Companion Metadata — sample_1_hindi_redactions.json
json
{
  "input": "sample_1_hindi.wav",
  "language": "hi",
  "redactions": [
    {
      "start_sec": 7.41,
      "end_sec": 8.65,
      "type": "name",
      "text": "प्रिया शर्मा"
    },
    {
      "start_sec": 9.46,
      "end_sec": 10.56,
      "type": "name",
      "text": "प्रिया"
    },
    {
      "start_sec": 14.51,
      "end_sec": 19.33,
      "type": "mobile",
      "text": "9845012345"
    },
    {
      "start_sec": 23.97,
      "end_sec": 26.51,
      "type": "pincode",
      "text": "560034"
    }
  ]
}
5. Speed & Accuracy Presets (--preset)
Preset	Whisper Model & Decoding	Cold CPU Speed (30 s call, 8 cores)	Repeat Run (Cached)	Worst PII Leak on Test Suite
fast (default)	large-v3-turbo, beam 5 (int8)	~9 – 11 s (RTF 0.42x)	~0.05 s	10 ms (0 leaks)
balanced	large-v3, beam 1 (int8)	~12 – 24 s (RTF 0.62x)	~0.05 s	10 ms (0 leaks)
accurate	large-v3, beam 5 (int8)	~22 – 47 s (RTF 1.13x)	~0.05 s	10 ms (0 leaks)
powershell
# Switch presets or bypass the SHA-256 transcript cache
.\.venv\Scripts\python.exe -m pii_anonymizer call.wav --lang hi --preset accurate
.\.venv\Scripts\python.exe -m pii_anonymizer call.wav --lang hi --no-cache
6. CLI Reference
Flag	Default	Description
input	(required)	Input .wav/.mp3 file or directory of recordings.
-l, --lang	(required)	Language code (hi, ta, kn).
-o, --output, --out-dir	input folder	Direct .wav output filename (single file) or output directory.
--open	false	Open the beeped .wav copy in the OS default media player.
--play	false	Play the beeped .wav copy directly in the terminal.
--no-json	false	Write only the primary beeped .wav copy (skip *_redactions.json).
-r, --recursive	false	Recursively scan subfolders when input is a directory.
-p, --preset	fast	Speed/model preset: fast, balanced, or accurate.
--device	auto	Execution device: auto, cpu, or cuda.
--no-cache	false	Force fresh ASR transcription (ignore cached transcript).
--pad-before / --pad-after	0.08 / 0.10	Safety margin (seconds) added before/after each acoustic PII span.
--beep-freq	1000.0	Beep sine-wave frequency in Hz.
--save-transcript	false	Also dump word-level ASR timestamps & detection reasons (*_transcript.json).
--list-languages	false	List installed declarative language packs and exit.
7. Testing & Ground-Truth Evaluation
powershell
# 1. Run the automated pytest suite (13 unit & integration tests)
.\.venv\Scripts\python.exe -m pytest -v -W ignore::DeprecationWarning
# 2. Evaluate redacted audio in samples\ against exact segment-level ground truth
#    Verifies voiced PII leak (ms), extra non-PII beep (s), span order, and frame-exact length
.\.venv\Scripts\python.exe scripts\evaluate.py samples samples

Included test calls in samples/:

sample_1_hindi.wav — Brief Sample 1 (Hindi name + agent repeat + grouped 10-digit mobile + 6-digit PIN).
sample_2_tamil.wav — Brief Sample 2 (Tamil name + agent repeat + colloquial mobile with "double ஆறு" + English PIN).
sample_3_hindi_codemix.wav — Hinglish code-mixed digits, "double", full agent mobile read-back, and two agent name repeats.
sample_4_tamil_colloquial.wav — Colloquial Tamil numerals, "triple", honorifics, and native Tamil PIN code.
8. Repository Structure
text
.
├── setup.bat / setup.sh         # One-command environment & model setup (Windows / Linux / macOS)
├── run.bat   / run.sh           # One-command CLI runner -> writes to outputs/
├── run_ui.bat                   # Launches local Browser Audio Player & Studio (http://127.0.0.1:8765)
├── pyproject.toml               # Package metadata & console entry point (pii-anonymizer)
├── requirements.txt             # Pinned runtime & test dependencies
├── pii_anonymizer/
│   ├── __init__.py              # Top-level Python API (anonymize, Anonymizer, create_beeped_copy)
│   ├── cli.py                   # Command-line interface, presets, batch runner, --open / --play
│   ├── pipeline.py              # End-to-end orchestration & frame/rate/channel verification
│   ├── audio.py                 # Native-dtype WAV/MP3 I/O, speaker-matched 1 kHz beep & OS playback
│   ├── asr.py                   # Deterministic faster-whisper backend (lazy-loaded)
│   ├── cache.py                 # SHA-256 content-addressed ASR transcript cache
│   ├── textnorm.py              # Script normalization & cross-script phonetic key
│   ├── tokens.py                # Sub-word numeral splitter with proportional timing
│   ├── numbers.py               # Spoken digits, compounds, tens, repeaters, partial read-backs
│   ├── names.py                 # Cues, frames, honorifics, phonetic gazetteer, call memory
│   ├── spans.py                 # 10 ms RMS acoustic boundary aligner & span merger
│   ├── tts.py                   # Neural 2-speaker dialogue synthesizer for custom test speeches
│   ├── web.py                   # Zero-dependency local web audio player, mic recorder & TTS UI
│   └── languages/
│       ├── __init__.py          # YAML pack loader with lexicon collision checks
│       ├── en.yaml              # Shared English code-mix base pack
│       ├── hi.yaml              # Hindi language pack
│       ├── ta.yaml              # Tamil language pack (literary + colloquial spoken forms)
│       ├── kn.yaml              # Kannada language pack (3rd-language extensibility pack)
│       └── gazetteer_names.txt  # Pan-Indian & Tamil given names and surnames
├── samples/                     # Test calls (*.wav), primary outputs (*_redacted.wav), JSONs & truth
├── scripts/
│   ├── generate_samples.py      # Rebuilds the 4 two-speaker benchmark calls + ground truth
│   ├── make_speech.py           # Creates a new speech WAV from custom text & anonymizes it
│   └── evaluate.py              # Acoustic leak (ms) & collateral extra-beep (s) evaluator
└── tests/
    └── test_anonymizer.py       # 13 automated unit, audio-invariant, CLI & web-streaming tests
9. Adding Another Indian Language

All language-specific lexicons and cues live in declarative YAML files under pii_anonymizer/languages/—no edits to numbers.py, names.py, spans.py, or pipeline.py are needed:

Copy pii_anonymizer/languages/kn.yaml to pii_anonymizer/languages/<code>.yaml (e.g., mr.yaml for Marathi or te.yaml for Telugu) and keep include: [en] to inherit English code-mixed digits and cues automatically.
Populate numbers (digits, compounds, tens, hundred, repeat), keywords (mobile, pincode), fillers, and names (cues, weak_cues, frames, honorifics_before, honorifics_after, stopwords).
If the language uses a new Unicode script (e.g., Telugu or Bengali; 0 lines needed for Devanagari languages like Marathi), add a 3-line sanscript branch in pii_anonymizer/textnorm.py so cross-script phonetic matching works automatically against gazetteer_names.txt.
Run pytest to validate the pack for lexicon collisions, then anonymize calls with python -m pii_anonymizer call.wav --lang <code>.
4. Run `python -m pii_anonymizer call.wav --lang te`.
