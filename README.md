# Audio-PII-Anonymizer

A fast, deterministic command-line program and Python library that **produces a copy of an Indian-language call recording (`.wav` or `.mp3`) in which every spoken customer name, 10-digit mobile number, and 6-digit PIN code is covered by a loudness-matched 1 kHz beep** as its primary output, while leaving every non-PII audio sample untouched.


## Quick Start 

### Windows (PowerShell or Command Prompt)

```powershell
# 1. Open terminal in the project folder
cd "c:\Users\kanis\OneDrive\Desktop\internship 1st project"

# 2. One-time setup (creates .venv, installs packages, pre-caches the speech model)
.\setup.bat

# 3. Produce beeped recording copies in outputs\ (<name>_redacted.wav + <name>_redactions.json)
.\run.bat samples\sample_1_hindi.wav hi
.\run.bat samples\sample_2_tamil.wav ta

# 4. Produce AND open the beeped recording in Windows Media Player (--open) or play in terminal (--play)
.\run.bat samples\sample_1_hindi.wav hi --open
.\run.bat samples\sample_1_hindi.wav hi --play

# 5. Launch the interactive Browser Audio Player & Anonymizer Studio (http://127.0.0.1:8765)
#    Includes: Upload .wav/.mp3, Record Live Microphone, and Type Text -> Generate Speech
.\run_ui.bat

# 6. Run on an entire folder of recordings
.\run.bat my_hindi_calls\ hi
```

### Adding a New Speech (3 Easy Ways)

1. **Use any `.wav` or `.mp3` audio file** (recorded on your phone, Windows Sound Recorder, or a call-center sample):
   ```powershell
   .\run.bat "C:\path\to\your_speech.wav" hi --open
   .\run.bat "C:\path\to\tamil_call.mp3" ta --open
   ```
   *(Or open `.\run_ui.bat` -> **Upload .wav / .mp3 File** -> click **Anonymize & Play Beeped Copy**.)*

2. **Record your own voice live from your microphone**:
   - Run `.\run_ui.bat`, click **🎙 2. Record Your Voice (Mic)** -> **Start Microphone Recording**, speak in Hindi/Tamil/Kannada (include a name, mobile number, or PIN code), and click **Stop & Anonymize My Voice**.

3. **Type custom dialogue text to synthesize a new 2-speaker speech `.wav`**:
   - In `.\run_ui.bat`, click **✍ 3. Type Text to Create New Speech**, edit the `Agent:` / `Customer:` lines, and click **Generate Speech WAV & Anonymize**.
   - Or from PowerShell via [`scripts/make_speech.py`](scripts/make_speech.py):
     ```powershell
     .\.venv\Scripts\python.exe scripts\make_speech.py --lang hi --name my_call --anonymize --open --text "Agent: नमस्ते, अपना नाम बताइए।`nCustomer: मेरा नाम अमित पटेल है।`nAgent: धन्यवाद अमित जी, मोबाइल नंबर बताइए।`nCustomer: नौ आठ एक शून्य दो तीन चार पाँच छह सात।"
     ```

Or invoke the Python CLI directly:

```powershell
# Default: writes <stem>_redacted.wav (primary output) next to the input recording
.\.venv\Scripts\python.exe -m pii_anonymizer samples\sample_1_hindi.wav --lang hi
.\.venv\Scripts\python.exe -m pii_anonymizer samples\sample_2_tamil.wav --lang ta

# Custom primary output WAV path (and optional --no-json to emit ONLY the beeped audio copy)
.\.venv\Scripts\python.exe -m pii_anonymizer samples\sample_1_hindi.wav --lang hi -o outputs\hindi_beeped.wav
.\.venv\Scripts\python.exe -m pii_anonymizer samples\sample_1_hindi.wav --lang hi -o outputs\hindi_beeped.wav --no-json

# Whole folder -> custom output folder
.\.venv\Scripts\python.exe -m pii_anonymizer samples\ --lang hi -o outputs

# Installed console command (after activating .venv)
.\.venv\Scripts\Activate.ps1
pii-anonymizer call_001.wav --lang hi
```

### Python API (Returns the Beeped Recording Path)

```python
from pii_anonymizer import anonymize

beeped_wav = anonymize("samples/sample_1_hindi.wav", language="hi")
print("Primary output:", beeped_wav)  # samples/sample_1_hindi_redacted.wav
```

### Linux / macOS

```bash
chmod +x setup.sh run.sh
./setup.sh                                   # one-time setup
./run.sh samples/sample_1_hindi.wav hi       # writes outputs/sample_1_hindi_redacted.wav
./run.sh samples/sample_2_tamil.wav ta
```

---

## Speed & Accuracy Presets

Speech recognition accounts for >95% of runtime on CPU. Three built-in presets let you choose the right speed/model trade-off (`--preset` / `-p`), and an automatic **SHA-256 transcript cache** (`~/.cache/pii_anonymizer`) makes repeat runs instant (`< 0.1s`) without even loading Whisper into memory:

| Preset | Model & Decoding | CPU Speed (30s call, 8 cores) | Repeat Run (Cached) | PII Leak on Test Suite |
| :--- | :--- | :---: | :---: | :---: |
| **`fast`** *(default)* | `large-v3-turbo`, beam 5 (`int8`) | **~9 – 11 s** *(RTF 0.42)* | **~0.05 s** | **0 – 10 ms (0 leaks)** |
| **`balanced`** | `large-v3`, beam 1 (`int8`) | **~12 – 24 s** *(RTF 0.62)* | **~0.05 s** | **0 – 10 ms (0 leaks)** |
| **`accurate`** | `large-v3`, beam 5 (`int8`) | **~22 – 47 s** *(RTF 1.13)* | **~0.05 s** | **0 – 10 ms (0 leaks)** |

```powershell
# Fastest preset (default: ~10s per 30s call on CPU, ~1-2s on CUDA GPU)
.\.venv\Scripts\python.exe -m pii_anonymizer call_001.wav --lang hi --preset fast

# Full 32-layer Whisper large-v3 greedy or beam-5
.\.venv\Scripts\python.exe -m pii_anonymizer call_001.wav --lang hi --preset balanced
.\.venv\Scripts\python.exe -m pii_anonymizer call_001.wav --lang hi --preset accurate

# Force fresh speech recognition (bypass transcript cache)
.\.venv\Scripts\python.exe -m pii_anonymizer call_001.wav --lang hi --no-cache
```

---

## Output Format & Guarantees

For an input file `call_001.wav` (or `.mp3`), the program never modifies the original file; it produces:

1. **Primary Output — `call_001_redacted.wav`** *(or the custom `.wav` path given via `-o`)*
   - **Independent copy with exact length**: identical frame count, sample rate, and channel count as the input (`0.0 ms` duration drift, asserted after writing).
   - **Bit-identical non-PII audio**: for PCM/float WAV inputs, every sample outside a beep interval is `100%` bit-for-bit identical to the original recording.
   - **Irreversible, speaker-matched beep replacement**: inside each PII interval, original samples are completely overwritten (never mixed) with a steady 1 kHz sine tone matched to that speaker's local active speech RMS (`0.0 dB` loudness delta, clamped smoothly around call-wide speech RMS and capped at `-1 dBFS`), shaped with 5 ms raised-cosine ramps to prevent clicks.
2. **Companion Metadata — `call_001_redactions.json`** *(optional override via `--no-json`)*
   ```json
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
   ```

---

## Verification & Testing Commands

```powershell
# 1. Run unit & audio-invariant test suite (11 tests: Hindi, Tamil, Kannada, code-mix, MP3, cache, bit-exactness, primary beep copy)
.\.venv\Scripts\python.exe -m pytest -v -W ignore::DeprecationWarning

# 2. Score redacted audio against exact segment-level ground truth (checks LEAK ms, EXTRA s, order, length)
.\.venv\Scripts\python.exe scripts\evaluate.py samples samples

# 3. List installed language packs
.\.venv\Scripts\python.exe -m pii_anonymizer --list-languages
```

---

## Repository Layout

```text
.
├── setup.bat / run.bat          # One-command Windows setup & runner
├── setup.sh  / run.sh           # One-command Linux/macOS setup & runner
├── pii_anonymizer/
│   ├── cli.py                   # CLI parser, presets, batch folder runner
│   ├── pipeline.py              # End-to-end pipeline & frame-exact output check
│   ├── asr.py                   # Lazy-loaded faster-whisper backend (deterministic)
│   ├── cache.py                 # Content-addressed SHA-256 ASR transcript cache
│   ├── textnorm.py              # Script-aware normalization & cross-script phonetic key
│   ├── tokens.py                # Sub-word tokenizer with proportional timing
│   ├── numbers.py               # Spoken digits, compounds, tens, double/triple, read-backs
│   ├── names.py                 # Cues, frames, honorifics, phonetic gazetteer, call memory
│   ├── spans.py                 # 10 ms RMS acoustic boundary alignment & span merging
│   ├── audio.py                 # Native-dtype WAV/MP3 I/O & loudness-matched beep generator
│   └── languages/
│       ├── en.yaml              # Shared English code-mix base pack
│       ├── hi.yaml              # Hindi pack
│       ├── ta.yaml              # Tamil pack (literary + colloquial spoken forms)
│       ├── kn.yaml              # Kannada pack (3rd-language extensibility example)
│       └── gazetteer_names.txt  # Pan-Indian & Tamil given names + surnames
├── samples/                     # Test recordings, *_expected.json truth, *_redacted.wav & *_redactions.json
├── scripts/
│   ├── generate_samples.py      # Generates two-speaker test calls + exact ground-truth timestamps
│   └── evaluate.py              # Measures leaked voiced PII (ms) & extra non-PII beeped (s)
├── tests/
│   └── test_anonymizer.py       # Automated pytest suite
```

---

## Adding Another Indian Language

Adding Telugu (`te`), Marathi (`mr`), or Bengali (`bn`) requires **one YAML file** (`pii_anonymizer/languages/<code>.yaml` with `include: [en]`) and **zero changes** to the detection engine:

1. Copy [`pii_anonymizer/languages/kn.yaml`](pii_anonymizer/languages/kn.yaml) to `pii_anonymizer/languages/te.yaml`.
2. Fill in `numbers` (`digits`, `compounds`, `tens`, `hundred`, `repeat`), `keywords` (`mobile`, `pincode`), `fillers`, and `names` (`cues`, `weak_cues`, `gazetteer_cues`, `frames`, `honorifics_before`, `honorifics_after`, `stopwords`).
3. If the language uses a new Indic script, add a 3-line `indic_transliteration.sanscript` mapping in [`pii_anonymizer/textnorm.py`](pii_anonymizer/textnorm.py) so cross-script phonetic name matching works automatically against `gazetteer_names.txt`.
4. Run `python -m pii_anonymizer call.wav --lang te`.
