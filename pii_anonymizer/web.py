"""Zero-dependency local browser UI for listening to, recording, generating,
and running the Audio PII Anonymizer.

Usage:
    python -m pii_anonymizer.web
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
import threading
import time
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .asr import DEFAULT_PRESET, PRESETS, FasterWhisperASR
from .cache import TranscriptCache
from .languages import available_languages, load_language
from .pipeline import AUDIO_EXTENSIONS, Anonymizer, PipelineConfig, list_audio_files
from .tts import synthesize_dialogue

ROOT_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT_DIR / "samples"
OUTPUTS_DIR = ROOT_DIR / "outputs"
UPLOADS_DIR = OUTPUTS_DIR / "_uploads"

_KNOWN_LABELS = {
    "sample_1_hindi.wav": ("hi", "Sample 1 — Hindi (Brief)"),
    "sample_2_tamil.wav": ("ta", "Sample 2 — Tamil (Brief)"),
    "sample_3_hindi_codemix.wav": ("hi", "Sample 3 — Hindi Code-Mix"),
    "sample_4_tamil_colloquial.wav": ("ta", "Sample 4 — Tamil Colloquial"),
}

_LOCK = threading.Lock()
_ASR_CACHE: dict[str, FasterWhisperASR] = {}


def _list_samples() -> list[dict[str, str]]:
    if not SAMPLES_DIR.is_dir():
        return []
    out: list[dict[str, str]] = []
    for p in list_audio_files(SAMPLES_DIR):
        if p.name in _KNOWN_LABELS:
            lang, label = _KNOWN_LABELS[p.name]
        else:
            low = p.stem.lower()
            lang = "ta" if ("ta" in low or "tamil" in low) else ("kn" if ("kn" in low or "kannada" in low) else "hi")
            label = f"{p.stem} ({lang})"
        out.append({"file": p.name, "lang": lang, "label": label})
    return out


def _get_anonymizer(preset: str) -> Anonymizer:
    p = PRESETS.get(preset, PRESETS[DEFAULT_PRESET])
    key = f"{p['model']}:{p['beam_size']}"
    if key not in _ASR_CACHE:
        _ASR_CACHE[key] = FasterWhisperASR(model=p["model"], beam_size=p["beam_size"])
    return Anonymizer(_ASR_CACHE[key], PipelineConfig(), cache=TranscriptCache(enabled=True))


_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Audio PII Anonymizer — Beeped Audio Studio</title>
<style>
  :root {
    --bg: #0f172a;
    --panel: #1e293b;
    --panel2: #0b1220;
    --border: #334155;
    --text: #f8fafc;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --primary: #10b981;
    --beep: #f43f5e;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.5;
  }
  .wrap { max-width: 1020px; margin: 0 auto; padding: 26px 20px 60px; }
  h1 { margin: 0 0 6px; font-size: 1.65rem; }
  .sub { color: var(--muted); margin: 0 0 20px; font-size: 0.95rem; }
  .card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 14px; padding: 20px; margin-bottom: 18px;
  }
  .card.primary-card {
    border: 2px solid var(--primary);
    box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.12);
  }
  .badge {
    display: inline-block; font-size: 0.75rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.05em;
    padding: 3px 9px; border-radius: 999px;
    background: rgba(16, 185, 129, 0.18); color: #34d399;
    margin-bottom: 8px;
  }
  .samples { display: flex; flex-wrap: wrap; gap: 9px; margin-bottom: 14px; }
  button, .btn {
    cursor: pointer; border: 1px solid var(--border); background: #273549;
    color: var(--text); padding: 9px 14px; border-radius: 9px;
    font-size: 0.89rem; font-weight: 600; text-decoration: none;
    display: inline-flex; align-items: center; gap: 6px;
  }
  button:hover, .btn:hover { border-color: var(--accent); background: #334155; }
  button.run-btn {
    background: var(--primary); border-color: var(--primary); color: #042f2e;
  }
  button.run-btn:hover { filter: brightness(1.08); }
  button.rec-btn {
    background: #dc2626; border-color: #ef4444; color: #fff;
  }
  .tabs { display: flex; gap: 8px; margin: 14px 0 12px; flex-wrap: wrap; }
  .tab {
    background: var(--panel2); border: 1px solid var(--border);
    color: var(--muted); padding: 8px 14px; border-radius: 9px;
  }
  .tab.active {
    background: #1d4ed8; border-color: #3b82f6; color: #fff;
  }
  .tab-pane { display: none; padding-top: 6px; }
  .tab-pane.active { display: block; }
  .row { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
  select, input[type="file"], input[type="text"], textarea {
    background: var(--panel2); color: var(--text);
    border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px;
    font-family: inherit; font-size: 0.9rem;
  }
  textarea { width: 100%; min-height: 118px; margin: 10px 0; resize: vertical; }
  audio { width: 100%; margin-top: 10px; height: 46px; }
  .timeline {
    position: relative; height: 22px; background: var(--panel2);
    border-radius: 6px; overflow: hidden; margin: 14px 0 8px;
    border: 1px solid var(--border); cursor: pointer;
  }
  .beep-bar { position: absolute; top: 0; bottom: 0; background: var(--beep); opacity: 0.88; }
  .playhead { position: absolute; top: 0; bottom: 0; width: 2px; background: #fff; pointer-events: none; }
  .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
  .chip {
    background: rgba(244, 63, 94, 0.14); border: 1px solid rgba(244, 63, 94, 0.45);
    color: #fda4af; font-size: 0.84rem; padding: 6px 11px; border-radius: 8px;
  }
  .chip:hover { background: rgba(244, 63, 94, 0.28); border-color: var(--beep); }
  pre {
    background: var(--panel2); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px; overflow-x: auto;
    font-size: 0.84rem; color: #cbd5e1; margin: 10px 0 0;
  }
  .status { color: var(--accent); font-weight: 600; font-size: 0.92rem; }
  .meta { color: var(--muted); font-size: 0.88rem; margin-top: 4px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Audio PII Anonymizer</h1>
  <p class="sub">Produces a copy of the recording with customer names, mobile numbers, and PIN codes covered by a 1 kHz beep.</p>

  <div class="card">
    <div style="font-weight:700; margin-bottom:8px;">Saved Recordings in <code>samples/</code> (1-Click Play)</div>
    <div class="samples" id="sampleBtns"></div>

    <div class="row" style="margin-top:8px;">
      <label>Language:
        <select id="langSelect" onchange="maybeUpdateTemplate()"></select>
      </label>
      <label>Speed Preset:
        <select id="presetSelect">
          <option value="fast" selected>fast (large-v3-turbo)</option>
          <option value="balanced">balanced (large-v3 beam 1)</option>
          <option value="accurate">accurate (large-v3 beam 5)</option>
        </select>
      </label>
      <span class="status" id="statusMsg"></span>
    </div>

    <div class="tabs">
      <button class="tab active" onclick="switchTab('upload', this)">📁 1. Upload .wav / .mp3 File</button>
      <button class="tab" onclick="switchTab('mic', this)">🎙 2. Record Your Voice (Mic)</button>
      <button class="tab" onclick="switchTab('tts', this)">✍ 3. Type Text to Create New Speech</button>
    </div>

    <!-- Tab 1: Upload -->
    <div class="tab-pane active" id="pane-upload">
      <div class="row">
        <input type="file" id="fileInput" accept=".wav,.mp3,audio/*" />
        <button class="run-btn" onclick="runUpload()">Anonymize &amp; Play Beeped Copy</button>
      </div>
    </div>

    <!-- Tab 2: Live Microphone -->
    <div class="tab-pane" id="pane-mic">
      <div class="meta" style="margin-bottom:10px;">
        Speak a sentence in the selected language containing a name, 10-digit mobile number, or 6-digit PIN code:
      </div>
      <div class="row">
        <button id="micStartBtn" class="run-btn" onclick="startMic()">🎙 Start Microphone Recording</button>
        <button id="micStopBtn" class="rec-btn" style="display:none;" onclick="stopMicAndAnonymize()">⏹ Stop &amp; Anonymize My Voice</button>
        <span class="meta" id="micTimer"></span>
      </div>
    </div>

    <!-- Tab 3: Text-to-Speech Generator -->
    <div class="tab-pane" id="pane-tts">
      <div class="row" style="justify-content:space-between;">
        <span class="meta">Write <code>Agent: ...</code> and <code>Customer: ...</code> lines to synthesize a realistic 2-speaker call in <code>samples/</code>:</span>
        <div class="row">
          <button onclick="fillTemplate('hi')">Load Hindi Example</button>
          <button onclick="fillTemplate('ta')">Load Tamil Example</button>
          <button onclick="fillTemplate('kn')">Load Kannada Example</button>
        </div>
      </div>
      <textarea id="ttsScript"></textarea>
      <div class="row">
        <label>Save as:
          <input type="text" id="ttsName" value="my_new_speech" placeholder="my_new_speech" />
        </label>
        <button class="run-btn" onclick="runCreateSpeech()">🔊 Generate Speech WAV &amp; Anonymize</button>
      </div>
    </div>
  </div>

  <div id="resultArea" style="display:none;">
    <div class="card primary-card">
      <span class="badge">Primary Output — Beeped Recording Copy</span>
      <div class="row" style="justify-content:space-between;">
        <div>
          <div style="font-size:1.15rem; font-weight:700;" id="outTitle"></div>
          <div class="meta" id="outMeta"></div>
        </div>
        <a class="btn" id="downloadBtn" download>⬇ Download Beeped WAV</a>
      </div>

      <audio id="redactedAudio" controls preload="auto"></audio>

      <div class="timeline" id="timeline" title="Red segments = 1 kHz PII Beeps (click anywhere to seek)">
        <div class="playhead" id="playhead" style="left:0%"></div>
      </div>
      <div class="meta">Click any detected PII span below to jump right before that beep and hear it:</div>
      <div class="chips" id="beepChips"></div>
    </div>

    <div class="card">
      <div style="font-weight:700;">Compare with Original Unredacted Recording</div>
      <div class="meta" id="origName"></div>
      <audio id="origAudio" controls preload="metadata"></audio>
    </div>

    <div class="card">
      <div style="font-weight:700;">Companion Redaction Metadata JSON</div>
      <pre id="jsonBox"></pre>
    </div>
  </div>
</div>

<script>
const TEMPLATES = {
  hi: `Agent: नमस्ते, अपना नाम बताइए।
Customer: मेरा नाम अमित पटेल है।
Agent: धन्यवाद अमित जी। आपका मोबाइल नंबर बताइए।
Customer: नौ आठ एक शून्य, दो तीन चार, पाँच छह सात।
Agent: ठीक है, और आपका पिन कोड क्या है?
Customer: चार शून्य शून्य शून्य शून्य एक।`,
  ta: `Agent: வணக்கம். உங்க பெயர் என்ன?
Customer: என் பெயர் சுரேஷ் குமார்.
Agent: நன்றி சுரேஷ். உங்க mobile number சொல்லுங்க.
Customer: ஒன்பது எட்டு நாலு பூஜ்யம், double அஞ்சு, ஒன்னு ரெண்டு மூணு ஆறு.
Agent: சரி, உங்க pin code சொல்லுங்க.
Customer: six zero zero zero two eight.`,
  kn: `Agent: ನಮಸ್ಕಾರ, ನಿಮ್ಮ ಹೆಸರು ಹೇಳಿ.
Customer: ನನ್ನ ಹೆಸರು ರಮೇಶ್ ಶರ್ಮಾ.
Agent: ಧನ್ಯವಾದ ರಮೇಶ್. ನಿಮ್ಮ ಮೊಬೈಲ್ ನಂಬರ್ ಹೇಳಿ.
Customer: ಒಂಬತ್ತು ಎಂಟು ನಾಲ್ಕು ಐದು ಸೊನ್ನೆ ಒಂದು ಎರಡು ಮೂರು ನಾಲ್ಕು ಐದು.
Agent: ನಿಮ್ಮ ಪಿನ್ ಕೋಡ್ ಹೇಳಿ.
Customer: ಐದು ಆರು ಸೊನ್ನೆ ಸೊನ್ನೆ ಮೂರು ನಾಲ್ಕು.`
};

let currentDuration = 1;
const redAudio = document.getElementById('redactedAudio');
const origAudio = document.getElementById('origAudio');

redAudio.addEventListener('play', () => origAudio.pause());
origAudio.addEventListener('play', () => redAudio.pause());
redAudio.addEventListener('timeupdate', () => {
  const pct = Math.min(100, (redAudio.currentTime / Math.max(currentDuration, 0.01)) * 100);
  document.getElementById('playhead').style.left = pct + '%';
});
document.getElementById('timeline').addEventListener('click', (e) => {
  const rect = e.currentTarget.getBoundingClientRect();
  const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  redAudio.currentTime = frac * currentDuration;
  redAudio.play();
});

function switchTab(id, btn) {
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('pane-' + id).classList.add('active');
}

function fillTemplate(code) {
  document.getElementById('langSelect').value = code;
  document.getElementById('ttsScript').value = TEMPLATES[code] || TEMPLATES.hi;
}
function maybeUpdateTemplate() {
  const ta = document.getElementById('ttsScript');
  if (!ta.value.trim()) fillTemplate(document.getElementById('langSelect').value);
}

async function refreshSamples(autoRunFirst = false) {
  const r = await fetch('/api/info');
  const info = await r.json();
  const langSel = document.getElementById('langSelect');
  if (!langSel.options.length) {
    info.languages.forEach(l => {
      const o = document.createElement('option');
      o.value = l.code; o.textContent = `${l.code} — ${l.name}`;
      langSel.appendChild(o);
    });
    fillTemplate('hi');
  }
  const sb = document.getElementById('sampleBtns');
  sb.innerHTML = '';
  info.samples.forEach(s => {
    const b = document.createElement('button');
    b.textContent = '▶ ' + s.label;
    b.onclick = () => runSample(s.file, s.lang);
    sb.appendChild(b);
  });
  if (autoRunFirst && info.samples.length > 0) {
    runSample(info.samples[0].file, info.samples[0].lang, false);
  }
}

async function runSample(file, lang, autoPlay = true) {
  const st = document.getElementById('statusMsg');
  document.getElementById('langSelect').value = lang;
  st.textContent = 'Processing ' + file + ' ...';
  const preset = document.getElementById('presetSelect').value;
  const r = await fetch('/api/anonymize-sample', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({file, lang, preset})
  });
  const data = await r.json();
  if (!r.ok) { st.textContent = 'Error: ' + (data.error || r.statusText); return; }
  st.textContent = `Ready in ${data.elapsed_sec}s (${data.cached ? 'cached' : 'transcribed'})`;
  renderResult(data, autoPlay);
}

function bytesToBase64(bytes) {
  let bin = '';
  for (let i = 0; i < bytes.length; i += 8192) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 8192));
  }
  return btoa(bin);
}

async function sendUploadBytes(filename, bytes) {
  const st = document.getElementById('statusMsg');
  st.textContent = 'Anonymizing ' + filename + ' ...';
  const lang = document.getElementById('langSelect').value;
  const preset = document.getElementById('presetSelect').value;
  const r = await fetch('/api/anonymize-upload', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({filename, data_b64: bytesToBase64(bytes), lang, preset})
  });
  const data = await r.json();
  if (!r.ok) { st.textContent = 'Error: ' + (data.error || r.statusText); return; }
  st.textContent = `Anonymized in ${data.elapsed_sec}s`;
  renderResult(data, true);
}

async function runUpload() {
  const fi = document.getElementById('fileInput');
  if (!fi.files.length) { alert('Choose a .wav or .mp3 recording first.'); return; }
  const f = fi.files[0];
  const buf = await f.arrayBuffer();
  await sendUploadBytes(f.name, new Uint8Array(buf));
}

async function runCreateSpeech() {
  const script = document.getElementById('ttsScript').value.trim();
  const name = document.getElementById('ttsName').value.trim() || 'custom_speech';
  const lang = document.getElementById('langSelect').value;
  const preset = document.getElementById('presetSelect').value;
  if (!script) { alert('Enter dialogue text first.'); return; }
  const st = document.getElementById('statusMsg');
  st.textContent = 'Synthesizing new speech WAV & anonymizing ...';
  const r = await fetch('/api/create-speech', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name, script, lang, preset})
  });
  const data = await r.json();
  if (!r.ok) { st.textContent = 'Error: ' + (data.error || r.statusText); return; }
  st.textContent = `Created samples/${data.input_name} & anonymized in ${data.elapsed_sec}s`;
  await refreshSamples(false);
  renderResult(data, true);
}

// --- Live Microphone -> 16-bit PCM WAV encoder ---
let micCtx = null, micStream = null, micProc = null, micChunks = [], micStartTs = 0, micTick = null;

async function startMic() {
  try {
    micStream = await navigator.mediaDevices.getUserMedia({audio: true});
  } catch (err) {
    alert('Microphone permission denied or unavailable: ' + err.message);
    return;
  }
  micCtx = new (window.AudioContext || window.webkitAudioContext)({sampleRate: 16000});
  const src = micCtx.createMediaStreamSource(micStream);
  micProc = micCtx.createScriptProcessor(4096, 1, 1);
  micChunks = [];
  micProc.onaudioprocess = (e) => {
    micChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };
  src.connect(micProc);
  micProc.connect(micCtx.destination);
  micStartTs = Date.now();
  document.getElementById('micStartBtn').style.display = 'none';
  document.getElementById('micStopBtn').style.display = 'inline-flex';
  micTick = setInterval(() => {
    const s = ((Date.now() - micStartTs) / 1000).toFixed(1);
    document.getElementById('micTimer').textContent = `🔴 Recording... ${s}s`;
  }, 100);
}

function encodeWavPCM16(samples, sr) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buf);
  const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sr, true);
  view.setUint32(28, sr * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  let off = 44;
  for (let i = 0; i < samples.length; i++, off += 2) {
    const v = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, v < 0 ? v * 32768 : v * 32767, true);
  }
  return new Uint8Array(buf);
}

async function stopMicAndAnonymize() {
  clearInterval(micTick);
  document.getElementById('micTimer').textContent = '';
  document.getElementById('micStartBtn').style.display = 'inline-flex';
  document.getElementById('micStopBtn').style.display = 'none';
  const sr = micCtx ? micCtx.sampleRate : 16000;
  if (micProc) micProc.disconnect();
  if (micStream) micStream.getTracks().forEach(t => t.stop());
  if (micCtx) await micCtx.close();

  const total = micChunks.reduce((a, c) => a + c.length, 0);
  if (total < sr * 0.4) { alert('Recording too short — please speak for at least 1 second.'); return; }
  const merged = new Float32Array(total);
  let pos = 0;
  for (const c of micChunks) { merged.set(c, pos); pos += c.length; }
  const wavBytes = encodeWavPCM16(merged, sr);
  const fname = `mic_recording_${Date.now()}.wav`;
  await sendUploadBytes(fname, wavBytes);
}

function renderResult(data, autoPlay) {
  document.getElementById('resultArea').style.display = 'block';
  currentDuration = data.duration || 1;
  document.getElementById('outTitle').textContent = data.redacted_name;
  document.getElementById('outMeta').textContent =
    `${data.duration.toFixed(2)}s | ${data.samplerate} Hz | ${data.channels} ch | ${data.report.redactions.length} beep(s) applied`;
  document.getElementById('origName').textContent = data.input_name;

  const bust = '?t=' + Date.now();
  redAudio.src = data.redacted_url + bust;
  origAudio.src = data.input_url + bust;
  const dl = document.getElementById('downloadBtn');
  dl.href = data.redacted_url;
  dl.download = data.redacted_name;

  const tl = document.getElementById('timeline');
  tl.querySelectorAll('.beep-bar').forEach(el => el.remove());
  const chips = document.getElementById('beepChips');
  chips.innerHTML = '';

  data.report.redactions.forEach((red, idx) => {
    const bar = document.createElement('div');
    bar.className = 'beep-bar';
    bar.style.left = ((red.start_sec / currentDuration) * 100) + '%';
    bar.style.width = Math.max(0.6, ((red.end_sec - red.start_sec) / currentDuration) * 100) + '%';
    bar.title = `${red.start_sec.toFixed(2)}s - ${red.end_sec.toFixed(2)}s [${red.type}] ${red.text}`;
    tl.appendChild(bar);

    const chip = document.createElement('button');
    chip.className = 'chip';
    chip.textContent = `▶ #${idx + 1}  ${red.start_sec.toFixed(2)}s–${red.end_sec.toFixed(2)}s  ${red.type} (${red.text})`;
    chip.onclick = () => {
      redAudio.currentTime = Math.max(0, red.start_sec - 0.6);
      redAudio.play();
    };
    chips.appendChild(chip);
  });

  document.getElementById('jsonBox').textContent = JSON.stringify(data.report, null, 2);
  if (autoPlay) {
    redAudio.play().catch(() => {});
  }
}

refreshSamples(true);
</script>
</body>
</html>
"""

_RANGE_RE = re.compile(r"bytes=(\d+)-(\d*)")


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        return

    def _send_json(self, payload: dict, status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _serve_audio_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Audio file not found")
            return
        size = path.stat().st_size
        ctype = mimetypes.guess_type(path.name)[0] or "audio/wav"
        range_hdr = self.headers.get("Range", "")
        m = _RANGE_RE.match(range_hdr)
        if m:
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else size - 1
            end = min(end, size - 1)
            if start >= size or start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            length = end - start + 1
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            with path.open("rb") as fh:
                fh.seek(start)
                self.wfile.write(fh.read(length))
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(size))
        self.end_headers()
        with path.open("rb") as fh:
            self.wfile.write(fh.read())

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        route = urllib.parse.unquote(parsed.path)
        if route == "/":
            raw = _HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if route == "/api/info":
            langs = [
                {"code": c, "name": load_language(c).name}
                for c in available_languages()
                if c != "en"
            ]
            self._send_json({"languages": langs, "samples": _list_samples()})
            return
        if route.startswith("/media/samples/"):
            fname = Path(route.removeprefix("/media/samples/")).name
            self._serve_audio_file(SAMPLES_DIR / fname)
            return
        if route.startswith("/media/outputs/"):
            fname = Path(route.removeprefix("/media/outputs/")).name
            self._serve_audio_file(OUTPUTS_DIR / fname)
            return
        if route.startswith("/media/uploads/"):
            fname = Path(route.removeprefix("/media/uploads/")).name
            self._serve_audio_file(UPLOADS_DIR / fname)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        try:
            if self.path == "/api/anonymize-sample":
                fname = Path(body["file"]).name
                in_path = SAMPLES_DIR / fname
                if not in_path.is_file():
                    self._send_json({"error": f"Sample not found: {fname}"}, 404)
                    return
                res, elapsed, cached = self._run(in_path, body["lang"], body.get("preset", DEFAULT_PRESET))
                self._send_json(self._format_res(res, elapsed, cached, f"/media/samples/{in_path.name}"))
                return

            if self.path == "/api/anonymize-upload":
                raw_name = Path(body["filename"]).name
                suffix = Path(raw_name).suffix.lower()
                if suffix not in AUDIO_EXTENSIONS:
                    self._send_json({"error": "Only .wav and .mp3 files are supported"}, 400)
                    return
                UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
                in_path = UPLOADS_DIR / raw_name
                in_path.write_bytes(base64.b64decode(body["data_b64"]))
                res, elapsed, cached = self._run(in_path, body["lang"], body.get("preset", DEFAULT_PRESET))
                self._send_json(self._format_res(res, elapsed, cached, f"/media/uploads/{in_path.name}"))
                return

            if self.path == "/api/create-speech":
                raw_stem = re.sub(r"[^A-Za-z0-9_\-]", "_", Path(body.get("name", "custom_speech")).stem).strip("_")
                stem = raw_stem or "custom_speech"
                lang = body["lang"]
                script = body["script"]
                SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
                in_path = SAMPLES_DIR / f"{stem}.wav"
                t0 = time.perf_counter()
                synthesize_dialogue(script, lang, in_path)
                res, _, cached = self._run(in_path, lang, body.get("preset", DEFAULT_PRESET))
                total_elapsed = round(time.perf_counter() - t0, 2)
                self._send_json(self._format_res(res, total_elapsed, cached, f"/media/samples/{in_path.name}"))
                return
        except Exception as exc:
            self._send_json({"error": str(exc)}, 500)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    @staticmethod
    def _run(in_path: Path, lang: str, preset: str):
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            t0 = time.perf_counter()
            anon = _get_anonymizer(preset)
            res = anon.process_file(in_path, lang, out_dir=OUTPUTS_DIR)
            elapsed = round(time.perf_counter() - t0, 2)
            return res, elapsed, anon.last_asr_cached

    @staticmethod
    def _format_res(res, elapsed: float, cached: bool, input_url: str) -> dict:
        report = json.loads(res.json_path.read_text(encoding="utf-8"))
        return {
            "input_name": res.input_path.name,
            "input_url": input_url,
            "redacted_name": res.primary_output.name,
            "redacted_url": f"/media/outputs/{res.primary_output.name}",
            "duration": res.duration,
            "samplerate": res.samplerate,
            "channels": res.channels,
            "elapsed_sec": elapsed,
            "cached": cached,
            "report": report,
        }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Launch the browser audio player for Audio PII Anonymizer.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Audio PII Anonymizer Studio running at: {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
