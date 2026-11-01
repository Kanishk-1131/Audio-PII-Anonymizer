"""On-disk cache of ASR results.

Speech recognition is >95% of the run time. The words recognised for a file
depend only on (audio bytes, model, language, decoding settings), so they are
cached under that key. Effects:

* Re-running the program on the same file is instant.
* Repeatability is guaranteed: a second run reuses exactly the same words.
* Tuning detection settings (padding, lexicons) never re-runs the ASR.

Delete the cache folder (or pass --no-cache) to force re-transcription.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .types import Word

CACHE_VERSION = 1


def default_cache_dir() -> Path:
    env = os.environ.get("PII_ANONYMIZER_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "pii_anonymizer"


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TranscriptCache:
    def __init__(self, directory: str | Path | None = None, enabled: bool = True):
        self.dir = Path(directory) if directory else default_cache_dir()
        self.enabled = enabled

    def key(self, audio_sha: str, settings: dict, language: str) -> str:
        blob = json.dumps({"v": CACHE_VERSION, "audio": audio_sha, "lang": language,
                           "asr": settings}, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def get(self, key: str) -> list[Word] | None:
        if not self.enabled:
            return None
        p = self.dir / f"{key}.json"
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return [Word.from_dict(d) for d in data["words"]]
        except Exception:
            return None  # corrupt entry -> recompute

    def put(self, key: str, words: list[Word], meta: dict) -> None:
        if not self.enabled:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f"{key}.tmp"
        tmp.write_text(json.dumps({"meta": meta, "words": [w.to_dict() for w in words]},
                                  ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.dir / f"{key}.json")
