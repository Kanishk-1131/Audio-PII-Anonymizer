"""Core data types shared by the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Word:
    """One word from the ASR with its time span (seconds)."""
    text: str
    start: float
    end: float
    probability: float = 1.0

    def to_dict(self) -> dict:
        return {"text": self.text, "start": round(self.start, 3),
                "end": round(self.end, 3), "probability": round(self.probability, 4)}

    @classmethod
    def from_dict(cls, d: dict) -> "Word":
        return cls(str(d["text"]), float(d["start"]), float(d["end"]),
                   float(d.get("probability", 1.0)))


@dataclass
class Token:
    """A normalised sub-word. One ASR word may yield several tokens."""
    raw: str            # text as written by the ASR (punctuation kept)
    norm: str           # textnorm.normalize(raw)
    start: float
    end: float
    word_index: int
    sentence_end: bool = False  # raw token carried . ? ! , etc.
    quantity: bool = False      # contains ₹ % : etc. -> never a phone digit


@dataclass
class Detection:
    """A PII mention expressed as an inclusive token range."""
    type: str                 # "name" | "mobile" | "pincode"
    first: int                # first token index
    last: int                 # last token index (inclusive)
    text: str                 # value reported in the JSON
    reason: str = ""          # why it was detected (debug / write-up)
    extra_before: list[int] = field(default_factory=list)
    extra_after: list[int] = field(default_factory=list)


@dataclass
class Redaction:
    start: float
    end: float
    type: str
    text: str

    def to_dict(self) -> dict:
        return {"start_sec": round(self.start, 3), "end_sec": round(self.end, 3),
                "type": self.type, "text": self.text}
