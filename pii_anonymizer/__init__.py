"""Audio PII Anonymizer - produces a copy of an Indian-language call recording
in which customer names, mobile numbers, and PIN codes are covered by a beep."""

from .audio import create_beeped_copy
from .pipeline import Anonymizer, FileResult, PipelineConfig, anonymize

__version__ = "1.0.0"
__all__ = [
    "Anonymizer",
    "FileResult",
    "PipelineConfig",
    "anonymize",
    "create_beeped_copy",
    "__version__",
]
