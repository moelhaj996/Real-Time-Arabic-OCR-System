"""Training modules for Arabic OCR."""

from .losses import HybridLoss, CTCLoss, CrossEntropyLoss
from .metrics import calculate_cer, calculate_wer, OCRMetrics

__all__ = [
    "HybridLoss",
    "CTCLoss",
    "CrossEntropyLoss",
    "calculate_cer",
    "calculate_wer",
    "OCRMetrics",
]
