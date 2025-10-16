"""Metrics for OCR evaluation."""

import torch
import numpy as np
from typing import List, Tuple
import editdistance


def calculate_cer(predictions: List[str], targets: List[str]) -> float:
    """
    Calculate Character Error Rate.

    Args:
        predictions: List of predicted texts
        targets: List of target texts

    Returns:
        CER as percentage
    """
    total_chars = 0
    total_errors = 0

    for pred, target in zip(predictions, targets):
        # Calculate edit distance
        distance = editdistance.eval(pred, target)
        total_errors += distance
        total_chars += len(target)

    if total_chars == 0:
        return 0.0

    cer = (total_errors / total_chars) * 100
    return cer


def calculate_wer(predictions: List[str], targets: List[str]) -> float:
    """
    Calculate Word Error Rate.

    Args:
        predictions: List of predicted texts
        targets: List of target texts

    Returns:
        WER as percentage
    """
    total_words = 0
    total_errors = 0

    for pred, target in zip(predictions, targets):
        # Split into words
        pred_words = pred.split()
        target_words = target.split()

        # Calculate edit distance on words
        distance = editdistance.eval(pred_words, target_words)
        total_errors += distance
        total_words += len(target_words)

    if total_words == 0:
        return 0.0

    wer = (total_errors / total_words) * 100
    return wer


def calculate_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Calculate exact match accuracy.

    Args:
        predictions: List of predicted texts
        targets: List of target texts

    Returns:
        Accuracy as percentage
    """
    correct = sum(1 for pred, target in zip(predictions, targets) if pred == target)
    accuracy = (correct / len(targets)) * 100
    return accuracy


class OCRMetrics:
    """Metrics tracker for OCR training."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.confidences = []

    def update(
        self,
        predictions: List[str],
        targets: List[str],
        confidences: List[float] = None,
    ):
        """
        Update metrics with new batch.

        Args:
            predictions: Predicted texts
            targets: Target texts
            confidences: Prediction confidences
        """
        self.predictions.extend(predictions)
        self.targets.extend(targets)

        if confidences is not None:
            self.confidences.extend(confidences)

    def compute(self) -> dict:
        """
        Compute all metrics.

        Returns:
            Dictionary of metrics
        """
        if not self.predictions:
            return {}

        metrics = {
            "cer": calculate_cer(self.predictions, self.targets),
            "wer": calculate_wer(self.predictions, self.targets),
            "accuracy": calculate_accuracy(self.predictions, self.targets),
        }

        if self.confidences:
            metrics["mean_confidence"] = np.mean(self.confidences)

        metrics["num_samples"] = len(self.predictions)

        return metrics

    def __repr__(self) -> str:
        """String representation."""
        metrics = self.compute()
        return (
            f"OCRMetrics(CER={metrics.get('cer', 0):.2f}%, "
            f"WER={metrics.get('wer', 0):.2f}%, "
            f"Acc={metrics.get('accuracy', 0):.2f}%)"
        )


def calculate_sequence_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int = 0,
) -> float:
    """
    Calculate token-level sequence accuracy.

    Args:
        predictions: Predicted tokens [B, T]
        targets: Target tokens [B, T]
        pad_idx: Padding index

    Returns:
        Accuracy as percentage
    """
    # Create mask for non-padding tokens
    mask = targets != pad_idx

    # Calculate accuracy
    correct = (predictions == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float()

    return accuracy.item() * 100


def calculate_confidence_metrics(
    confidences: List[float],
    predictions: List[str],
    targets: List[str],
    threshold: float = 0.8,
) -> dict:
    """
    Calculate confidence-based metrics.

    Args:
        confidences: Prediction confidences
        predictions: Predicted texts
        targets: Target texts
        threshold: Confidence threshold for filtering

    Returns:
        Dictionary of confidence metrics
    """
    confidences = np.array(confidences)

    # High confidence samples
    high_conf_mask = confidences >= threshold
    high_conf_correct = sum(
        1
        for conf, pred, target in zip(confidences, predictions, targets)
        if conf >= threshold and pred == target
    )

    high_conf_total = high_conf_mask.sum()

    metrics = {
        "mean_confidence": confidences.mean(),
        "median_confidence": np.median(confidences),
        "std_confidence": confidences.std(),
        "min_confidence": confidences.min(),
        "max_confidence": confidences.max(),
        "high_confidence_ratio": high_conf_total / len(confidences) if len(confidences) > 0 else 0,
        "high_confidence_accuracy": (
            high_conf_correct / high_conf_total if high_conf_total > 0 else 0
        ),
    }

    return metrics
