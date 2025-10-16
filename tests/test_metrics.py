"""Tests for metrics."""

import pytest
from src.training.metrics import calculate_cer, calculate_wer, calculate_accuracy


def test_cer_exact_match():
    """Test CER with exact match."""
    predictions = ["مرحبا"]
    targets = ["مرحبا"]
    cer = calculate_cer(predictions, targets)
    assert cer == 0.0


def test_cer_complete_mismatch():
    """Test CER with complete mismatch."""
    predictions = ["xyz"]
    targets = ["abc"]
    cer = calculate_cer(predictions, targets)
    assert cer > 0


def test_wer_exact_match():
    """Test WER with exact match."""
    predictions = ["مرحبا بك"]
    targets = ["مرحبا بك"]
    wer = calculate_wer(predictions, targets)
    assert wer == 0.0


def test_accuracy():
    """Test accuracy calculation."""
    predictions = ["a", "b", "c", "d"]
    targets = ["a", "b", "x", "d"]
    acc = calculate_accuracy(predictions, targets)
    assert acc == 75.0


def test_empty_predictions():
    """Test with empty predictions."""
    predictions = [""]
    targets = ["مرحبا"]
    cer = calculate_cer(predictions, targets)
    assert cer == 100.0
