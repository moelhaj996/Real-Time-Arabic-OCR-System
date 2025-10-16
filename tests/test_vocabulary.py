"""Tests for vocabulary module."""

import pytest
from src.models.vocabulary import ArabicVocabulary


def test_vocabulary_creation():
    """Test vocabulary creation."""
    vocab = ArabicVocabulary()
    assert vocab.vocab_size > 0
    assert vocab.pad_idx == 0
    assert vocab.sos_idx > 0
    assert vocab.eos_idx > 0


def test_vocabulary_encoding():
    """Test text encoding."""
    vocab = ArabicVocabulary()
    text = "مرحبا"
    tokens = vocab.encode(text)
    assert len(tokens) > 0
    assert all(isinstance(t, int) for t in tokens)


def test_vocabulary_decoding():
    """Test text decoding."""
    vocab = ArabicVocabulary()
    text = "مرحبا"
    tokens = vocab.encode(text)
    decoded = vocab.decode(tokens, remove_special=False)
    assert decoded == text


def test_vocabulary_roundtrip():
    """Test encode-decode roundtrip."""
    vocab = ArabicVocabulary()
    text = "مرحبا بك في النظام"
    tokens = vocab.encode(text)
    decoded = vocab.decode(tokens, remove_special=False)
    assert decoded == text


def test_vocabulary_special_tokens():
    """Test special token handling."""
    vocab = ArabicVocabulary()
    text = "مرحبا"
    tokens = vocab.encode(text, add_sos=True, add_eos=True)

    assert tokens[0] == vocab.sos_idx
    assert tokens[-1] == vocab.eos_idx


def test_vocabulary_save_load(tmp_path):
    """Test saving and loading vocabulary."""
    vocab1 = ArabicVocabulary()
    vocab_path = tmp_path / "vocab.json"

    vocab1.save(str(vocab_path))
    vocab2 = ArabicVocabulary.load(str(vocab_path))

    assert vocab1.vocab_size == vocab2.vocab_size
    assert vocab1.char2idx == vocab2.char2idx
