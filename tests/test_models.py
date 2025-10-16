"""Tests for model modules."""

import pytest
import torch
from src.models import ArabicVocabulary, ArabicOCRModel


@pytest.fixture
def vocab():
    """Create vocabulary fixture."""
    return ArabicVocabulary()


@pytest.fixture
def model(vocab):
    """Create model fixture."""
    return ArabicOCRModel(
        vocab_size=vocab.vocab_size,
        encoder_type="vit",
        encoder_name="vit_base_patch16_384",
        d_model=768,
        decoder_layers=2,  # Smaller for testing
    )


def test_model_creation(model):
    """Test model creation."""
    assert model is not None
    assert model.vocab_size > 0


def test_model_forward(model, vocab):
    """Test forward pass."""
    batch_size = 2
    images = torch.randn(batch_size, 3, 384, 384)
    tokens = torch.randint(0, vocab.vocab_size, (batch_size, 20))

    logits = model(images, tokens)

    assert logits.shape == (batch_size, 20, vocab.vocab_size)


def test_model_greedy_decode(model, vocab):
    """Test greedy decoding."""
    batch_size = 2
    images = torch.randn(batch_size, 3, 384, 384)

    sequences = model.greedy_decode(
        images,
        sos_idx=vocab.sos_idx,
        eos_idx=vocab.eos_idx,
        max_length=50,
    )

    assert sequences.shape[0] == batch_size
    assert sequences.shape[1] <= 50


def test_model_parameter_count(model):
    """Test parameter counting."""
    total_params = model.get_num_parameters()
    trainable_params = model.get_num_parameters(trainable_only=True)

    assert total_params > 0
    assert trainable_params > 0
    assert trainable_params <= total_params


def test_model_freeze_unfreeze(model):
    """Test encoder freezing."""
    # Initial state
    initial_trainable = model.get_num_parameters(trainable_only=True)

    # Freeze
    model.freeze_encoder()
    frozen_trainable = model.get_num_parameters(trainable_only=True)

    assert frozen_trainable < initial_trainable

    # Unfreeze
    model.unfreeze_encoder()
    unfrozen_trainable = model.get_num_parameters(trainable_only=True)

    assert unfrozen_trainable == initial_trainable
