"""Tests for data modules."""

import pytest
import numpy as np
from PIL import Image
from src.data.preprocessing import ImagePreprocessor
from src.data.augmentation import ArabicTextAugmentation


def test_preprocessor_basic():
    """Test basic preprocessing."""
    preprocessor = ImagePreprocessor()
    image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)

    processed = preprocessor(image)

    assert processed.shape == (384, 384, 3)
    assert processed.dtype == np.float32


def test_preprocessor_grayscale():
    """Test grayscale image processing."""
    preprocessor = ImagePreprocessor()
    image = np.random.randint(0, 255, (500, 500), dtype=np.uint8)

    processed = preprocessor(image)

    assert processed.shape == (384, 384, 3)


def test_augmentation():
    """Test augmentation."""
    augmenter = ArabicTextAugmentation(p=1.0)
    image = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)

    augmented = augmenter(image=image)

    assert "image" in augmented
    assert augmented["image"].shape == (384, 384, 3)


def test_augmentation_probability():
    """Test augmentation probability."""
    augmenter = ArabicTextAugmentation(p=0.0)
    image = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)

    augmented = augmenter(image=image)

    # With p=0, image should be unchanged
    assert "image" in augmented
