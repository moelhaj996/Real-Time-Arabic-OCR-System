"""PyTorch Dataset for Arabic OCR."""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from PIL import Image
import numpy as np
from typing import Optional, Callable, List, Dict, Tuple
import logging

from .preprocessing import ImagePreprocessor
from .augmentation import ArabicTextAugmentation

logger = logging.getLogger(__name__)


class ArabicOCRDataset(Dataset):
    """Dataset for Arabic OCR training."""

    def __init__(
        self,
        data_dir: str,
        annotations_file: str,
        vocabulary,
        transform: Optional[Callable] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
        max_seq_length: int = 256,
        training: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing images
            annotations_file: Path to annotations JSON file
            vocabulary: Vocabulary object
            transform: Optional transform to apply
            preprocessor: Image preprocessor
            max_seq_length: Maximum sequence length
            training: Training mode (enables augmentation)
        """
        self.data_dir = Path(data_dir)
        self.vocabulary = vocabulary
        self.transform = transform
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.max_seq_length = max_seq_length
        self.training = training

        # Load annotations
        with open(annotations_file, "r", encoding="utf-8") as f:
            self.annotations = json.load(f)

        logger.info(f"Loaded {len(self.annotations)} samples from {annotations_file}")

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item by index.

        Args:
            idx: Item index

        Returns:
            Dictionary with image, tokens, text, etc.
        """
        # Get annotation
        ann = self.annotations[idx]

        # Load image
        if ann.get("image_filename"):
            image_path = self.data_dir / "images" / ann["image_filename"]
            image = Image.open(image_path).convert("RGB")
            image = np.array(image)
        else:
            # Handle case where image is embedded in annotations
            raise ValueError("Image data not found in annotation")

        # Apply augmentation (training only)
        if self.training and self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Preprocess
        image = self.preprocessor(image)

        # Convert to tensor [3, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Get text and encode
        text = ann["text"]
        tokens = self.vocabulary.encode(text, add_sos=True, add_eos=True)

        # Pad/truncate to max length
        if len(tokens) > self.max_seq_length:
            tokens = tokens[: self.max_seq_length]
        else:
            tokens = tokens + [self.vocabulary.pad_idx] * (
                self.max_seq_length - len(tokens)
            )

        tokens = torch.tensor(tokens, dtype=torch.long)

        return {
            "image": image,
            "tokens": tokens,
            "text": text,
            "length": len(text),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Args:
        batch: List of samples

    Returns:
        Batched dictionary
    """
    images = torch.stack([item["image"] for item in batch])
    tokens = torch.stack([item["tokens"] for item in batch])
    texts = [item["text"] for item in batch]
    lengths = torch.tensor([item["length"] for item in batch])

    return {
        "images": images,
        "tokens": tokens,
        "texts": texts,
        "lengths": lengths,
    }


def create_dataloaders(
    train_data_dir: str,
    val_data_dir: str,
    vocabulary,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (384, 384),
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        train_data_dir: Training data directory
        val_data_dir: Validation data directory
        vocabulary: Vocabulary object
        batch_size: Batch size
        num_workers: Number of workers
        img_size: Target image size

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create augmentation for training
    train_transform = ArabicTextAugmentation(p=0.8)

    # Create datasets
    train_dataset = ArabicOCRDataset(
        data_dir=train_data_dir,
        annotations_file=str(Path(train_data_dir) / "annotations.json"),
        vocabulary=vocabulary,
        transform=train_transform,
        training=True,
    )

    val_dataset = ArabicOCRDataset(
        data_dir=val_data_dir,
        annotations_file=str(Path(val_data_dir) / "annotations.json"),
        vocabulary=vocabulary,
        transform=None,
        training=False,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader


class TextImageDataset(Dataset):
    """Simple dataset for text-image pairs (for quick testing)."""

    def __init__(
        self,
        texts: List[str],
        images: List[np.ndarray],
        vocabulary,
        preprocessor: Optional[ImagePreprocessor] = None,
        max_seq_length: int = 256,
    ):
        """
        Initialize simple dataset.

        Args:
            texts: List of text strings
            images: List of images
            vocabulary: Vocabulary object
            preprocessor: Image preprocessor
            max_seq_length: Maximum sequence length
        """
        assert len(texts) == len(images), "Texts and images must have same length"

        self.texts = texts
        self.images = images
        self.vocabulary = vocabulary
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get image and preprocess
        image = self.images[idx]
        image = self.preprocessor(image)
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Encode text
        text = self.texts[idx]
        tokens = self.vocabulary.encode(text, add_sos=True, add_eos=True)

        # Pad
        if len(tokens) > self.max_seq_length:
            tokens = tokens[: self.max_seq_length]
        else:
            tokens = tokens + [self.vocabulary.pad_idx] * (
                self.max_seq_length - len(tokens)
            )

        tokens = torch.tensor(tokens, dtype=torch.long)

        return {
            "image": image,
            "tokens": tokens,
            "text": text,
            "length": len(text),
        }
