"""Synthetic Arabic text image generator."""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import arabic_reshaper
from bidi.algorithm import get_display
import json
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class SyntheticArabicGenerator:
    """Generate synthetic Arabic text images for training."""

    # Arabic character ranges
    ARABIC_LETTERS = "".join(chr(i) for i in range(0x0600, 0x06FF))
    ARABIC_SUPPLEMENT = "".join(chr(i) for i in range(0x0750, 0x077F))

    # Common Arabic words and phrases
    COMMON_WORDS = [
        "السلام", "عليكم", "الله", "محمد", "العربية", "الكتاب", "المدرسة",
        "البيت", "الطالب", "المعلم", "الدرس", "القلم", "الورقة", "الباب",
        "النافذة", "الشمس", "القمر", "النجم", "البحر", "الجبل", "الشجرة"
    ]

    def __init__(
        self,
        fonts_dir: str = "/System/Library/Fonts",  # macOS default
        output_size: Tuple[int, int] = (384, 384),
        min_font_size: int = 16,
        max_font_size: int = 48,
    ):
        """
        Initialize synthetic data generator.

        Args:
            fonts_dir: Directory containing Arabic fonts
            output_size: Output image size (H, W)
            min_font_size: Minimum font size
            max_font_size: Maximum font size
        """
        self.fonts_dir = Path(fonts_dir)
        self.output_size = output_size
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size

        # Load fonts
        self.fonts = self._load_arabic_fonts()
        if not self.fonts:
            logger.warning("No Arabic fonts found. Using default font.")
            self.fonts = [None]  # Will use default PIL font

    def _load_arabic_fonts(self) -> List[Path]:
        """
        Load Arabic font files.

        Returns:
            List of font file paths
        """
        fonts = []

        # Common Arabic font names
        arabic_font_patterns = [
            "*Arabic*", "*Geeza*", "*Baghdad*", "*Kufi*", "*Naskh*",
            "*Thuluth*", "*Diwani*", "*Farisi*", "*DecoType*"
        ]

        for pattern in arabic_font_patterns:
            fonts.extend(self.fonts_dir.glob(f"**/{pattern}.ttf"))
            fonts.extend(self.fonts_dir.glob(f"**/{pattern}.ttc"))

        return list(set(fonts))[:50]  # Limit to 50 fonts

    def generate_text(
        self,
        min_words: int = 1,
        max_words: int = 10,
        include_diacritics: bool = False
    ) -> str:
        """
        Generate random Arabic text.

        Args:
            min_words: Minimum number of words
            max_words: Maximum number of words
            include_diacritics: Whether to include diacritics

        Returns:
            Generated Arabic text
        """
        num_words = random.randint(min_words, max_words)
        words = random.choices(self.COMMON_WORDS, k=num_words)
        text = " ".join(words)

        if include_diacritics:
            text = self._add_diacritics(text)

        return text

    def _add_diacritics(self, text: str) -> str:
        """
        Add random diacritics to text.

        Args:
            text: Input text

        Returns:
            Text with diacritics
        """
        diacritics = ["\u064B", "\u064C", "\u064D", "\u064E", "\u064F", "\u0650"]
        result = []

        for char in text:
            result.append(char)
            if random.random() < 0.3 and char in self.ARABIC_LETTERS:
                result.append(random.choice(diacritics))

        return "".join(result)

    def render_text(
        self,
        text: str,
        font_size: Optional[int] = None,
        font_path: Optional[Path] = None,
        text_color: Tuple[int, int, int] = (0, 0, 0),
        bg_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> Tuple[np.ndarray, str]:
        """
        Render Arabic text to image.

        Args:
            text: Arabic text to render
            font_size: Font size (random if None)
            font_path: Path to font file (random if None)
            text_color: RGB text color
            bg_color: RGB background color

        Returns:
            Tuple of (rendered image, reshaped text)
        """
        # Reshape Arabic text for proper display
        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped_text)

        # Random font and size
        if font_size is None:
            font_size = random.randint(self.min_font_size, self.max_font_size)

        if font_path is None and self.fonts and self.fonts[0] is not None:
            font_path = random.choice(self.fonts)

        # Load font
        try:
            if font_path:
                font = ImageFont.truetype(str(font_path), font_size)
            else:
                font = ImageFont.load_default()
        except Exception as e:
            logger.warning(f"Failed to load font {font_path}: {e}")
            font = ImageFont.load_default()

        # Create image
        img = Image.new("RGB", self.output_size, bg_color)
        draw = ImageDraw.Draw(img)

        # Get text bounding box
        try:
            bbox = draw.textbbox((0, 0), bidi_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            # Fallback for older PIL versions
            text_width, text_height = draw.textsize(bidi_text, font=font)

        # Center text
        x = (self.output_size[1] - text_width) // 2
        y = (self.output_size[0] - text_height) // 2

        # Draw text
        draw.text((x, y), bidi_text, font=font, fill=text_color)

        # Convert to numpy array
        img_array = np.array(img)

        return img_array, text

    def generate_sample(
        self,
        apply_augmentation: bool = True,
        **kwargs
    ) -> Tuple[np.ndarray, str, Dict]:
        """
        Generate a single training sample.

        Args:
            apply_augmentation: Whether to apply augmentations
            **kwargs: Additional arguments for text generation

        Returns:
            Tuple of (image, text, metadata)
        """
        # Generate text
        text = self.generate_text(**kwargs)

        # Random colors
        text_color = tuple(random.randint(0, 50) for _ in range(3))
        bg_color = tuple(random.randint(200, 255) for _ in range(3))

        # Render
        image, text = self.render_text(text, text_color=text_color, bg_color=bg_color)

        # Apply augmentations
        if apply_augmentation:
            image = self._apply_transformations(image)

        # Metadata
        metadata = {
            "text": text,
            "length": len(text),
            "num_words": len(text.split()),
            "image_shape": image.shape,
        }

        return image, text, metadata

    def _apply_transformations(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random transformations to image.

        Args:
            image: Input image

        Returns:
            Transformed image
        """
        # Rotation
        if random.random() < 0.5:
            angle = random.uniform(-10, 10)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))

        # Blur
        if random.random() < 0.3:
            kernel_size = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        # Noise
        if random.random() < 0.3:
            noise = np.random.normal(0, 10, image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)

        # Brightness
        if random.random() < 0.5:
            factor = random.uniform(0.7, 1.3)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)

        return image

    def generate_dataset(
        self,
        num_samples: int,
        output_dir: str,
        split: str = "train",
        save_images: bool = True,
        **kwargs
    ) -> None:
        """
        Generate a complete dataset.

        Args:
            num_samples: Number of samples to generate
            output_dir: Output directory
            split: Dataset split (train/val/test)
            save_images: Whether to save images to disk
            **kwargs: Additional arguments for sample generation
        """
        output_path = Path(output_dir) / split
        output_path.mkdir(parents=True, exist_ok=True)

        images_dir = output_path / "images"
        images_dir.mkdir(exist_ok=True)

        annotations = []

        logger.info(f"Generating {num_samples} samples for {split} split...")

        for i in tqdm(range(num_samples)):
            # Generate sample
            image, text, metadata = self.generate_sample(**kwargs)

            # Save image
            if save_images:
                image_filename = f"{split}_{i:06d}.jpg"
                image_path = images_dir / image_filename
                cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            else:
                image_filename = None

            # Add to annotations
            annotations.append({
                "image_id": i,
                "image_filename": image_filename,
                "text": text,
                "metadata": metadata,
            })

        # Save annotations
        annotations_file = output_path / "annotations.json"
        with open(annotations_file, "w", encoding="utf-8") as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)

        logger.info(f"Dataset saved to {output_path}")
        logger.info(f"Annotations: {annotations_file}")


if __name__ == "__main__":
    # Example usage
    generator = SyntheticArabicGenerator()

    # Generate training set
    generator.generate_dataset(
        num_samples=1000,
        output_dir="data/augmented",
        split="train",
        min_words=1,
        max_words=5,
        include_diacritics=True
    )
