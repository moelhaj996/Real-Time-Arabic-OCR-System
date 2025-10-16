"""Image preprocessing for Arabic OCR."""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing."""

    target_size: Tuple[int, int] = (384, 384)
    maintain_aspect_ratio: bool = True
    use_clahe: bool = True
    clip_limit: float = 2.0
    tile_grid_size: Tuple[int, int] = (8, 8)
    denoise: bool = True
    denoise_h: float = 10.0
    deskew: bool = True
    binarize: bool = False
    normalize: bool = True
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class ImagePreprocessor:
    """Preprocessor for Arabic OCR images."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the preprocessor.

        Args:
            config: Preprocessing configuration. Uses defaults if None.
        """
        self.config = config or PreprocessingConfig()

    def __call__(
        self, image: np.ndarray, return_debug: bool = False
    ) -> np.ndarray | Dict[str, np.ndarray]:
        """
        Apply preprocessing pipeline to image.

        Args:
            image: Input image (BGR or grayscale)
            return_debug: If True, return dict with intermediate results

        Returns:
            Preprocessed image or dict with debug info
        """
        debug_images = {} if return_debug else None

        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if return_debug:
            debug_images["original"] = image.copy()

        # Denoise
        if self.config.denoise:
            image = self.denoise_image(image)
            if return_debug:
                debug_images["denoised"] = image.copy()

        # CLAHE enhancement
        if self.config.use_clahe:
            image = self.apply_clahe(image)
            if return_debug:
                debug_images["clahe"] = image.copy()

        # Deskew
        if self.config.deskew:
            image = self.deskew_image(image)
            if return_debug:
                debug_images["deskewed"] = image.copy()

        # Binarize
        if self.config.binarize:
            image = self.binarize_image(image)
            if return_debug:
                debug_images["binarized"] = image.copy()

        # Resize
        image = self.resize_image(image)
        if return_debug:
            debug_images["resized"] = image.copy()

        # Normalize
        if self.config.normalize:
            image = self.normalize_image(image)
            if return_debug:
                debug_images["normalized"] = image.copy()

        if return_debug:
            debug_images["final"] = image
            return debug_images

        return image

    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply non-local means denoising.

        Args:
            image: Input image

        Returns:
            Denoised image
        """
        return cv2.fastNlMeansDenoisingColored(
            image, None, self.config.denoise_h, self.config.denoise_h, 7, 21
        )

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            image: Input RGB image

        Returns:
            Enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clip_limit, tileGridSize=self.config.tile_grid_size
        )
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        # Convert back to RGB
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Deskew image using Hough transform.

        Args:
            image: Input image

        Returns:
            Deskewed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Threshold
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Find coordinates of non-zero pixels
        coords = np.column_stack(np.where(thresh > 0))

        if len(coords) < 100:
            return image

        # Calculate minimum area rectangle
        angle = cv2.minAreaRect(coords)[-1]

        # Adjust angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Rotate image if angle is significant
        if abs(angle) > 0.5:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(
                image,
                M,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )

        return image

    def binarize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Binarize image using adaptive thresholding.

        Args:
            image: Input image

        Returns:
            Binarized image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Convert back to RGB
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size.

        Args:
            image: Input image

        Returns:
            Resized image
        """
        target_h, target_w = self.config.target_size
        h, w = image.shape[:2]

        if not self.config.maintain_aspect_ratio:
            return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Calculate aspect ratio
        aspect = w / h

        if aspect > 1:
            # Width is larger
            new_w = target_w
            new_h = int(target_w / aspect)
        else:
            # Height is larger
            new_h = target_h
            new_w = int(target_h * aspect)

        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target size
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )

        return padded

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image using mean and std.

        Args:
            image: Input image (0-255)

        Returns:
            Normalized image
        """
        # Convert to float and scale to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Normalize
        mean = np.array(self.config.mean, dtype=np.float32)
        std = np.array(self.config.std, dtype=np.float32)
        image = (image - mean) / std

        return image

    def remove_borders(self, image: np.ndarray, threshold: int = 250) -> np.ndarray:
        """
        Remove white borders from image.

        Args:
            image: Input image
            threshold: Threshold for border detection

        Returns:
            Image with borders removed
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        # Find non-white pixels
        coords = np.argwhere(gray < threshold)

        if len(coords) == 0:
            return image

        # Get bounding box
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1

        # Crop
        return image[y0:y1, x0:x1]


def batch_preprocess(
    images: list[np.ndarray], config: Optional[PreprocessingConfig] = None
) -> np.ndarray:
    """
    Preprocess a batch of images.

    Args:
        images: List of input images
        config: Preprocessing configuration

    Returns:
        Batch of preprocessed images as numpy array
    """
    preprocessor = ImagePreprocessor(config)
    processed = [preprocessor(img) for img in images]
    return np.stack(processed, axis=0)
