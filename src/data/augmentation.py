"""Data augmentation for Arabic OCR."""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any, Optional
import random


class ArabicTextAugmentation:
    """Augmentation pipeline specifically designed for Arabic text images."""

    def __init__(
        self,
        rotation_limit: int = 15,
        brightness_range: tuple = (0.7, 1.3),
        contrast_range: tuple = (0.8, 1.2),
        blur_limit: int = 3,
        noise_var: float = 0.01,
        perspective_scale: float = 0.05,
        elastic_alpha: float = 1.0,
        elastic_sigma: float = 50.0,
        p: float = 0.8,
    ):
        """
        Initialize augmentation pipeline.

        Args:
            rotation_limit: Max rotation angle in degrees
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment
            blur_limit: Max blur kernel size
            noise_var: Variance for Gaussian noise
            perspective_scale: Scale for perspective transform
            elastic_alpha: Alpha for elastic deformation
            elastic_sigma: Sigma for elastic deformation
            p: Probability of applying augmentations
        """
        self.p = p

        # Create augmentation pipeline
        self.transform = A.Compose(
            [
                # Geometric transformations
                A.Rotate(limit=rotation_limit, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=255),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=rotation_limit,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=255,
                    p=0.3,
                ),
                A.Perspective(scale=perspective_scale, p=0.3, pad_val=255),
                A.ElasticTransform(
                    alpha=elastic_alpha,
                    sigma=elastic_sigma,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=255,
                    p=0.2,
                ),
                # Color transformations
                A.RandomBrightnessContrast(
                    brightness_limit=(brightness_range[0] - 1, brightness_range[1] - 1),
                    contrast_limit=(contrast_range[0] - 1, contrast_range[1] - 1),
                    p=0.5,
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.HueSaturationValue(
                    hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.2
                ),
                # Quality degradation (simulates real-world conditions)
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=blur_limit, p=1.0),
                        A.GaussianBlur(blur_limit=blur_limit, p=1.0),
                        A.MedianBlur(blur_limit=blur_limit, p=1.0),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(10, 50), p=1.0),
                        A.ISONoise(p=1.0),
                        A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
                    ],
                    p=0.3,
                ),
                A.ImageCompression(quality_lower=75, quality_upper=100, p=0.3),
                # Coarse dropout (simulates occlusion)
                A.CoarseDropout(
                    max_holes=8,
                    max_height=8,
                    max_width=8,
                    min_holes=1,
                    min_height=4,
                    min_width=4,
                    fill_value=255,
                    p=0.2,
                ),
                # Shadow and lighting effects
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),
                    num_shadows_lower=1,
                    num_shadows_upper=2,
                    shadow_dimension=5,
                    p=0.2,
                ),
            ]
        )

    def __call__(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Apply augmentations to image.

        Args:
            image: Input image (RGB)
            **kwargs: Additional parameters

        Returns:
            Dictionary with augmented image
        """
        if random.random() > self.p:
            return {"image": image}

        augmented = self.transform(image=image)
        return augmented

    def get_training_transform(self, target_size: tuple = (384, 384)) -> A.Compose:
        """
        Get full training transformation pipeline including resize.

        Args:
            target_size: Target image size (H, W)

        Returns:
            Albumentations Compose object
        """
        return A.Compose(
            [
                self.transform,
                A.Resize(height=target_size[0], width=target_size[1], p=1.0),
            ]
        )

    def get_validation_transform(self, target_size: tuple = (384, 384)) -> A.Compose:
        """
        Get validation transformation pipeline (no augmentation).

        Args:
            target_size: Target image size (H, W)

        Returns:
            Albumentations Compose object
        """
        return A.Compose(
            [
                A.Resize(height=target_size[0], width=target_size[1], p=1.0),
            ]
        )


class MixupAugmentation:
    """Mixup augmentation for OCR training."""

    def __init__(self, alpha: float = 0.2, prob: float = 0.5):
        """
        Initialize Mixup.

        Args:
            alpha: Beta distribution parameter
            prob: Probability of applying mixup
        """
        self.alpha = alpha
        self.prob = prob

    def __call__(
        self, image1: np.ndarray, image2: np.ndarray, label1: str, label2: str
    ) -> tuple:
        """
        Apply mixup to two images.

        Args:
            image1: First image
            image2: Second image
            label1: First label
            label2: Second label

        Returns:
            Mixed image and labels
        """
        if random.random() > self.prob:
            return image1, label1, 1.0

        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)

        # Mix images
        mixed_image = lam * image1 + (1 - lam) * image2
        mixed_image = mixed_image.astype(image1.dtype)

        return mixed_image, (label1, label2), lam


class CutMixAugmentation:
    """CutMix augmentation adapted for OCR."""

    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        """
        Initialize CutMix.

        Args:
            alpha: Beta distribution parameter
            prob: Probability of applying cutmix
        """
        self.alpha = alpha
        self.prob = prob

    def __call__(
        self, image1: np.ndarray, image2: np.ndarray, label1: str, label2: str
    ) -> tuple:
        """
        Apply cutmix to two images.

        Args:
            image1: First image
            image2: Second image
            label1: First label
            label2: Second label

        Returns:
            Mixed image and labels
        """
        if random.random() > self.prob:
            return image1, label1, 1.0

        # Sample lambda
        lam = np.random.beta(self.alpha, self.alpha)

        # Get image dimensions
        h, w = image1.shape[:2]

        # Get random box
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        # Uniform sampling
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        # Apply cutmix
        mixed_image = image1.copy()
        mixed_image[bby1:bby2, bbx1:bbx2] = image2[bby1:bby2, bbx1:bbx2]

        # Adjust lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))

        return mixed_image, (label1, label2), lam


def simulate_real_world_conditions(image: np.ndarray) -> np.ndarray:
    """
    Simulate real-world image capture conditions.

    Args:
        image: Input image

    Returns:
        Image with simulated conditions
    """
    # Random lighting conditions
    if random.random() < 0.3:
        # Simulate low light
        image = (image * random.uniform(0.3, 0.7)).astype(np.uint8)
    elif random.random() < 0.3:
        # Simulate bright light
        image = np.clip(image * random.uniform(1.2, 1.5), 0, 255).astype(np.uint8)

    # Simulate camera shake (motion blur)
    if random.random() < 0.2:
        kernel_size = random.choice([3, 5, 7])
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel /= kernel_size
        image = cv2.filter2D(image, -1, kernel)

    # Simulate old/worn text (erosion)
    if random.random() < 0.2:
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)

    # Simulate ink bleeding (dilation)
    if random.random() < 0.2:
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)

    return image


def add_background_texture(image: np.ndarray, texture_type: str = "random") -> np.ndarray:
    """
    Add background texture to simulate different paper types.

    Args:
        image: Input image
        texture_type: Type of texture (paper, canvas, fabric, random)

    Returns:
        Image with texture
    """
    h, w = image.shape[:2]

    if texture_type == "random":
        texture_type = random.choice(["paper", "canvas", "fabric", "none"])

    if texture_type == "none":
        return image

    # Create texture
    if texture_type == "paper":
        texture = np.random.normal(240, 10, (h, w, 3)).astype(np.uint8)
    elif texture_type == "canvas":
        texture = np.random.normal(235, 15, (h, w, 3)).astype(np.uint8)
        texture = cv2.GaussianBlur(texture, (3, 3), 0)
    elif texture_type == "fabric":
        texture = np.random.normal(245, 8, (h, w, 3)).astype(np.uint8)
        texture = cv2.medianBlur(texture, 3)
    else:
        return image

    # Blend with original image
    mask = (image < 240).astype(np.float32)
    result = image * mask + texture * (1 - mask)
    return result.astype(np.uint8)
