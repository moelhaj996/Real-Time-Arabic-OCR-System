"""Inference predictor for Arabic OCR."""

import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from PIL import Image
from typing import Union, List, Dict, Optional
import yaml
import time

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models import ArabicVocabulary, ArabicOCRModel
from src.data.preprocessing import ImagePreprocessor


class ArabicOCRPredictor:
    """Predictor for Arabic OCR inference."""

    def __init__(
        self,
        model_path: str,
        vocab_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = "auto",
        beam_width: int = 5,
        temperature: float = 1.0,
    ):
        """
        Initialize predictor.

        Args:
            model_path: Path to trained model checkpoint
            vocab_path: Path to vocabulary file
            config_path: Path to inference config
            device: Device to use (auto, cpu, cuda, mps)
            beam_width: Beam width for decoding
            temperature: Temperature for sampling
        """
        self.model_path = Path(model_path)
        self.beam_width = beam_width
        self.temperature = temperature

        # Set device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load vocabulary
        if vocab_path is None:
            vocab_path = self.model_path.parent / "vocabulary.json"

        self.vocabulary = ArabicVocabulary.load(str(vocab_path))
        print(f"Loaded vocabulary: {self.vocabulary.vocab_size} tokens")

        # Load config
        if config_path:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()

        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor()

        # Load model
        self._load_model()

    def _default_config(self) -> dict:
        """Get default configuration."""
        return {
            "preprocessing": {
                "target_size": [384, 384],
                "use_clahe": True,
                "denoise": True,
                "deskew": True,
            },
            "recognition": {
                "decoding_method": "beam_search",
                "beam_width": 5,
                "min_confidence": 0.5,
            },
        }

    def _load_model(self):
        """Load model from checkpoint."""
        print(f"Loading model from {self.model_path}...")

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Create model
        if "model_config" in checkpoint:
            model_config = checkpoint["model_config"]
        else:
            # Default config
            model_config = {
                "encoder": {"name": "vit_base_patch16_384", "embed_dim": 768},
                "decoder": {"num_layers": 6, "num_heads": 8},
            }

        self.model = ArabicOCRModel.from_config(model_config, self.vocabulary.vocab_size)

        # Load weights
        if "state_dict" in checkpoint:
            # PyTorch Lightning checkpoint
            state_dict = {
                k.replace("model.", ""): v
                for k, v in checkpoint["state_dict"].items()
                if k.startswith("model.")
            }
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully ({self.model.get_num_parameters():,} parameters)")

    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        return_confidence: bool = True,
        return_alternatives: bool = False,
    ) -> Dict:
        """
        Predict text from image.

        Args:
            image: Input image (path or array)
            return_confidence: Return confidence score
            return_alternatives: Return alternative predictions

        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()

        # Load and preprocess image
        image_array = self._load_image(image)
        processed = self.preprocessor(image_array)

        # Convert to tensor
        image_tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        # Predict
        if self.config["recognition"]["decoding_method"] == "beam_search":
            sequences, scores = self.model.predict(
                image_tensor,
                beam_width=self.beam_width,
                temperature=self.temperature,
                sos_idx=self.vocabulary.sos_idx,
                eos_idx=self.vocabulary.eos_idx,
                pad_idx=self.vocabulary.pad_idx,
            )
        else:
            sequences = self.model.greedy_decode(
                image_tensor,
                sos_idx=self.vocabulary.sos_idx,
                eos_idx=self.vocabulary.eos_idx,
            )
            scores = torch.zeros(sequences.size(0))

        # Decode
        text = self.vocabulary.decode(sequences[0].tolist(), remove_special=True)
        confidence = torch.exp(scores[0]).item() if return_confidence else None

        # Prepare result
        result = {
            "text": text,
            "processing_time": time.time() - start_time,
        }

        if return_confidence:
            result["confidence"] = confidence

        if return_alternatives and self.beam_width > 1:
            alternatives = []
            for seq, score in zip(sequences[1:], scores[1:]):
                alt_text = self.vocabulary.decode(seq.tolist(), remove_special=True)
                alternatives.append({
                    "text": alt_text,
                    "confidence": torch.exp(score).item(),
                })
            result["alternatives"] = alternatives

        return result

    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[str, Path, np.ndarray, Image.Image]],
        batch_size: int = 8,
    ) -> List[Dict]:
        """
        Predict text from multiple images.

        Args:
            images: List of images
            batch_size: Batch size for processing

        Returns:
            List of prediction results
        """
        results = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]

            # Preprocess batch
            processed_batch = []
            for img in batch_images:
                img_array = self._load_image(img)
                processed = self.preprocessor(img_array)
                processed_batch.append(processed)

            # Stack into batch tensor
            batch_tensor = torch.stack([
                torch.from_numpy(img).permute(2, 0, 1)
                for img in processed_batch
            ])
            batch_tensor = batch_tensor.to(self.device)

            # Predict
            sequences = self.model.greedy_decode(
                batch_tensor,
                sos_idx=self.vocabulary.sos_idx,
                eos_idx=self.vocabulary.eos_idx,
            )

            # Decode
            for seq in sequences:
                text = self.vocabulary.decode(seq.tolist(), remove_special=True)
                results.append({"text": text})

        return results

    def _load_image(
        self, image: Union[str, Path, np.ndarray, Image.Image]
    ) -> np.ndarray:
        """
        Load image from various formats.

        Args:
            image: Input image

        Returns:
            Image as numpy array
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
            return np.array(image)
        elif isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                # Grayscale to RGB
                return np.stack([image] * 3, axis=-1)
            return image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def __call__(self, image: Union[str, Path, np.ndarray, Image.Image]) -> str:
        """
        Shorthand for predict().

        Args:
            image: Input image

        Returns:
            Predicted text
        """
        result = self.predict(image, return_confidence=False)
        return result["text"]


def load_predictor(
    checkpoint_dir: str = "models/checkpoints",
    device: str = "auto",
) -> ArabicOCRPredictor:
    """
    Load predictor from checkpoint directory.

    Args:
        checkpoint_dir: Directory containing checkpoints
        device: Device to use

    Returns:
        Loaded predictor
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Find best checkpoint
    checkpoints = list(checkpoint_dir.glob("*.ckpt")) + list(checkpoint_dir.glob("*.pth"))

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    # Use most recent or "best" checkpoint
    best_ckpt = None
    for ckpt in checkpoints:
        if "best" in ckpt.name.lower() or "last" in ckpt.name.lower():
            best_ckpt = ckpt
            break

    if best_ckpt is None:
        best_ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime)

    print(f"Loading checkpoint: {best_ckpt.name}")

    predictor = ArabicOCRPredictor(
        model_path=str(best_ckpt),
        device=device,
    )

    return predictor
