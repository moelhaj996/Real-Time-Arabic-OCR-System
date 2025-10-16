"""Complete Arabic OCR Model combining encoder and decoder."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from .vit_encoder import create_encoder
from .transformer_decoder import TransformerDecoder, BeamSearchDecoder
from .vocabulary import ArabicVocabulary


class ArabicOCRModel(nn.Module):
    """Complete Arabic OCR model."""

    def __init__(
        self,
        vocab_size: int,
        encoder_type: str = "vit",
        encoder_name: str = "vit_base_patch16_384",
        img_size: int = 384,
        encoder_pretrained: bool = True,
        encoder_freeze: bool = False,
        d_model: int = 768,
        decoder_layers: int = 6,
        decoder_heads: int = 8,
        decoder_dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 256,
    ):
        """
        Initialize OCR model.

        Args:
            vocab_size: Vocabulary size
            encoder_type: Type of encoder ("vit" or "cnn")
            encoder_name: Name of encoder model
            img_size: Input image size
            encoder_pretrained: Use pretrained encoder
            encoder_freeze: Freeze encoder weights
            d_model: Model dimension
            decoder_layers: Number of decoder layers
            decoder_heads: Number of attention heads
            decoder_dim_feedforward: FFN dimension
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # Encoder
        self.encoder = create_encoder(
            encoder_type=encoder_type,
            model_name=encoder_name,
            img_size=img_size,
            pretrained=encoder_pretrained,
            freeze=encoder_freeze,
            embed_dim=d_model,
        )

        # Decoder
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            dim_feedforward=decoder_dim_feedforward,
            dropout=dropout,
            max_seq_length=max_seq_length,
        )

    def forward(
        self,
        images: torch.Tensor,
        target_tokens: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass (for training).

        Args:
            images: Input images [B, 3, H, W]
            target_tokens: Target token indices [B, T]
            target_mask: Target mask

        Returns:
            Logits [B, T, vocab_size]
        """
        # Encode images
        memory, _ = self.encoder(images)  # [B, S, D]

        # Decode to text
        logits = self.decoder(target_tokens, memory, target_mask)

        return logits

    def predict(
        self,
        images: torch.Tensor,
        beam_width: int = 5,
        temperature: float = 1.0,
        sos_idx: int = 1,
        eos_idx: int = 2,
        pad_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate predictions using beam search.

        Args:
            images: Input images [B, 3, H, W]
            beam_width: Beam width for search
            temperature: Sampling temperature
            sos_idx: Start token index
            eos_idx: End token index
            pad_idx: Padding token index

        Returns:
            Tuple of (sequences, scores)
        """
        # Encode
        with torch.no_grad():
            memory, _ = self.encoder(images)

        # Decode with beam search
        beam_decoder = BeamSearchDecoder(
            self.decoder,
            beam_width=beam_width,
            max_length=self.max_seq_length,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
        )

        sequences, scores = beam_decoder.decode(memory, temperature)

        return sequences, scores

    def greedy_decode(
        self,
        images: torch.Tensor,
        sos_idx: int = 1,
        eos_idx: int = 2,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Greedy decoding (faster but less accurate).

        Args:
            images: Input images [B, 3, H, W]
            sos_idx: Start token index
            eos_idx: End token index
            max_length: Maximum length

        Returns:
            Token sequences [B, T]
        """
        if max_length is None:
            max_length = self.max_seq_length

        batch_size = images.size(0)
        device = images.device

        # Encode
        with torch.no_grad():
            memory, _ = self.encoder(images)

            # Start with SOS token
            tokens = torch.full(
                (batch_size, 1), sos_idx, dtype=torch.long, device=device
            )

            for _ in range(max_length - 1):
                # Get predictions
                logits = self.decoder(tokens, memory)  # [B, T, V]

                # Get next token (greedy)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]

                # Append to sequence
                tokens = torch.cat([tokens, next_token], dim=1)

                # Check if all sequences have EOS
                if (tokens == eos_idx).any(dim=1).all():
                    break

        return tokens

    @classmethod
    def from_config(cls, config: Dict, vocab_size: int) -> "ArabicOCRModel":
        """
        Create model from configuration dict.

        Args:
            config: Configuration dictionary
            vocab_size: Vocabulary size

        Returns:
            Model instance
        """
        encoder_config = config.get("encoder", {})
        decoder_config = config.get("decoder", {})

        return cls(
            vocab_size=vocab_size,
            encoder_type=config.get("type", "vit"),
            encoder_name=encoder_config.get("name", "vit_base_patch16_384"),
            img_size=encoder_config.get("img_size", 384),
            encoder_pretrained=encoder_config.get("pretrained", True),
            encoder_freeze=encoder_config.get("freeze", False),
            d_model=encoder_config.get("embed_dim", 768),
            decoder_layers=decoder_config.get("num_layers", 6),
            decoder_heads=decoder_config.get("num_heads", 8),
            decoder_dim_feedforward=decoder_config.get("ffn_dim", 2048),
            dropout=decoder_config.get("dropout", 0.1),
            max_seq_length=decoder_config.get("max_seq_length", 256),
        )

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """
        Get number of parameters.

        Args:
            trainable_only: Count only trainable parameters

        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True


def create_model_from_config(
    model_config_path: str,
    vocab: ArabicVocabulary,
) -> ArabicOCRModel:
    """
    Create model from YAML config file.

    Args:
        model_config_path: Path to model config YAML
        vocab: Vocabulary object

    Returns:
        Model instance
    """
    import yaml

    with open(model_config_path, "r") as f:
        config = yaml.safe_load(f)

    model = ArabicOCRModel.from_config(config["model"], vocab.vocab_size)

    return model
