"""Vision Transformer Encoder for Arabic OCR."""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import timm
from einops import rearrange


class VisionTransformerEncoder(nn.Module):
    """Vision Transformer encoder for feature extraction."""

    def __init__(
        self,
        model_name: str = "vit_base_patch16_384",
        img_size: int = 384,
        pretrained: bool = True,
        embed_dim: int = 768,
        freeze: bool = False,
    ):
        """
        Initialize ViT encoder.

        Args:
            model_name: Name of pretrained ViT model from timm
            img_size: Input image size
            pretrained: Use pretrained weights
            embed_dim: Embedding dimension
            freeze: Freeze encoder weights
        """
        super().__init__()

        self.img_size = img_size
        self.embed_dim = embed_dim

        # Load pretrained ViT from timm
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,  # Remove classification head
        )

        # Get patch embedding info
        self.patch_embed = self.vit.patch_embed
        self.num_patches = self.patch_embed.num_patches
        self.patch_size = self.patch_embed.patch_size

        # Freeze if requested
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            Tuple of:
            - features: Extracted features [B, num_patches, embed_dim]
            - cls_token: CLS token [B, embed_dim]
        """
        # Forward through ViT
        # Returns features from all layers
        features = self.vit.forward_features(x)

        # features shape: [B, num_patches + 1, embed_dim]
        # First token is CLS token, rest are patch tokens

        cls_token = features[:, 0]  # [B, embed_dim]
        patch_features = features[:, 1:]  # [B, num_patches, embed_dim]

        return patch_features, cls_token

    def get_num_patches(self) -> int:
        """Get number of patches."""
        return self.num_patches

    def get_embed_dim(self) -> int:
        """Get embedding dimension."""
        return self.embed_dim


class CNNEncoder(nn.Module):
    """Alternative CNN-based encoder (lightweight)."""

    def __init__(
        self,
        model_name: str = "efficientnet_b3",
        pretrained: bool = True,
        embed_dim: int = 512,
    ):
        """
        Initialize CNN encoder.

        Args:
            model_name: Name of CNN model from timm
            pretrained: Use pretrained weights
            embed_dim: Output embedding dimension
        """
        super().__init__()

        # Load backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[4],  # Use last feature map
        )

        # Get feature map channels
        dummy_input = torch.randn(1, 3, 384, 384)
        with torch.no_grad():
            dummy_out = self.backbone(dummy_input)[0]
            self.feature_channels = dummy_out.shape[1]
            self.feature_h = dummy_out.shape[2]
            self.feature_w = dummy_out.shape[3]

        # Project to embed_dim
        self.projection = nn.Conv2d(
            self.feature_channels,
            embed_dim,
            kernel_size=1,
        )

        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Forward pass.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            Tuple of:
            - features: Extracted features [B, H*W, embed_dim]
            - None (for compatibility with ViT)
        """
        # Extract features
        features = self.backbone(x)[0]  # [B, C, H, W]

        # Project
        features = self.projection(features)  # [B, embed_dim, H, W]

        # Reshape to sequence
        B, C, H, W = features.shape
        features = rearrange(features, "b c h w -> b (h w) c")

        return features, None

    def get_num_patches(self) -> int:
        """Get number of spatial locations."""
        return self.feature_h * self.feature_w

    def get_embed_dim(self) -> int:
        """Get embedding dimension."""
        return self.embed_dim


def create_encoder(
    encoder_type: str = "vit",
    **kwargs
) -> nn.Module:
    """
    Factory function to create encoder.

    Args:
        encoder_type: Type of encoder ("vit" or "cnn")
        **kwargs: Arguments for encoder

    Returns:
        Encoder module
    """
    if encoder_type == "vit":
        return VisionTransformerEncoder(**kwargs)
    elif encoder_type == "cnn":
        return CNNEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
