"""Loss functions for Arabic OCR training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CTCLoss(nn.Module):
    """CTC Loss for sequence alignment."""

    def __init__(self, blank_idx: int = 0, reduction: str = "mean"):
        """
        Initialize CTC loss.

        Args:
            blank_idx: Index of blank token
            reduction: Reduction method
        """
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction=reduction, zero_infinity=True)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute CTC loss.

        Args:
            logits: Model output [B, T, C]
            targets: Target tokens [B, T]
            input_lengths: Input sequence lengths
            target_lengths: Target sequence lengths

        Returns:
            Loss value
        """
        # Get dimensions
        batch_size, seq_len, vocab_size = logits.shape

        # Default lengths if not provided
        if input_lengths is None:
            input_lengths = torch.full(
                (batch_size,), seq_len, dtype=torch.long, device=logits.device
            )

        if target_lengths is None:
            # Calculate actual target lengths (excluding padding)
            target_lengths = (targets != 0).sum(dim=1)

        # CTC expects [T, B, C]
        logits = logits.permute(1, 0, 2)
        log_probs = F.log_softmax(logits, dim=-1)

        # Compute loss
        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)

        return loss


class CrossEntropyLoss(nn.Module):
    """Cross-entropy loss for sequence generation."""

    def __init__(
        self,
        pad_idx: int = 0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        """
        Initialize cross-entropy loss.

        Args:
            pad_idx: Padding token index
            label_smoothing: Label smoothing factor
            reduction: Reduction method
        """
        super().__init__()
        self.pad_idx = pad_idx
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss.

        Args:
            logits: Model output [B, T, C]
            targets: Target tokens [B, T]

        Returns:
            Loss value
        """
        # Reshape for cross-entropy
        # logits: [B*T, C], targets: [B*T]
        batch_size, seq_len, vocab_size = logits.shape

        logits = logits.reshape(-1, vocab_size)
        targets = targets.reshape(-1)

        loss = self.criterion(logits, targets)

        return loss


class HybridLoss(nn.Module):
    """Hybrid loss combining CTC and Cross-Entropy."""

    def __init__(
        self,
        ctc_weight: float = 0.3,
        ce_weight: float = 0.7,
        pad_idx: int = 0,
        blank_idx: int = 0,
        label_smoothing: float = 0.0,
    ):
        """
        Initialize hybrid loss.

        Args:
            ctc_weight: Weight for CTC loss
            ce_weight: Weight for cross-entropy loss
            pad_idx: Padding token index
            blank_idx: CTC blank token index
            label_smoothing: Label smoothing for CE
        """
        super().__init__()

        assert abs(ctc_weight + ce_weight - 1.0) < 1e-6, "Weights must sum to 1.0"

        self.ctc_weight = ctc_weight
        self.ce_weight = ce_weight

        self.ctc_loss = CTCLoss(blank_idx=blank_idx)
        self.ce_loss = CrossEntropyLoss(pad_idx=pad_idx, label_smoothing=label_smoothing)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute hybrid loss.

        Args:
            logits: Model output [B, T, C]
            targets: Target tokens [B, T]
            input_lengths: Input sequence lengths
            target_lengths: Target sequence lengths

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Compute individual losses
        ctc_loss_val = self.ctc_loss(logits, targets, input_lengths, target_lengths)
        ce_loss_val = self.ce_loss(logits, targets)

        # Combine
        total_loss = self.ctc_weight * ctc_loss_val + self.ce_weight * ce_loss_val

        loss_dict = {
            "loss": total_loss.item(),
            "ctc_loss": ctc_loss_val.item(),
            "ce_loss": ce_loss_val.item(),
        }

        return total_loss, loss_dict


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pad_idx: int = 0,
        reduction: str = "mean",
    ):
        """
        Initialize focal loss.

        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
            pad_idx: Padding token index
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pad_idx = pad_idx
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Model output [B, T, C]
            targets: Target tokens [B, T]

        Returns:
            Loss value
        """
        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather target log probs
        target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        # Compute focal weight
        probs = torch.exp(target_log_probs)
        focal_weight = (1 - probs) ** self.gamma

        # Compute loss
        loss = -self.alpha * focal_weight * target_log_probs

        # Mask padding
        mask = targets != self.pad_idx
        loss = loss * mask

        if self.reduction == "mean":
            return loss.sum() / mask.sum()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
