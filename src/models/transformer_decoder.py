"""Transformer Decoder for sequence generation in Arabic OCR."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, : x.size(1), :]


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with cross-attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            tgt: Target sequence [B, T, D]
            memory: Encoder output [B, S, D]
            tgt_mask: Causal mask for target
            memory_mask: Mask for memory

        Returns:
            Output tensor [B, T, D]
        """
        # Self-attention with causal mask
        attn_out, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout(attn_out))

        # Cross-attention to encoder features
        attn_out, _ = self.cross_attn(tgt, memory, memory, key_padding_mask=memory_mask)
        tgt = self.norm2(tgt + self.dropout(attn_out))

        # Feedforward
        ffn_out = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(ffn_out))

        return tgt


class TransformerDecoder(nn.Module):
    """Transformer decoder for Arabic OCR."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 256,
    ):
        """
        Initialize decoder.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_layers: Number of decoder layers
            num_heads: Number of attention heads
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
        """
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)

        # Decoder layers
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model, num_heads, dim_feedforward, dropout
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.output_projection.weight, mean=0, std=0.02)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            tgt: Target tokens [B, T]
            memory: Encoder features [B, S, D]
            tgt_mask: Causal mask for target

        Returns:
            Logits [B, T, vocab_size]
        """
        # Embed tokens
        tgt_emb = self.token_embedding(tgt) * math.sqrt(self.d_model)

        # Add positional encoding
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)

        # Generate causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_causal_mask(tgt.size(1)).to(tgt.device)

        # Pass through decoder layers
        output = tgt_emb
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask)

        # Project to vocabulary
        logits = self.output_projection(output)

        return logits

    @staticmethod
    def generate_causal_mask(seq_len: int) -> torch.Tensor:
        """
        Generate causal mask for autoregressive generation.

        Args:
            seq_len: Sequence length

        Returns:
            Causal mask [seq_len, seq_len]
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def decode_step(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        cache: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Single decoding step (for autoregressive generation).

        Args:
            tgt: Target tokens [B, T]
            memory: Encoder features [B, S, D]
            cache: Cached states

        Returns:
            Tuple of (logits, cache)
        """
        # For simplicity, we recompute everything
        # In production, implement KV caching for efficiency
        logits = self.forward(tgt, memory)

        return logits[:, -1:, :], cache


class BeamSearchDecoder:
    """Beam search decoder for sequence generation."""

    def __init__(
        self,
        model: TransformerDecoder,
        beam_width: int = 5,
        max_length: int = 256,
        length_penalty: float = 0.6,
        sos_idx: int = 1,
        eos_idx: int = 2,
        pad_idx: int = 0,
    ):
        """
        Initialize beam search.

        Args:
            model: Transformer decoder model
            beam_width: Beam width
            max_length: Maximum sequence length
            length_penalty: Length penalty factor
            sos_idx: Start-of-sequence token index
            eos_idx: End-of-sequence token index
            pad_idx: Padding token index
        """
        self.model = model
        self.beam_width = beam_width
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

    @torch.no_grad()
    def decode(
        self,
        memory: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Beam search decoding.

        Args:
            memory: Encoder features [B, S, D]
            temperature: Temperature for sampling

        Returns:
            Tuple of (sequences, scores)
        """
        batch_size = memory.size(0)
        device = memory.device

        # Initialize with SOS token
        sequences = torch.full(
            (batch_size * self.beam_width, 1),
            self.sos_idx,
            dtype=torch.long,
            device=device,
        )

        # Expand memory for beam width
        memory = memory.unsqueeze(1).repeat(1, self.beam_width, 1, 1)
        memory = memory.view(batch_size * self.beam_width, memory.size(2), memory.size(3))

        # Initialize scores
        scores = torch.zeros(batch_size * self.beam_width, device=device)

        # Track finished sequences
        finished = torch.zeros(batch_size * self.beam_width, dtype=torch.bool, device=device)

        for step in range(self.max_length - 1):
            # Get logits for current sequence
            logits = self.model(sequences, memory)[:, -1, :]  # [B*beam, vocab]

            # Apply temperature
            logits = logits / temperature

            # Get log probabilities
            log_probs = F.log_softmax(logits, dim=-1)

            # Add previous scores
            log_probs = log_probs + scores.unsqueeze(1)

            # Reshape for beam search
            log_probs = log_probs.view(batch_size, self.beam_width * self.model.vocab_size)

            # Get top k
            topk_scores, topk_indices = torch.topk(log_probs, self.beam_width, dim=-1)

            # Update sequences and scores
            beam_indices = topk_indices // self.model.vocab_size
            token_indices = topk_indices % self.model.vocab_size

            # Reorder sequences
            sequences = sequences.view(batch_size, self.beam_width, -1)
            sequences = torch.gather(
                sequences,
                1,
                beam_indices.unsqueeze(-1).expand(-1, -1, sequences.size(2)),
            )
            sequences = sequences.view(batch_size * self.beam_width, -1)

            # Append new tokens
            new_tokens = token_indices.view(batch_size * self.beam_width, 1)
            sequences = torch.cat([sequences, new_tokens], dim=1)

            # Update scores
            scores = topk_scores.view(batch_size * self.beam_width)

            # Check for EOS
            finished = finished | (new_tokens.squeeze(-1) == self.eos_idx)

            if finished.all():
                break

        # Return best sequences
        sequences = sequences.view(batch_size, self.beam_width, -1)
        scores = scores.view(batch_size, self.beam_width)

        # Apply length penalty
        lengths = (sequences != self.pad_idx).sum(dim=-1).float()
        scores = scores / (lengths ** self.length_penalty)

        # Get best sequences
        best_scores, best_indices = scores.max(dim=1)
        best_sequences = torch.gather(
            sequences,
            1,
            best_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, sequences.size(2)),
        ).squeeze(1)

        return best_sequences, best_scores
