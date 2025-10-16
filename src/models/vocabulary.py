"""Arabic text vocabulary for OCR."""

from typing import List, Dict, Optional
import json
from pathlib import Path


class ArabicVocabulary:
    """Vocabulary for Arabic OCR."""

    # Special tokens
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    # Arabic Unicode ranges
    ARABIC_LETTERS_START = 0x0600
    ARABIC_LETTERS_END = 0x06FF
    ARABIC_SUPPLEMENT_START = 0x0750
    ARABIC_SUPPLEMENT_END = 0x077F

    # Arabic diacritics
    DIACRITICS = [
        "\u064B",  # Fathatan
        "\u064C",  # Dammatan
        "\u064D",  # Kasratan
        "\u064E",  # Fatha
        "\u064F",  # Damma
        "\u0650",  # Kasra
        "\u0651",  # Shadda
        "\u0652",  # Sukun
        "\u0653",  # Maddah
        "\u0654",  # Hamza above
        "\u0655",  # Hamza below
    ]

    def __init__(
        self,
        include_diacritics: bool = True,
        include_english: bool = True,
        include_numbers: bool = True,
        include_punctuation: bool = True,
    ):
        """
        Initialize vocabulary.

        Args:
            include_diacritics: Include Arabic diacritics
            include_english: Include English letters
            include_numbers: Include numbers
            include_punctuation: Include punctuation marks
        """
        self.include_diacritics = include_diacritics
        self.include_english = include_english
        self.include_numbers = include_numbers
        self.include_punctuation = include_punctuation

        # Build vocabulary
        self._build_vocabulary()

    def _build_vocabulary(self) -> None:
        """Build the character vocabulary."""
        chars = []

        # Add special tokens
        chars.extend([self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN])

        # Add Arabic letters
        for code in range(self.ARABIC_LETTERS_START, self.ARABIC_LETTERS_END + 1):
            chars.append(chr(code))

        # Add Arabic supplement
        for code in range(self.ARABIC_SUPPLEMENT_START, self.ARABIC_SUPPLEMENT_END + 1):
            chars.append(chr(code))

        # Add diacritics
        if self.include_diacritics:
            chars.extend(self.DIACRITICS)

        # Add space
        chars.append(" ")

        # Add English letters
        if self.include_english:
            chars.extend([chr(i) for i in range(ord('a'), ord('z') + 1)])
            chars.extend([chr(i) for i in range(ord('A'), ord('Z') + 1)])

        # Add numbers
        if self.include_numbers:
            chars.extend([chr(i) for i in range(ord('0'), ord('9') + 1)])
            # Arabic-Indic digits
            chars.extend([chr(i) for i in range(0x0660, 0x066A)])

        # Add punctuation
        if self.include_punctuation:
            punctuation = ".,!?;:()[]{}\"'-،؛؟"
            chars.extend(punctuation)

        # Remove duplicates and sort
        chars = sorted(list(set(chars)))

        # Create mappings
        self.char2idx = {char: idx for idx, char in enumerate(chars)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}

        # Cache special token indices
        self.pad_idx = self.char2idx[self.PAD_TOKEN]
        self.sos_idx = self.char2idx[self.SOS_TOKEN]
        self.eos_idx = self.char2idx[self.EOS_TOKEN]
        self.unk_idx = self.char2idx[self.UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.char2idx)

    def encode(self, text: str, add_sos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Encode text to indices.

        Args:
            text: Input text
            add_sos: Add start-of-sequence token
            add_eos: Add end-of-sequence token

        Returns:
            List of character indices
        """
        indices = []

        if add_sos:
            indices.append(self.sos_idx)

        for char in text:
            indices.append(self.char2idx.get(char, self.unk_idx))

        if add_eos:
            indices.append(self.eos_idx)

        return indices

    def decode(
        self,
        indices: List[int],
        remove_special: bool = True,
        remove_duplicates: bool = False
    ) -> str:
        """
        Decode indices to text.

        Args:
            indices: List of character indices
            remove_special: Remove special tokens
            remove_duplicates: Remove consecutive duplicates (for CTC)

        Returns:
            Decoded text
        """
        chars = []
        prev_idx = None

        for idx in indices:
            # Skip padding
            if remove_special and idx == self.pad_idx:
                continue

            # Skip special tokens
            if remove_special and idx in [self.sos_idx, self.eos_idx]:
                continue

            # Skip duplicates (for CTC blank handling)
            if remove_duplicates and idx == prev_idx:
                continue

            # Get character
            char = self.idx2char.get(idx, self.UNK_TOKEN)

            # Skip unknown tokens if removing special
            if remove_special and char == self.UNK_TOKEN:
                continue

            chars.append(char)
            prev_idx = idx

        return "".join(chars)

    def save(self, path: str) -> None:
        """
        Save vocabulary to file.

        Args:
            path: Output file path
        """
        data = {
            "char2idx": self.char2idx,
            "config": {
                "include_diacritics": self.include_diacritics,
                "include_english": self.include_english,
                "include_numbers": self.include_numbers,
                "include_punctuation": self.include_punctuation,
            }
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ArabicVocabulary":
        """
        Load vocabulary from file.

        Args:
            path: Input file path

        Returns:
            Loaded vocabulary
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = cls(**data["config"])
        vocab.char2idx = data["char2idx"]
        vocab.idx2char = {int(idx): char for char, idx in data["char2idx"].items()}

        # Update cached indices
        vocab.pad_idx = vocab.char2idx[vocab.PAD_TOKEN]
        vocab.sos_idx = vocab.char2idx[vocab.SOS_TOKEN]
        vocab.eos_idx = vocab.char2idx[vocab.EOS_TOKEN]
        vocab.unk_idx = vocab.char2idx[vocab.UNK_TOKEN]

        return vocab

    def __len__(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size

    def __repr__(self) -> str:
        """String representation."""
        return f"ArabicVocabulary(vocab_size={self.vocab_size})"
