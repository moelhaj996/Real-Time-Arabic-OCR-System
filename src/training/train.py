"""Training script for Arabic OCR model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
from pathlib import Path
import argparse
from typing import Optional

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models import ArabicVocabulary, ArabicOCRModel
from src.data.dataset import create_dataloaders
from src.training.losses import HybridLoss
from src.training.metrics import OCRMetrics, calculate_cer


class ArabicOCRLightningModule(pl.LightningModule):
    """PyTorch Lightning module for Arabic OCR."""

    def __init__(
        self,
        model: ArabicOCRModel,
        vocabulary: ArabicVocabulary,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 10000,
        ctc_weight: float = 0.3,
        ce_weight: float = 0.7,
        label_smoothing: float = 0.1,
    ):
        """
        Initialize Lightning module.

        Args:
            model: OCR model
            vocabulary: Vocabulary object
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_steps: Warmup steps
            ctc_weight: CTC loss weight
            ce_weight: Cross-entropy loss weight
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "vocabulary"])

        self.model = model
        self.vocabulary = vocabulary

        # Loss
        self.criterion = HybridLoss(
            ctc_weight=ctc_weight,
            ce_weight=ce_weight,
            pad_idx=vocabulary.pad_idx,
            blank_idx=vocabulary.pad_idx,
            label_smoothing=label_smoothing,
        )

        # Metrics
        self.train_metrics = OCRMetrics()
        self.val_metrics = OCRMetrics()

    def forward(self, images, targets):
        """Forward pass."""
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        """Training step."""
        images = batch["images"]
        tokens = batch["tokens"]

        # Teacher forcing: use targets as decoder input (shifted)
        decoder_input = tokens[:, :-1]
        decoder_target = tokens[:, 1:]

        # Forward pass
        logits = self.model(images, decoder_input)

        # Calculate loss
        loss, loss_dict = self.criterion(logits, decoder_target)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_ctc_loss", loss_dict["ctc_loss"], on_step=True, on_epoch=True)
        self.log("train_ce_loss", loss_dict["ce_loss"], on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images = batch["images"]
        tokens = batch["tokens"]
        texts = batch["texts"]

        # Forward pass with teacher forcing
        decoder_input = tokens[:, :-1]
        decoder_target = tokens[:, 1:]
        logits = self.model(images, decoder_input)

        # Calculate loss
        loss, loss_dict = self.criterion(logits, decoder_target)

        # Generate predictions (greedy decoding)
        with torch.no_grad():
            pred_tokens = self.model.greedy_decode(
                images, sos_idx=self.vocabulary.sos_idx, eos_idx=self.vocabulary.eos_idx
            )

        # Decode predictions
        predictions = []
        for tokens in pred_tokens:
            text = self.vocabulary.decode(tokens.tolist(), remove_special=True)
            predictions.append(text)

        # Update metrics
        self.val_metrics.update(predictions, texts)

        # Log
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return {"val_loss": loss, "predictions": predictions, "targets": texts}

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        metrics = self.val_metrics.compute()

        self.log("val_cer", metrics.get("cer", 0), prog_bar=True)
        self.log("val_wer", metrics.get("wer", 0), prog_bar=True)
        self.log("val_accuracy", metrics.get("accuracy", 0), prog_bar=True)

        # Reset metrics
        self.val_metrics.reset()

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-7
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


def train(
    model_config_path: str,
    train_config_path: str,
    output_dir: str = "models/checkpoints",
    log_dir: str = "logs",
):
    """
    Main training function.

    Args:
        model_config_path: Path to model config
        train_config_path: Path to training config
        output_dir: Output directory for checkpoints
        log_dir: Log directory
    """
    # Load configs
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    with open(train_config_path, "r") as f:
        train_config = yaml.safe_load(f)

    # Create vocabulary
    vocab = ArabicVocabulary(
        include_diacritics=model_config["vocabulary"]["diacritics"],
        include_english=True,
        include_numbers=True,
    )
    vocab.save(Path(output_dir) / "vocabulary.json")

    print(f"Vocabulary size: {vocab.vocab_size}")

    # Create model
    model = ArabicOCRModel.from_config(model_config["model"], vocab.vocab_size)
    print(f"Model parameters: {model.get_num_parameters():,}")

    # Create dataloaders
    data_config = train_config["data"]
    train_loader, val_loader = create_dataloaders(
        train_data_dir=data_config["train_data_path"],
        val_data_dir=data_config["val_data_path"],
        vocabulary=vocab,
        batch_size=train_config["training"]["batch_size"],
        num_workers=train_config["hardware"]["num_workers"],
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create Lightning module
    lightning_module = ArabicOCRLightningModule(
        model=model,
        vocabulary=vocab,
        learning_rate=train_config["training"]["optimizer"]["learning_rate"],
        weight_decay=train_config["training"]["optimizer"]["weight_decay"],
        ctc_weight=train_config["loss"]["ctc_weight"],
        ce_weight=train_config["loss"]["ce_weight"],
        label_smoothing=train_config["loss"]["label_smoothing"],
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="arabic-ocr-{epoch:02d}-{val_cer:.2f}",
        monitor="val_cer",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_cer",
        patience=train_config["evaluation"]["early_stopping"]["patience"],
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Logger
    logger = TensorBoardLogger(log_dir, name="arabic_ocr")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=train_config["training"]["num_epochs"],
        accelerator="auto",
        devices=train_config["hardware"]["gpus"],
        precision=train_config["hardware"]["precision"],
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=logger,
        gradient_clip_val=train_config["training"]["max_grad_norm"],
        accumulate_grad_batches=train_config["training"]["gradient_accumulation_steps"],
        log_every_n_steps=train_config["logging"]["log_every_n_steps"],
        val_check_interval=train_config["evaluation"]["eval_every_n_steps"],
    )

    # Train
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50 + "\n")

    trainer.fit(lightning_module, train_loader, val_loader)

    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Best model saved to: {checkpoint_callback.best_model_path}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Arabic OCR model")
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model config",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/checkpoints",
        help="Output directory",
    )
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")

    args = parser.parse_args()

    train(
        model_config_path=args.model_config,
        train_config_path=args.train_config,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
    )
