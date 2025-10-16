#!/bin/bash

# Training script for Arabic OCR Model
# Usage: bash scripts/train_model.sh

set -e

echo "===== Arabic OCR Model Training ====="
echo ""

# Configuration
MODEL_CONFIG="configs/model_config.yaml"
TRAIN_CONFIG="configs/training_config.yaml"
OUTPUT_DIR="models/checkpoints"
LOG_DIR="logs"

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# Check if data exists
if [ ! -d "data/processed/train" ] && [ ! -d "data/augmented/train" ]; then
    echo "❌ No training data found!"
    echo "   Run: python -m src.data.synthetic_generator"
    exit 1
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "⚠️  No GPU detected. Training will use CPU (slower)."
    echo ""
fi

# Start training
echo "Starting training..."
echo "Model config: $MODEL_CONFIG"
echo "Train config: $TRAIN_CONFIG"
echo "Output dir: $OUTPUT_DIR"
echo ""

python -m src.training.train \
    --model-config $MODEL_CONFIG \
    --train-config $TRAIN_CONFIG \
    --output-dir $OUTPUT_DIR \
    --log-dir $LOG_DIR \
    "$@"

echo ""
echo "✅ Training complete!"
echo "📁 Checkpoints saved to: $OUTPUT_DIR"
