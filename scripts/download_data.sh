#!/bin/bash

# Script to download Arabic OCR datasets
# Usage: bash scripts/download_data.sh

set -e

echo "===== Arabic OCR Dataset Downloader ====="
echo ""

DATA_DIR="data/raw"
mkdir -p $DATA_DIR

echo "📥 Downloading datasets..."
echo ""

# Note: These are placeholder URLs
# Replace with actual dataset URLs or implement custom download logic

echo "ℹ️  SARD Dataset"
echo "   Please download manually from: https://..."
echo ""

echo "ℹ️  KHATT Dataset"
echo "   Please download manually from: http://..."
echo ""

echo "ℹ️  Arabic Fonts"
echo "   Checking system fonts..."
if [ "$(uname)" == "Darwin" ]; then
    echo "   macOS fonts location: /System/Library/Fonts/"
    ls -1 /System/Library/Fonts/*Arabic*.ttf 2>/dev/null || echo "   No Arabic fonts found"
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    echo "   Linux fonts location: /usr/share/fonts/"
    fc-list :lang=ar | head -n 5
fi

echo ""
echo "✅ Dataset preparation instructions:"
echo "   1. Download SARD and KHATT datasets"
echo "   2. Place in data/raw/"
echo "   3. Run preprocessing scripts"
echo ""
echo "Or generate synthetic data:"
echo "   python -m src.data.synthetic_generator --num-samples 10000"
