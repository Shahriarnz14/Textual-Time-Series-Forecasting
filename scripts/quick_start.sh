#!/bin/bash
#
# Quick Start Script for Textual Time Series Forecasting
# =====================================================
#
# This script provides a quick demonstration of the framework capabilities
# using smaller models and reduced training times for rapid prototyping.
#
# Usage:
#   bash quick_start.sh [data_dir] [test_dir]
#
# Author: Textual Time Series Forecasting Team

set -e

# Configuration
DATA_DIR=${1:-"data/sample"}
TEST_DIR=${2:-"data/sample"}
OUTPUT_DIR="quick_start_$(date +%Y%m%d_%H%M%S)"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}"
echo "=================================================="
echo "  Textual Time Series Forecasting - Quick Start"
echo "=================================================="
echo -e "${NC}"

echo "🚀 Running quick demonstration experiments..."
echo "📂 Data Directory: $DATA_DIR"
echo "📂 Test Directory: $TEST_DIR"
echo "📁 Output Directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${YELLOW}Step 1: Testing Encoder Models${NC}"
echo "Running lightweight encoder experiments..."

cd encoder_llm

# Quick encoder test with BERT
echo "  📊 Testing BERT on time window classification..."
python run_encoder_experiments.py \
  --task time_window \
  --model bert \
  --data_dir "../$DATA_DIR" \
  --test_dir "../$TEST_DIR" \
  --output_dir "../$OUTPUT_DIR" \
  --epochs 3 \
  --batch_size 8 \
  --patience 2 \
  --dry_run

# Quick encoder test with RoBERTa
echo "  📊 Testing RoBERTa on concordance..."
python run_encoder_experiments.py \
  --task concordance \
  --model roberta \
  --data_dir "../$DATA_DIR" \
  --test_dir "../$TEST_DIR" \
  --output_dir "../$OUTPUT_DIR" \
  --epochs 3 \
  --batch_size 8 \
  --patience 2 \
  --dry_run

cd ..

echo -e "${YELLOW}Step 2: Testing Decoder Models${NC}"
echo "Running lightweight decoder experiments..."

cd decoder_llm

# Quick decoder test with small Llama
echo "  🤖 Testing Llama 3.2 1B with MLP head..."
python run_decoder_experiments.py \
  --approach MLP \
  --model llama-3.2-1b \
  --data_dir "../$DATA_DIR" \
  --test_dir "../$TEST_DIR" \
  --forecast_window 24 \
  --epochs 10 \
  --batch_size 4 \
  --run_dir "../$OUTPUT_DIR/runs" \
  --dry_run

# Quick prompt test
echo "  💭 Testing prompt-based inference..."
python run_decoder_experiments.py \
  --approach PROMPT:window \
  --model llama-3.2-1b \
  --test_dir "../$TEST_DIR" \
  --prompt_template window_system_text.txt \
  --eval_mode \
  --run_dir "../$OUTPUT_DIR/runs" \
  --dry_run

cd ..

echo -e "${GREEN}"
echo "✅ Quick start demonstration completed!"
echo ""
echo "📋 Summary:"
echo "  - Demonstrated encoder approaches (BERT, RoBERTa)"
echo "  - Demonstrated decoder approaches (MLP head, prompting)"
echo "  - Used dry_run mode for fast validation"
echo ""
echo "🚀 To run actual experiments:"
echo "  1. Prepare your data in CSV format"
echo "  2. Remove --dry_run flags from commands above"
echo "  3. Adjust epochs and batch sizes as needed"
echo "  4. Run: bash scripts/run_all_experiments.sh data/train data/test"
echo ""
echo "📖 For more details, see:"
echo "  - README.md (main repository)"
echo "  - encoder_llm/README.md (encoder approaches)"  
echo "  - decoder_llm/README.md (decoder approaches)"
echo -e "${NC}"
