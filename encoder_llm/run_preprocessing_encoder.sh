#!/bin/bash

TIME_OF_INTEREST=168
MODELS=(
  "google-bert/bert-base-uncased"
  "FacebookAI/roberta-base"
  "microsoft/deberta-v3-small"
  "answerdotai/ModernBERT-base"
  "answerdotai/ModernBERT-large"
)

DATASETS=(
  "phe_deceased/t2s2_train"
  "phe_deceased/t2s2_test"
  "phe_deceased/sepsis10"
  "phe_deceased/sepsis100"
)

TOTAL=$(( ${#MODELS[@]} * ${#DATASETS[@]} ))
COUNT=1

for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "[$COUNT/$TOTAL] Running preprocessing for:"
    echo "  → Model:   $MODEL"
    echo "  → Dataset: $DATASET"
    echo "  → Time:    $(date '+%Y-%m-%d %H:%M:%S')"
    echo "-------------------------------------------"
    python preprocess_survival_data.py --data_folder "$DATASET" --model "$MODEL" --time_of_interest "$TIME_OF_INTEREST"
    COUNT=$((COUNT + 1))
  done
done

echo ""
echo "All preprocessing jobs completed!"