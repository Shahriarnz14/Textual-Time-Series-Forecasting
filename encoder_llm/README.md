# Encoder-Based Textual Time Series Forecasting

<div align="center">

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.50-yellow.svg)](https://huggingface.co/transformers)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)

*Encoder-only transformer models for clinical time series forecasting*

</div>

## 🧠 Overview

This module implements encoder-based approaches for textual time series forecasting using pre-trained encoder-only transformer models. Our framework transforms irregular clinical event sequences into structured prediction tasks, enabling accurate temporal forecasting with encoder-only architectures.

### 🎯 Key Features

- **Fine-tuned Classification**: Task-specific heads on pre-trained encoders
- **Zero-Shot Masking**: Masked language modeling for temporal reasoning
- **Flexible Architectures**: Support for BERT, RoBERTa, DeBERTa, ModernBERT, and Bioclinical ModernBERT 
- **Clinical Applications**: Designed for irregular medical event sequences
- **Robust Evaluation**: Comprehensive metrics including C-Index and F1-Score

## 📦 Data Format

Our framework expects clinical time series data where each case report contains a sequence of timestamped events:

```csv
event,timestamp
"Patient admitted to ICU",0.0
"Elevated troponin levels observed",2.5
"Cardiac catheterization performed",6.0
"Patient transferred to ward",24.0
"Discharge planning initiated",72.0
```

### Data Requirements
- **Event**: Free-text clinical observation or intervention
- **Timestamp**: Time in hours relative to admission or reference point
- **Format**: CSV files with consistent column naming
- **Structure**: One file per case report/patient encounter

## 🛠 Input Representation

Clinical events are serialized into transformer-compatible sequences:

```
[CLS] (0.0): Patient admitted to ICU [SEP] (2.5): Elevated troponin levels [SEP] (6.0): Cardiac catheterization [SEP]
```

### Preprocessing Pipeline
1. **Event Extraction**: Parse (event, timestamp) pairs from CSV files
2. **Temporal Grouping**: Group events occurring at identical timestamps  
3. **Sequence Generation**: Create sliding windows with history and forecast segments
4. **Text Serialization**: Format as transformer input with special tokens
5. **Truncation Strategy**: Left-truncation to fit within model context limits

## 🧪 Prediction Tasks

We implement two complementary binary classification tasks:

### 1. Event Ordering (Concordance) 🔄
**Objective**: Given pairs of future events, predict their temporal ordering

**Input Format**:
```
[CLS] history_events [SEP] event_A [SEP] event_B [SEP]
```

**Label**: `1` if event_A occurs before event_B, `0` otherwise

**Evaluation**: Concordance Index (C-Index) measuring ranking quality

**Applications**:
- Treatment sequence planning
- Clinical pathway optimization
- Risk progression modeling

### 2. Time-Window Classification ⏰
**Objective**: Predict whether events occur within specified time horizons

**Input Format**:
```
[CLS] history_events [SEP] target_event [SEP]
```

**Label**: `1` if event occurs within H hours, `0` otherwise

**Evaluation**: Macro-averaged F1-Score with optimal threshold selection

**Applications**:
- Early warning systems
- Resource allocation planning
- Intervention timing

## 🤖 Model Architectures

### Standard Classification Models

#### BERT (`bert-base-uncased`)
- **Architecture**: 12-layer encoder with 768 hidden dimensions
- **Tokenizer**: WordPiece with 30K vocabulary
- **Context Window**: 512 tokens
- **Strengths**: Robust baseline performance, well-studied

#### RoBERTa (`roberta-base`)
- **Architecture**: 12-layer encoder optimized training procedure
- **Tokenizer**: Byte-level BPE with 50K vocabulary  
- **Context Window**: 512 tokens
- **Strengths**: Improved pre-training, better downstream performance

#### DeBERTa (`microsoft/deberta-v3-base`)
- **Architecture**: Disentangled attention mechanism
- **Tokenizer**: SentencePiece with relative position encoding
- **Context Window**: 512 tokens  
- **Strengths**: Enhanced positional understanding, state-of-the-art results

#### ModernBERT (`answerdotai/ModernBERT-base`)
- **Architecture**: Modern training techniques and architecture improvements
- **Tokenizer**: Updated vocabulary and tokenization strategy
- **Context Window**: 512 tokens
- **Strengths**: Latest improvements in encoder architectures

#### Bioclinical ModernBERT (`thomas-sounack/BioClinical-ModernBERT-base`)
- **Architecture**: Modern training techniques and architecture improvements finetuned on Biomedical Text
- **Tokenizer**: Updated vocabulary and tokenization strategy
- **Context Window**: 512 tokens
- **Strengths**: Latest improvements in encoder architectures

### Self-Supervised Masked Models

#### Masked Time Window Classification
- **Approach**: MLM-style training with `[MASK]` token prediction
- **Input**: `[CLS] history [SEP] Will "event" happen within H hours? [MASK] [SEP]`
- **Target**: Predict "yes"/"no" tokens for the `[MASK]` position
- **Advantage**: Leverages pre-training knowledge directly

#### Masked Concordance Prediction  
- **Approach**: Temporal relationship prediction via masking
- **Input**: `[CLS] history [SEP] event_A [MASK] event_B [SEP]`
- **Target**: Predict "before"/"after" tokens for relative timing
- **Advantage**: Natural language reasoning for temporal ordering

## 🚀 Training & Evaluation

### Training Procedure
1. **Data Preparation**: Generate sliding window examples from time series
2. **Train/Val Split**: 80/20 stratified split maintaining temporal structure
3. **Optimization**: AdamW optimizer with learning rate scheduling
4. **Early Stopping**: Patience-based stopping on validation metrics
5. **Checkpointing**: Best model selection based on validation performance

### Training Configuration
```python
# Default hyperparameters
learning_rate = 1e-5
batch_size = 16  # Model-dependent
epochs = 10
patience = 3
max_length = 512
K = 8  # Number of forecast events
H = 24  # Time window in hours
```

### Evaluation Protocol
- **Metrics**: C-Index for concordance, F1-Score for time windows
- **Cross-Validation**: Temporal split to avoid data leakage
- **Threshold Selection**: Optimal F1 threshold on validation set

## 📁 File Structure

```
encoder_llm/
├── 📄 encoder_time_window.py          # Time window classification with MLP heads
├── 📄 encoder_concordance.py          # Event ordering with MLP heads  
├── 📄 encoder_mask_time_window.py     # Self-supervised time window prediction
├── 📄 encoder_mask_concordance.py     # Self-supervised concordance prediction
├── 📄 run_encoder_experiments.py      # Unified experiment runner
├── 📄 README.md                       # This file
└── 📂 pycox/                          # Survival analysis utilities
    ├── 📂 datasets/                   # Data loading utilities
    ├── 📂 models/                     # Model implementations
    ├── 📂 evaluation/                 # Evaluation metrics
    └── 📂 preprocessing/              # Data preprocessing tools
```

## 🛠 Usage

### Quick Start
```bash
# Install environment
conda env create -f ../environment_tts_forecasting.yml
conda activate tts_forecasting

# Run time window classification with DeBERTa
python run_encoder_experiments.py \
  --task time_window \
  --model deberta \
  --data_dir data/train \
  --test_dir data/test \
  --epochs 15 \
  --batch_size 12

# Run concordance task with RoBERTa
python run_encoder_experiments.py \
  --task concordance \
  --model roberta \
  --data_dir data/train \
  --test_dir data/test \
  --H 48

# Evaluate all models on masked time window task
python run_encoder_experiments.py \
  --task mask_time_window \
  --model all \
  --data_dir data/train \
  --eval_only \
  --checkpoint_path checkpoints/best.pt
```

### Advanced Configuration
```bash
# Custom hyperparameters
python run_encoder_experiments.py \
  --task time_window \
  --model deberta \
  --data_dir data/train \
  --K 12 \
  --H 72 \
  --lr 2e-5 \
  --batch_size 8 \
  --epochs 20 \
  --patience 5 \
  --timestep_drop_rate 0.1

# Load from configuration file
python run_encoder_experiments.py --config configs/experiment_config.json
```

### Individual Model Scripts
```bash
# Direct script execution
python encoder_time_window.py \
  --model_name microsoft/deberta-v3-base \
  --train_data_directory data/train \
  --test_data_directory data/test \
  --K 8 \
  --H 24 \
  --epochs 10 \
  --batch_size 16

# Self-supervised masked training
python encoder_mask_time_window.py \
  --model_name bert-base-uncased \
  --train_data_directory data/train \
  --test_data_directory data/test \
  --max_length 512 \
  --lr 1e-5
```

## 📚 Dependencies

### Core Requirements
- `torch >= 2.6.0`
- `transformers >= 4.50.0`
- `numpy >= 2.2.0`
- `pandas >= 2.2.0`
- `scikit-learn >= 1.6.0`

### Additional Utilities
- `tensorboard` for experiment tracking
- `tqdm` for progress bars
- `matplotlib` for visualization
- `seaborn` for statistical plots
