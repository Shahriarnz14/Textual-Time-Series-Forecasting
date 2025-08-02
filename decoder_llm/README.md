# Decoder-Based Textual Time Series Forecasting

<div align="center">

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.50-yellow.svg)](https://huggingface.co/transformers)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)

*Decoder-only transformer models for clinical textual time series forecasting*

</div>

## Overview

This module implements decoder-based approaches for textual time series forecasting using large language models (LLMs). Our framework leverages the generative capabilities of decoder-only architectures to perform structured prediction over clinical event sequences through both fine-tuning and prompt-based approaches.

### Key Features

- **Dual Approach Framework**: MLP head fine-tuning and prompt-based inference
- **Large Language Model Support**: Llama, DeepSeek, and other state-of-the-art models
- **Structured Generation**: Multi-label probability outputs and natural language responses
- **Flexible Forecasting**: Variable time windows and event counts
- **Zero/Few-Shot Capabilities**: Prompt-based inference without parameter updates

## Data Format

Our framework processes clinical time series data with timestamped events and generates structured forecasts:

### Input Format
```csv
event,timestamp
"Patient admitted to emergency department",0.0
"Blood pressure 180/95 recorded",1.5
"Chest X-ray shows infiltrates",3.0
"Antibiotic therapy initiated",4.5
"Patient transferred to ICU",8.0
```

### Output Formats

#### MLP Head Approach
Multi-label probability vector for K events:
```python
[0.8, 0.3, 0.1, 0.9, 0.2, 0.6, 0.4, 0.7]  # 8 event probabilities
```

#### Prompt-Based Approach
Structured text response for time window prediction:
```
0.8 | 0.3 | 0.1 | 0.9 | 0.2 | 0.6 | 0.4 | 0.7
```

## Architecture Approaches

### 1. MLP Head Fine-Tuning

**Concept**: Freeze pre-trained LLM and train lightweight classification head

**Architecture**:
```
[Input Text] → LLM (Frozen) → Hidden States → MLP Head → Predictions
```

**Training**: 
- Gradient updates only to MLP parameters
- BCE loss for multi-label classification
- Memory efficient with large models

**Advantages**:
- Fast training and inference
- Preserves pre-trained knowledge
- Scalable to very large models

### 2. Prompt-Based Inference

**Concept**: Use natural language prompts to elicit structured predictions

**Zero-Shot Example**:
```
You are an expert physician. Reply with structured predictions in a k-item, 
bar-separated row. For example, if k=3 events (A, B, C) and only B occurs 
in the time window, then: 0 | 1 | 0

Given: [Patient history...]
Predict: Which events occur in the next 24 hours?
```

**Few-Shot Example**: Includes demonstration examples before the target prediction

**Advantages**:
- No parameter updates required
- Leverages model's reasoning capabilities
- Interpretable prediction process

## Supported Models

### Llama Family

#### Llama 3.3 70B (`meta-llama/Llama-3.3-70B-Instruct`)
- **Parameters**: 70B
- **Context Length**: 8K tokens
- **Strengths**: State-of-the-art reasoning, medical knowledge
- **Use Case**: High-accuracy applications, complex reasoning

#### Llama 3.1 8B (`meta-llama/Llama-3.1-8B-Instruct`)
- **Parameters**: 8B  
- **Context Length**: 8K tokens
- **Strengths**: Efficient, good performance-to-size ratio
- **Use Case**: Resource-constrained environments

#### Llama 3.2 1B (`meta-llama/Llama-3.2-1B-Instruct`)
- **Parameters**: 1B
- **Context Length**: 2K tokens
- **Strengths**: Very fast inference, minimal resources
- **Use Case**: Edge deployment, rapid prototyping

### DeepSeek Family

#### DeepSeek R1 70B (`deepseek-ai/DeepSeek-R1-Distill-Llama-70B`)
- **Parameters**: 70B (distilled)
- **Context Length**: 8K tokens  
- **Strengths**: Reasoning-optimized, efficient training
- **Use Case**: Complex clinical reasoning tasks

#### DeepSeek R1 8B (`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`)
- **Parameters**: 8B (distilled)
- **Context Length**: 8K tokens
- **Strengths**: Compact reasoning capabilities
- **Use Case**: Balanced performance and efficiency


### Open-Source LLMs

#### OLMO 32B Instruct (`allenai/OLMo-2-0325-32B-Instruct`)
#### MediPhi-PubMed (`microsoft/MediPhi-PubMed`)
#### RedPajama-INCITE 7B Instruct (`togethercomputer/RedPajama-INCITE-7B-Instruct`)

## Prediction Tasks

### Time Window Forecasting

**Objective**: Predict which events occur within specified time horizons

**Input Representation**:
```
Patient admitted at 0h. Chest pain reported at 2h. ECG abnormal at 3h.
Predict events in next 24h: [cardiac_cath, icu_transfer, surgery, discharge]
```

**Output**: Binary probability for each event occurring within the window

**Metrics**: 
- F1-Score (macro-averaged)
- AUROC per event
- Concordance with clinical timelines

### Event Ordering Prediction

**Objective**: Predict temporal ordering of future clinical events

**Input Representation**:
```
Given history: [admission, symptoms, diagnosis]
Order these future events: [treatment_A, treatment_B, outcome_C]
```

**Output**: Pairwise ordering probabilities or ranking scores

**Metrics**:
- Concordance Index (C-Index)
- Ranking accuracy

## Training & Evaluation

### MLP Head Training Pipeline

```python
# 1. Load pre-trained LLM (frozen)
model = AutoModelForCausalLM.from_pretrained(model_name)
for param in model.parameters():
    param.requires_grad = False

# 2. Add classification head  
hidden_size = model.config.hidden_size
mlp_head = MLPHead(hidden_size, num_labels)

# 3. Train head only
optimizer = torch.optim.Adam(mlp_head.parameters(), lr=1e-3)
```

### Training Configuration
```python
# Default hyperparameters
learning_rate = 1e-3       # Higher LR for head-only training
epochs = 1000              # More epochs due to small parameter set
batch_size = 4             # Large model constraint  
accumulation_steps = 5     # Effective batch size increase
checkpoint_interval = 20   # Regular checkpointing
max_length = 2048          # Extended context for clinical notes
```

### Prompt Engineering

#### System Prompts
Located in `prompts/` directory:

- `window_system_text.txt`: Zero-shot window prediction
- `window_system_fewshot.txt`: Few-shot with examples
- `window_system_text_1to5.txt`: Rating scale (1-5) format
- `window_system_fewshot_1to5.txt`: Few-shot rating format

#### Prompt Structure
```
[SYSTEM MESSAGE]
You are an expert physician...

[TASK DESCRIPTION] 
Predict which events occur in the next X hours...

[FEW-SHOT EXAMPLES] (if applicable)
Example 1: Given: [...] Answer: 0 | 1 | 0
Example 2: Given: [...] Answer: 1 | 0 | 1

[TARGET QUESTION]
Given: [Patient history]
Predict: [Event list]
```

## File Structure

```
decoder_llm/
├── 📄 decoder.py                      # Main training/evaluation script
├── 📄 run_decoder_experiments.py      # Unified experiment runner
├── 📄 generate_args.py                # Experiment configuration generator
├── 📄 json_comment_reader.py          # JSON config parser with comments
├── 📂 prompts/                        # Prompt templates
│   ├── window_system_text.txt         # Zero-shot window prompts
│   ├── window_system_fewshot.txt      # Few-shot window prompts  
│   ├── window_system_text_1to5.txt    # Rating scale prompts
│   └── window_system_fewshot_1to5.txt # Few-shot rating prompts
└── 📄 README.md                       # This file
```

## Usage

### Quick Start

#### MLP Head Training
```bash
# Install environment
conda env create -f ../environment_tts_forecasting.yml
conda activate tts_forecasting

# Train MLP head on Llama 3.3 70B
python run_decoder_experiments.py \
  --approach MLP \
  --model llama-3.3-70b \
  --data_dir data/train \
  --test_dir data/test \
  --forecast_window 24 \
  --epochs 500

# Train with custom parameters
python run_decoder_experiments.py \
  --approach MLP \
  --model llama-3.1-8b \
  --forecast_window 168 \
  --num_labels 12 \
  --batch_size 8 \
  --lr 5e-4
```

#### Prompt-Based Evaluation
```bash
# Zero-shot evaluation
python run_decoder_experiments.py \
  --approach PROMPT:window \
  --model llama-3.3-70b \
  --prompt_template window_system_text.txt \
  --eval_mode \
  --test_dir data/test

# Few-shot evaluation  
python run_decoder_experiments.py \
  --approach PROMPT:window \
  --model llama-3.3-70b \
  --prompt_template window_system_fewshot.txt \
  --eval_mode \
  --test_dir data/test
```

### Advanced Configuration

#### Custom Experiment Setup
```bash
# Multi-model comparison
python run_decoder_experiments.py \
  --approach MLP \
  --model all \
  --forecast_window 24 \
  --data_dir data/train \
  --test_dir data/test

# Custom cache directory for large models
python run_decoder_experiments.py \
  --approach MLP \
  --model deepseek-r1-70b \
  --cache_dir /path/to/hf_cache \
  --forecast_window 72

# Load from configuration file
python run_decoder_experiments.py --config configs/decoder_config.json
```

#### Direct Script Usage
```bash
# Run decoder.py directly
python decoder.py \
  --approach MLP \
  --base_model meta-llama/Llama-3.3-70B-Instruct \
  --data_dir data/train \
  --test_dir data/test \
  --forecast_window 24 \
  --num_labels 8 \
  --learning_rate 1e-3 \
  --epochs 1000 \
  --batch_size 4

# Prompt-based inference
python decoder.py \
  --approach "PROMPT:window" \
  --base_model meta-llama/Llama-3.3-70B-Instruct \
  --systext_file prompts/window_system_fewshot.txt \
  --test_dir data/test \
  --eval_mode
```
