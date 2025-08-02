# Forecasting Clinical Risk from Textual Time Series: Structuring Narratives for Temporal AI in Healthcare

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.50-yellow.svg)](https://huggingface.co/transformers)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

## 🌟 Overview

This repository presents a novel framework for forecasting clinical risk from textual time series using both **encoder-based** and **decoder-based** transformer models. Our approach transforms irregular, event-based clinical narratives into structured temporal predictions, enabling automated clinical decision support and early risk detection.

### 🎯 Key Contributions

- **Unified Framework**: A comprehensive approach handling both encoder-only (BERT, RoBERTa, DeBERTa-v3, ModernBERT, Bioclinical ModernBERT) and decoder-only (Llama, DeepSeek, OLMO, RedPajama-INCITE, MediPhi-PubMed) architectures
- **Dual Task Learning**: Event ordering prediction (concordance) and time-window classification for robust temporal understanding
- **Clinical Applications**: Real-world evaluation on clinical case reports with irregular time series data
- **Novel Representations**: Innovative text serialization strategies for temporal clinical data

## 🏗️ Architecture

Our framework supports two complementary approaches:

### 📥 Encoder Models (`encoder_llm/`)
- **Fine-tuned Classification**: Task-specific heads on pre-trained encoders
- **Zero-Shot Masking**: Masked language modeling for temporal reasoning
- **Tasks**: Binary classification for event ordering and time-window prediction

### 📤 Decoder Models (`decoder_llm/`)
- **MLP Head Adaptation**: Lightweight heads on frozen language models  
- **Prompt-Based Inference**: Zero-shot and few-shot structured prediction
- **Flexible Output**: Multi-label probability distributions for forecast windows

## 🚀 Quick Start

### Prerequisites
```bash
# Clone the repository
git clone https://github.com/Shahriarnz14/Textual-Time-Series-Forecasting.git
cd Textual-Time-Series-Forecasting

# Create and activate conda environment
conda env create -f environment_tts_forecasting.yml
conda activate tts_forecasting
```

### 🏃‍♂️ Running Experiments

#### Encoder Models
```bash
# Time window classification
cd encoder_llm
python run_encoder_experiments.py --task time_window --model deberta --data_dir data/train --test_dir data/test

# Event ordering (concordance)
python run_encoder_experiments.py --task concordance --model roberta --data_dir data/train --test_dir data/test
```

#### Decoder Models
```bash
# MLP head training
cd decoder_llm  
python run_decoder_experiments.py --approach MLP --model llama-3.3-70b --forecast_window 24

# Prompt-based evaluation
python run_decoder_experiments.py --approach PROMPT:window --model llama-3.3-70b --eval_mode
```

## 📊 Data Format

Our framework expects clinical time series data in CSV format:

```csv
event,timestamp
"Patient admitted to ICU",0
"Elevated troponin levels observed",2.5
"Cardiac catheterization performed",6.0
"Patient transferred to ward",24.0
```

Each case report contains:
- **Event**: Free-text clinical observation
- **Timestamp**: Hours relative to admission

## 🔬 Experimental Setup

### Evaluation Tasks

1. **Event Ordering (Concordance)**: Given pairs of future events, predict temporal ordering
2. **Time Window Classification**: Predict if events occur within specified time horizons

### Models Evaluated

#### Encoder Models
- BERT (`bert-base-uncased`)
- RoBERTa (`roberta-base`) 
- DeBERTa (`microsoft/deberta-v3-small`)
- ModernBERT (`answerdotai/ModernBERT-base` and `large`)
- Bioclinical ModernBERT (`thomas-sounack/BioClinical-ModernBERT-base` and `large`)

#### Decoder Models  
- Llama 3.3 70B (`meta-llama/Llama-3.3-70B-Instruct`)
- Llama 3.1 8B (`meta-llama/Llama-3.1-8B-Instruct`)
- DeepSeek R1 (`deepseek-ai/DeepSeek-R1-Distill-Llama-70B`)
- *Open-Source LLMs*:
  - OLMO 32B Instruct (`allenai/OLMo-2-0325-32B-Instruct`)
  - MediPhi-PubMed (`microsoft/MediPhi-PubMed`)
  - RedPajama-INCITE 7B Instruct (`togethercomputer/RedPajama-INCITE-7B-Instruct`)

### Metrics
- **Concordance Index (C-Index)**: For event ordering tasks
- **F1-Score**: For time window classification  
- **AUROC**: For binary prediction performance

## 📁 Repository Structure

```
Textual-Time-Series-Forecasting/
├── 📂 encoder_llm/           # Encoder-based models
│   ├── encoder_time_window.py
│   ├── encoder_concordance.py  
│   ├── encoder_mask_*.py
│   └── run_encoder_experiments.py
├── 📂 decoder_llm/           # Decoder-based models
│   ├── decoder.py
│   ├── generate_args.py
│   ├── prompts/
│   └── run_decoder_experiments.py
├── 📂 scripts/               # Batch experiment scripts
├── 📄 environment_tts_forecasting.yml
└── 📄 README.md
```

## 🔬 Results Summary

Our approach demonstrates strong performance across clinical forecasting tasks:

- **Encoder Models**: Achieve 0.72-0.78 C-Index for temporal ordering
- **Decoder Models**: 0.65-0.82 F1-Score for time window prediction  
- **Prompt-Based**: Competitive zero-shot performance with minimal examples

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@article{noroozizadeh2025forecasting,
  title={Forecasting from clinical textual time series: Adaptations of the encoder and decoder language model families},
  author={Noroozizadeh, Shahriar and Kumar, Sayantan and Weiss, Jeremy C},
  journal={arXiv preprint arXiv:2504.10340},
  year={2025}
}
```
