#!/bin/bash
#
# Benchmarking Script for Textual Time Series Forecasting
# =======================================================
#
# This script runs comprehensive benchmarks across all models and configurations
# to generate publication-ready results and comparisons.
#
# Usage:
#   bash benchmark_suite.sh [data_dir] [test_dir] [num_runs]
#
# Author: Textual Time Series Forecasting Team

set -e

# Configuration
DATA_DIR=${1:-"data/train"}
TEST_DIR=${2:-"data/test"}
NUM_RUNS=${3:-3}  # Number of runs for statistical significance
OUTPUT_DIR="benchmarks/$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$OUTPUT_DIR/logs"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/benchmark.log"
}

echo -e "${PURPLE}"
echo "================================================================="
echo "  Textual Time Series Forecasting - Comprehensive Benchmarks"  
echo "================================================================="
echo -e "${NC}"

log "Starting comprehensive benchmarking suite"
log "Data Directory: $DATA_DIR"
log "Test Directory: $TEST_DIR"
log "Number of runs per experiment: $NUM_RUNS"
log "Output Directory: $OUTPUT_DIR"

# Model configurations for benchmarking
declare -A ENCODER_MODELS=(
    ["bert"]="bert-base-uncased"
    ["roberta"]="roberta-base"  
    ["deberta"]="microsoft/deberta-v3-base"
    ["deberta-small"]="microsoft/deberta-v3-small"
    ["modernbert"]="answerdotai/ModernBERT-base"
)

declare -A DECODER_MODELS=(
    ["llama-3.2-1b"]="meta-llama/Llama-3.2-1B-Instruct"
    ["llama-3.1-8b"]="meta-llama/Llama-3.1-8B-Instruct"
    ["llama-3.3-70b"]="meta-llama/Llama-3.3-70B-Instruct"
    ["deepseek-r1-8b"]="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    ["deepseek-r1-70b"]="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
)

ENCODER_TASKS=("time_window" "concordance" "mask_time_window" "mask_concordance")
DECODER_APPROACHES=("MLP" "PROMPT:window")
FORECAST_WINDOWS=(1 24 168)  # 1 hour, 1 day, 1 week

# Function to run encoder benchmarks
run_encoder_benchmarks() {
    echo -e "\n${BLUE}Starting Encoder Model Benchmarks${NC}"
    
    cd encoder_llm
    
    for task in "${ENCODER_TASKS[@]}"; do
        echo -e "\n${YELLOW}Benchmarking task: $task${NC}"
        
        for model in "${!ENCODER_MODELS[@]}"; do
            for run in $(seq 1 $NUM_RUNS); do
                experiment_name="${task}_${model}_run${run}"
                log "Running $experiment_name"
                
                if timeout 3600 python run_encoder_experiments.py \
                    --task "$task" \
                    --model "$model" \
                    --data_dir "../$DATA_DIR" \
                    --test_dir "../$TEST_DIR" \
                    --output_dir "../$OUTPUT_DIR/encoder" \
                    --experiment_name "$experiment_name" \
                    --epochs 20 \
                    --patience 5 \
                    --verbose > "../$LOG_DIR/encoder_${experiment_name}.log" 2>&1; then
                    log "SUCCESS: $experiment_name"
                else
                    log "FAILED: $experiment_name (timeout or error)"
                fi
            done
        done
    done
    
    cd ..
}

# Function to run decoder benchmarks
run_decoder_benchmarks() {
    echo -e "\n${BLUE}Starting Decoder Model Benchmarks${NC}"
    
    cd decoder_llm
    
    # MLP head benchmarks
    echo -e "\n${YELLOW}Benchmarking MLP head approach${NC}"
    
    for model in "${!DECODER_MODELS[@]}"; do
        for window in "${FORECAST_WINDOWS[@]}"; do
            for run in $(seq 1 $NUM_RUNS); do
                experiment_name="MLP_${model}_fw${window}_run${run}"
                log "Running $experiment_name"
                
                # Set batch size based on model size
                if [[ "$model" == *"70b"* ]]; then
                    batch_size=2
                    epochs=50
                elif [[ "$model" == *"8b"* ]]; then
                    batch_size=4
                    epochs=100
                else
                    batch_size=8
                    epochs=200
                fi
                
                if timeout 7200 python run_decoder_experiments.py \
                    --approach MLP \
                    --model "$model" \
                    --data_dir "../$DATA_DIR" \
                    --test_dir "../$TEST_DIR" \
                    --forecast_window "$window" \
                    --epochs "$epochs" \
                    --batch_size "$batch_size" \
                    --run_dir "../$OUTPUT_DIR/decoder" \
                    --experiment_name "$experiment_name" \
                    --verbose > "../$LOG_DIR/decoder_${experiment_name}.log" 2>&1; then
                    log "SUCCESS: $experiment_name"
                else
                    log "FAILED: $experiment_name (timeout or error)"
                fi
            done
        done
    done
    
    # Prompt-based benchmarks
    echo -e "\n${YELLOW}Benchmarking prompt-based approach${NC}"
    
    PROMPT_TEMPLATES=("window_system_text.txt" "window_system_fewshot.txt")
    
    for model in llama-3.1-8b llama-3.3-70b deepseek-r1-8b; do
        for template in "${PROMPT_TEMPLATES[@]}"; do
            for run in $(seq 1 $NUM_RUNS); do
                experiment_name="PROMPT_${model}_${template%.txt}_run${run}"
                log "Running $experiment_name"
                
                if timeout 1800 python run_decoder_experiments.py \
                    --approach PROMPT:window \
                    --model "$model" \
                    --test_dir "../$TEST_DIR" \
                    --prompt_template "$template" \
                    --eval_mode \
                    --run_dir "../$OUTPUT_DIR/decoder" \
                    --experiment_name "$experiment_name" \
                    --verbose > "../$LOG_DIR/decoder_${experiment_name}.log" 2>&1; then
                    log "SUCCESS: $experiment_name"
                else
                    log "FAILED: $experiment_name (timeout or error)"
                fi
            done
        done
    done
    
    cd ..
}

# Function to collect and analyze results
analyze_results() {
    echo -e "\n${BLUE}Analyzing Benchmark Results${NC}"
    
    python3 << EOF
import os
import pandas as pd
import numpy as np
from pathlib import Path
import re

def parse_log_files(log_dir):
    """Parse log files to extract performance metrics."""
    results = []
    
    for log_file in Path(log_dir).glob("*.log"):
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Extract experiment details from filename
            filename = log_file.stem
            if filename.startswith('encoder_'):
                parts = filename.replace('encoder_', '').split('_')
                model_type = 'encoder'
                task = parts[0]
                model = parts[1]
                run = parts[2] if len(parts) > 2 else 'run1'
            elif filename.startswith('decoder_'):
                parts = filename.replace('decoder_', '').split('_')
                model_type = 'decoder'
                approach = parts[0]
                model = parts[1]
                run = parts[-1] if parts[-1].startswith('run') else 'run1'
            else:
                continue
            
            # Extract metrics from log content
            metrics = {}
            
            # F1 Score
            f1_match = re.search(r'[Ff]1[- ][Ss]core[:\s]+([\d.]+)', content)
            if f1_match:
                metrics['f1_score'] = float(f1_match.group(1))
            
            # Concordance Index
            c_index_match = re.search(r'[Cc]oncordance[:\s]+([\d.]+)', content)
            if c_index_match:
                metrics['c_index'] = float(c_index_match.group(1))
            
            # Training time (if available)
            time_match = re.search(r'Training.*?(\d+)\s*min', content)
            if time_match:
                metrics['training_time_min'] = int(time_match.group(1))
            
            if metrics:  # Only add if we found some metrics
                result = {
                    'model_type': model_type,
                    'model': model,
                    'run': run,
                    **metrics
                }
                
                if model_type == 'encoder':
                    result['task'] = task
                else:
                    result['approach'] = approach
                
                results.append(result)
                
        except Exception as e:
            print(f"Error parsing {log_file}: {e}")
    
    return pd.DataFrame(results)

def generate_benchmark_report(results_df, output_dir):
    """Generate comprehensive benchmark report."""
    
    report_path = Path(output_dir) / 'benchmark_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# Textual Time Series Forecasting - Benchmark Report\\n\\n")
        f.write(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        
        if results_df.empty:
            f.write("⚠️ No results found. Check log files for errors.\\n")
            return
        
        # Encoder results
        encoder_results = results_df[results_df['model_type'] == 'encoder']
        if not encoder_results.empty:
            f.write("## 📊 Encoder Model Results\\n\\n")
            
            # Group by task and model, compute statistics
            for task in encoder_results['task'].unique():
                task_data = encoder_results[encoder_results['task'] == task]
                f.write(f"### {task.replace('_', ' ').title()}\\n\\n")
                f.write("| Model | F1 Score | C-Index | Avg±Std |\\n")
                f.write("|-------|----------|---------|---------|\\n")
                
                for model in task_data['model'].unique():
                    model_data = task_data[task_data['model'] == model]
                    
                    # Compute statistics
                    f1_scores = model_data['f1_score'].dropna()
                    c_indices = model_data['c_index'].dropna()
                    
                    f1_mean = f1_scores.mean() if not f1_scores.empty else np.nan
                    f1_std = f1_scores.std() if not f1_scores.empty else np.nan
                    c_mean = c_indices.mean() if not c_indices.empty else np.nan
                    c_std = c_indices.std() if not c_indices.empty else np.nan
                    
                    f.write(f"| {model} | {f1_mean:.3f}±{f1_std:.3f} | {c_mean:.3f}±{c_std:.3f} | - |\\n")
                
                f.write("\\n")
        
        # Decoder results
        decoder_results = results_df[results_df['model_type'] == 'decoder']
        if not decoder_results.empty:
            f.write("## 🤖 Decoder Model Results\\n\\n")
            
            for approach in decoder_results['approach'].unique():
                approach_data = decoder_results[decoder_results['approach'] == approach]
                f.write(f"### {approach} Approach\\n\\n")
                f.write("| Model | F1 Score | C-Index | Avg±Std |\\n")
                f.write("|-------|----------|---------|---------|\\n")
                
                for model in approach_data['model'].unique():
                    model_data = approach_data[approach_data['model'] == model]
                    
                    f1_scores = model_data['f1_score'].dropna()
                    c_indices = model_data['c_index'].dropna()
                    
                    f1_mean = f1_scores.mean() if not f1_scores.empty else np.nan
                    f1_std = f1_scores.std() if not f1_scores.empty else np.nan
                    c_mean = c_indices.mean() if not c_indices.empty else np.nan
                    c_std = c_indices.std() if not c_indices.empty else np.nan
                    
                    f.write(f"| {model} | {f1_mean:.3f}±{f1_std:.3f} | {c_mean:.3f}±{c_std:.3f} | - |\\n")
                
                f.write("\\n")
        
        f.write("## 📈 Key Findings\\n\\n")
        f.write("- **Best Encoder**: [Analysis needed]\\n")
        f.write("- **Best Decoder**: [Analysis needed]\\n")
        f.write("- **Efficiency Winner**: [Analysis needed]\\n")
        f.write("- **Clinical Applicability**: [Analysis needed]\\n\\n")
        
        f.write("## 🔍 Statistical Analysis\\n\\n")
        f.write("All results are averaged across {num_runs} independent runs.\\n".format(num_runs=$NUM_RUNS))
        f.write("Standard deviations indicate result stability across runs.\\n\\n")
    
    print(f"✅ Benchmark report generated: {report_path}")

# Main analysis
results_df = parse_log_files("$LOG_DIR")
generate_benchmark_report(results_df, "$OUTPUT_DIR")

# Save raw results
results_df.to_csv("$OUTPUT_DIR/raw_results.csv", index=False)
print(f"📊 Raw results saved: $OUTPUT_DIR/raw_results.csv")

print(f"\\n📋 Summary Statistics:")
print(f"Total experiments: {len(results_df)}")
print(f"Encoder experiments: {len(results_df[results_df['model_type'] == 'encoder'])}")
print(f"Decoder experiments: {len(results_df[results_df['model_type'] == 'decoder'])}")
EOF
}

# Main execution
main() {
    # Check if data directories exist
    if [[ ! -d "$DATA_DIR" ]]; then
        echo -e "${RED}Error: Data directory not found: $DATA_DIR${NC}"
        exit 1
    fi
    
    if [[ ! -d "$TEST_DIR" ]]; then
        echo -e "${RED}Error: Test directory not found: $TEST_DIR${NC}"
        exit 1
    fi
    
    # Check if conda environment is activated
    if [[ "$CONDA_DEFAULT_ENV" != "tts_forecasting" ]]; then
        echo -e "${RED}Warning: tts_forecasting conda environment not activated${NC}"
        echo "Please run: conda activate tts_forecasting"
        exit 1
    fi
    
    log "Starting benchmark suite execution"
    
    # Run benchmarks
    run_encoder_benchmarks
    run_decoder_benchmarks
    
    # Analyze results
    analyze_results
    
    echo -e "\n${GREEN}🎉 Benchmark Suite Completed! 🎉${NC}"
    echo -e "📊 Results: $OUTPUT_DIR/benchmark_report.md"
    echo -e "📈 Raw Data: $OUTPUT_DIR/raw_results.csv"
    echo -e "📝 Logs: $LOG_DIR/"
    
    log "Benchmark suite completed successfully"
}

# Execute main function
main "$@"
