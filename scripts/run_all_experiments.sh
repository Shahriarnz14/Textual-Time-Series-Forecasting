#!/bin/bash
#
# Textual Time Series Forecasting - Complete Experiment Suite
# ==========================================================
#
# This script runs comprehensive experiments across both encoder and decoder approaches
# for textual time series forecasting. It demonstrates the full experimental pipeline
# from basic model evaluation to advanced multi-model comparisons.
#
# Usage:
#   bash run_all_experiments.sh [data_dir] [test_dir]
#   bash run_all_experiments.sh /path/to/train /path/to/test
#
# Author: Textual Time Series Forecasting Team

set -e  # Exit on any error

# Configuration
DATA_DIR=${1:-"data/train"}
TEST_DIR=${2:-"data/test"}
OUTPUT_DIR="experiments/$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$OUTPUT_DIR/logs"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/main.log"
}

log_section() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
    log "Starting: $1"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
    log "SUCCESS: $1"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
    log "ERROR: $1"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    log "WARNING: $1"
}

# Check prerequisites
check_prerequisites() {
    log_section "Checking Prerequisites"
    
    # Check if conda environment is activated
    if [[ "$CONDA_DEFAULT_ENV" != "tts_forecasting" ]]; then
        log_warning "tts_forecasting conda environment not activated"
        log "Please run: conda activate tts_forecasting"
        exit 1
    fi
    
    # Check if data directories exist
    if [[ ! -d "$DATA_DIR" ]]; then
        log_error "Training data directory not found: $DATA_DIR"
        exit 1
    fi
    
    if [[ ! -d "$TEST_DIR" ]]; then
        log_error "Test data directory not found: $TEST_DIR"
        exit 1
    fi
    
    # Check if required scripts exist
    if [[ ! -f "encoder_llm/run_encoder_experiments.py" ]]; then
        log_error "Encoder experiment runner not found"
        exit 1
    fi
    
    if [[ ! -f "decoder_llm/run_decoder_experiments.py" ]]; then
        log_error "Decoder experiment runner not found"
        exit 1
    fi
    
    log_success "All prerequisites satisfied"
}

# Run encoder experiments
run_encoder_experiments() {
    log_section "Running Encoder-Based Experiments"
    
    cd encoder_llm
    
    # Time window classification experiments
    log "Starting time window classification experiments..."
    
    for model in bert roberta deberta; do
        log "Running time window classification with $model..."
        if python run_encoder_experiments.py \
            --task time_window \
            --model "$model" \
            --data_dir "../$DATA_DIR" \
            --test_dir "../$TEST_DIR" \
            --output_dir "../$OUTPUT_DIR/encoder" \
            --epochs 15 \
            --patience 5 > "../$LOG_DIR/encoder_time_window_$model.log" 2>&1; then
            log_success "Time window classification with $model completed"
        else
            log_error "Time window classification with $model failed"
        fi
    done
    
    # Concordance experiments
    log "Starting concordance experiments..."
    
    for model in bert roberta deberta; do
        log "Running concordance with $model..."
        if python run_encoder_experiments.py \
            --task concordance \
            --model "$model" \
            --data_dir "../$DATA_DIR" \
            --test_dir "../$TEST_DIR" \
            --output_dir "../$OUTPUT_DIR/encoder" \
            --epochs 15 \
            --patience 5 > "../$LOG_DIR/encoder_concordance_$model.log" 2>&1; then
            log_success "Concordance with $model completed"
        else
            log_error "Concordance with $model failed"
        fi
    done
    
    # Masked experiments (self-supervised)
    log "Starting masked self-supervised experiments..."
    
    for task in mask_time_window mask_concordance; do
        for model in bert roberta; do
            log "Running $task with $model..."
            if python run_encoder_experiments.py \
                --task "$task" \
                --model "$model" \
                --data_dir "../$DATA_DIR" \
                --test_dir "../$TEST_DIR" \
                --output_dir "../$OUTPUT_DIR/encoder" \
                --epochs 12 \
                --patience 4 > "../$LOG_DIR/encoder_${task}_$model.log" 2>&1; then
                log_success "$task with $model completed"
            else
                log_error "$task with $model failed"
            fi
        done
    done
    
    cd ..
    log_success "All encoder experiments completed"
}

# Run decoder experiments  
run_decoder_experiments() {
    log_section "Running Decoder-Based Experiments"
    
    cd decoder_llm
    
    # MLP head training experiments
    log "Starting MLP head training experiments..."
    
    # Test with smaller models first to ensure setup works
    for model in llama-3.2-1b llama-3.1-8b; do
        for window in 24 168; do
            log "Running MLP training with $model, window $window hours..."
            if python run_decoder_experiments.py \
                --approach MLP \
                --model "$model" \
                --data_dir "../$DATA_DIR" \
                --test_dir "../$TEST_DIR" \
                --forecast_window "$window" \
                --epochs 200 \
                --run_dir "../$OUTPUT_DIR/decoder/runs" \
                --batch_size 8 > "../$LOG_DIR/decoder_mlp_${model}_w${window}.log" 2>&1; then
                log_success "MLP training with $model (window $window) completed"
            else
                log_error "MLP training with $model (window $window) failed"
            fi
        done
    done
    
    # Large model experiments (if resources available)
    log "Attempting large model experiments..."
    
    for model in llama-3.3-70b deepseek-r1-70b; do
        log "Running MLP training with $model (24h window)..."
        if python run_decoder_experiments.py \
            --approach MLP \
            --model "$model" \
            --data_dir "../$DATA_DIR" \
            --test_dir "../$TEST_DIR" \
            --forecast_window 24 \
            --epochs 100 \
            --run_dir "../$OUTPUT_DIR/decoder/runs" \
            --batch_size 2 > "../$LOG_DIR/decoder_mlp_${model}_w24.log" 2>&1; then
            log_success "MLP training with $model completed"
        else
            log_warning "MLP training with $model failed (likely resource constraints)"
        fi
    done
    
    # Prompt-based experiments
    log "Starting prompt-based experiments..."
    
    for model in llama-3.1-8b llama-3.3-70b; do
        for prompt in window_system_text.txt window_system_fewshot.txt; do
            log "Running prompt-based evaluation with $model using $prompt..."
            if python run_decoder_experiments.py \
                --approach PROMPT:window \
                --model "$model" \
                --test_dir "../$TEST_DIR" \
                --prompt_template "$prompt" \
                --eval_mode \
                --run_dir "../$OUTPUT_DIR/decoder/runs" > "../$LOG_DIR/decoder_prompt_${model}_${prompt%.txt}.log" 2>&1; then
                log_success "Prompt-based evaluation with $model ($prompt) completed"
            else
                log_warning "Prompt-based evaluation with $model ($prompt) failed"
            fi
        done
    done
    
    cd ..
    log_success "All decoder experiments completed"
}

# Generate experiment summary
generate_summary() {
    log_section "Generating Experiment Summary"
    
    summary_file="$OUTPUT_DIR/experiment_summary.md"
    
    cat > "$summary_file" << EOF
# Textual Time Series Forecasting - Experiment Summary

**Experiment Date**: $(date '+%Y-%m-%d %H:%M:%S')
**Data Directory**: $DATA_DIR
**Test Directory**: $TEST_DIR
**Output Directory**: $OUTPUT_DIR

## Experiment Configuration

### Encoder Experiments
- **Models Tested**: BERT, RoBERTa, DeBERTa
- **Tasks**: Time Window Classification, Concordance, Masked variants
- **Training Epochs**: 12-15 per model
- **Early Stopping**: Patience of 4-5 epochs

### Decoder Experiments  
- **Models Tested**: Llama 3.2 1B, Llama 3.1 8B, Llama 3.3 70B, DeepSeek R1 70B
- **Approaches**: MLP Head Training, Prompt-based Inference
- **Forecast Windows**: 24h, 168h (7 days)
- **Prompt Templates**: Zero-shot, Few-shot variants

## Results Summary

### Encoder Results
EOF

    # Add encoder results if available
    if [[ -d "$OUTPUT_DIR/encoder" ]]; then
        echo "Results files generated in: $OUTPUT_DIR/encoder/" >> "$summary_file"
        find "$OUTPUT_DIR/encoder" -name "*.txt" -exec echo "- {}" \; >> "$summary_file"
    else
        echo "No encoder results found." >> "$summary_file"
    fi

    cat >> "$summary_file" << EOF

### Decoder Results
EOF

    # Add decoder results if available
    if [[ -d "$OUTPUT_DIR/decoder" ]]; then
        echo "Results files generated in: $OUTPUT_DIR/decoder/" >> "$summary_file"
        find "$OUTPUT_DIR/decoder" -name "*.txt" -exec echo "- {}" \; >> "$summary_file"
    else
        echo "No decoder results found." >> "$summary_file"
    fi

    cat >> "$summary_file" << EOF

## Log Files
All detailed logs are available in: $LOG_DIR/

### Key Log Files:
EOF

    # List all log files
    if [[ -d "$LOG_DIR" ]]; then
        find "$LOG_DIR" -name "*.log" -exec echo "- {}" \; >> "$summary_file"
    fi

    cat >> "$summary_file" << EOF

## Next Steps

1. **Review Results**: Check individual result files for detailed metrics
2. **Compare Models**: Use TensorBoard to visualize training curves
3. **Error Analysis**: Review failed experiment logs for debugging
4. **Hyperparameter Tuning**: Adjust parameters based on initial results

## TensorBoard Visualization

To view training progress:
\`\`\`bash
tensorboard --logdir $OUTPUT_DIR/decoder/runs
\`\`\`

## Re-running Individual Experiments

### Encoder Example:
\`\`\`bash
cd encoder_llm
python run_encoder_experiments.py --task time_window --model deberta --data_dir $DATA_DIR --test_dir $TEST_DIR
\`\`\`

### Decoder Example:  
\`\`\`bash
cd decoder_llm
python run_decoder_experiments.py --approach MLP --model llama-3.1-8b --data_dir $DATA_DIR --test_dir $TEST_DIR --forecast_window 24
\`\`\`
EOF

    log_success "Experiment summary generated: $summary_file"
}

# Main execution
main() {
    log_section "Textual Time Series Forecasting - Complete Experiment Suite"
    log "Starting comprehensive experiments..."
    log "Data Directory: $DATA_DIR"
    log "Test Directory: $TEST_DIR"
    log "Output Directory: $OUTPUT_DIR"
    
    # Check prerequisites
    check_prerequisites
    
    # Run experiments
    run_encoder_experiments
    run_decoder_experiments
    
    # Generate summary
    generate_summary
    
    log_section "All Experiments Completed"
    log_success "Comprehensive experiment suite finished!"
    log "Results summary: $OUTPUT_DIR/experiment_summary.md"
    log "Detailed logs: $LOG_DIR/"
    
    echo -e "\n${GREEN}🎉 All experiments completed successfully! 🎉${NC}"
    echo -e "📊 Results: $OUTPUT_DIR/"
    echo -e "📋 Summary: $OUTPUT_DIR/experiment_summary.md"
    echo -e "📝 Logs: $LOG_DIR/"
}

# Run main function
main "$@"
