#!/usr/bin/env python3
"""
Textual Time Series Forecasting - Decoder Models Runner
======================================================

This script provides a unified interface for running all decoder-based experiments
for textual time series forecasting, including MLP head training and prompt-based
inference approaches.

Usage:
    python run_decoder_experiments.py --approach MLP --model llama-3.3-70b --forecast_window 24
    python run_decoder_experiments.py --approach PROMPT:window --model llama-3.3-70b --eval_mode
    python run_decoder_experiments.py --config configs/experiment_config.json

Author: Textual Time Series Forecasting Team
"""

import argparse
import json
import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Available models and their configurations
AVAILABLE_MODELS = {
    'llama-3.3-70b': {
        'model_name': 'meta-llama/Llama-3.3-70B-Instruct',
        'cache_dir': None,
        'batch_size': 4,
        'max_len': 2048
    },
    'llama-3.1-8b': {
        'model_name': 'meta-llama/Llama-3.1-8B-Instruct', 
        'cache_dir': None,
        'batch_size': 8,
        'max_len': 2048
    },
    'llama-3.2-1b': {
        'model_name': 'meta-llama/Llama-3.2-1B-Instruct',
        'cache_dir': None,
        'batch_size': 16,
        'max_len': 2048
    },
    'deepseek-r1-70b': {
        'model_name': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
        'cache_dir': None,
        'batch_size': 4,
        'max_len': 2048
    },
    'deepseek-r1-8b': {
        'model_name': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
        'cache_dir': None,
        'batch_size': 8,
        'max_len': 2048
    }
}

# Available approaches
AVAILABLE_APPROACHES = {
    'MLP': {
        'description': 'MLP head on frozen language model',
        'requires_training': True
    },
    'PROMPT:window': {
        'description': 'Prompt-based time window prediction',
        'requires_training': False
    },
    'PROMPT:ordering': {
        'description': 'Prompt-based event ordering prediction', 
        'requires_training': False
    }
}

# Available prompt templates
PROMPT_TEMPLATES = {
    'window_system_text.txt': 'Zero-shot window prediction',
    'window_system_fewshot.txt': 'Few-shot window prediction',
    'window_system_text_1to5.txt': 'Zero-shot 1-to-5 prediction',
    'window_system_fewshot_1to5.txt': 'Few-shot 1-to-5 prediction'
}

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run decoder-based textual time series forecasting experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # MLP head training
  python run_decoder_experiments.py --approach MLP --model llama-3.3-70b --forecast_window 24

  # Prompt-based evaluation
  python run_decoder_experiments.py --approach PROMPT:window --model llama-3.3-70b --eval_mode

  # Custom hyperparameters
  python run_decoder_experiments.py --approach MLP --model llama-3.1-8b --epochs 100 --lr 1e-4

  # All models on MLP approach
  python run_decoder_experiments.py --approach MLP --model all --forecast_window 168

  # Load from config file
  python run_decoder_experiments.py --config configs/decoder_config.json
        """
    )
    
    # Model and approach selection
    parser.add_argument('--approach', type=str, choices=list(AVAILABLE_APPROACHES.keys()),
                       help='Approach to use (MLP or PROMPT variants)')
    parser.add_argument('--model', type=str, choices=list(AVAILABLE_MODELS.keys()) + ['all'],
                       help='Model type to use (or "all" for all models)')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, help='Training data directory')
    parser.add_argument('--test_dir', type=str, help='Test data directory')
    
    # Model configuration
    parser.add_argument('--forecast_window', type=int, default=24, 
                       help='Forecast window in hours')
    parser.add_argument('--num_labels', type=int, default=8,
                       help='Number of labels/events to predict')
    parser.add_argument('--max_len', type=int, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    
    # Training configuration
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--train_epochs_per_eval', type=int, default=10,
                       help='Training epochs per evaluation')
    parser.add_argument('--accumulation_steps', type=int, default=5,
                       help='Gradient accumulation steps')
    parser.add_argument('--checkpoint_interval', type=int, default=20,
                       help='Checkpoint save interval')
    
    # Decoder-specific options
    parser.add_argument('--separate_labels', action='store_true',
                       help='Use separate labels for concordance and window tasks')
    parser.add_argument('--sof', type=int, default=20,
                       help='Step out factor for evaluation')
    parser.add_argument('--cache_dir', type=str, help='HuggingFace cache directory')
    
    # Prompt-based options
    parser.add_argument('--systext_file', type=str, help='System text file for prompting')
    parser.add_argument('--prompt_template', type=str, choices=list(PROMPT_TEMPLATES.keys()),
                       help='Pre-defined prompt template to use')
    
    # Execution modes
    parser.add_argument('--eval_mode', action='store_true',
                       help='Run evaluation only (no training)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show commands that would be executed without running them')
    
    # Checkpoint and output
    parser.add_argument('--checkpoint_loadpath', type=str, 
                       help='Path to checkpoint for evaluation or continued training')
    parser.add_argument('--run_dir', type=str, default='runs',
                       help='Directory for TensorBoard runs and checkpoints')
    parser.add_argument('--experiment_name', type=str, help='Custom experiment name')
    
    # Config file option
    parser.add_argument('--config', type=str, help='Path to JSON config file')
    
    # Utility options
    parser.add_argument('--list_models', action='store_true', help='List available models and exit')
    parser.add_argument('--list_approaches', action='store_true', help='List available approaches and exit')
    parser.add_argument('--list_prompts', action='store_true', help='List available prompt templates and exit')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    return parser.parse_args()

def load_config_file(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config file {config_path}: {e}")
        sys.exit(1)

def list_models():
    """List all available models."""
    print("\n🤖 Available Models:")
    print("=" * 70)
    for model_key, model_info in AVAILABLE_MODELS.items():
        print(f"  {model_key:20} -> {model_info['model_name']}")
        print(f"  {'':20}    Batch Size: {model_info['batch_size']}, Max Length: {model_info['max_len']}")
    print()

def list_approaches():
    """List all available approaches."""
    print("\n🎯 Available Approaches:")
    print("=" * 60)
    for approach_key, approach_info in AVAILABLE_APPROACHES.items():
        print(f"  {approach_key:15} -> {approach_info['description']}")
        print(f"  {'':15}    Training Required: {approach_info['requires_training']}")
    print()

def list_prompts():
    """List all available prompt templates."""
    print("\n📝 Available Prompt Templates:")
    print("=" * 60)
    for template, description in PROMPT_TEMPLATES.items():
        print(f"  {template:30} -> {description}")
    print()

def validate_arguments(args: argparse.Namespace) -> bool:
    """Validate provided arguments."""
    if args.list_models or args.list_approaches or args.list_prompts:
        return True
        
    if not args.approach:
        logger.error("Approach must be specified")
        return False
        
    if not args.model:
        logger.error("Model must be specified")
        return False
    
    # Validate approach-specific requirements
    approach_info = AVAILABLE_APPROACHES[args.approach]
    
    if approach_info['requires_training'] and not args.eval_mode and not args.data_dir:
        logger.error("Training data directory must be specified for training approaches")
        return False
        
    if args.eval_mode and approach_info['requires_training'] and not args.checkpoint_loadpath:
        logger.error("Checkpoint path must be specified for evaluation-only mode with training approaches")
        return False
        
    if args.approach.startswith('PROMPT') and not args.systext_file and not args.prompt_template:
        logger.error("System text file or prompt template must be specified for prompt-based approaches")
        return False
        
    # Validate paths exist
    if args.data_dir and not os.path.exists(args.data_dir):
        logger.error(f"Training data directory does not exist: {args.data_dir}")
        return False
        
    if args.test_dir and not os.path.exists(args.test_dir):
        logger.error(f"Test data directory does not exist: {args.test_dir}")
        return False
        
    if args.systext_file and not os.path.exists(args.systext_file):
        logger.error(f"System text file does not exist: {args.systext_file}")
        return False
        
    return True

def setup_prompt_file(args: argparse.Namespace) -> Optional[str]:
    """Setup the prompt file based on template or custom file."""
    if args.systext_file:
        return args.systext_file
        
    if args.prompt_template:
        prompt_path = os.path.join('prompts', args.prompt_template)
        if os.path.exists(prompt_path):
            return prompt_path
        else:
            logger.error(f"Prompt template not found: {prompt_path}")
            return None
            
    return None

def generate_experiment_name(approach: str, model: str, forecast_window: int, timestamp: str) -> str:
    """Generate a unique experiment name."""
    return f"{approach.replace(':', '_')}_{model}_fw{forecast_window}_{timestamp}"

def build_command(args: argparse.Namespace, model_key: str, experiment_name: str) -> List[str]:
    """Build the command to execute for a specific model."""
    model_info = AVAILABLE_MODELS[model_key]
    
    # Start with the script
    cmd = ['python', 'decoder.py']
    
    # Basic parameters
    cmd.extend(['--approach', args.approach])
    cmd.extend(['--base_model', model_info['model_name']])
    cmd.extend(['--forecast_window', str(args.forecast_window)])
    cmd.extend(['--num_labels', str(args.num_labels)])
    cmd.extend(['--infix', experiment_name])
    
    # Model-specific parameters
    max_len = args.max_len if args.max_len else model_info['max_len']
    batch_size = args.batch_size if args.batch_size else model_info['batch_size']
    cmd.extend(['--max_len', str(max_len)])
    cmd.extend(['--batch_size', str(batch_size)])
    
    # Cache directory
    cache_dir = args.cache_dir if args.cache_dir else model_info['cache_dir']
    if cache_dir:
        cmd.extend(['--cache_dir', cache_dir])
    
    # Training parameters
    cmd.extend(['--learning_rate', str(args.lr)])
    cmd.extend(['--epochs', str(args.epochs)])
    cmd.extend(['--train_epochs_per_eval', str(args.train_epochs_per_eval)])
    cmd.extend(['--accumulation_steps', str(args.accumulation_steps)])
    cmd.extend(['--checkpoint_interval', str(args.checkpoint_interval)])
    cmd.extend(['--sof', str(args.sof)])
    
    # Data directories
    if args.data_dir:
        cmd.extend(['--data_dir', args.data_dir])
    if args.test_dir:
        cmd.extend(['--test_dir', args.test_dir])
    elif args.data_dir:
        # Use training dir as test dir if test dir not specified
        cmd.extend(['--test_dir', args.data_dir])
        
    # Run directory
    run_dir = os.path.join(args.run_dir, experiment_name)
    cmd.extend(['--run_dir', run_dir])
    
    # Prompt-based options
    if args.approach.startswith('PROMPT'):
        systext_file = setup_prompt_file(args)
        if systext_file:
            cmd.extend(['--systext_file', systext_file])
    
    # Optional flags
    if args.separate_labels:
        cmd.append('--separate_labels')
        
    if args.eval_mode:
        cmd.append('--eval_mode')
        
    if args.checkpoint_loadpath:
        cmd.extend(['--checkpoint_loadpath', args.checkpoint_loadpath])
    
    return cmd

def run_experiment(cmd: List[str], experiment_name: str, dry_run: bool = False) -> bool:
    """Run a single experiment."""
    logger.info(f"🚀 Starting experiment: {experiment_name}")
    
    if dry_run:
        logger.info(f"DRY RUN - Command: {' '.join(cmd)}")
        return True
        
    try:
        # Log the command
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Change to decoder_llm directory
        original_dir = os.getcwd()
        decoder_dir = os.path.join(os.path.dirname(__file__), '..', 'decoder_llm')
        if os.path.exists(decoder_dir):
            os.chdir(decoder_dir)
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)  # 4 hour timeout
        
        # Return to original directory
        os.chdir(original_dir)
        
        if result.returncode == 0:
            logger.info(f"✅ Experiment {experiment_name} completed successfully")
            if result.stdout:
                logger.info(f"Output: {result.stdout[-500:]}")  # Last 500 chars
            return True
        else:
            logger.error(f"❌ Experiment {experiment_name} failed")
            logger.error(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ Experiment {experiment_name} timed out")
        return False
    except Exception as e:
        logger.error(f"💥 Experiment {experiment_name} crashed: {e}")
        return False
    finally:
        # Ensure we return to original directory
        try:
            os.chdir(original_dir)
        except:
            pass

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Handle utility commands
    if args.list_models:
        list_models()
        return
        
    if args.list_approaches:
        list_approaches()
        return
        
    if args.list_prompts:
        list_prompts()
        return
    
    # Load config file if provided
    if args.config:
        config = load_config_file(args.config)
        # Update args with config values
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine which models to run
    if args.model == 'all':
        models_to_run = list(AVAILABLE_MODELS.keys())
    else:
        models_to_run = [args.model]
    
    # Track results
    results = {
        'successful': [],
        'failed': []
    }
    
    logger.info(f"🎯 Running {args.approach} approach on {len(models_to_run)} model(s)")
    logger.info(f"📊 Models: {', '.join(models_to_run)}")
    logger.info(f"⏰ Forecast Window: {args.forecast_window} hours")
    
    # Run experiments
    for model_key in models_to_run:
        experiment_name = generate_experiment_name(args.approach, model_key, 
                                                 args.forecast_window, timestamp)
        
        # Build command
        cmd = build_command(args, model_key, experiment_name)
        
        # Run experiment
        success = run_experiment(cmd, experiment_name, args.dry_run)
        
        if success:
            results['successful'].append(f"{args.approach}_{model_key}")
        else:
            results['failed'].append(f"{args.approach}_{model_key}")
    
    # Print summary
    print("\n" + "="*60)
    print("🏁 EXPERIMENT SUMMARY")
    print("="*60)
    print(f"✅ Successful: {len(results['successful'])}")
    for exp in results['successful']:
        print(f"   - {exp}")
    
    print(f"\n❌ Failed: {len(results['failed'])}")
    for exp in results['failed']:
        print(f"   - {exp}")
    
    if results['failed']:
        print(f"\n⚠️  {len(results['failed'])} experiment(s) failed. Check logs for details.")
        sys.exit(1)
    else:
        print(f"\n🎉 All {len(results['successful'])} experiment(s) completed successfully!")

if __name__ == "__main__":
    main()
