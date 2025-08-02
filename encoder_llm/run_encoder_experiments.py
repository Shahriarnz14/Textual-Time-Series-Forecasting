#!/usr/bin/env python3
"""
Textual Time Series Forecasting - Encoder Models Runner
======================================================

This script provides a unified interface for running all encoder-based experiments
for textual time series forecasting, including time window classification and 
event ordering (concordance) tasks.

Usage:
    python run_encoder_experiments.py --task time_window --model deberta --data_dir data/train
    python run_encoder_experiments.py --task concordance --model roberta --eval_only
    python run_encoder_experiments.py --config configs/experiment_config.json

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
    'bert': {
        'model_name': 'bert-base-uncased',
        'max_length': 512,
        'batch_size': 16
    },
    'roberta': {
        'model_name': 'roberta-base', 
        'max_length': 512,
        'batch_size': 16
    },
    'deberta': {
        'model_name': 'microsoft/deberta-v3-base',
        'max_length': 512,
        'batch_size': 12
    },
    'deberta-small': {
        'model_name': 'microsoft/deberta-v3-small',
        'max_length': 512,
        'batch_size': 20
    },
    'modernbert': {
        'model_name': 'answerdotai/ModernBERT-base',
        'max_length': 512,
        'batch_size': 16
    },
    'modernbert-large': {
        'model_name': 'answerdotai/ModernBERT-large',
        'max_length': 512,
        'batch_size': 8
    }
}

# Available tasks and their script mappings
AVAILABLE_TASKS = {
    'time_window': {
        'script': 'encoder_time_window.py',
        'description': 'Time window classification task'
    },
    'concordance': {
        'script': 'encoder_concordance.py', 
        'description': 'Event ordering concordance task'
    },
    'mask_time_window': {
        'script': 'encoder_mask_time_window.py',
        'description': 'Masked time window classification task'
    },
    'mask_concordance': {
        'script': 'encoder_mask_concordance.py',
        'description': 'Masked event ordering concordance task'
    }
}

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run encoder-based textual time series forecasting experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic time window classification
  python run_encoder_experiments.py --task time_window --model deberta --data_dir data/train

  # Evaluation only mode
  python run_encoder_experiments.py --task concordance --model roberta --eval_only --checkpoint_path checkpoints/best.pt

  # Custom hyperparameters
  python run_encoder_experiments.py --task time_window --model bert --epochs 20 --lr 2e-5 --batch_size 24

  # All models on time window task
  python run_encoder_experiments.py --task time_window --model all --data_dir data/train

  # Load from config file
  python run_encoder_experiments.py --config configs/experiment_config.json
        """
    )
    
    # Model and task selection
    parser.add_argument('--task', type=str, choices=list(AVAILABLE_TASKS.keys()),
                       help='Task type to run')
    parser.add_argument('--model', type=str, choices=list(AVAILABLE_MODELS.keys()) + ['all'],
                       help='Model type to use (or "all" for all models)')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, help='Training data directory')
    parser.add_argument('--test_dir', type=str, help='Test data directory')
    parser.add_argument('--val_dir', type=str, help='Validation data directory (optional)')
    
    # Model configuration
    parser.add_argument('--K', type=int, default=8, help='Number of next events to consider')
    parser.add_argument('--H', type=int, default=24, help='Time window for classification (hours)')
    parser.add_argument('--max_length', type=int, help='Maximum token length (model default if not specified)')
    parser.add_argument('--batch_size', type=int, help='Batch size (model default if not specified)')
    
    # Training configuration
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--timestep_drop_rate', type=float, default=0.0, 
                       help='Rate of randomly dropping timesteps')
    
    # Execution modes
    parser.add_argument('--eval_only', action='store_true', 
                       help='Run evaluation only (no training)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show commands that would be executed without running them')
    
    # Checkpoint and output
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint for evaluation')
    parser.add_argument('--output_dir', type=str, default='experiments',
                       help='Output directory for results')
    parser.add_argument('--experiment_name', type=str, help='Custom experiment name')
    
    # Config file option
    parser.add_argument('--config', type=str, help='Path to JSON config file')
    
    # Utility options
    parser.add_argument('--list_models', action='store_true', help='List available models and exit')
    parser.add_argument('--list_tasks', action='store_true', help='List available tasks and exit')
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
    print("=" * 50)
    for model_key, model_info in AVAILABLE_MODELS.items():
        print(f"  {model_key:15} -> {model_info['model_name']}")
        print(f"  {'':15}    Max Length: {model_info['max_length']}, Batch Size: {model_info['batch_size']}")
    print()

def list_tasks():
    """List all available tasks."""
    print("\n🎯 Available Tasks:")
    print("=" * 50)
    for task_key, task_info in AVAILABLE_TASKS.items():
        print(f"  {task_key:20} -> {task_info['description']}")
        print(f"  {'':20}    Script: {task_info['script']}")
    print()

def validate_arguments(args: argparse.Namespace) -> bool:
    """Validate provided arguments."""
    if args.list_models or args.list_tasks:
        return True
        
    if not args.task:
        logger.error("Task must be specified")
        return False
        
    if not args.model:
        logger.error("Model must be specified")
        return False
        
    if not args.eval_only and not args.data_dir:
        logger.error("Training data directory must be specified for training")
        return False
        
    if args.eval_only and not args.checkpoint_path:
        logger.error("Checkpoint path must be specified for evaluation-only mode")
        return False
        
    # Validate paths exist
    if args.data_dir and not os.path.exists(args.data_dir):
        logger.error(f"Training data directory does not exist: {args.data_dir}")
        return False
        
    if args.test_dir and not os.path.exists(args.test_dir):
        logger.error(f"Test data directory does not exist: {args.test_dir}")
        return False
        
    return True

def generate_experiment_name(task: str, model: str, timestamp: str) -> str:
    """Generate a unique experiment name."""
    return f"{task}_{model}_{timestamp}"

def build_command(args: argparse.Namespace, model_key: str, experiment_name: str) -> List[str]:
    """Build the command to execute for a specific model."""
    task_info = AVAILABLE_TASKS[args.task]
    model_info = AVAILABLE_MODELS[model_key]
    
    # Start with the script
    cmd = ['python', task_info['script']]
    
    # Add basic parameters
    cmd.extend(['--K', str(args.K)])
    cmd.extend(['--H', str(args.H)])
    cmd.extend(['--model_name', model_info['model_name']])
    
    # Model-specific parameters
    max_length = args.max_length if args.max_length else model_info['max_length']
    batch_size = args.batch_size if args.batch_size else model_info['batch_size']
    cmd.extend(['--max_length', str(max_length)])
    cmd.extend(['--batch_size', str(batch_size)])
    
    # Training parameters
    cmd.extend(['--lr', str(args.lr)])
    cmd.extend(['--epochs', str(args.epochs)])
    cmd.extend(['--patience', str(args.patience)])
    cmd.extend(['--timestep_drop_rate', str(args.timestep_drop_rate)])
    
    # Data directories
    if args.data_dir:
        cmd.extend(['--train_data_directory', args.data_dir])
    if args.test_dir:
        cmd.extend(['--test_data_directory', args.test_dir])
    elif args.data_dir:
        # Use training dir as test dir if test dir not specified
        cmd.extend(['--test_data_directory', args.data_dir])
        
    # Checkpoint and output paths
    output_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    if 'time_window' in args.task:
        checkpoint_path = os.path.join(output_dir, f'checkpoint_{model_key}_time_window.pt')
        cmd.extend(['--checkpoint_path_time_window', checkpoint_path])
        
    if 'concordance' in args.task:
        checkpoint_path = os.path.join(output_dir, f'checkpoint_{model_key}_concordance.pt')
        cmd.extend(['--checkpoint_path_concordance', checkpoint_path])
        
    if 'mask' in args.task:
        if 'time_window' in args.task:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_{model_key}_mask_time_window.pt')
            cmd.extend(['--checkpoint_path_mask_time_window', checkpoint_path])
        else:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_{model_key}_mask_concordance.pt')
            cmd.extend(['--checkpoint_path_mask_concordance', checkpoint_path])
    
    # Results file
    results_file = os.path.join(output_dir, f'results_{model_key}.txt')
    cmd.extend(['--test_results_text_file', results_file])
    
    # Evaluation mode
    if args.eval_only:
        cmd.append('--eval_only')
        if args.checkpoint_path:
            cmd.extend(['--checkpoint_loadpath', args.checkpoint_path])
    
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
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
        
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

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Handle utility commands
    if args.list_models:
        list_models()
        return
        
    if args.list_tasks:
        list_tasks()
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
    
    logger.info(f"🎯 Running {args.task} task on {len(models_to_run)} model(s)")
    logger.info(f"📊 Models: {', '.join(models_to_run)}")
    
    # Run experiments
    for model_key in models_to_run:
        experiment_name = generate_experiment_name(args.task, model_key, timestamp)
        
        # Build command
        cmd = build_command(args, model_key, experiment_name)
        
        # Run experiment
        success = run_experiment(cmd, experiment_name, args.dry_run)
        
        if success:
            results['successful'].append(f"{args.task}_{model_key}")
        else:
            results['failed'].append(f"{args.task}_{model_key}")
    
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
