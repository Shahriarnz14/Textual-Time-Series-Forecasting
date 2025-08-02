import json
import os

# Define base configurations for each experiment type
base_configs = {
    "expt250326_dsr1": {
        "--forecast_window": 1,
        "--test_dir": "[ANONYMIZED_PATH]/tts_files/DSR1Q_annotations/ordered_new_DStrain/",
        "--data_dir": "[ANONYMIZED_PATH]/tts_files/DSR1Q_annotations/ordered_new_DStrain/",
        "--epochs": "1000",
        "--sof": 40
    },
    "expt250326_dsr1_eval": {
        "--test_dir": "[ANONYMIZED_PATH]/tts_files/DSR1Q_annotations/ordered_new_DStest/",
        "--data_dir": "[ANONYMIZED_PATH]/tts_files/DSR1Q_annotations/ordered_new_DStrain/",
        "--eval_mode": True,
        "--epochs": "1000",
        "--sof": 99999
    },
    "expt250328_dsr1_prompting": {
        "--approach": "PROMPT:window",
        "--eval_mode": True,
        "--epochs": 1,
        "--sof": 99999
    }
}

# Define base output directories
base_output_dirs = {
    "expt250326_dsr1": "scripts/args/expt250326_dsr1/",
    "expt250326_dsr1_eval": "scripts/args/expt250326_dsr1_eval/",
    "expt250328_dsr1_prompting": "scripts/args/expt250328_dsr1_prompting/"
}

# Define the sets for the arrangement
forecast_windows = [1, 24, 168]
models = {
    'dsl8': {
        'base_model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
        'infix_prefix': 'dsl8'
    },
    'l33': {
        'base_model': 'meta-llama/Llama-3.3-70B-Instruct',
        'infix_prefix': 'l33'
    },
    'dsl33': {
        'base_model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
        'infix_prefix': 'dsl33'
    },
    'l318': {
        'base_model': 'meta-llama/Llama-3.1-8B-Instruct',
        'infix_prefix': 'l318'
    }
}

# Build the new experiments list
new_experiments = []

# Generate training and evaluation for each window and model
for window in forecast_windows:
    for model_key, model_info in models.items():
        infix_prefix = model_info['infix_prefix']
        base_model = model_info['base_model']
        infix = f"dsr1_{infix_prefix}_fw{window}"
        
        # Training experiment
        training_overrides = {
            "--infix": infix,
            "--forecast_window": window,
            "--base_model": base_model
        }
        if model_key == 'dsl33':
            training_overrides["--cache_dir"] = "[ANONYMIZED_PATH]/.cache/huggingface/hub/"
        
        new_experiments.append({
            "type": "expt250326_dsr1",
            "subdir": f"w{window}",
            "variant": model_key,
            "filename": f"args_tts0.7_{model_key}.json",
            "overrides": training_overrides
        })
        
        # Evaluation experiment
        eval_overrides = {
            "--infix": infix,
            "--forecast_window": window,
            "--base_model": base_model,
            "--checkpoint_loadpath": f"[ANONYMIZED_PATH]/scripts/checkpoints/{infix}/best.pt"
        }
        new_experiments.append({
            "type": "expt250326_dsr1_eval",
            "subdir": f"w{window}",
            "variant": model_key,
            "filename": f"args_tts0.7_{model_key}.json",
            "overrides": eval_overrides
        })

# Append the existing prompting experiments (as in the original list, the last 3)
subdirs = ["sepsis-100", "sepsis-10", "t2s2"]
variants = {
    "l33fewshot1to5": {
        "--infix": "dsl33_l33fewshot_1to5",
        "--systext_file": "[ANONYMIZED_PATH]/scripts/window_system_fewshot_1to5.txt",
        "--base_model": "meta-llama/Llama-3.3-70B-Instruct"
    },
    "l33fewshot": {
        "--infix": "dsl33_l33fewshot",
        "--systext_file": "[ANONYMIZED_PATH]/scripts/window_system_fewshot.txt",
        "--base_model": "meta-llama/Llama-3.3-70B-Instruct"
    },
    "l330shot1to5": {
        "--infix": "dsl33_l330shot_1to5",
        "--systext_file": "[ANONYMIZED_PATH]/scripts/window_system_text_1to5.txt",
        "--base_model": "meta-llama/Llama-3.3-70B-Instruct"
    },
    "l330shot": {
        "--infix": "dsl33_l330",
        "--systext_file": "[ANONYMIZED_PATH]/scripts/window_system_text.txt",
        "--base_model": "meta-llama/Llama-3.3-70B-Instruct"
    }
}

prompting_experiments = []
for subdir in subdirs:
    for variant_key, variant_overrides in variants.items():
        exp = {
            "type": "expt250328_dsr1_prompting",
            "subdir": subdir,
            "variant": variant_key,
            "filename": f"args_{variant_key}.json",
            "overrides": {
                "--test_dir": "",  # will set below
                "--data_dir": "[ANONYMIZED_PATH]/heads/ordered_l33train/",
                **variant_overrides
            }
        }
        if subdir == "sepsis-100":
            exp["overrides"]["--test_dir"] = "[ANONYMIZED_PATH]/tts_files/DSR1Q_annotations/ordered_new_sepsis90/"
        elif subdir == "sepsis-10":
            exp["overrides"]["--test_dir"] = "[ANONYMIZED_PATH]/tts_files/DSR1Q_annotations/ordered_new_sepsis10/"
        
        prompting_experiments.append(exp)

new_experiments += prompting_experiments

# Now set the experiments list to the new_experiments
experiments = new_experiments

# Generate the files
for exp in experiments:
    # Get the base config for this experiment type
    base_config = base_configs[exp["type"]]
    
    # Get the base output directory
    base_output_dir = base_output_dirs[exp["type"]]
    
    # Determine the full output directory: base_output_dir + subdir
    output_dir = os.path.join(base_output_dir, exp["subdir"])
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the full file path
    filepath = os.path.join(output_dir, exp["filename"])
    
    # Merge the base config with the overrides
    config = {**base_config, **exp["overrides"]}
    
    # Write the JSON file
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)