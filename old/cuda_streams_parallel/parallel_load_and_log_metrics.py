#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to load saved metrics and upload them to wandb.
This can be used to upload metrics after a SLURM job timeout or failure.


# Metrics Loading Script

This script allows you to load metrics saved during training and upload them to Weights & Biases (wandb) after a SLURM job timeout or failure.

## Background

When running long training jobs on HPC clusters with SLURM, jobs can sometimes be terminated early due to time limits. To prevent loss of training metrics when this happens, the `parallel_training.py` script now automatically saves metrics to disk after each epoch. This script can then be used to upload those saved metrics to wandb after the job has ended.

## How it Works

During training, the `parallel_training.py` script saves all metrics to a single file (`metrics.pt`) in each model's directory. This file is updated after each epoch, accumulating metrics throughout training. If a SLURM job is interrupted, you can still access all metrics that were saved up to that point.

## Usage

After a training job has been interrupted by SLURM, you can run:

```bash
python load_and_log_metrics.py --path /path/to/your/saved/models
```

### Required Arguments:

- `--path`: Path to the directory containing model metrics (should be the same as the `--path` argument used during training)

### Optional Arguments:

- `--wandb_project`: Wandb project name (default: 'Equilibrium-Propagation')
- `--wandb_entity`: Wandb entity (username or team name)
- `--wandb_name`: Wandb run name prefix
- `--wandb_group`: Wandb group name
- `--wandb_mode`: Wandb logging mode (online, offline, disabled)

## Example

1. Your SLURM job was running with:
   ```bash
   python main.py --path /path/to/output --wandb_mode online --epochs 100
   ```

2. The job was terminated after 50 epochs due to time limits.

3. You can now load and upload the saved metrics with:
   ```bash
   python load_and_log_metrics.py --path /path/to/output
   ```

4. This will load all metrics saved during training and upload them to wandb.

## Notes

- The script will create new wandb runs for each model, using the same naming convention as the original training script.
- It will mark the runs as not completed (`training_completed: False`) to indicate they were interrupted.
- Metrics for each epoch will be uploaded with the correct timestamp.
- The metrics file is updated incrementally during training, so you'll have access to all metrics up to the point where SLURM terminated the job. 
"""

import os
import argparse
import torch
import wandb
import glob
from collections import defaultdict
import copy # Import copy

def parse_args():
    parser = argparse.ArgumentParser(description='Load and log metrics to wandb')
    
    # Required parameters
    parser.add_argument('--path', type=str, required=True,
                       help='Path to the directory containing model metrics')
    
    # Wandb parameters
    parser.add_argument('--wandb_project', type=str, default='Equilibrium-Propagation',
                       help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='Wandb entity (username or team name)')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='Wandb run name (prefix)')
    parser.add_argument('--wandb_group', type=str, default=None,
                       help='Wandb group name')
    parser.add_argument('--wandb_mode', type=str, default='online',
                       choices=['online', 'offline', 'disabled'],
                       help='Wandb logging mode')
    
    return parser.parse_args()

def load_metrics_from_path(path):
    """Load all metrics and associated config saved in the given path"""
    all_metrics = defaultdict(list)
    model_seeds = {}
    model_configs = {} # Store config per model
    
    # Look for model directories
    model_dirs = glob.glob(os.path.join(path, "model_*"))
    
    if not model_dirs:
        print(f"No model directories found in {path}")
        return all_metrics, model_seeds, model_configs # Return empty dict
    
    print(f"Found {len(model_dirs)} potential model directories. Checking for metrics...")
    
    for model_dir in model_dirs:
        try:
            # Extract potential model index from directory name
            try:
                potential_model_idx = int(os.path.basename(model_dir).split('_')[-1])
            except ValueError:
                print(f"Skipping directory (could not parse index): {model_dir}")
                continue

            metrics_file = os.path.join(model_dir, "metrics.pt")
            
            if not os.path.exists(metrics_file):
                # print(f"No metrics file found for model {potential_model_idx} in {model_dir}") # Verbose, maybe omit
                continue
                
            print(f"Loading metrics and config file for model index {potential_model_idx} from {metrics_file}")
            
            # Load metrics file
            try:
                data = torch.load(metrics_file, weights_only=False, map_location='cpu') # Load to CPU
                
                # --- Get definitive model index from file --- 
                if 'model_idx' not in data:
                     print(f"Warning: 'model_idx' not found in {metrics_file}. Skipping.")
                     continue
                model_idx = data['model_idx']
                # --- --- 
                
                # Store seed for this model
                if 'seed' in data:
                    model_seeds[model_idx] = data['seed']
                
                # Store metrics if present
                if 'metrics' in data and data['metrics']: # Check if list is not empty
                    all_metrics[model_idx] = data['metrics']
                    print(f"  Loaded {len(data['metrics'])} metric entries for model {model_idx}")
                else:
                    print(f"  No metric entries found in file for model {model_idx}")
                    # If no metrics, we might still want the config later?
                    # For now, skip if no metrics, as we can't log anything.
                    continue 
                    
                # --- Store wandb config for this model --- 
                model_config = {
                    'wandb_project': data.get('wandb_project'),
                    'wandb_entity': data.get('wandb_entity'),
                    'wandb_name': data.get('wandb_name'),
                    'wandb_group': data.get('wandb_group'),
                    # Add seed and index for convenience
                    'seed': model_seeds.get(model_idx),
                    'model_index': model_idx 
                }
                model_configs[model_idx] = model_config
                print(f"  Stored config for model {model_idx}: project='{model_config['wandb_project']}', name='{model_config['wandb_name']}', group='{model_config['wandb_group']}'")
                # --- --- 
                
            except Exception as e:
                print(f"Error loading metrics file {metrics_file}: {e}")
        except Exception as e:
            print(f"Error processing model directory {model_dir}: {e}")
    
    print(f"Finished loading. Found metrics for {len(all_metrics)} models.")
    return all_metrics, model_seeds, model_configs

def log_metrics_to_wandb(all_metrics, model_configs, args, mark_completed=False):
    """Log loaded metrics to wandb using per-model config where available."""
    if args.wandb_mode == 'disabled':
        print("Wandb logging is disabled")
        return
    
    num_logged = 0
    num_skipped = 0
    
    # Log metrics for each model found
    for model_idx, metrics_list in all_metrics.items():
        if not metrics_list:
            print(f"No metrics found for model {model_idx}, skipping logging.")
            num_skipped += 1
            continue
            
        # --- Get config for this specific model --- 
        config_for_model = model_configs.get(model_idx, {}) # Get specific config or empty dict
        
        # Determine project, entity, name, group for this run
        # Priority: 1. Config from metrics.pt, 2. Script args, 3. Script defaults
        project = config_for_model.get('wandb_project') or args.wandb_project
        entity = config_for_model.get('wandb_entity') or args.wandb_entity
        name_base = config_for_model.get('wandb_name') or args.wandb_name or 'run' # Fallback to 'run' if absolutely no name found
        group_base = config_for_model.get('wandb_group') or args.wandb_group or f"{name_base}-group" # Fallback group
        
        # Use the wandb mode specified by *this script's* arguments
        mode = args.wandb_mode 
        # --- ---
            
        print(f"Logging {len(metrics_list)} metrics for model {model_idx} (Project: '{project}', Name: '{name_base}-model{model_idx}', Group: '{group_base}', Mode: '{mode}')...")
        
        # Construct run name and group name
        run_name = f"{name_base}-model{model_idx}"
        group_name = group_base # Group name usually doesn't include model index
        
        # Create wandb config object (include seed if available)
        wandb_config = {
            'model_index': model_idx,
            'seed': config_for_model.get('seed') # Use seed from config if available
        }
        
        # Initialize wandb run
        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            group=group_name,
            config=wandb_config,
            mode=mode,
            reinit=True,  # Need this since we're creating multiple runs
        )
        
        try:
            # Sort metrics by epoch to ensure correct order
            sorted_metrics = sorted(metrics_list, key=lambda x: x['epoch'])
            max_epoch = 0
            
            # Log each metric with correct epoch
            for metric_item in sorted_metrics:
                epoch = metric_item['epoch']
                metrics = metric_item['metrics']
                
                # Update max epoch seen
                max_epoch = max(max_epoch, epoch)
                
                # Log metrics
                run.log(metrics, step=epoch)
            
            # Log completion metrics
            run.log({
                "training_completed": mark_completed,
                "last_logged_epoch": max_epoch
            })
            
            print(f"  Successfully logged metrics for model {model_idx} to WandB (marked completed: {mark_completed})")
            num_logged += 1
        except Exception as e:
            print(f"  Error logging metrics for model {model_idx} to WandB: {e}")
            num_skipped += 1
        finally:
            # Finish the run
            if run:
                 run.finish()
                 
    print(f"\nLogging finished. Logged: {num_logged}, Skipped/Errors: {num_skipped}")

def main():
    args = parse_args()
    
    # Now configuration is loaded per-model from metrics.pt
    print(f"Loading metrics and config from {args.path}...")
    all_metrics, model_seeds, model_configs = load_metrics_from_path(args.path)

    num_models = len(all_metrics)
    print(f"Found metrics for {num_models} models")

    # Log metrics to wandb using per-model config found in metrics.pt
    # Pass the script's args as well for fallback values and the desired wandb_mode
    # When run standalone, mark_completed remains False by default
    if num_models > 0:
        print("\nLogging metrics to WandB...")
        log_metrics_to_wandb(all_metrics, model_configs, args) # Standalone call uses default mark_completed=False
    else:
        print("No metrics found, nothing to log to wandb")

if __name__ == "__main__":
    main() 