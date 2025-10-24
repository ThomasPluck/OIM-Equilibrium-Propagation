"""
Main script for launching multiple independent training runs using multiprocessing.
"""

import torch
import torch.multiprocessing as mp
import random
import argparse
import numpy as np
import os
import sys
import time
import json
import itertools
from datetime import datetime
from copy import deepcopy

# Import the ModelTrainer class directly
from model_trainer import ModelTrainer

### ARGUMENTS ###
def get_args():
    """Parses and returns command line arguments."""
    parser = argparse.ArgumentParser(description='Eqprop - Multiprocess Launcher')
    
    parser.add_argument('--wandb-project', type=str, default='Equilibrium-Propagation', help='WandB project name')
    parser.add_argument('--wandb-entity', type=str, default='alexgower-team', help='WandB entity/username')
    parser.add_argument('--wandb-name', type=str, default=None, help='WandB run name base (seed will be appended)')
    parser.add_argument('--wandb-group', type=str, default=None, help='WandB group name for organizing related runs')
    parser.add_argument('--wandb-mode', type=str, default='disabled', help='WandB mode (online/offline/disabled)')
     
    parser.add_argument('--model',type = str, default = 'MLP', metavar = 'm', help='model e.g. MLP, OIM_MLP, CNN') 
    parser.add_argument('--act',type = str, default = 'cos', metavar = 'a', help='activation function, their default was mysig') 
    parser.add_argument('--task',type = str, default = 'MNIST', metavar = 't', help='task (MNIST or FashionMNIST)')
    parser.add_argument('--optim', type = str, default = 'sgd', metavar = 'opt', help='optimizer for training')
    parser.add_argument('--loss', type = str, default = 'mse', metavar = 'lss', help='loss for training')
    parser.add_argument('--alg', type = str, default = 'EP', metavar = 'al', help='EP or BPTT')
    parser.add_argument('--thirdphase', default = False, action = 'store_true', help='add third phase for higher order evaluation of the gradient (default: False)')
    parser.add_argument('--save', default = False, action = 'store_true', help='saving results')
    parser.add_argument('--todo', type=str, default='train', help='training task - always train for multiprocessing')
    parser.add_argument('--load-path', type = str, default = '', metavar = 'l', help='load a model')
    parser.add_argument('--load-checkpoint', type = str, default = 'final', choices=['final', 'best'], metavar = 'lc', help='which checkpoint to load: "final" (most recent) or "best" (highest test accuracy)')
    parser.add_argument('--device',type = int, default = 0, metavar = 'd', help='device')
    
    parser.add_argument('--T1', type=int, default=20, metavar = 'T1', help='Time of first phase')
    parser.add_argument('--T2', type=int, default=4, metavar = 'T2', help='Time of second phase (and third phase if applicable)')
    parser.add_argument('--betas', nargs='+', type = float, default = [0.0, 0.01], metavar = 'Bs', help='Betas in EP phases')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Step size for OIM dynamics')
    parser.add_argument('--noise-level', type=float, default=0.0, help='Noise level for phase dynamics')
    parser.add_argument('--N-data-train', type=int, default=60000, help='Number of training data points')
    parser.add_argument('--N-data-test', type=int, default=10000, help='Number of test data points')
    
    parser.add_argument('--archi', nargs='+', type = int, default = [784, 512, 10], metavar = 'A', help='architecture of the network')
    parser.add_argument('--weight-lrs', nargs='+', type = float, default = [0.01], metavar = 'wl', help='Layer-wise learning rates for weights.')
    parser.add_argument('--bias-lrs', nargs='+', type = float, default = None, metavar = 'bl', help='Layer-wise learning rates for biases (OIM). Defaults to weight_lrs.')
    parser.add_argument('--sync-lrs', nargs='+', type = float, default = None, metavar = 'sl', help='Layer-wise learning rates for sync params (OIM). Defaults to weight_lrs.')
    parser.add_argument('--epochs',type = int, default = 10, metavar = 'EPT',help='Number of epochs')
    parser.add_argument('--weight-scale', nargs='+', type=float, default=None, metavar='wg', help='Scale factors for weight init')
    parser.add_argument('--bias-scale', nargs='+', type=float, default=None, metavar='bg', help='Scale factors for bias init (defaults to weight_scale)')
    parser.add_argument('--mbs',type = int, default = 20, metavar = 'M', help='minibatch size')

    parser.add_argument('--plot', default = False, action = 'store_true', help='Enable plotting of phase dynamics')
    parser.add_argument('--debug', default=False, action='store_true', help='Debug mode')
    parser.add_argument('--check-thm', default=False, action='store_true', help='Check GDU alignment during training')
    parser.add_argument('--profiling', default=False, action='store_true', help='Enable PyTorch profiler (for rank 0 process)')

    parser.add_argument('--mmt',type = float, default = 0.0, metavar = 'mmt', help='Momentum for SGD optimizer')
    parser.add_argument('--wds', nargs='+', type = float, default = None, metavar = 'wd', help='Layer-wise weight decays.')
    parser.add_argument('--lr-decay', default = False, action = 'store_true', help='enabling learning rate decay')
    
    parser.add_argument('--random-phase-initialisation', default=False, action='store_true', help='Initialize phases randomly (OIM)')
    parser.add_argument('--intralayer-connections', default=False, action='store_true', help='Add trainable intralayer synapses')
    parser.add_argument('--reinitialise-neurons', default=False, action='store_true', help='Reinitialize neurons before phase 2/3')
    parser.add_argument('--input-positive-negative-mapping', default=False, action='store_true', help='Remap input pixels to [-1,1]')
    parser.add_argument('--random-sign', default = False, action = 'store_true', help='randomly switch beta_2 sign (EP variant)')
    parser.add_argument('--data-aug', default = False, action = 'store_true', help='Enable data augmentation (e.g., for cifar10)')
    parser.add_argument('--softmax', default = False, action = 'store_true', help='Use softmax output layer (potentially affects loss choice)')
    
    # Quantization parameters for physical system modeling
    parser.add_argument('--quantisation-bits', type=int, default=0, help='Number of bits for parameter quantization (0 means no quantization)')
    parser.add_argument('--neuron-quantisation-bits', type=int, default=0, help='Number of bits for neural state quantization (0 means no quantization)')
    parser.add_argument('--J-max', type=float, default=1.0, help='Maximum absolute value for synaptic weights')
    parser.add_argument('--h-max', type=float, default=1.0, help='Maximum absolute value for bias parameters')
    parser.add_argument('--sync-max', type=float, default=1.0, help='Maximum absolute value for synchronization parameters')
    parser.add_argument('--float64', default=False, action='store_true', help='Use 64-bit float precision instead of default 32-bit')
    
    # Multiprocessing-specific arguments
    parser.add_argument('--num-repeats', type=int, default=5, help='Number of parallel processes to run (default: 5)')
    parser.add_argument('--base-seed', type=int, default=1, help='Base seed for random number generation, processes will use base_seed+rank (default: 42)')
    
    # Performance optimization arguments
    parser.add_argument('--cache-to-gpu', action='store_true', help='Cache entire dataset to GPU VRAM (eliminates PCIe bottleneck, recommended for MNIST/CIFAR)')
    parser.add_argument('--multi-gpu', action='store_true', help='Distribute processes across all available GPUs in round-robin fashion')
    
    # Experiment grid search
    parser.add_argument('--experiments-json', type=str, default=None, help='Path to JSON file defining experiment grid (overrides individual hyperparameters)')
    parser.add_argument('--start-index', type=int, default=0, help='Index to start/resume experiments from (0-based)')
    # parser.add_argument('--pools', type = str, default = 'mm', metavar = 'p', help='pooling') 
    # parser.add_argument('--channels', nargs='+', type = int, default = [32, 64], metavar = 'C', help='channels of the convnet')
    # parser.add_argument('--kernels', nargs='+', type = int, default = [5, 5], metavar = 'K', help='kernels sizes of the convnet')
    # parser.add_argument('--strides', nargs='+', type = int, default = [1, 1], metavar = 'S', help='strides of the convnet')
    # parser.add_argument('--paddings', nargs='+', type = int, default = [0, 0], metavar = 'P', help='paddings of the conv layers')
    # parser.add_argument('--fc', nargs='+', type = int, default = [10], metavar = 'S', help='linear classifier of the convnet')


    return parser.parse_args()

def prepare_args_for_process(args, rank):
    """Prepare args for a specific process"""
    proc_args = deepcopy(args)
    
    # Set unique seed for this process
    proc_args.seed = args.base_seed + rank
    proc_args.rank = rank
    
    # Multi-GPU assignment: Distribute processes across available GPUs
    if args.multi_gpu and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        proc_args.device = rank % num_gpus
        print(f"[Process {rank}] Assigned to GPU {proc_args.device} (of {num_gpus} total)", flush=True)
    else:
        # Single GPU mode - use specified device
        proc_args.device = args.device

    # Do different things depending on whether we are loading or not
    if args.load_path == '': # New run
        # Use the shared timestamp instead of generating a new one
        overall_name = f"{args.wandb_name}_{args.shared_timestamp}"

        # Set as wandb_name and wandb_id
        proc_args.wandb_name = f"{overall_name}_model_{rank}"
        proc_args.wandb_id = f"{overall_name}_model_{rank}"

        proc_args.path = f'results/{args.wandb_group}/{overall_name}/model_{rank}'
    else: # Loading run
        overall_name = os.path.basename(args.load_path)
        
        # Set wandb name and ID to match the previous run format
        proc_args.wandb_name = f"{overall_name}_model_{rank}"
        proc_args.wandb_id = f"{overall_name}_model_{rank}"

        # For loading, derive the specific model path
        proc_args.path = f'{args.load_path}/model_{rank}'

    return proc_args

def run_process(rank, args):
    """Function to run in a separate process"""
    try:
        # Set process-specific args
        proc_args = prepare_args_for_process(args, rank)
        
        # Initialize the trainer
        trainer = ModelTrainer(proc_args)
        
        # Run the appropriate function based on the todo argument
        if proc_args.todo == 'gducheck':
            print(f"Process {rank}: Running GDU check")
            trainer.run_gdu_check()
        elif proc_args.todo == 'evaluate':
            print(f"Process {rank}: Running evaluation")
            trainer.run_evaluate()
        else:  # Default is 'train'
            print(f"Process {rank}: Running training")
            trainer.run_training()
        
    except Exception as e:
        import traceback
        print(f"Process {rank} failed with error: {e}")
        traceback.print_exc()

### EXPERIMENT GRID SEARCH FUNCTIONS ###

def load_experiment_grid(json_path, start_index=0):
    """
    Load experiment grid from JSON file.
    Returns list of experiment configurations.
    """
    with open(json_path, 'r') as f:
        config = json.load(f)
    
    experiments = []
    
    for exp_group in config.get('experiments', []):
        group_name = exp_group.get('name', 'unnamed')
        base_config = exp_group.get('base', {})
        grid = exp_group.get('grid', {})
        num_seeds = exp_group.get('num_seeds', 1)
        
        # Merge base_config from top-level and experiment-level
        merged_base = config.get('base_config', {}).copy()
        merged_base.update(base_config)
        
        # Generate all combinations from grid
        if grid:
            keys = list(grid.keys())
            values = list(grid.values())
            combinations = list(itertools.product(*values))
            
            for combo in combinations:
                exp_config = merged_base.copy()
                for key, val in zip(keys, combo):
                    exp_config[key] = val
                
                # Add multiple seeds for this configuration
                for seed_offset in range(num_seeds):
                    exp_with_seed = exp_config.copy()
                    exp_with_seed['_seed_offset'] = seed_offset
                    exp_with_seed['_group_name'] = group_name
                    experiments.append(exp_with_seed)
        else:
            # No grid, just use base config with multiple seeds
            for seed_offset in range(num_seeds):
                exp_with_seed = merged_base.copy()
                exp_with_seed['_seed_offset'] = seed_offset
                exp_with_seed['_group_name'] = group_name
                experiments.append(exp_with_seed)
                
    experiments = experiments[start_index:]
    
    return experiments

def create_args_from_config(base_args, config_dict, exp_idx):
    """
    Create args namespace from experiment configuration dictionary.
    """
    args = deepcopy(base_args)
    
    # Apply all config values to args
    for key, value in config_dict.items():
        if key.startswith('_'):
            # Skip internal keys (like _seed_offset, _group_name)
            continue
        
        # Handle special cases
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            # Add new attribute if it doesn't exist
            setattr(args, key, value)
    
    # Set seed
    seed_offset = config_dict.get('_seed_offset', 0)
    args.seed = base_args.base_seed + exp_idx  # Use exp_idx for unique seed
    
    # Set rank and experiment ID
    args.rank = exp_idx
    
    return args

def print_experiment_summary(experiments):
    """Print summary of experiments to run"""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT GRID SUMMARY")
    print(f"{'='*70}")
    print(f"Total experiments: {len(experiments)}\n")
    
    # Group by experiment name
    groups = {}
    for exp in experiments:
        group_name = exp.get('_group_name', 'unnamed')
        if group_name not in groups:
            groups[group_name] = 0
        groups[group_name] += 1
    
    for group_name, count in groups.items():
        print(f"  {group_name}: {count} experiments")
    
    print(f"{'='*70}\n")

def run_experiment_grid(base_args):
    """
    Run experiment grid from JSON file.
    Manages process pool across all experiments.
    """
    # Load experiments from JSON
    print(f"Loading experiments from: {base_args.experiments_json}")
    experiments = load_experiment_grid(base_args.experiments_json, base_args.start_index)
    
    # Print summary
    print_experiment_summary(experiments)
    
    # Generate shared timestamp
    date = datetime.now().strftime('%Y-%m-%d')
    time_str = datetime.now().strftime('%H-%M-%S')
    shared_timestamp = f"{date}_{time_str}"
    base_args.shared_timestamp = shared_timestamp
    
    # Set float precision
    if base_args.float64:
        torch.set_default_dtype(torch.float64)
        print('Using 64-bit floating point precision')
    else:
        print('Using default 32-bit floating point precision')
    
    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    start_time = time.time()
    
    if base_args.multi_gpu and torch.cuda.is_available():
        # Multi-GPU mode: Process pool
        num_gpus = torch.cuda.device_count()
        print(f"MULTI-GPU MODE: Running {len(experiments)} experiments on {num_gpus} GPUs")
        print(f"Batch size: {num_gpus} concurrent processes\n")
        
        active_processes = {}  # {process: (exp_idx, exp_config)}
        remaining_experiments = list(enumerate(experiments))
        completed_count = 0
        
        # Start initial batch
        while len(active_processes) < num_gpus and remaining_experiments:
            exp_idx, exp_config = remaining_experiments.pop(0)
            
            # Create args for this experiment
            exp_args = create_args_from_config(base_args, exp_config, exp_idx)
            exp_args.device = exp_idx % num_gpus
            
            # Set experiment-specific paths
            group_name = exp_config.get('_group_name', 'unnamed')
            exp_name = f"{group_name}_exp{exp_idx:04d}"
            exp_args.path = f'results/{base_args.wandb_group or "experiment_grid"}/{shared_timestamp}/{exp_name}'
            exp_args.wandb_name = exp_name
            exp_args.wandb_id = exp_name
            
            p = mp.Process(target=run_process, args=(exp_idx, exp_args))
            p.start()
            active_processes[p] = (exp_idx, exp_config)
            print(f"[STARTED] Experiment {exp_idx}/{len(experiments)} ({group_name}) on GPU {exp_idx % num_gpus} (PID {p.pid})")
        
        # As processes complete, start new ones
        while active_processes:
            for p in list(active_processes.keys()):
                if not p.is_alive():
                    completed_idx, completed_config = active_processes.pop(p)
                    p.join()
                    completed_count += 1
                    group_name = completed_config.get('_group_name', 'unnamed')
                    print(f"[COMPLETED] Experiment {completed_idx} ({group_name}) - {completed_count}/{len(experiments)} total")
                    
                    # Start next experiment if any remain
                    if remaining_experiments:
                        exp_idx, exp_config = remaining_experiments.pop(0)
                        
                        exp_args = create_args_from_config(base_args, exp_config, exp_idx)
                        exp_args.device = exp_idx % num_gpus
                        
                        group_name = exp_config.get('_group_name', 'unnamed')
                        exp_name = f"{group_name}_exp{exp_idx:04d}"
                        exp_args.path = f'results/{base_args.wandb_group or "experiment_grid"}/{shared_timestamp}/{exp_name}'
                        exp_args.wandb_name = exp_name
                        exp_args.wandb_id = exp_name
                        
                        new_p = mp.Process(target=run_process, args=(exp_idx, exp_args))
                        new_p.start()
                        active_processes[new_p] = (exp_idx, exp_config)
                        print(f"[STARTED] Experiment {exp_idx}/{len(experiments)} ({group_name}) on GPU {exp_idx % num_gpus} (PID {new_p.pid})")
                    break
            
            if active_processes:
                import time as sleep_time
                sleep_time.sleep(0.1)
    else:
        # Single-GPU mode: Run experiments sequentially
        print(f"SINGLE-GPU MODE: Running {len(experiments)} experiments sequentially\n")
        
        for exp_idx, exp_config in enumerate(experiments):
            # Create args for this experiment
            exp_args = create_args_from_config(base_args, exp_config, exp_idx)
            
            # Set experiment-specific paths
            group_name = exp_config.get('_group_name', 'unnamed')
            exp_name = f"{group_name}_exp{exp_idx:04d}"
            exp_args.path = f'results/{base_args.wandb_group or "experiment_grid"}/{shared_timestamp}/{exp_name}'
            exp_args.wandb_name = exp_name
            exp_args.wandb_id = exp_name
            
            print(f"[STARTED] Experiment {exp_idx+1}/{len(experiments)} ({group_name})")
            
            p = mp.Process(target=run_process, args=(exp_idx, exp_args))
            p.start()
            p.join()
            
            print(f"[COMPLETED] Experiment {exp_idx+1}/{len(experiments)} ({group_name})")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"ALL {len(experiments)} EXPERIMENTS COMPLETED")
    print(f"Total time: {elapsed/3600:.2f} hours ({elapsed/60:.2f} minutes)")
    print(f"{'='*70}\n")

def main():
    """Main function for multiprocessing training"""
    ### ARGUMENTS ###
    args = get_args()
    
    # Check if running experiment grid
    if args.experiments_json:
        run_experiment_grid(args)
        return
    
    # Original single-experiment mode
    # AUTO-ADJUST: If multi-GPU is enabled, set num_repeats to match GPU count
    if args.multi_gpu and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if args.num_repeats != num_gpus:
            print(f"Multi-GPU mode: Overriding --num-repeats from {args.num_repeats} to {num_gpus} (one process per GPU)")
            args.num_repeats = num_gpus
    
    # Generate a single shared timestamp for all processes
    date = datetime.now().strftime('%Y-%m-%d')
    time_str = datetime.now().strftime('%H-%M-%S')
    shared_timestamp = f"{date}_{time_str}"
    args.shared_timestamp = shared_timestamp
    print(f"Using shared timestamp for all processes: {shared_timestamp}")

    ### INITIAL PRINTING ###
    print('\n')
    print(' '.join(sys.argv))
    print('\n')
    print('##################################################################')
    print('\nargs\tmbs\tT1\tT2\tepochs\tactivation\tbetas\tprocesses')
    print('\t', args.mbs, '\t', args.T1, '\t', args.T2, '\t', args.epochs, '\t', 
          args.act, '\t', args.betas, '\t', args.num_repeats)
    print('\n')
    
    ### PERFORMANCE OPTIMIZATION INFO ###
    if args.multi_gpu and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print('##################################################################')
        print(f'MULTI-GPU MODE ENABLED')
        print(f'Available GPUs: {num_gpus}')
        print(f'Total processes: {args.num_repeats}')
        print(f'Batch size: {num_gpus} concurrent processes')
        print(f'GPU assignment: Round-robin (process_id % {num_gpus})')
        
        # Show first batch
        first_batch = min(num_gpus, args.num_repeats)
        for i in range(first_batch):
            print(f'  Process {i} → GPU {i % num_gpus}')
        if args.num_repeats > num_gpus:
            print(f'  ... (processes will queue and run in batches of {num_gpus})')
        print('##################################################################\n')
    
    if args.cache_to_gpu:
        print('##################################################################')
        print('VRAM CACHING ENABLED')
        print('Dataset will be cached to GPU memory')
        print('Expected speedup: 5-10× per epoch')
        print('##################################################################\n')

    ### FLOAT PRECISION ###
    if args.float64:
        torch.set_default_dtype(torch.float64)
        print('Using 64-bit floating point precision')
    else:
        print('Using default 32-bit floating point precision')
    print('Default dtype :\t', torch.get_default_dtype(), '\n')


    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)

    # Start processes
    start_time = time.time()

    # Clear CUDA cache before starting processes
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.multi_gpu and torch.cuda.is_available():
        # Multi-GPU mode: Use process pool to run in batches
        num_gpus = torch.cuda.device_count()
        active_processes = {}  # {process: rank}
        remaining_ranks = list(range(args.num_repeats))
        
        print(f"Starting {args.num_repeats} training processes in batches of {num_gpus}...\n")
        
        # Start initial batch (up to num_gpus processes)
        while len(active_processes) < num_gpus and remaining_ranks:
            rank = remaining_ranks.pop(0)
            p = mp.Process(target=run_process, args=(rank, args))
            p.start()
            active_processes[p] = rank
            print(f"[STARTED] Process {rank} on GPU {rank % num_gpus} (PID {p.pid})")
        
        # As processes complete, start new ones
        while active_processes:
            # Wait for any process to finish
            for p in list(active_processes.keys()):
                if not p.is_alive():
                    completed_rank = active_processes.pop(p)
                    p.join()
                    print(f"[COMPLETED] Process {completed_rank}")
                    
                    # Start next process if any remain
                    if remaining_ranks:
                        rank = remaining_ranks.pop(0)
                        new_p = mp.Process(target=run_process, args=(rank, args))
                        new_p.start()
                        active_processes[new_p] = rank
                        print(f"[STARTED] Process {rank} on GPU {rank % num_gpus} (PID {new_p.pid})")
                    break
            
            # Small sleep to avoid busy-waiting
            if active_processes:
                import time as sleep_time
                sleep_time.sleep(0.1)
    else:
        # Single-GPU mode: Spawn all processes at once (original behavior)
        processes = []
        print(f"Starting {args.num_repeats} training processes...")
        for rank in range(args.num_repeats):
            p = mp.Process(target=run_process, args=(rank, args))
            p.start()
            processes.append(p)
            print(f"Started process {rank} with PID {p.pid}")
        
        # Wait for all processes to finish
        for p in processes:
            p.join()

    # Report completion
    print("\n==== Training Complete ====")
    print(f"Processes completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 