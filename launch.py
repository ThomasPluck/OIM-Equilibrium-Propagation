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
from datetime import datetime
import copy

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
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers per process for data loading (default: 4)')
    

    # parser.add_argument('--pools', type = str, default = 'mm', metavar = 'p', help='pooling') 
    # parser.add_argument('--channels', nargs='+', type = int, default = [32, 64], metavar = 'C', help='channels of the convnet')
    # parser.add_argument('--kernels', nargs='+', type = int, default = [5, 5], metavar = 'K', help='kernels sizes of the convnet')
    # parser.add_argument('--strides', nargs='+', type = int, default = [1, 1], metavar = 'S', help='strides of the convnet')
    # parser.add_argument('--paddings', nargs='+', type = int, default = [0, 0], metavar = 'P', help='paddings of the conv layers')
    # parser.add_argument('--fc', nargs='+', type = int, default = [10], metavar = 'S', help='linear classifier of the convnet')


    return parser.parse_args()

def prepare_args_for_process(args, rank):
    """Prepare args for a specific process"""
    proc_args = copy.deepcopy(args)
    
    # Set unique seed for this process
    proc_args.seed = args.base_seed + rank
    proc_args.rank = rank

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

        
            
    # When using multiple processes, adjust data loader workers to prevent system resource exhaustion
    if args.num_repeats > 1:
        # Calculate a reasonable number of workers per process based on CPU count
        max_workers = max(1, (mp.cpu_count() // args.num_repeats) - 1)
        # Cap the workers to the maximum reasonable value, but respect the user-provided value
        proc_args.num_workers = min(args.num_workers, max_workers)
        
        if proc_args.num_workers != args.num_workers and args.num_workers > max_workers:
            print(f"Process {rank}: Limiting num_workers from {args.num_workers} to {proc_args.num_workers} to prevent resource exhaustion")

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

def main():
    """Main function for multiprocessing training"""
    ### ARGUMENTS ###
    args = get_args()
    
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
    processes = []
    start_time = time.time()

    # Clear CUDA cache before starting processes
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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