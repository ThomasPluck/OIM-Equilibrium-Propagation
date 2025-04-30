#!/usr/bin/env python3
"""
Parallel Training for Equilibrium Propagation

This script provides functionality for training multiple independent models 
(of the same design but different seeds) in parallel on a single GPU.
"""

import sys
import argparse
import torch
import os
from datetime import datetime

from parallel_training import train_models_in_parallel


def add_arguments(parser):
    """
    Add standard command line arguments for Equilibrium Propagation
    
    Parameters:
    - parser: ArgumentParser instance
    
    Returns:
    - parser: Updated ArgumentParser
    """
    # WandB arguments
    parser.add_argument('--wandb_project', type=str, default='Equilibrium-Propagation', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity/username')
    parser.add_argument('--wandb_name', type=str, default=None, help='WandB run name')
    parser.add_argument('--wandb_group', type=str, default=None, help='WandB group name for organizing related runs')
    parser.add_argument('--wandb_mode', type=str, default='disabled', help='WandB mode (online/offline/disabled)')
    parser.add_argument('--wandb_id', type=str, default=None, help='WandB run ID for continuing a crashed run')

    parser.add_argument('--model', type=str, default='MLP', metavar='m', help='model e.g. MLP, OIM_MLP, CNN') 
    parser.add_argument('--act', type=str, default='cos', metavar='a', help='activation function, their default was mysig')
    parser.add_argument('--task', type=str, default='MNIST', metavar='t', choices=['MNIST', 'CIFAR10'], help='task')
    parser.add_argument('--optim', type=str, default='sgd', metavar='opt', choices=['sgd', 'adam'], help='optimizer for training')
    parser.add_argument('--loss', type=str, default='mse', metavar='lss', choices=['mse', 'cel'], help='loss for training')
    parser.add_argument('--alg', type=str, default='EP', metavar='al', help='EP or BPTT')
    parser.add_argument('--thirdphase', default=False, action='store_true', help='add third phase for higher order evaluation of the gradient (default: False)')
    parser.add_argument('--save', default=False, action='store_true', help='saving results')
    parser.add_argument('--todo', type=str, default='train', metavar='tr', choices=['train', 'test', 'gducheck', 'evaluate'], help='training or plot gdu curves or evaluate')
    parser.add_argument('--load-path', type=str, default='', metavar='l', help='load a model')
    parser.add_argument('--seed', type=int, default=42, metavar='s', help='random seed')
    parser.add_argument('--device', type=int, default=0, metavar='d', help='device')

    parser.add_argument('--T1', type=int, default=20, metavar='T1', help='Time of first phase')
    parser.add_argument('--T2', type=int, default=4, metavar='T2', help='Time of second phase (and third phase if applicable)')
    parser.add_argument('--betas', nargs='+', type=float, default=[0.0, 0.01], metavar='Bs', help='Betas in first and second (and third if applicable) phase')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Step size for OIM dynamics (default=0.1)')
    parser.add_argument('--noise_level', type=float, default=0.0, help='Noise level for phase dynamics (default=0.0)')
    parser.add_argument('--N_data_train', type=int, default=1000, help='Number of training data points (default: 1000)')
    parser.add_argument('--N_data_test', type=int, default=100, help='Number of test data points (default: 100)')

    parser.add_argument('--archi', nargs='+', type=int, default=[784, 512, 10], metavar='A', help='architecture of the network')
    parser.add_argument('--weight_lrs', nargs='+', type=float, default=[], metavar='l', help='layer wise lr')
    parser.add_argument('--bias_lrs', nargs='+', type=float, default=[], metavar='bl', help='layer wise lr for biases (only applies to OIM models)')
    parser.add_argument('--sync_lrs', nargs='+', type=float, default=[], metavar='sl', help='layer wise lr for sync parameters (only applies to OIM models)')
    parser.add_argument('--epochs', type=int, default=1, metavar='EPT', help='Number of epochs per tasks')
    parser.add_argument('--weight_scale', nargs='+', type=float, default=None, metavar='wg', help='scale factors for weight init (single float or list per layer)')
    parser.add_argument('--bias_scale', nargs='+', type=float, default=None, metavar='bg', help='scale factors for bias init (single float or list per layer, defaults to weight_scale)')
    
    parser.add_argument('--mbs', type=int, default=20, metavar='M', help='minibatch size')
    parser.add_argument('--plot', default=False, action='store_true', help='Enable plotting of phase dynamics during training and evaluation')
    parser.add_argument('--debug', default=False, action='store_true', help='Debug mode (default: False)')
    parser.add_argument('--check-thm', default=False, action='store_true', help='checking the gdu while training')
    
    parser.add_argument('--mmt', type=float, default=0.0, metavar='mmt', help='Momentum for sgd')
    parser.add_argument('--wds', nargs='+', type=float, default=None, metavar='l', help='layer weight decays')
    
    parser.add_argument('--lr-decay', default=False, action='store_true', help='enabling learning rate decay')
    parser.add_argument('--random_phase_initialisation', default=False, action='store_true', help='Initialize phases randomly between 0 and 2π (default: False)')
    parser.add_argument('--intralayer_connections', default=False, action='store_true', help='Add trainable synaptic connections within each hidden layer (default: False)')
    parser.add_argument('--reinitialise_neurons', default=False, action='store_true', help='Reinitialize neurons before second phase (and third if applicable) (default: False)')
    parser.add_argument('--input_positive_negative_mapping', default=False, action='store_true', help='Remap input pixel values from [0,1] to [-1,1] (default: False)')

    # Quantization parameters for physical system modeling
    parser.add_argument('--quantisation_bits', type=int, default=0, help='Number of bits for parameter quantization (0 means no quantization)')
    parser.add_argument('--J_max', type=float, default=1.0, help='Maximum absolute value for synaptic weights')
    parser.add_argument('--h_max', type=float, default=1.0, help='Maximum absolute value for bias parameters')
    parser.add_argument('--sync_max', type=float, default=1.0, help='Maximum absolute value for synchronization parameters')

    parser.add_argument('--random_sign', default=False, action='store_true', help='randomly switch beta_2 sign')
    parser.add_argument('--data_aug', default=False, action='store_true', help='enabling data augmentation for cifar10')
    parser.add_argument('--softmax', default=False, action='store_true', help='softmax loss with parameters (default: False)')

    parser.add_argument('--float64', default=False, action='store_true', help='Use 64-bit float precision instead of default 32-bit')

    # CNN parameters (commented out as in main.py)
    # parser.add_argument('--pools', type=str, default='mm', metavar='p', help='pooling') 
    # parser.add_argument('--channels', nargs='+', type=int, default=[32, 64], metavar='C', help='channels of the convnet')
    # parser.add_argument('--kernels', nargs='+', type=int, default=[5, 5], metavar='K', help='kernels sizes of the convnet')
    # parser.add_argument('--strides', nargs='+', type=int, default=[1, 1], metavar='S', help='strides of the convnet')
    # parser.add_argument('--paddings', nargs='+', type=int, default=[0, 0], metavar='P', help='paddings of the conv layers')
    # parser.add_argument('--fc', nargs='+', type=int, default=[10], metavar='S', help='linear classifier of the convnet')
    
    return parser

def add_parallel_arguments(parser):
    """
    Add parallel training specific arguments to the parser
    
    Parameters:
    - parser: ArgumentParser instance
    
    Returns:
    - parser: Updated ArgumentParser
    """
    # Add parallel training specific arguments
    parser.add_argument('--simultaneous_parallel_models', type=int, default=1, metavar='SPM',
                       help='Number of models to train in parallel (default: 1)')
    parser.add_argument('--parallel_base_seed', type=int, default=1, metavar='PBS',
                       help='Base seed for parallel training (default: 42)')
    
    return parser

def main():
    """
    Main function for parallel training
    """

    #### PARSE ARGUMENTS ###
    parser = argparse.ArgumentParser(description='Equilibrium Propagation with Parallel Training')
    parser = add_arguments(parser)
    parser = add_parallel_arguments(parser)
    args = parser.parse_args()


    
    #### PRINT INITIAL SUMMARY ###
    print('\n')
    print(' '.join(sys.argv))
    print('\n')
    print('##################################################################')
    print('\nargs\tmbs\tT1\tT2\tepochs\tactivation\tbetas\tparallel')
    print('\t', args.mbs, '\t', args.T1, '\t', args.T2, '\t', args.epochs, '\t', 
          args.act, '\t', args.betas, '\t', args.simultaneous_parallel_models)
    print('\n')
    
    # Print training mode
    if args.load_path:
        if os.path.exists(f"{args.load_path}/parallel_checkpoint.tar"):
            print(f"Continuing training from checkpoint at {args.load_path}/parallel_checkpoint.tar")
            print("Training mode: Resume")
        else:
            raise ValueError(f"No checkpoint found at {args.load_path}/parallel_checkpoint.tar")
    else:
        print("Training mode: New")





    ### DO MANDATORY PRE-TRAINING TASKS ###
    
    # Set global precision based on args.float64
    # This has to be done in advance
    if args.float64:
        torch.set_default_dtype(torch.float64)
        print('Using 64-bit floating point precision')
    

    
    ### TRAIN MODELS IN PARALLEL ###
    trainer, results = train_models_in_parallel(args)
    




    ### PRINT SUMMARY ###
    print(f"\n===== Training Complete =====")
    print(f"Trained {args.simultaneous_parallel_models} models")
    if args.save:
        print(f"Models saved to {args.path}")
    
    if args.simultaneous_parallel_models > 1:
        print(f"Final test accuracy for each model:")
        for idx, result in results.items():
            print(f"  Model {idx} (seed {trainer.seeds[idx]}): {result['test_accs'][-1]/100:.4f}")
    else:
        print(f"Final test accuracy: {results[0]['test_accs'][-1]/100:.4f}")


if __name__ == "__main__":
    main() 