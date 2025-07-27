#!/usr/bin/env python3
"""
Feedforward Neural Network Training Script for MNIST/FashionMNIST
A comparison script that reuses existing codebase modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import os
import time
import random
from datetime import datetime
import wandb
from contextlib import nullcontext

# Import from existing modules
from data_utils import generate_mnist, generate_fashion_mnist
from model_utils import my_init
from metric_utils import *

class FeedforwardMLP(nn.Module):
    """
    Simple feedforward MLP that reuses initialization from model_utils
    """
    def __init__(self, archi, activation='tanh', weight_scale=None, bias_scale=None):
        super(FeedforwardMLP, self).__init__()
        
        self.archi = archi
        self.nc = archi[-1]  # Number of classes
        
        # Set activation function
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'cos':
            self.activation = torch.cos
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Create layers (reusing naming convention from existing code)
        # Match EP architecture: bias=False in Linear layers
        self.synapses = nn.ModuleList()
        for idx in range(len(archi) - 1):
            self.synapses.append(nn.Linear(archi[idx], archi[idx+1], bias=False))
        
        # Add separate bias parameters like EP (if bias_scale provided)
        if bias_scale is not None:
            self.biases = nn.ParameterList()
            for idx in range(len(archi) - 1):
                bias = nn.Parameter(torch.zeros(archi[idx+1]))
                self.biases.append(bias)
        else:
            self.biases = None
        
        # Initialize weights using existing initialization
        self._initialize_weights(weight_scale, bias_scale)
    
    def _initialize_weights(self, weight_scale, bias_scale):
        """Initialize weights using the existing my_init function"""
        # Initialize weights
        if weight_scale is not None:
            for i, layer in enumerate(self.synapses):
                my_init(layer.weight, weight_scale[i])
        
        # Initialize separate biases if they exist
        if self.biases is not None and bias_scale is not None:
            for i, bias in enumerate(self.biases):
                my_init(bias, bias_scale[i])
    
    def forward(self, x):
        """Standard feedforward pass with separate bias handling"""
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Pass through layers
        for i, layer in enumerate(self.synapses):
            x = layer(x)
            
            # Add separate bias if available (matching EP architecture)
            if self.biases is not None:
                x = x + self.biases[i]
            
            # Apply activation to all layers except the last
            if i < len(self.synapses) - 1:
                x = self.activation(x)
        
        return x

def train_epoch(model, train_loader, optimizer, criterion, device, args):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(data)
        
        # Calculate loss
        if args.loss == 'mse':
            # Convert to one-hot and transform to [-1,1] if needed
            y_one_hot = F.one_hot(targets, num_classes=model.nc).float()
            if args.input_positive_negative_mapping:
                y_one_hot = y_one_hot * 2 - 1  # Transform to [-1,1]
            loss = 0.5 * criterion(outputs, y_one_hot).sum(dim=1).mean()
        else:  # Cross-entropy
            loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        pred = torch.argmax(outputs, dim=1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy

def test_epoch(model, test_loader, criterion, device, args):
    """Test for one epoch"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            
            # Calculate loss
            if args.loss == 'mse':
                y_one_hot = F.one_hot(targets, num_classes=model.nc).float()
                if args.input_positive_negative_mapping:
                    y_one_hot = y_one_hot * 2 - 1
                loss = 0.5 * criterion(outputs, y_one_hot).sum(dim=1).mean()
            else:
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy

def create_results_directory(args):
    """Create results directory structure similar to existing codebase"""
    if not args.save:
        return None
        
    date = datetime.now().strftime('%Y-%m-%d')
    time_str = datetime.now().strftime('%H-%M-%S')
    
    results_dir = f"results/Feedforward/{args.loss}/{date}"
    os.makedirs(results_dir, exist_ok=True)
    
    model_dir = f"{results_dir}/{time_str}_feedforward"
    os.makedirs(model_dir, exist_ok=True)
    
    return model_dir

def createHyperparametersFile(save_dir, args, model, additional_info=""):
    """Save hyperparameters file using existing format"""
    if save_dir is None:
        return
        
    with open(os.path.join(save_dir, 'hyperparameters.txt'), 'w') as f:
        f.write("=== Feedforward Neural Network Training ===\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write all arguments
        f.write("=== Arguments ===\n")
        for key, value in sorted(vars(args).items()):
            f.write(f"{key}: {value}\n")
        
        f.write(f"\n=== Model Architecture ===\n")
        f.write(f"{model}\n")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"\nTotal parameters: {total_params}\n")
        f.write(f"Trainable parameters: {trainable_params}\n")
        
        if additional_info:
            f.write(f"\n=== Additional Info ===\n")
            f.write(additional_info)

def set_seeds(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def initialize_wandb(args):
    """Initialize Weights & Biases logging"""
    if args.wandb_mode == 'disabled':
        return None
    
    config = vars(args).copy()
    
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        group=args.wandb_group,
        config=config,
        mode=args.wandb_mode
    )
    
    print(f"WandB initialized with run name: {args.wandb_name}")
    return run

def get_args():
    """Parse command line arguments - similar to existing launch.py"""
    parser = argparse.ArgumentParser(description='Feedforward Neural Network Training')
    
    # WandB arguments
    parser.add_argument('--wandb-project', type=str, default='Feedforward-Comparison', help='WandB project name')
    parser.add_argument('--wandb-entity', type=str, default='alexgower-team', help='WandB entity/username')
    parser.add_argument('--wandb-name', type=str, default=None, help='WandB run name')
    parser.add_argument('--wandb-group', type=str, default=None, help='WandB group name')
    parser.add_argument('--wandb-mode', type=str, default='disabled', help='WandB mode (online/offline/disabled)')
    
    # Model arguments (matching existing codebase)
    parser.add_argument('--act', type=str, default='tanh', help='Activation function (tanh, relu, sigmoid, cos)')
    parser.add_argument('--task', type=str, default='MNIST', help='Task (MNIST or FashionMNIST)')
    parser.add_argument('--optim', type=str, default='sgd', help='Optimizer (sgd, adam)')
    parser.add_argument('--loss', type=str, default='mse', help='Loss function (mse, cel)')
    parser.add_argument('--save', default=False, action='store_true', help='Save results')
    parser.add_argument('--device', type=int, default=0, help='Device ID')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Data arguments (matching existing naming)
    parser.add_argument('--N-data-train', type=int, default=1000, help='Number of training data points')
    parser.add_argument('--N-data-test', type=int, default=100, help='Number of test data points')
    parser.add_argument('--input-positive-negative-mapping', default=False, action='store_true', 
                       help='Remap input pixels to [-1,1]')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loading')
    
    # Architecture arguments (matching existing naming)
    parser.add_argument('--archi', nargs='+', type=int, default=[784, 120, 10], help='Network architecture')
    parser.add_argument('--weight-lrs', nargs='+', type=float, default=[0.01, 0.001], help='Layer-wise learning rates')
    parser.add_argument('--bias-lrs', nargs='+', type=float, default=None, help='Layer-wise bias learning rates')
    parser.add_argument('--weight-scale', nargs='+', type=float, default=None, help='Weight initialization scale')
    parser.add_argument('--bias-scale', nargs='+', type=float, default=None, help='Bias initialization scale')
    
    # Training arguments (matching existing naming)
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--mbs', type=int, default=20, help='Mini-batch size')
    parser.add_argument('--mmt', type=float, default=0.0, help='Momentum for SGD')
    parser.add_argument('--wds', nargs='+', type=float, default=None, help='Layer-wise weight decay')
    parser.add_argument('--lr-decay', default=False, action='store_true', help='Enable learning rate decay')
    
    # Debug arguments
    parser.add_argument('--debug', default=False, action='store_true', help='Debug mode')
    parser.add_argument('--plot', default=False, action='store_true', help='Enable plotting')
    
    return parser.parse_args()

def main():
    """Main training function"""
    args = get_args()
    
    # Set random seeds
    set_seeds(args.seed)
    
    # Print startup information (similar to existing codebase)
    print('\n' + '='*60)
    print('FEEDFORWARD NEURAL NETWORK TRAINING')
    print('='*60)
    print(f"Task: {args.task}")
    print(f"Architecture: {args.archi}")
    print(f"Epochs: {args.epochs}, Batch size: {args.mbs}")
    print(f"Learning rates: {args.weight_lrs}")
    print(f"Activation: {args.act}, Loss: {args.loss}, Optimizer: {args.optim}")
    print('='*60 + '\n')
    
    # Set device
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
        print(f"Using GPU: {device}")
    else:
        device = torch.device('cpu')
        print(f"Using CPU")
    
    # Create datasets using existing functions
    if args.task == 'MNIST':
        train_loader, test_loader = generate_mnist(args)
    elif args.task == 'FashionMNIST':
        train_loader, test_loader = generate_fashion_mnist(args)
    else:
        raise ValueError(f"Unsupported task: {args.task}")
    
    # Create model
    model = FeedforwardMLP(
        archi=args.archi,
        activation=args.act,
        weight_scale=args.weight_scale,
        bias_scale=args.bias_scale
    ).to(device)
    
    print(f"Model: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create loss criterion
    if args.loss == 'mse':
        criterion = nn.MSELoss(reduction='none')  # Use reduction='none' to get per-element losses
    elif args.loss == 'cel':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss: {args.loss}")
    
    # Create optimizer with layer-wise learning rates (similar to existing code)
    optim_params = []
    
    # Add weight parameters
    for idx, layer in enumerate(model.synapses):
        lr = args.weight_lrs[idx] if idx < len(args.weight_lrs) else args.weight_lrs[-1]
        wd = 0.0
        if args.wds is not None:
            wd = args.wds[idx] if idx < len(args.wds) else args.wds[-1]
        optim_params.append({'params': layer.weight, 'lr': lr, 'weight_decay': wd})
    
    # Add bias parameters if they exist
    if model.biases is not None and args.bias_lrs is not None:
        for idx, bias in enumerate(model.biases):
            lr = args.bias_lrs[idx] if idx < len(args.bias_lrs) else args.bias_lrs[-1]
            wd = 0.0
            if args.wds is not None:
                wd = args.wds[idx] if idx < len(args.wds) else args.wds[-1]
            optim_params.append({'params': bias, 'lr': lr, 'weight_decay': wd})
    
    if args.optim == 'sgd':
        optimizer = optim.SGD(optim_params, momentum=args.mmt)
    elif args.optim == 'adam':
        optimizer = optim.Adam(optim_params)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optim}")
    
    # Create scheduler
    scheduler = None
    if args.lr_decay:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
    # Create results directory
    save_dir = create_results_directory(args)
    if save_dir:
        createHyperparametersFile(save_dir, args, model)
        print(f"Results will be saved to: {save_dir}")
    
    # Initialize WandB
    wandb_run = initialize_wandb(args)
    
    # Training loop
    print("Starting training...")
    best_test_acc = 0.0
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, args)
        
        # Test
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device, args)
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Log results
        print(f"Epoch {epoch+1:3d}/{args.epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # WandB logging
        if wandb_run:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'test_loss': test_loss,
                'test_accuracy': test_acc,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            if save_dir:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_test_acc': best_test_acc,
                    'args': args
                }, os.path.join(save_dir, 'best_model.pth'))
    
    total_time = time.time() - start_time
    
    print(f"\nTraining completed in {total_time:.2f} seconds!")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    
    # Close WandB
    if wandb_run:
        wandb.finish()

if __name__ == '__main__':
    main() 