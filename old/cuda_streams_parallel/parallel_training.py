"""
Parallel Training Module for Equilibrium Propagation

This module allows training multiple independent models simultaneously on a single GPU.
Each model will have its own seed, random initialization, and wandb logging.
"""

import torch
import numpy as np
import random
import time
import wandb
import os
from copy import copy
from datetime import datetime
from contextlib import nullcontext
import math # Ensure math is imported
import torch.profiler # Import profiler

# Import functions from train_evaluate.py to reuse code
from train_evaluate import *
from data_utils import *
from model_utils import *
# Import functions for final logging
from parallel_load_and_log_metrics import load_metrics_from_path, log_metrics_to_wandb



class ParallelTrainer:
    """
    Class for training multiple independent models simultaneously on a single GPU.
    """
    
    def __init__(self, args, num_models=1, base_seed=1):
        """
        Initialize the parallel trainer with multiple models
        
        Parameters:
        - args: Command line arguments
        - num_models: Number of models to train in parallel
        - base_seed: Base seed to derive seeds for each model
        """
        
        ### BASIC SETUP ###
        self.args = args
        self.num_models = num_models
        self.base_seed = base_seed
        self.seeds = [base_seed + i for i in range(num_models)]
        
        ### DEVICE CONFIGURATION ###
        # Set up device
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{args.device}')
        else:
            self.device = torch.device('cpu')
            
        # Create CUDA streams for parallel execution if using CUDA
        if torch.cuda.is_available():
            if num_models > 1:
                print(f"Initializing {num_models} CUDA streams for parallel training")
                self.use_cuda_streams = True
                self.streams = [torch.cuda.Stream(device=self.device) for _ in range(num_models)]
            else:
                self.use_cuda_streams = False
                self.streams = None
                print("CUDA available but streams not used - only one model being trained")
        else:
            self.use_cuda_streams = False
            self.streams = None
            if num_models > 1:
                print("CUDA not available - parallel training will be sequential")
                
        # Verify GPU memory if available
        if torch.cuda.is_available():
            self._check_gpu_memory()
        
        ### RESULTS DIRECTORY SETUP ###
        # Create results directory if saving is enabled
        if args.save:
            date = datetime.now().strftime('%Y-%m-%d')
            time_str = datetime.now().strftime('%H-%M-%S')
            if args.load_path=='':
                # Format architecture as string (e.g. "784-512-10")
                arch_str = '-'.join(str(x) for x in args.archi)
                # New path format using wandb group, name, and timestamp appended to name
                path = f'results/{args.wandb_group}/{args.wandb_name}_{date}_{time_str}'
            else:
                path = args.load_path
            if not(os.path.exists(path)):
                os.makedirs(path)
            args.path = path
        else:
            args.path = ''
        
        ### MODEL AND OPTIMIZER INITIALIZATION ###
        # Initialize empty collections
        self.models = []
        self.optimizers = []
        self.schedulers = []
        
        # Set up model class and loss function
        self._setup_model_class()
        self._setup_criterion()
        
        ### METRICS AND WANDB SETUP ###
        # Store metrics for all models - we'll only live log one of them
        self.stored_metrics = {}
        for idx in range(num_models):
            self.stored_metrics[idx] = []
        
        # Only the first model will be live-logged 
        self.live_wandb_model_idx = 0 if args.wandb_mode != 'disabled' else None
        # Single active wandb run (only for the live model)
        self.active_wandb_run = None
        
        ### PRINT INITIALIZATION SUMMARY ###
        print(f"\n===== Initializing Parallel Training =====")
        print(f"Training {num_models} independent models simultaneously")
        print(f"Model type: {args.model}")
        print(f"Base seed: {base_seed}")
        print(f"Device: {self.device}")
        print(f"Parallel execution: {self.use_cuda_streams}")
        if self.live_wandb_model_idx is not None:
            print(f"Live WandB logging: Model {self.live_wandb_model_idx} (others will be logged at the end)")
        else:
            print("WandB logging: Disabled")
        print(f"==========================================\n")
    
    def _setup_model_class(self):
        """Set the model class based on args.model"""
        if self.args.model == 'OIM_MLP':
            self.model_class = OIM_MLP
        elif self.args.model == 'MLP':
            self.model_class = P_MLP
        # elif self.args.model == 'CNN_EP':
            # self.model_class = CNN_EP
        else:
            raise ValueError(f"Model {self.args.model} not supported for parallel training")
    
    def _setup_criterion(self):
        """Create the loss function based on args.loss"""
        if self.args.loss == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='none').to(self.device)
        elif self.args.loss == 'cel':
            self.criterion = torch.nn.CrossEntropyLoss(reduction='none').to(self.device)
        else:
            raise ValueError(f"Loss {self.args.loss} not supported")
        print('loss =', self.criterion)
    
    def _check_gpu_memory(self):
        """Check if the GPU has enough memory for the requested number of models"""
        gpu_mem = torch.cuda.get_device_properties(self.device).total_memory
        gpu_mem_gb = gpu_mem / (1024**3)
        
        # Rough estimate - 1GB base + 0.5GB per model with 128 batch size
        estimated_mem_needed = 1 + (0.5 * self.num_models)
        
        if estimated_mem_needed > gpu_mem_gb * 0.9:  # Leave 10% headroom
            print(f"WARNING: You may be trying to run too many models for your GPU memory!")
            print(f"Estimated memory needed: {estimated_mem_needed:.2f} GB")
            print(f"GPU memory available: {gpu_mem_gb:.2f} GB")
            print(f"Consider reducing the number of parallel models or batch size.")
    
    def _initialize_wandb(self, model_idx, seed):
        """Initialize a wandb run for a specific model"""
        if self.args.wandb_mode != 'disabled':
            # Only initialize the live model during training
            if model_idx == self.live_wandb_model_idx:
                # Clean up name and group - remove any quotes
                name_base = self.args.wandb_name or 'run'
                group_base = self.args.wandb_group or 'parallel-group'
                
                # Remove quotes and backslashes
                name_base = name_base.replace('"', '').replace("'", '').replace('\\', '')
                group_base = group_base.replace('"', '').replace("'", '').replace('\\', '')
                
                # Use '-live' suffix for the live model
                run_name = f"{name_base}-live"
                # Keep group name based on the base name
                group_name = f"{group_base}"

                # Copy args into config
                config = vars(self.args).copy()
                config['seed'] = seed
                config['model_index'] = model_idx # Still log the actual model index in config

                # Initialize wandb run
                run = wandb.init(
                    project=self.args.wandb_project,
                    entity=self.args.wandb_entity,
                    name=run_name, # Use the new run_name with -live suffix
                    group=group_name,
                    config=config,
                    mode=self.args.wandb_mode,
                    # id=run_name, # Let wandb generate ID, avoid using name as ID
                    # resume='allow', # Resume might be tricky if name changes
                    settings=wandb.Settings(start_method='thread')
                )

                # Log info
                print(f"Initialized LIVE WandB run for model {model_idx}: name={run_name}, id={run.id}")
                return run
            else:
                # Non-live models are initialized later or via the recovery script
                # print(f"Skipping WandB initialization for model {model_idx} (not the live model)")
                return None
        return None
    
    def _create_optimizer(self, model):
        """
        Create an optimizer for the given model with proper parameter groups,
        matching the behavior in main.py
        """
        optim_params = []

        # Add synapse parameters with their specific learning rates
        for idx in range(len(model.synapses)):
            # Get learning rate, defaulting to last one if index is out of range
            lr = self.args.weight_lrs[idx] if idx < len(self.args.weight_lrs) else self.args.weight_lrs[-1]
            
            # Add weight decay if specified
            if hasattr(self.args, 'wds') and self.args.wds is not None:
                wd = self.args.wds[idx] if idx < len(self.args.wds) else self.args.wds[-1]
                optim_params.append({'params': model.synapses[idx].parameters(), 'lr': lr, 'weight_decay': wd})
            else:
                optim_params.append({'params': model.synapses[idx].parameters(), 'lr': lr})

        # Add intralayer synapses to the optimizer if they exist
        if hasattr(model, 'intralayer_connections') and model.intralayer_connections and hasattr(model, 'intralayer_synapses'):
            for idx, synapse in enumerate(model.intralayer_synapses):
                # Get learning rate, defaulting to last one if index is out of range
                lr = self.args.weight_lrs[idx] if idx < len(self.args.weight_lrs) else self.args.weight_lrs[-1]
                
                # Add weight decay if specified
                if hasattr(self.args, 'wds') and self.args.wds is not None:
                    wd = self.args.wds[idx] if idx < len(self.args.wds) else self.args.wds[-1]
                    optim_params.append({'params': synapse.parameters(), 'lr': lr, 'weight_decay': wd})
                else:
                    optim_params.append({'params': synapse.parameters(), 'lr': lr})

        # Handle OIM_MLP specific parameters (biases and syncs)
        if isinstance(model, OIM_MLP):
            # Add bias parameters with custom learning rates
            if hasattr(self.args, 'bias_lrs') and self.args.bias_lrs:
                for idx, bias in enumerate(model.biases):
                    lr = self.args.bias_lrs[idx] if idx < len(self.args.bias_lrs) else self.args.bias_lrs[-1]
                    
                    # Add weight decay if specified
                    if hasattr(self.args, 'wds') and self.args.wds is not None:
                        wd = self.args.wds[idx] if idx < len(self.args.wds) else self.args.wds[-1]
                        optim_params.append({'params': [bias], 'lr': lr, 'weight_decay': wd})
                    else:
                        optim_params.append({'params': [bias], 'lr': lr})
            
            # Add sync parameters with custom learning rates
            if hasattr(self.args, 'sync_lrs') and self.args.sync_lrs:
                for idx, sync in enumerate(model.syncs):
                    lr = self.args.sync_lrs[idx] if idx < len(self.args.sync_lrs) else self.args.sync_lrs[-1]
                    
                    # Add weight decay if specified
                    if hasattr(self.args, 'wds') and self.args.wds is not None:
                        wd = self.args.wds[idx] if idx < len(self.args.wds) else self.args.wds[-1]
                        optim_params.append({'params': [sync], 'lr': lr, 'weight_decay': wd})
                    else:
                        optim_params.append({'params': [sync], 'lr': lr})

        # Create the optimizer with the parameter groups
        if self.args.optim == 'sgd':
            return torch.optim.SGD(optim_params, momentum=self.args.mmt)
        elif self.args.optim == 'adam':
            return torch.optim.Adam(optim_params)
        else:
            raise ValueError(f"Optimizer {self.args.optim} not supported")
    
    def _create_scheduler(self, optimizer):
        """Create a learning rate scheduler that matches the original implementation"""
        if self.args.lr_decay:
            # Use the same scheduler configuration as in main.py
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=100,  # 100 epochs cycle as in main.py
                eta_min=1e-5  # Minimum learning rate of 1e-5 as in main.py
            )
        return None
    
    def _get_activation(self, act_name):
        """
        Get activation function by name.
        
        Parameters:
        - act_name: Name of the activation function
        
        Returns:
        - Activation function
        """
        activation_map = {
            # OIM_MLP activations
            'tanh': torch.tanh,
            'cos': torch.cos,
            # MLP activations
            'mysig': my_sigmoid,
            'sigmoid': torch.sigmoid,
            'hard_sigmoid': hard_sigmoid,
            'my_hard_sig': my_hard_sig,
            'ctrd_hard_sig': ctrd_hard_sig,
        }
        
        if act_name in activation_map:
            return activation_map[act_name]
        else:
            raise ValueError(f"Activation {act_name} not supported")
    
    def _initialize_new_model(self, idx, seed):
        """Initialize a new model with the given index and seed"""
        # Get activation function based on args.act
        activation = self._get_activation(self.args.act) if self.args.act else None
            
        # Create model
        model = self.model_class(
            self.args.archi, 
            epsilon=self.args.epsilon,
            random_phase_initialisation=self.args.random_phase_initialisation,
            activation=activation, 
            path=f"{self.args.path}/model_{idx}" if self.args.path else None,
            intralayer_connections=self.args.intralayer_connections,
            quantisation_bits=self.args.quantisation_bits,
            J_max=self.args.J_max,
            h_max=self.args.h_max,
            sync_max=self.args.sync_max
        )
        
        # Move model to device
        model.to(self.device)
        
        # Apply weight and bias scale initialization if specified
        if hasattr(self.args, 'weight_scale') and self.args.weight_scale is not None:
            model.apply(my_init(self.args.weight_scale, self.args.bias_scale))
        
        return model

    def _initialize_models(self):
        """
        Initialize all models with different seeds or load existing models if a load path is specified
        """
        models = []
        optimizers = []
        schedulers = []
        
        # For each model index
        for idx, seed in enumerate(self.seeds):
            # Set the seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            # Only initialize wandb for the live model
            if idx == self.live_wandb_model_idx:
                self.active_wandb_run = self._initialize_wandb(idx, seed)
            
            # Check if we should load an existing model
            model_loaded = False
            if hasattr(self.args, 'load_path') and self.args.load_path:
                model_path = f"{self.args.load_path}/model_{idx}/model.pt"
                checkpoint_path = f"{self.args.load_path}/model_{idx}/checkpoint.tar"
                
                if os.path.exists(model_path):
                    print(f"Loading model {idx} from {model_path}")
                    # Load the model
                    model = torch.load(model_path, map_location=self.device, weights_only=False)
                    
                    # Update model.path to match the current load_path
                    if hasattr(model, 'path') and model.path != f"{self.args.load_path}/model_{idx}":
                        print(f"Updating model path from '{model.path}' to '{self.args.load_path}/model_{idx}'")
                        model.path = f"{self.args.load_path}/model_{idx}"
                    
                    # Move model to device
                    model.to(self.device)
                    model_loaded = True
                    
                    # Create optimizer
                    optimizer = self._create_optimizer(model)
                    
                    # Load optimizer state if checkpoint exists
                    if os.path.exists(checkpoint_path):
                        print(f"Loading checkpoint for model {idx} from {checkpoint_path}")
                        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                        
                        # Load optimizer state
                        if 'opt' in checkpoint:
                            optimizer.load_state_dict(checkpoint['opt'])
                        
                        # Create and load scheduler if needed
                        scheduler = self._create_scheduler(optimizer)
                        if scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
                            scheduler.load_state_dict(checkpoint['scheduler'])
                    else:
                        # If no checkpoint, just create a scheduler
                        print(f"No checkpoint found for model {idx}, initializing new optimizer")
                        scheduler = self._create_scheduler(optimizer)
                    
                    print(f"Model {idx} loaded from {model_path}")
            
            # Initialize a new model if not loaded
            if not model_loaded:
                if hasattr(self.args, 'load_path') and self.args.load_path:
                    print(f"No model file found at {model_path}, initializing new model {idx}")
                
                # Initialize new model
                model = self._initialize_new_model(idx, seed)
                
                # Create optimizer and scheduler
                optimizer = self._create_optimizer(model)
                scheduler = self._create_scheduler(optimizer)
            
            # Add to our collections
            models.append(model)
            optimizers.append(optimizer)
            schedulers.append(scheduler)
            
            # Print status
            status = "loaded" if model_loaded else "initialized"
            print(f"Model {idx} {status} with seed {seed}")
            
            # Load existing metrics from disk into memory if they exist
            metrics_file = f"{model.path}/metrics.pt" if model.path else None
            if metrics_file and os.path.exists(metrics_file):
                try:
                    data = torch.load(metrics_file, map_location='cpu', weights_only=False) # Load to CPU to avoid GPU mem issues
                    self.stored_metrics[idx] = data.get('metrics', [])
                    print(f"Loaded {len(self.stored_metrics[idx])} existing metric entries for model {idx} from {metrics_file}")
                except Exception as e:
                    print(f"Warning: Could not load existing metrics for model {idx}: {e}")
                    self.stored_metrics[idx] = [] # Start fresh if loading failed
            else:
                # Ensure stored_metrics[idx] exists even if no file was found
                self.stored_metrics[idx] = []
        
        # Store as instance variables
        self.models = models
        self.optimizers = optimizers
        self.schedulers = schedulers
        
        action = "prepared" if hasattr(self.args, 'load_path') and self.args.load_path else "initialized"
        print(f"All {self.num_models} models {action}")
        
        # Create the hyperparameters file now that models are initialized
        if self.args.save and getattr(self.args, 'load_path', '') == '':
            # Get command line arguments to save in the hyperparameters file
            import sys
            command_line = ' '.join(sys.argv)
            
            # Save global hyperparameters for the entire run
            createHyperparametersFile(self.args.path, self.args, self.models[0], command_line)
            
            # Also save hyperparameters for each individual model
            for idx, model in enumerate(self.models):
                model_dir = f"{self.args.path}/model_{idx}"
                os.makedirs(model_dir, exist_ok=True)
                
                # Create a modified args object with model-specific seed
                import copy as python_copy
                model_args = python_copy.deepcopy(self.args)
                model_args.seed = self.seeds[idx]
                createHyperparametersFile(model_dir, model_args, model, command_line)

    def _get_datasets(self):
        """
        Generate datasets based on args.task
        
        Returns:
        - train_loader: DataLoader for training data
        - test_loader: DataLoader for test data
        """
        if self.args.task == 'MNIST':
            train_loader, test_loader = generate_mnist(self.args)
        else:
            raise ValueError(f"Task {self.args.task} not supported")
            
        return train_loader, test_loader
    
    def _train_single_model_minibatch(self, model, optimizer, data, targets, idx, epoch, batch_idx=0, epoch_metrics=None, batch_metrics=None):
        """
        Train a single model on one batch of data using the train_minibatch function
        
        Parameters:
        - model: The model to train
        - optimizer: The optimizer for the model
        - data: Input data batch
        - targets: Target labels batch
        - idx: Model index
        - epoch: Current epoch
        - batch_idx: Current batch index (passed directly to train_minibatch)
        - epoch_metrics: Dictionary to update with epoch metrics
        - batch_metrics: Dictionary to update with batch metrics (for first batch only)
        
        Returns:
        - batch_result: Full result dictionary from train_minibatch
        """
        # Context manager for CUDA stream (no-op if stream is None)
        stream_ctx = torch.cuda.stream(self.streams[idx]) if self.use_cuda_streams else nullcontext()
        
        with stream_ctx:
            # Use the train_minibatch function from train_evaluate.py
            # train_minibatch already decides whether to collect metrics based on batch_idx
            batch_result = train_minibatch(
                model, 
                optimizer, 
                data, 
                targets, 
                self.args, 
                self.criterion, 
                epoch_idx=epoch, 
                batch_idx=batch_idx
            )
            
            # Update metrics within the stream context if provided
            # Note: train_minibatch now returns tensors, aggregation happens after sync
            # if epoch_metrics is not None:
            #     epoch_metrics[idx]['correct'] += batch_result['correct'] # <-- This was incorrect if relying on .item() inside train_minibatch
            #     epoch_metrics[idx]['total'] += batch_result['total']
            #     epoch_metrics[idx]['loss'] += batch_result['loss']
            
            # Store metrics from first batch for wandb logging if provided
            if batch_metrics is not None and batch_idx == 0:
                batch_metrics[idx]['convergence_metrics'] = batch_result.get('convergence_metrics', {})
                batch_metrics[idx]['binarization_metrics'] = batch_result.get('binarization_metrics', {})
                batch_metrics[idx]['network_metrics'] = batch_result.get('network_metrics')
                batch_metrics[idx]['gradient_metrics'] = batch_result.get('gradient_metrics')
        
        return batch_result # Return the dictionary containing tensors

    def _check_gdu_single_model(self, model, data, targets, idx, error_dicts=None):
        """
        Check the GDU theorem for a single model, optionally in a dedicated CUDA stream.
        
        Parameters:
        - model: The model to check
        - data: Input data batch
        - targets: Target labels batch
        - idx: Model index
        - error_dicts: Optional list to append the error dictionary to within the CUDA stream
        
        Returns:
        - Dictionary with RMSE results from GDU check
        """
        # Context manager for CUDA stream (no-op if stream is None)
        stream_ctx = torch.cuda.stream(self.streams[idx]) if self.use_cuda_streams else nullcontext()
        
        with stream_ctx:
            # Run the GDU check for this model
            BPTT, EP = check_gdu(model, data, targets, self.args, self.criterion, betas=self.args.betas)
            
            # If third phase is enabled, run that too
            EP_2 = None
            if self.args.thirdphase:
                beta_1, beta_2 = self.args.betas
                _, EP_2 = check_gdu(model, data, targets, self.args, self.criterion, 
                                  betas=(beta_1, -beta_2))
                
                # Plot GDU results if enabled (keeping syntax consistent with train_evaluate.py)
                if self.args.plot and self.args.save:
                    # Ensure model directory exists
                    model_dir = self._ensure_model_directory_exists(idx)
                    plot_gdu(BPTT, EP, model_dir, 
                           EP_2=EP_2 if self.args.thirdphase else None, alg=self.args.alg)
                    plot_gdu_instantaneous(BPTT, EP, self.args, 
                                         EP_2=EP_2 if self.args.thirdphase else None,
                                         path=model_dir)
            
            # Calculate RMSE metrics
            error_dict = RMSE(BPTT, EP)
            
            # If a list was provided, append the result to it within the stream context
            if error_dicts is not None:
                error_dicts.append(error_dict)


    def _evaluate_single_model(self, model, test_loader, idx, should_get_metrics=False, should_plot=False, eval_results=None):
        """
        Evaluate a single model on the test set, optionally in a dedicated CUDA stream.
        
        Parameters:
        - model: The model to evaluate
        - test_loader: DataLoader for test data
        - idx: Model index
        - should_get_metrics: Whether to collect detailed metrics
        - should_plot: Whether to generate plots
        - eval_results: Optional list to append the evaluation results to within the CUDA stream
        
        Returns:
        - Dictionary with evaluation results if eval_results is None
        """
        # Context manager for CUDA stream (no-op if stream is None)
        stream_ctx = torch.cuda.stream(self.streams[idx]) if self.use_cuda_streams else nullcontext()
        
        with stream_ctx:
            # Evaluate model
            # Note: evaluate function now aggregates results internally and returns scalars/final dicts
            test_correct, test_loss_current, test_velocities, final_neurons = evaluate(
                model, test_loader, self.args.T1, self.device, 
                plot=should_plot, return_velocities=should_get_metrics,
                criterion=self.criterion, noise_level=self.args.noise_level
            )
            
            # evaluate() already calculates accuracy internally, but we recalculate here for consistency?
            # This seems redundant if evaluate already calculates and prints it.
            # Let's trust evaluate's calculation for now.
            test_acc_current = test_correct/(len(test_loader.dataset)) # Might be slightly different due to internal aggregation in evaluate
            
            # Create a flat result dictionary (primarily for metrics)
            result = {
                'test_correct': test_correct, # Scalar from evaluate
                'test_loss': test_loss_current, # Scalar from evaluate
                'test_acc': test_acc_current, # Scalar calculated above
                # Potentially large tensors below - only needed if should_get_metrics is True
                'test_velocities': test_velocities if should_get_metrics else None, 
                'final_neurons': final_neurons if should_get_metrics else None 
            }
            
            # Add metrics directly to the result dictionary if metrics collection is enabled
            # These are calculated based on the tensors returned by evaluate
            if should_get_metrics:
                # Check if tensors were actually returned (they are None otherwise)
                if test_velocities is not None:
                    convergence_metrics = get_convergence_metrics(test_velocities, phase_type="test")
                    for key, value in convergence_metrics.items():
                        result[f"test_convergence_{key}"] = value
                
                if final_neurons is not None:
                    binarization_metrics = get_binarization_metrics(model, final_neurons, phase_type="test")
                    for key, value in binarization_metrics.items():
                        result[f"test_binarization_{key}"] = value
            
            # If a list was provided, append the result dictionary to it within the stream context
            # This dictionary now contains scalars for loss/acc/correct and potentially tensors for metrics
            if eval_results is not None:
                eval_results.append(result)
                

    def _ensure_model_directory_exists(self, model_idx):
        """Ensure the model-specific directory exists before saving files"""
        if self.args.path:
            model_dir = f"{self.args.path}/model_{model_idx}"
            os.makedirs(model_dir, exist_ok=True)
            return model_dir
        return None

    def _save_metrics_to_disk(self, idx, metrics, epoch):
        """
        Save metrics to disk for a single model to preserve data in case of SLURM timeout
        
        Parameters:
        - idx: Model index
        - metrics: Dictionary of metrics to save
        - epoch: Current epoch number
        """
        # Ensure model directory exists
        model_dir = self._ensure_model_directory_exists(idx)
        if model_dir is None:
            return
            
        # Path for the metrics file
        metrics_file = f"{model_dir}/metrics.pt"
        
        # Load existing metrics if file exists
        all_metrics = []
        # We also need to load existing config to preserve it
        existing_config = {}
        if os.path.exists(metrics_file):
            try:
                data = torch.load(metrics_file, weights_only=False)
                all_metrics = data.get('metrics', [])
                # Load existing config keys if they exist in the file
                for key in ['wandb_project', 'wandb_entity', 'wandb_name', 'wandb_group']:
                    if key in data:
                        existing_config[key] = data[key]
            except Exception as e:
                print(f"Warning: Could not load existing metrics file for config/metrics: {e}")
        
        # Add current metrics
        all_metrics.append({
            'epoch': epoch,
            'metrics': metrics
        })
        
        # Prepare data to save, including config
        data_to_save = {
            'model_idx': idx,
            'seed': self.seeds[idx],
            'metrics': all_metrics,
            # Add wandb config from the main args used for this trainer
            'wandb_project': getattr(self.args, 'wandb_project', None),
            'wandb_entity': getattr(self.args, 'wandb_entity', None),
            'wandb_name': getattr(self.args, 'wandb_name', None),
            'wandb_group': getattr(self.args, 'wandb_group', None)
        }
        
        # Update with existing config ONLY if the current args value is None (to avoid overwriting older valid config)
        # This handles the case where maybe the args changed mid-run, though unlikely.
        # A better approach might be to just always save the current args.
        for key, value in existing_config.items():
            if data_to_save.get(key) is None:
                 data_to_save[key] = value
        
        # Save updated data
        torch.save(data_to_save, metrics_file)

    def train(self, checkpoint=None):
        """
        Train all models in parallel
        
        Parameters:
        - checkpoint: Optional checkpoint to resume training from
        
        Returns:
        - Dictionary of training results for each model
        """


        
        ### INITIALIZATION PHASE ###
        # Initialize models
        self._initialize_models()
        
        # Verify wandb runs are properly initialized
        if self.args.wandb_mode != 'disabled':
            print("\nVerifying wandb run:")
            if self.active_wandb_run is not None:
                print(f"  Live model {self.live_wandb_model_idx}: Active (ID: {self.active_wandb_run.id})")
            else:
                print(f"  Live model {self.live_wandb_model_idx}: Not initialized")
            print()
        
        # Get datasets
        train_loader, test_loader = self._get_datasets()



        
        
        ### SETUP TRAINING STATE ###
        # Setup for storing results and resume from checkpoint if provided
        results = {}
        
        if checkpoint is None:
            # Fresh start for all models
            for idx in range(self.num_models):
                results[idx] = {
                    'train_losses': [],
                    'test_losses': [],
                    'train_accs': [10.0],  # Initialize with dummy value as in train_evaluate.py
                    'test_accs': [10.0],   # Initialize with dummy value as in train_evaluate.py
                    'best': 0.0
                }
            epoch_sofar = 0
        else:
            # Resume from checkpoint
            results = checkpoint['results']
            epoch_sofar = checkpoint['epoch']
            print(f"Resuming training from epoch {epoch_sofar}")
        
        # Calculate iterations per epoch for progress updates
        mbs = train_loader.batch_size
        iter_per_epochs = math.ceil(len(train_loader.dataset)/mbs)
        start = time.time()
        




        
        ### TRAINING LOOP ###
        for epoch in range(self.args.epochs):
            # Training metrics for this epoch
            epoch_metrics = {}
            for idx in range(self.num_models):
                epoch_metrics[idx] = {
                    'correct': 0,
                    'total': 0,
                    'loss': 0.0
                }
            
            # Set all models to train mode
            for model in self.models:
                model.train()
            
            # Store batch metrics for each model (only collected from first batch)
            batch_metrics = {}
            for idx in range(self.num_models):
                batch_metrics[idx] = {
                    'convergence_metrics': {},
                    'binarization_metrics': {},
                    'network_metrics': None,
                    'gradient_metrics': None
                }

            # --- Start Profiling (Moved outside batch loop) ---
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2), # Profile a few batches per epoch
                # Add flush=True to the print statement
                on_trace_ready=lambda prof: print(f"\n--- PROFILER TRACE READY (End of Batch {batch_idx}) ---\n{prof.key_averages().table(sort_by='cpu_time_total', row_limit=10)}\n", flush=True),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                
                ### BATCH TRAINING LOOP ###
                for batch_idx, (data, targets) in enumerate(train_loader):
                    data, targets = data.to(self.device), targets.to(self.device)

                    # The profiler context is now outside this specific block
                    try:
                        # Initialize lists to store tensor results for this batch across models
                        batch_correct_tensors = []
                        batch_loss_tensors = []
                        batch_sizes = []
                        
                        ### TRAIN ALL MODELS ON CURRENT BATCH ###
                        # Train all models on this batch (either with or without CUDA streams)
                        for idx, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
                            # Train each model (stream context is handled inside the method)
                            batch_result = self._train_single_model_minibatch(
                                model, optimizer, data, targets, idx, epoch, batch_idx=batch_idx, 
                                # epoch_metrics are updated *after* sync now
                                batch_metrics=batch_metrics
                            )
                            # Store tensors from result
                            batch_correct_tensors.append(batch_result['correct_tensor'])
                            batch_loss_tensors.append(batch_result['batch_loss_tensor'])
                            batch_sizes.append(batch_result['batch_size'])
                        
                        # Synchronize all streams to ensure operations are complete (only if using CUDA streams)
                        if self.use_cuda_streams:
                            torch.cuda.synchronize(self.device)
                        
                        # Aggregate results after synchronization
                        for idx in range(self.num_models):
                            epoch_metrics[idx]['correct'] += batch_correct_tensors[idx].item()
                            # Calculate weighted loss sum
                            epoch_metrics[idx]['loss'] += batch_loss_tensors[idx].item() * batch_sizes[idx]
                            epoch_metrics[idx]['total'] += batch_sizes[idx]
                        
                        print(f"--- Calling prof.step() for batch {batch_idx} ---") # Debug print
                        # Signal profiler that a step is complete (inside the profiler context)
                        prof.step()
                        print(f"--- Finished prof.step() for batch {batch_idx} ---") # Debug print
                    
                    except Exception as e:
                        print(f"\n !!! EXCEPTION INSIDE PROFILER BLOCK (batch {batch_idx}) !!!")
                        print(e)
                        import traceback
                        traceback.print_exc()
                        # Optionally re-raise if needed, or just continue
                        # raise e 
                # --- End Profiling --- 

                ### PROGRESS REPORTING ###
                if iter_per_epochs < 10 or ((batch_idx%(iter_per_epochs//10)==0) or (batch_idx==iter_per_epochs-1)):
                    # Only print first model's progress to avoid excessive output
                    # Use epoch_metrics which are now correctly aggregated after sync
                    train_acc_current = epoch_metrics[0]['correct']/epoch_metrics[0]['total'] if epoch_metrics[0]['total'] > 0 else 0.0 # Avoid division by zero
                    avg_loss = epoch_metrics[0]['loss']/epoch_metrics[0]['total'] if epoch_metrics[0]['total'] > 0 else 0.0 # Avoid division by zero
                    fractional_epoch = round(epoch_sofar+epoch+(batch_idx/iter_per_epochs), 2)
                    
                    print('##### Epoch :', fractional_epoch, ' #####',
                          '\tRun train acc :', round(train_acc_current,3),'\t('+str(epoch_metrics[0]['correct'])+'/'+str(epoch_metrics[0]['total'])+')\t',
                          '\tRun train loss:', round(avg_loss,3),
                          timeSince(start, ((batch_idx+1)+(epoch_sofar+epoch)*iter_per_epochs)/((epoch_sofar+self.args.epochs)*iter_per_epochs)))
                    
                    # Log fractional epoch metrics to wandb for the live model
                    if self.args.wandb_mode != "disabled" and self.live_wandb_model_idx == 0 and self.active_wandb_run is not None:
                        self.active_wandb_run.log({
                            "train/running_epoch_accuracy": train_acc_current,
                            "train/running_epoch_loss": avg_loss,
                            "epoch_decimal": fractional_epoch
                        })
                    
                    
                    ### DEBUG OUTPUT ###
                    if self.args.debug and batch_idx == 0:
                        print(f"\n### DEBUGGING INFORMATION AFTER EPOCH {epoch_sofar+epoch} ###")
                        
                        if batch_metrics[0]['network_metrics']:
                            print_network_metrics(batch_metrics[0]['network_metrics'])
                        
                        if batch_metrics[0]['gradient_metrics']:
                            print_gradient_metrics(batch_metrics[0]['gradient_metrics'])
                        
                        # Print convergence and binarization metrics
                        for phase, metrics in batch_metrics[0]['convergence_metrics'].items():
                            print(f"\n{phase.capitalize()} phase convergence metrics:")
                            print_convergence_metrics(metrics)
                        
                        for phase, metrics in batch_metrics[0]['binarization_metrics'].items():
                            print(f"\n{phase.capitalize()} phase binarization metrics:")
                            print_binarization_metrics(metrics)
                        
                        print(f"\nTrain accuracy: {100*train_acc_current:.4f} ({epoch_metrics[0]['correct']}/{epoch_metrics[0]['total']})")
                        print(f"\nTrain loss: {avg_loss:.4f}")
                        print()
                        print("### ###")
                        print()
                        
                    
                    ### QUANTIZATION DEBUG ###
                    if hasattr(self.models[0], 'quantisation_bits') and self.models[0].quantisation_bits > 0:
                        flat_values = []
                        for synapse in self.models[0].synapses:
                            flat_values.append(synapse.weight.data.flatten())
                        all_values = torch.cat(flat_values)
                        unique_values = torch.unique(all_values)
                        print(f"Unique quantized synapse weights ({len(unique_values)}): {unique_values.tolist()}\n")
                    

                    
                    ### PLOT NEURAL ACTIVITY ###
                    if self.args.plot and self.args.save:
                        for idx, model in enumerate(self.models):
                            if batch_idx == 0 and 'neurons' in batch_result:
                                # Ensure model directory exists
                                self._ensure_model_directory_exists(idx)
                                plot_neural_activity(batch_result['neurons'], f"{self.args.path}/model_{idx}")
                




                
                ### GDU THEOREM CHECK ###
                if (batch_idx==iter_per_epochs-1) and self.args.check_thm:
                    # Run GDU checks for all models in parallel
                    error_dicts = []
                    subset_size = min(5, data.size(0))
                    subset_data = data[0:subset_size,:]
                    subset_targets = targets[0:subset_size]
                    
                    # Launch all GDU checks in parallel
                    for idx, model in enumerate(self.models):
                        # Pass error_dicts to allow appending within the CUDA stream
                        self._check_gdu_single_model(model, subset_data, subset_targets, idx, error_dicts)
                    
                    # Synchronize all streams if using CUDA
                    if self.use_cuda_streams:
                        torch.cuda.synchronize(self.device)
                    
                    # Process results and log RMSE after synchronization
                    for idx, error_dict in enumerate(error_dicts):
                        # Store the metric for all models
                        wandb_metrics = {}
                        for key, metrics in error_dict.items():
                            # Format the parameter name for wandb logging
                            param_name = key.replace('.', '_')
                            wandb_metrics[f'gdu_check/{param_name}/rmse'] = metrics['rmse']
                            wandb_metrics[f'gdu_check/{param_name}/sign_error'] = metrics['sign_error']
                        
                        # Save to stored metrics
                        self.stored_metrics[idx].append({
                            'type': 'gdu_check',
                            'epoch': epoch_sofar+epoch+1,
                            'metrics': wandb_metrics
                        })
                        
                        # Save GDU check metrics to disk
                        if self.args.save:
                            self._save_metrics_to_disk(idx, wandb_metrics, epoch_sofar+epoch+1)
                        
                        # Log live for the active model
                        if self.args.wandb_mode != "disabled" and idx == self.live_wandb_model_idx and self.active_wandb_run is not None:
                            wandb_metrics = {}
                            for key, metrics in error_dict.items():
                                # Format the parameter name for wandb logging
                                param_name = key.replace('.', '_')
                                wandb_metrics[f'gdu_check/{param_name}/rmse'] = metrics['rmse']
                                wandb_metrics[f'gdu_check/{param_name}/sign_error'] = metrics['sign_error']
                            self.active_wandb_run.log(wandb_metrics)


                            
            
            ### END OF EPOCH OPERATIONS ###
            # Learning rate decay after each epoch for each model
            for idx, scheduler in enumerate(self.schedulers):
                if scheduler is not None:
                    scheduler.step()
            


            
            ### MODEL EVALUATION ###
            # Evaluate all models on test set in parallel
            should_get_metrics = (self.args.debug or self.args.wandb_mode != "disabled")
            should_plot = self.args.plot
            
            # Launch all evaluations in parallel
            eval_results = []
            for idx, model in enumerate(self.models):
                # Pass eval_results to allow appending within the CUDA stream
                self._evaluate_single_model(model, test_loader, idx, should_get_metrics, should_plot, eval_results)
            
            # Synchronize all streams if using CUDA
            if self.use_cuda_streams:
                torch.cuda.synchronize(self.device)
            

            ### PROCESS EVALUATION RESULTS ###
            for idx, eval_result in enumerate(eval_results):
                test_correct = eval_result['test_correct']
                test_loss_current = eval_result['test_loss']
                test_acc_current = eval_result['test_acc']
                
                # Calculate train metrics for this epoch
                train_acc_current = epoch_metrics[idx]['correct']/epoch_metrics[idx]['total']
                train_loss_current = epoch_metrics[idx]['loss']/epoch_metrics[idx]['total']
                
                # Store results
                results[idx]['train_accs'].append(100*train_acc_current)
                results[idx]['train_losses'].append(train_loss_current)
                results[idx]['test_accs'].append(100*test_acc_current)
                results[idx]['test_losses'].append(test_loss_current)
                
                ### DEBUG TEST METRICS ###
                if should_get_metrics and self.args.debug and idx == 0:
                    print("\n### TEST PHASE METRICS ###")
                    print("Test phase convergence metrics:")
                    print_convergence_metrics({k.replace('test_convergence_', ''): v for k, v in eval_result.items() if k.startswith('test_convergence_')})
                    print("\nTest phase binarization metrics:")
                    print_binarization_metrics({k.replace('test_binarization_', ''): v for k, v in eval_result.items() if k.startswith('test_binarization_')})
                    print("### ###\n")
                
                ### PREPARE AND LOG METRICS ###
                metrics_to_log = {}
                
                if should_get_metrics:
                    # Combine training and test phase metrics for logging, similar to train_evaluate.py
                    if batch_metrics[idx]['convergence_metrics']:
                        for key, value in batch_metrics[idx]['convergence_metrics'].items():
                            metrics_to_log[f"convergence/{key}"] = value
                    
                    # Add test phase convergence metrics
                    for key, value in eval_result.items():
                        if key.startswith('test_convergence_'):
                            metrics_to_log[f"convergence/test/{key.replace('test_convergence_', '')}"] = value
                    
                    # Add training phase binarization metrics
                    if batch_metrics[idx]['binarization_metrics']:
                        for key, value in batch_metrics[idx]['binarization_metrics'].items():
                            metrics_to_log[f"binarization/{key}"] = value
                    
                    # Add test phase binarization metrics
                    for key, value in eval_result.items():
                        if key.startswith('test_binarization_'):
                            metrics_to_log[f"binarization/test/{key.replace('test_binarization_', '')}"] = value
                    
                    # Add network metrics if available
                    if batch_metrics[idx]['network_metrics']:
                        for key, value in batch_metrics[idx]['network_metrics'].items():
                            metrics_to_log[f"network/{key}"] = value
                    
                    # Add gradient metrics if available
                    if batch_metrics[idx]['gradient_metrics']:
                        for key, value in batch_metrics[idx]['gradient_metrics'].items():
                            metrics_to_log[f"gradient/{key}"] = value
                
                # Add basic training and testing metrics
                epoch_metrics_to_log = {
                    "train/accuracy": 100*train_acc_current,
                    "test/accuracy": 100*test_acc_current,
                    "train/loss": train_loss_current,
                    "test/loss": test_loss_current,
                    "epoch": epoch_sofar+epoch+1,
                    "learning_rate": self.optimizers[idx].param_groups[0]['lr']
                }
                
                # Merge with detailed metrics
                metrics_to_log.update(epoch_metrics_to_log)
                
                # Store metrics for all models
                self.stored_metrics[idx].append({
                    'type': 'epoch',
                    'epoch': epoch_sofar+epoch+1,
                    'metrics': metrics_to_log
                })
                
                # Save metrics to disk after each epoch to preserve data in case of SLURM timeout
                if self.args.save:
                    self._save_metrics_to_disk(idx, metrics_to_log, epoch_sofar+epoch+1)
                
                # Log metrics to wandb for live model only
                if self.args.wandb_mode != "disabled" and idx == self.live_wandb_model_idx and self.active_wandb_run is not None:
                    # Only log metrics for live model
                    self.active_wandb_run.log(metrics_to_log)
                




                
                ### SAVE MODEL CHECKPOINTS ###
                if self.args.save:
                    if test_correct > results[idx].get('best', 0):
                        results[idx]['best'] = test_correct
                        model_dir = f"{self.args.path}/model_{idx}"
                        os.makedirs(model_dir, exist_ok=True)
                        
                        # Save checkpoint
                        save_dic = {
                            'model_state_dict': model.state_dict(), 
                            'opt': self.optimizers[idx].state_dict(),
                            'train_acc': results[idx]['train_accs'], 
                            'test_acc': results[idx]['test_accs'], 
                            'train_loss': results[idx]['train_losses'], 
                            'test_loss': results[idx]['test_losses'],
                            'best': results[idx]['best'], 
                            'epoch': epoch_sofar+epoch+1,
                            'seed': self.seeds[idx]
                        }
                        
                        # Add scheduler if it exists
                        if self.schedulers[idx] is not None:
                            save_dic['scheduler'] = self.schedulers[idx].state_dict()
                        else:
                            save_dic['scheduler'] = None
                        
                        torch.save(save_dic, f"{model_dir}/checkpoint.tar")
                        torch.save(model, f"{model_dir}/model.pt")
                        


                        
                    ### PLOT TRAINING CURVES ###
                    # Ensure model directory exists
                    self._ensure_model_directory_exists(idx)
                    plot_acc(results[idx]['train_accs'], results[idx]['test_accs'], f"{self.args.path}/model_{idx}")
                    plot_loss(results[idx]['train_losses'], results[idx]['test_losses'], f"{self.args.path}/model_{idx}")
            


            
            ### EPOCH SUMMARY ###
            print(f"\n === Epoch {epoch_sofar+epoch+1}/{epoch_sofar+self.args.epochs} Summary ===")
            for idx in range(self.num_models):
                print(f"  Model {idx} (seed {self.seeds[idx]}): "
                      f"Train loss: {results[idx]['train_losses'][-1]:.4f}, "
                      f"Train acc: {results[idx]['train_accs'][-1]/100:.4f}, "
                      f"Test loss: {results[idx]['test_losses'][-1]:.4f}, "
                      f"Test acc: {results[idx]['test_accs'][-1]/100:.4f}")
            

            
            ### SAVE CHECKPOINT ###
            # Save overall checkpoint for resuming training
            if self.args.save:
                overall_checkpoint = {
                    'results': results,
                    'epoch': epoch_sofar+epoch+1,
                }
                torch.save(overall_checkpoint, f"{self.args.path}/parallel_checkpoint.tar")
        

        
        ### FINAL OPERATIONS ###
        # Save final models
        if self.args.save:
            self.save_models(results)
        


        
        ### LOG FINAL METRICS ###
        # Log the final completion message for live model
        if self.args.wandb_mode != "disabled" and self.active_wandb_run is not None:
            try:
                self.active_wandb_run.log({
                    "training_completed": True,
                    "final_epoch": self.args.epochs,
                    "final_test_acc": results[self.live_wandb_model_idx]['test_accs'][-1]/100
                })
                # Finish this run so we can log other models
                self.active_wandb_run.finish()
            except Exception as e:
                print(f"Warning: Could not log final metrics to wandb for live model: {e}")

        # Log final metrics for ALL models using the imported functions
        if self.args.wandb_mode != "disabled" and self.args.save:
            print(f"\nLoading metrics from disk ({self.args.path}) and logging all models to WandB...")
            try:
                # Load all metrics and configs from disk (this might include model 0's just saved metrics)
                all_metrics, _, model_configs = load_metrics_from_path(self.args.path)
                
                # Log them all, marking as completed
                # Pass self.args for wandb config fallback and mode
                log_metrics_to_wandb(all_metrics, model_configs, self.args, mark_completed=True) 
                print("Finished logging all models.")
            except Exception as e:
                print(f"Error during final logging of all models: {e}")
        
        return results

    def save_models(self, results):
        """
        Save all models and their results
        
        Parameters:
        - results: Dictionary of training results for each model
        """
        # Create base directory if it doesn't exist
        os.makedirs(self.args.path, exist_ok=True)
        
        for idx, model in enumerate(self.models):
            # Use the same model directory as during training
            model_dir = f"{self.args.path}/model_{idx}"
            os.makedirs(model_dir, exist_ok=True)
            
            # Save final model
            save_dic = {
                'model_state_dict': model.state_dict(), 
                'opt': self.optimizers[idx].state_dict(),
                'train_acc': results[idx]['train_accs'], 
                'test_acc': results[idx]['test_accs'], 
                'train_loss': results[idx]['train_losses'], 
                'test_loss': results[idx]['test_losses'],
                'best': results[idx].get('best', 0), 
                'epoch': self.args.epochs,
                'seed': self.seeds[idx]
            }
            
            # Add scheduler if it exists
            if self.schedulers[idx] is not None:
                save_dic['scheduler'] = self.schedulers[idx].state_dict()
            
            torch.save(save_dic, f"{model_dir}/final_checkpoint.tar")
            torch.save(model, f"{model_dir}/final_model.pt")
            
            # No need to save hyperparameters here - they're already saved at the beginning of training
            
            print(f"Final model {idx} saved to {model_dir}")
        
        print(f"All models saved successfully")


def train_models_in_parallel(args):
    """
    Train multiple models in parallel on a single GPU.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments containing training configuration.
        
    Returns:
    --------
    trainer : ParallelTrainer
        The trainer object after training.
    results : dict
        Dictionary containing training results for all models.
    """
    # Extract model count and base seed from args
    num_models = args.simultaneous_parallel_models
    base_seed = args.parallel_base_seed if num_models > 1 else args.seed
    
    # Check for existing checkpoint
    checkpoint = None
    if args.load_path and os.path.exists(f"{args.load_path}/parallel_checkpoint.tar"):
        checkpoint_path = f"{args.load_path}/parallel_checkpoint.tar"
        print(f"Loading parallel training checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu'), weights_only=False)
    
    # Initialize the parallel trainer
    trainer = ParallelTrainer(
        args=args,
        num_models=num_models,
        base_seed=base_seed
    )
    
    # Train all models in parallel
    results = trainer.train(checkpoint=checkpoint)
    
    # Return the trainer and results
    return trainer, results 
