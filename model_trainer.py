"""
Model Trainer Module for Equilibrium Propagation

This module provides a simplified trainer class for Equilibrium Propagation models.

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
import math
import torch.profiler

# Import functions from train_evaluate.py to reuse code
from train_evaluate import *
from data_utils import *
from model_utils import *


class ModelTrainer:
    """
    Class for training a single Equilibrium Propagation model.
    """
    
    def __init__(self, args):
        """
        Initialize the model trainer
        
        Parameters:
        - args: Command line arguments
        """
        
        ### BASIC SETUP ###
        self.args = args
 


        ### CREATE RESULTS DIRECTORY ###
        # Note args.path is model specific save path to save files
        # Note args.load_path is overarching path to resume training from
        # (may contain multiple model instances, or may not in which case args.path would = args.load_path anyway)
        if args.save and not(os.path.exists(args.path)):
            os.makedirs(args.path, exist_ok=True)
        
                
        ### DEVICE CONFIGURATION ###
        # Set up device
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{args.device}')
        else:
            self.device = torch.device('cpu')
            
        
        ### MODEL AND OPTIMIZER INITIALIZATION ###
        # Initialize model-related variables
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.checkpoint = None  
        
        # Set up model class and loss function
        self._setup_model_class()
        self._setup_criterion()

        # Initialize the model now
        self.model = self._initialize_model()

        # Check if we should save hyperparameters after model initialization
        if args.save and not(os.path.exists(os.path.join(args.path, 'hyperparameters.txt'))):
            createHyperparametersFile(self.args.path, self.args, self.model, "")
        
        ### METRICS AND WANDB SETUP ###
        # Store metrics for the model - list of dictionaries with epoch and metrics data
        self.stored_metrics = []
        
        # Load existing metrics if available
        self._load_existing_metrics()
        
        # Initialize wandb if enabled
        self.active_wandb_run = None
        if args.wandb_mode != 'disabled':
            self.active_wandb_run = self._initialize_wandb()
        

        ### PRINT INITIALIZATION SUMMARY ###
        print(f"\n===== Initializing Model Training =====")
        print(f"Model type: {args.model}")
        print(f"Seed: {args.seed}")
        print(f"Device: {self.device}")
        print(f"WandB logging: {'Enabled' if self.active_wandb_run else 'Disabled'}")
        print(f"Process Rank: {getattr(args, 'rank', 'N/A')}")
        print(f"==========================================\n")
    




    ##### INIT HELPER FUNCTIONS #####

    def _setup_model_class(self):
        """Set the model class based on args.model"""
        if self.args.model == 'OIM_MLP':
            self.model_class = OIM_MLP
        elif self.args.model == 'MLP':
            self.model_class = P_MLP
        # elif self.args.model == 'CNN_EP':
            # self.model_class = CNN_EP
        else:
            raise ValueError(f"Model {self.args.model} not supported for training")
    
    def _setup_criterion(self):
        """Create the loss function based on args.loss"""
        if self.args.loss == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='none').to(self.device)
        elif self.args.loss == 'cel':
            self.criterion = torch.nn.CrossEntropyLoss(reduction='none').to(self.device)
        else:
            raise ValueError(f"Loss {self.args.loss} not supported")
        print('loss =', self.criterion)
    

    def _initialize_wandb(self):
        """Initialize wandb for logging"""
        
        # Skip if wandb is disabled
        if self.args.wandb_mode == 'disabled':
            return None

        # Config
        config = vars(self.args).copy()

        # Initialize the run
        run = wandb.init(
            project=self.args.wandb_project,
            entity=self.args.wandb_entity,
            name=self.args.wandb_name,
            group=self.args.wandb_group,
            config=config,
            mode=self.args.wandb_mode,
            id=self.args.wandb_id,
            resume="allow"
        )

        print(f"WandB initialized with run name: {self.args.wandb_name}")
    
        return run
    
    ##### #####



    


    ##### MODEL INITIALISATION HELPER FUNCTIONS #####
    
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
                    
                    if lr != 0:
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
                    
                    if lr != 0:
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
                T_max=self.args.epochs, 
                eta_min=1e-5 
            )
        return None
    
    
    def _get_activation(self, act_name):
        """Get activation function based on name"""
        if act_name == 'tanh':
            return torch.tanh
        elif act_name == 'cos':
            return torch.cos
        elif act_name == 'mysig':
            return my_sigmoid
        elif act_name == 'sigmoid':
            return torch.sigmoid
        elif act_name == 'hard_sigmoid':
            return hard_sigmoid
        elif act_name == 'my_hard_sig':
            return my_hard_sig
        elif act_name == 'ctrd_hard_sig':
            return ctrd_hard_sig
        else:
            print(f"Warning: Unknown activation '{act_name}', defaulting to cos")
            return torch.cos
    

    def _initialize_new_model(self):
        """Initialize the model with the specified seed"""

        # Set the seed
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        
        # Get activation function based on args.act
        activation = self._get_activation(self.args.act)
            
        # Create model
        model = self.model_class(
            self.args.archi, 
            epsilon=self.args.epsilon,
            random_phase_initialisation=self.args.random_phase_initialisation,
            activation=activation, 
            path=self.args.path,
            intralayer_connections=self.args.intralayer_connections,
            quantisation_bits=self.args.quantisation_bits,
            J_max=self.args.J_max,
            h_max=self.args.h_max,
            sync_max=self.args.sync_max
        )
        
        # Move model to device
        model.to(self.device)
        print(model)
        
        # Apply weight and bias scale initialization if specified
        if self.args.weight_scale is not None:
            model.apply(my_init(self.args.weight_scale, self.args.bias_scale))


        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer(model)
        print(self.optimizer)
        self.scheduler = self._create_scheduler(self.optimizer)
        print(self.scheduler)
        return model


    def _load_model(self):
        """Load a model from a checkpoint or saved model file"""

        # Remember that args.path is model specific save path to save files
        # (args.load_path is overarching path to resume training from in this case)
        # so here we are just using args.load_path as boolean to check if we should load a model
        
        # Choose which checkpoint to load based on args.load_checkpoint
        if self.args.load_checkpoint == 'best':
            checkpoint_path = f"{self.args.path}/checkpoint.tar"
            model_path = f"{self.args.path}/model.pt"
            print(f"Loading best checkpoint from {checkpoint_path}")
        elif self.args.load_checkpoint == 'final':
            checkpoint_path = f"{self.args.path}/final_checkpoint.tar"
            model_path = f"{self.args.path}/final_model.pt"
            print(f"Loading final checkpoint from {checkpoint_path}")
        else:
            raise ValueError(f"Invalid load_checkpoint value: {self.args.load_checkpoint}. Must be 'best' or 'final'.")

        # Check if the requested files exist
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        ### LOAD MODEL ###
        model = torch.load(model_path, map_location=self.device, weights_only=False)
    
        # Update model path attribute
        if hasattr(model, 'path'):
            if self.args.path != model.path:
                print(f"Updating model path from '{model.path}' to '{self.args.path}'")
                model.path = self.args.path
        else:
            print(f"Adding model path attribute: '{self.args.path}'")
            model.path = self.args.path
            
        # Add load_path attribute to model (doesn't exist by default but needed for our logic)
        model.load_path = self.args.load_path

        # Move model to device
        model.to(self.device)
        print(model)

        ### ###


        ### LOAD STATE ###
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.optimizer = self._create_optimizer(model)
        self.optimizer.load_state_dict(self.checkpoint['opt'])
        print(self.optimizer)

        if self.checkpoint['scheduler'] is not None and self.args.lr_decay:
            self.scheduler = self._create_scheduler(self.optimizer)
            self.scheduler.load_state_dict(self.checkpoint['scheduler'])
            print(self.scheduler)
        ### ###

        return model

    def _load_existing_metrics(self):
        """Load existing metrics from disk if they exist"""
        if not self.args.save:
            return
            
        metrics_path = f"{self.args.path}/metrics.pt"
        if os.path.exists(metrics_path):
            try:
                data = torch.load(metrics_path, map_location='cpu', weights_only=False)
                self.stored_metrics = data.get('metrics', [])
                print(f"Loaded {len(self.stored_metrics)} existing metric entries from {metrics_path}")
            except Exception as e:
                print(f"Warning: Could not load existing metrics: {e}")
                self.stored_metrics = []

    def _save_metrics_to_disk(self, metrics, epoch):
        """
        Save metrics to disk to preserve data in case of job interruption
        
        Parameters:
        - metrics: Dictionary of metrics to save
        - epoch: Current epoch number
        """
        if not self.args.save:
            return
            
        # Path for the metrics file
        metrics_path = f"{self.args.path}/metrics.pt"
        
        # Add current metrics to stored metrics
        self.stored_metrics.append({
            'epoch': epoch,
            'metrics': metrics
        })
        
        # Prepare data to save, including config
        data_to_save = {
            'metrics': self.stored_metrics,
            # Add wandb config
            'wandb_project': getattr(self.args, 'wandb_project', None),
            'wandb_entity': getattr(self.args, 'wandb_entity', None),
            'wandb_name': getattr(self.args, 'wandb_name', None),
            'wandb_group': getattr(self.args, 'wandb_group', None)
        }
        
        # Add other useful info if it exists
        if hasattr(self.args, 'rank'):
            data_to_save['rank'] = self.args.rank
        
        # Save updated data
        torch.save(data_to_save, metrics_path)





    def _initialize_model(self):
        """Initialize the model with the specified seed"""
        
        if self.args.load_path: # i.e. if we are loading a model
            model = self._load_model()
        else:
            model = self._initialize_new_model()
    
        return model
    

    ##### #####



    ##### TRAINING HELPER FUNCTIONS #####

    def _get_datasets(self):
        """
        Generate datasets based on args.task
        
        Returns:
        - train_loader: DataLoader for training data
        - test_loader: DataLoader for test data
        """
        if self.args.task == 'MNIST':
            train_loader, test_loader = generate_mnist(self.args)
        elif self.args.task == 'FashionMNIST':
            train_loader, test_loader = generate_fashion_mnist(self.args)
        else:
            raise ValueError(f"Task {self.args.task} not supported")
            
        return train_loader, test_loader
    
    ##### #####




    ##### PUBLIC FUNCTIONS #####

    def run_training(self):
        """
        Run the full training process using the initialized model
        
        Returns:
            dict: Dictionary containing training results/metrics
        """
        # Generate data loaders
        train_loader, test_loader = self._get_datasets()
        
        # Set up profiling if enabled
        profiler_context = nullcontext()
        if self.args.profiling and hasattr(self.args, 'rank') and self.args.rank == 0:
            print(f"Enabling profiling for rank 0 process")
            profiler_context = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=3,
                    repeat=1
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{self.args.path}/profiler"),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )

        # Run training with appropriate context
        try:
            with profiler_context:
                # Run training loop
                print(f"\nStarting training for model with seed {self.args.seed}...\n")
                # The train function returns the metrics arrays
                train_accs, test_accs, train_losses, test_losses = train(
                    self.model,
                    self.optimizer,
                    train_loader,
                    test_loader,
                    self.args,
                    self.device,
                    self.criterion,
                    checkpoint=self.checkpoint,
                    scheduler=self.scheduler,
                    wandb_run=self.active_wandb_run,
                    save_metrics_callback=self._save_metrics_to_disk
                )
                
                # Print results
                print(f"Training completed for model with rank {self.args.rank} and seed {self.args.seed}")
                print(f"Final train accuracy: {train_accs[-1]}")
                print(f"Final test accuracy: {test_accs[-1]}")
                print(f"Best test accuracy: {max(test_accs)}")

    
        except Exception as e:
            import traceback
            print(f"Error during training for rank {self.args.rank} and seed {self.args.seed}: {e}")
            traceback.print_exc()
            
        # Clean up wandb
        if self.active_wandb_run:
            self.active_wandb_run.finish()
            
        
    def run_gdu_check(self):
        """
        Run GDU check to compare Equilibrium Propagation gradients with BPTT
        """
        # Get data loaders
        train_loader, _ = self._get_datasets()
        
        # Get a small batch of data for GDU check
        dataiter = iter(train_loader)
        images, labels = next(dataiter)
        images, labels = images[0:20,:], labels[0:20]  # Only use 20 samples
        images, labels = images.to(self.device), labels.to(self.device)
        
        # Run GDU check
        beta_1, beta_2 = self.args.betas
        BPTT, EP = check_gdu(self.model, images, labels, self.args, self.criterion, betas=(beta_1, beta_2))
        
        # Run second check if using third phase
        if self.args.thirdphase:
            _, EP_2 = check_gdu(self.model, images, labels, self.args, self.criterion, betas=(beta_1, -beta_2))
        
        # Print comparison results
        error_dict = RMSE(BPTT, EP)
        
        # Save results if requested
        if self.args.save:
            bptt_est = get_estimate(BPTT)
            ep_est = get_estimate(EP)
            torch.save(bptt_est, self.args.path+'/bptt.tar')
            torch.save(BPTT, self.args.path+'/BPTT.tar')
            torch.save(ep_est, self.args.path+'/ep.tar')
            torch.save(EP, self.args.path+'/EP.tar')
            
            if self.args.thirdphase:
                ep_2_est = get_estimate(EP_2)
                torch.save(ep_2_est, self.args.path+'/ep_2.tar')
                torch.save(EP_2, self.args.path+'/EP_2.tar')
                
                # Plot comparison visualizations
                compare_estimate(bptt_est, ep_est, ep_2_est, self.args.path)
                plot_gdu(BPTT, EP, self.args.path, EP_2=EP_2, alg=self.args.alg, pdf=True)
                plot_gdu_instantaneous(BPTT, EP, self.args, EP_2=EP_2, path=self.args.path)
            else:
                plot_gdu(BPTT, EP, self.args.path, alg=self.args.alg, pdf=True)
                plot_gdu_instantaneous(BPTT, EP, self.args, path=self.args.path)
        
        print('GDU check complete')
    
    def run_evaluate(self):
        """
        Run evaluation on the model
        """
        # Get data loaders
        train_loader, test_loader = self._get_datasets()
        
        # Evaluate on training set
        print("\nEvaluating on training set:")
        training_correct, training_loss, _, _ = evaluate(
            self.model, 
            train_loader, 
            self.args.T1, 
            self.device, 
            plot=self.args.plot, 
            criterion=self.criterion, 
            noise_level=self.args.noise_level
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set:")
        test_correct, test_loss, _, _ = evaluate(
            self.model, 
            test_loader, 
            self.args.T1, 
            self.device, 
            plot=self.args.plot, 
            criterion=self.criterion, 
            noise_level=self.args.noise_level
        )
        
        # Calculate and print metrics
        training_acc = training_correct / len(train_loader.dataset)
        test_acc = test_correct / len(test_loader.dataset)
        

        # Print results
        print(f"Training accuracy: {training_acc}")
        print(f"Test accuracy: {test_acc}")
        print(f"Training loss: {training_loss}")
        print(f"Test loss: {test_loss}")



    ##### #####





















