import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import wandb

from datetime import datetime
import time
import math
from data_utils import *

from itertools import repeat
from torch.nn.parameter import Parameter
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting

from metric_utils import *
from model_utils import *



 

### HELPER FUNCTIONS ###

def copy(neurons):
    copy = []
    for n in neurons:
        copy.append(torch.empty_like(n).copy_(n.data).requires_grad_())
    return copy

### ###
         



def train_minibatch(model, optimizer, data, targets, args, criterion, epoch_idx=0, batch_idx=0):
    """
    Train a model on a single minibatch of data.
    
    Args:
        model: The neural network model
        optimizer: The optimizer for updating model parameters
        data: Input data batch
        targets: Target labels batch
        args: Arguments object containing training parameters
        criterion: Loss function to use
        epoch_idx: Current epoch index (for metrics tracking)
        batch_idx: Current batch index (for metrics tracking)
        
    Returns:
        dict: Dictionary containing training metrics and debug information
    """
    ### SETUP ###
    # Dictionary to store outputs and metrics
    result = {
        'correct': 0,
        'total': 0,
        'loss': 0.0,
        'network_metrics': None,
        'gradient_metrics': None,
        'convergence_metrics': {},
        'binarization_metrics': {},
        'neurons': None  # Store final neurons for potential plotting
    }
    
    beta_1, beta_2 = args.betas
    device = next(model.parameters()).device
    
    # Determine if we should track metrics for this batch
    should_get_metrics = (args.debug or args.wandb_mode != "disabled") and batch_idx == 0
    should_plot = args.plot and batch_idx == 0
    
    # Clear gradients
    optimizer.zero_grad()
    
    ### INITIALIZE NEURONS ###
    neurons = model.init_neurons(data.size(0), device) # Initialize the neurons to zero, data.size(0) is the batch size
    
    ### TRAINING ALGORITHM SELECTION ###
    if args.alg == 'EP':
        ### FREE PHASE ###
        # Run free phase with or without velocity tracking
        neurons, velocities_free = model(data, targets, neurons, args.T1, 
                                       beta=beta_1, criterion=criterion, 
                                       plot=should_plot, phase_type="Free", 
                                       return_velocities=should_get_metrics, 
                                       noise_level=args.noise_level)
        
        # Calculate metrics if tracking is enabled
        if should_get_metrics:
            result['convergence_metrics']['free'] = get_convergence_metrics(velocities_free, phase_type="free")
            result['binarization_metrics']['free'] = get_binarization_metrics(model, neurons, phase_type="free")
        
        neurons_1 = copy(neurons)
        
        ### Determine whether to quantise neurons ###
        if args.neuron_quantisation_bits > 0:
            neurons_1 = quantize_neural_states(neurons_1, model, args.neuron_quantisation_bits)
        
        ### NUDGED/POSITIVE PHASE ###
        if args.random_sign and (beta_1 == 0.0):
            rnd_sgn = 2 * np.random.randint(2) - 1
            beta_2 = rnd_sgn * beta_2
        
        # Determine if should reinitialize neurons
        if args.reinitialise_neurons:
            neurons = model.init_neurons(data.size(0), device)
        else:
            neurons = copy(neurons_1)
        
        # Run nudged phase with or without velocity tracking
        neurons, velocities_positive = model(data, targets, neurons, args.T2, 
                                          beta=beta_2, criterion=criterion, 
                                          plot=should_plot, phase_type="Positive", 
                                          return_velocities=should_get_metrics, 
                                          noise_level=args.noise_level)
        
        # Calculate metrics if tracking is enabled
        if should_get_metrics:
            result['convergence_metrics']['positive'] = get_convergence_metrics(velocities_positive, phase_type="positive")
            result['binarization_metrics']['positive'] = get_binarization_metrics(model, neurons, phase_type="positive")
        
        neurons_2 = copy(neurons)
        
        ### Determine whether to quantise neurons ###
        if args.neuron_quantisation_bits > 0:
            neurons_2 = quantize_neural_states(neurons_2, model, args.neuron_quantisation_bits)
        
        ### OPTIONAL THIRD PHASE = NUDGED/NEGATIVE PHASE ###
        if args.thirdphase:
            # Determine if should reinitialize neurons
            if args.reinitialise_neurons:
                neurons = model.init_neurons(data.size(0), device)
            else:
                # Come back to the first equilibrium
                neurons = copy(neurons_1)
            
            # Run negative phase with or without velocity tracking
            neurons, velocities_negative = model(data, targets, neurons, args.T2, 
                                               beta=-beta_2, criterion=criterion, 
                                               plot=should_plot, phase_type="Negative", 
                                               return_velocities=should_get_metrics, 
                                               noise_level=args.noise_level)
            
            # Calculate metrics if tracking is enabled
            if should_get_metrics:
                result['convergence_metrics']['negative'] = get_convergence_metrics(velocities_negative, phase_type="negative")
                result['binarization_metrics']['negative'] = get_binarization_metrics(model, neurons, phase_type="negative")
            
            neurons_3 = copy(neurons)
            
            ### Determine whether to quantise neurons ###
            if args.neuron_quantisation_bits > 0:
                neurons_3 = quantize_neural_states(neurons_3, model, args.neuron_quantisation_bits)
            
            ### GRADIENT COMPUTATION ###
            # Compute parameter gradients with three phases
            model.compute_syn_grads(data, targets, neurons_2, neurons_3, (beta_2, -beta_2), criterion)
        else:
            ### GRADIENT COMPUTATION ###
            # Compute parameter gradients with two phases
            model.compute_syn_grads(data, targets, neurons_1, neurons_2, (beta_1, beta_2), criterion)
    
    elif args.alg == 'BPTT':
        ### FIRST PASS WITHOUT TRACKING GRADIENTS ###
        neurons, _ = model(data, targets, neurons, args.T1 - args.T2, beta=0.0, 
                          criterion=criterion, noise_level=args.noise_level)
        
        # Detach data and neurons from the graph
        data = data.detach()
        data.requires_grad = True
        for k in range(len(neurons)):
            neurons[k] = neurons[k].detach()
            neurons[k].requires_grad = True
            
        ### SECOND PASS WITH GRADIENT TRACKING ###
        neurons, _ = model(data, targets, neurons, args.T2, beta=0.0, 
                         criterion=criterion, check_thm=True, 
                         noise_level=args.noise_level)
        
        ### LOSS CALCULATION AND BACKPROPAGATION ###
        # Calculate model output
        model_output = model.activation(neurons[-1]) if hasattr(model, 'activation') else neurons[-1]
        
        # Calculate loss
        if criterion.__class__.__name__.find('MSE') != -1:
            y_one_hot = F.one_hot(targets, num_classes=model.nc).float()
            # Transform one-hot encoding from [0,1] to [-1,1] to match output range if using activation
            y_transformed = y_one_hot * 2 - 1 if hasattr(model, 'activation') else y_one_hot
            loss = 0.5 * criterion(model_output.float(), y_transformed).sum(dim=1).mean().squeeze()
        else:
            if not getattr(model, 'softmax', False):
                loss = criterion(model_output.float(), targets).mean().squeeze()
            else:
                loss = criterion(model.synapses[-1](neurons[-1].view(data.size(0),-1)).float(), targets).mean().squeeze()
        
        # Setting gradients field to zero before backward
        model.zero_grad()
        
        # Backpropagation through time
        loss.backward()
    
    ### PARAMETER UPDATE ###
    # Update parameters
    optimizer.step()
    
    # Apply quantization after optimization step if using OIM_MLP with quantization
    if hasattr(model, 'quantisation_bits') and model.quantisation_bits > 0:
        model.quantize_parameters()
    
    # Store neurons for potential plotting
    result['neurons'] = neurons
    
    ### METRICS CALCULATION ###
    with torch.no_grad():
        # For EP, ensure we use the free phase neurons for metrics
        eval_neurons = neurons_1 if args.alg == 'EP' else neurons
        
        ### OUTPUT CALCULATION ###
        # Calculate model outputs for both prediction and loss calculation
        if not model.softmax:
            # Check if this is an OIM model (which works with phases) or a standard model (which works with activations)
            if isinstance(model, OIM_MLP):
                # For OIM models, convert phases to activations
                model_output = model.activation(eval_neurons[-1])
            else:
                # For standard models like P_MLP, neurons already contain activations
                model_output = eval_neurons[-1]
        else:
            # WATCH OUT: prediction is different when softmax == True
            # Note we have an extra linear layer after the last layer of neurons
            # But note that this is only defined for CNN models!
            # For models with softmax
            model_output = F.softmax(model.synapses[-1](eval_neurons[-1].view(data.size(0),-1)), dim=1)
            
        ### PREDICTION AND ACCURACY ###
        # Use the output for prediction
        pred = torch.argmax(model_output, dim=1).squeeze()
        result['correct'] = (targets == pred).sum().item()
        result['total'] = targets.size(0)
        
        ### LOSS CALCULATION ###
        if criterion.__class__.__name__.find('MSE')!=-1: # i.e. if the loss is MSE
            y_one_hot = F.one_hot(targets, num_classes=model.nc).float()
            
            # Transform one-hot encoding from [0,1] to [-1,1] to match cosine output range if using OIM_MLP
            y_transformed = y_one_hot * 2 - 1 if isinstance(model, OIM_MLP) else y_one_hot
            
            batch_loss = 0.5*criterion(model_output.float(), y_transformed).sum(dim=1).mean().squeeze()
        else:
            if not model.softmax: # e.g. for cross entropy loss
                batch_loss = criterion(model_output.float(), targets).mean().squeeze()
            else: # i.e. apply extra linear layer to the output layer of neurons (only for CNN models)
                batch_loss = criterion(model.synapses[-1](model_output.view(data.size(0),-1)).float(), targets).mean().squeeze()
        
        result['loss'] = batch_loss.item() * targets.size(0)  # Accumulate weighted by batch size
        
    
    ### ADDITIONAL METRICS ###
    # Collect additional metrics if needed
    if should_get_metrics:
        result['network_metrics'] = get_network_metrics(model)
        result['gradient_metrics'] = get_gradient_metrics(model)

    return result






def train(model, optimizer, train_loader, test_loader, args, device, criterion, checkpoint=None, scheduler=None, wandb_run=None, save_metrics_callback=None):
    """Train the model using Equilibrium Propagation.
    
    Args:
        model: The neural network model
        optimizer: The optimizer for updating model parameters
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        args: Arguments object containing training parameters
        device: Device to run the model on
        criterion: Loss function to use
        checkpoint: Optional checkpoint to resume training from
        scheduler: Learning rate scheduler (optional)
        wandb_run: Optional wandb run object for logging (if None, will use global wandb)
        save_metrics_callback: Optional callback function for saving metrics during training
                              Function signature: save_metrics_callback(metrics_dict, epoch)
    """
    
    mbs = train_loader.batch_size
    start = time.time()
    iter_per_epochs = math.ceil(len(train_loader.dataset)/mbs)
    
    if checkpoint is None:
        train_acc = [10.0]
        test_acc = [10.0]
        train_loss = []  
        test_loss = []   
        best = 0.0
        epoch_sofar = 0
    else:
        train_acc = checkpoint['train_acc']
        test_acc = checkpoint['test_acc']    
        train_loss = checkpoint['train_loss']
        test_loss = checkpoint['test_loss']
        best = checkpoint['best']
        epoch_sofar = checkpoint['epoch']
    
    for epoch in range(args.epochs):
        run_correct = 0
        run_total = 0
        run_loss = 0.0
        model.train()
        
        # Variables to store first batch metrics for later use
        first_batch_metrics = None
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            # Train the model on this batch
            result = train_minibatch(model, optimizer, data, targets, args, criterion, epoch_idx=epoch, batch_idx=batch_idx)
            
            # Save metrics from the first batch
            if batch_idx == 0 and (args.debug or args.wandb_mode != "disabled"):
                first_batch_metrics = {
                    'convergence_metrics': result['convergence_metrics'],
                    'binarization_metrics': result['binarization_metrics'],
                    'network_metrics': result['network_metrics'],
                    'gradient_metrics': result['gradient_metrics']
                }
            
            # Update metrics
            run_correct += result['correct']
            run_total += result['total']
            run_loss += result['loss']
            
            # Print progress
            if ((batch_idx%(iter_per_epochs//10)==0) or (batch_idx==iter_per_epochs-1)):
                train_acc_current = run_correct/run_total
                avg_loss = run_loss/run_total
                current_epoch = epoch_sofar+epoch+(batch_idx/iter_per_epochs)
                print('##### Epoch :', round(current_epoch, 2), ' #####',
                     '\tRun acc :', round(train_acc_current,3),'\t('+str(run_correct)+'/'+str(run_total)+')\t',
                     '\tRun loss:', round(avg_loss,3),
                     timeSince(start, ((batch_idx+1)+(epoch_sofar+epoch)*iter_per_epochs)/((epoch_sofar+args.epochs)*iter_per_epochs)))
                
                # Log current training metrics to wandb if enabled
                if args.save:
                    current_metrics = {
                        "current_train/accuracy": 100*train_acc_current,
                        "current_train/loss": avg_loss,
                        "epoch": current_epoch
                    }
                    
                    if args.wandb_mode != "disabled" and wandb_run is not None:
                        wandb_run.log(current_metrics)
                    # Only log to wandb if both enabled and wandb_run is available
                    if args.wandb_mode != "disabled" and wandb_run is not None:
                        wandb_run.log(current_metrics)
                        
                    # Call the save_metrics_callback if provided
                    if save_metrics_callback is not None:
                        save_metrics_callback(current_metrics, current_epoch)
                
                # Print metrics if debug is enabled
                if args.debug and batch_idx == 0:
                    print("\n### DEBUGGING INFORMATION AFTER EPOCH", epoch_sofar+epoch, "###")
                    
                    # Print metrics if available
                    if result['network_metrics']:
                        print_network_metrics(result['network_metrics'])
                    
                    if result['gradient_metrics']:
                        print_gradient_metrics(result['gradient_metrics'])
                    
                    # Print convergence and binarization metrics
                    for phase, metrics in result['convergence_metrics'].items():
                        print(f"\n{phase.capitalize()} phase convergence metrics:")
                        print_convergence_metrics(metrics)
                    
                    for phase, metrics in result['binarization_metrics'].items():
                        print(f"\n{phase.capitalize()} phase binarization metrics:")
                        print_binarization_metrics(metrics)
                    
                    print(f"\nTrain accuracy: {100*train_acc_current:.4f} ({run_correct}/{run_total})")
                    print(f"\nTrain loss: {avg_loss:.4f}")
                    print()
                    print("### ###")
                    print()
                
                # Check GDU theorem at the end of each epoch
                if (batch_idx==iter_per_epochs-1) and args.check_thm and args.alg!='BPTT':
                    # Use only 5 examples for GDU check to reduce computation time
                    BPTT, EP = check_gdu(model, data[0:5,:], targets[0:5], args, criterion, betas=args.betas)
                    
                    if args.thirdphase:
                        beta_1, beta_2 = args.betas
                        _, EP_2 = check_gdu(model, data[0:5,:], targets[0:5], args, criterion, betas=(beta_1, -beta_2))
                        
                        if args.plot and args.save:
                            plot_gdu(BPTT, EP, args.path, 
                                    EP_2=EP_2 if args.thirdphase else None, alg=args.alg)
                            plot_gdu_instantaneous(BPTT, EP, args, 
                                                 EP_2=EP_2 if args.thirdphase else None,
                                                 path=args.path)
                    
                    error_dict = RMSE(BPTT, EP)
                    
                    # Log RMSE and sign error results to wandb if enabled
                    if args.save:
                        wandb_metrics = {}
                        for key, metrics in error_dict.items():
                            # Format the parameter name for wandb logging
                            param_name = key.replace('.', '_')
                            wandb_metrics[f'gdu_check/{param_name}/rmse'] = metrics['rmse']
                            wandb_metrics[f'gdu_check/{param_name}/sign_error'] = metrics['sign_error']
                        

                        # Use the provided wandb_run if available
                        if args.wandb_mode != "disabled" and wandb_run is not None:
                            wandb_run.log(wandb_metrics)
                            
                        # Call the save_metrics_callback if provided
                        if save_metrics_callback is not None:
                            save_metrics_callback(wandb_metrics, epoch_sofar+epoch)
                
                # Plot neural activity if enabled
                if args.plot and args.save:
                    plot_neural_activity(result['neurons'], args.path)
        
        # Learning rate decay after each epoch
        if scheduler is not None:
            scheduler.step()
        
        # Evaluate on test set
        test_correct, test_loss_current, test_velocities, final_neurons = evaluate(model, test_loader, args.T1, device, 
                                               plot=args.plot, 
                                               return_velocities=(args.debug or args.wandb_mode != "disabled"),
                                               criterion=criterion, noise_level=args.noise_level)
        test_acc_current = test_correct/(len(test_loader.dataset))
        
        # Calculate metrics if tracking is enabled
        if args.debug or args.wandb_mode != "disabled":
            test_phase_convergence_metrics = get_convergence_metrics(test_velocities, phase_type="test")
            test_phase_binarization_metrics = get_binarization_metrics(model, final_neurons, phase_type="test")
            
            # Print test phase metrics if debug is enabled
            if args.debug:
                print("\n### TEST PHASE METRICS ###")
                print("Test phase convergence metrics:")
                print_convergence_metrics(test_phase_convergence_metrics)
                print("\nTest phase binarization metrics:")
                print_binarization_metrics(test_phase_binarization_metrics)
                print("### ###\n")
            
            # Log test phase metrics to wandb if enabled
            if args.save:
                # Log convergence metrics
                test_metrics = {}
                for key, value in test_phase_convergence_metrics.items():
                    test_metrics[f"convergence/test/{key}"] = value
                
                # Log binarization metrics
                for key, value in test_phase_binarization_metrics.items():
                    test_metrics[f"binarization/test/{key}"] = value
                
                # Use the provided wandb_run if available
                if args.wandb_mode != "disabled" and wandb_run is not None:
                    wandb_run.log(test_metrics)
                    
                # Call the save_metrics_callback if provided
                if save_metrics_callback is not None:
                    save_metrics_callback(test_metrics, epoch_sofar+epoch)
        
        # Calculate train metrics for this epoch
        train_acc_current = run_correct/run_total
        train_loss_current = run_loss/run_total
        
        ### METRICS CALCULATION AND LOGGING SECTION
        if (args.debug or args.wandb_mode != "disabled") and args.alg != "BPTT" and first_batch_metrics:
            # Extract metrics from the first batch result
            free_phase_convergence_metrics = first_batch_metrics['convergence_metrics'].get('free', {})
            positive_phase_convergence_metrics = first_batch_metrics['convergence_metrics'].get('positive', {})
            negative_phase_convergence_metrics = first_batch_metrics['convergence_metrics'].get('negative', {}) if args.thirdphase else {}
            
            free_phase_binarization_metrics = first_batch_metrics['binarization_metrics'].get('free', {})
            positive_phase_binarization_metrics = first_batch_metrics['binarization_metrics'].get('positive', {})
            negative_phase_binarization_metrics = first_batch_metrics['binarization_metrics'].get('negative', {}) if args.thirdphase else {}
            
            network_metrics = first_batch_metrics['network_metrics']
            gradient_metrics = first_batch_metrics['gradient_metrics']
            
            # Combine all metrics for logging/printing
            all_convergence_metrics = {}
            all_convergence_metrics.update(free_phase_convergence_metrics)
            all_convergence_metrics.update(positive_phase_convergence_metrics)
            if args.thirdphase:
                all_convergence_metrics.update(negative_phase_convergence_metrics)
            all_convergence_metrics.update(test_phase_convergence_metrics)
            
            all_binarization_metrics = {}
            all_binarization_metrics.update(free_phase_binarization_metrics)
            all_binarization_metrics.update(positive_phase_binarization_metrics)
            if args.thirdphase:
                all_binarization_metrics.update(negative_phase_binarization_metrics)
            all_binarization_metrics.update(test_phase_binarization_metrics)
            
            # Log combined metrics to wandb if enabled
            if args.save:
                wandb_combined_metrics = {}
                # Log all metrics to wandb
                for key, value in all_convergence_metrics.items():
                    wandb_combined_metrics[f"convergence/{key}"] = value
                
                for key, value in all_binarization_metrics.items():
                    wandb_combined_metrics[f"binarization/{key}"] = value
                
                if network_metrics:
                    for key, value in network_metrics.items():
                        wandb_combined_metrics[f"network/{key}"] = value
                
                if gradient_metrics:
                    for key, value in gradient_metrics.items():
                        wandb_combined_metrics[f"gradient/{key}"] = value
                
                # Use the provided wandb_run if available
                if args.wandb_mode != "disabled" and wandb_run is not None:
                    wandb_run.log(wandb_combined_metrics)
                    
                # Call the save_metrics_callback if provided
                if save_metrics_callback is not None:
                    save_metrics_callback(wandb_combined_metrics, epoch_sofar+epoch)
        
        # Log metrics to wandb if enabled
        if args.save:
            epoch_metrics = {
                "train/accuracy": 100*train_acc_current,
                "test/accuracy": 100*test_acc_current,
                "train/loss": train_loss_current,
                "test/loss": test_loss_current,
                "epoch": epoch_sofar+epoch+1
            }
            
            # Use the provided wandb_run if available
            if args.wandb_mode != "disabled" and wandb_run is not None:
                wandb_run.log(epoch_metrics)
                
            # Call the save_metrics_callback if provided
            if save_metrics_callback is not None:
                save_metrics_callback(epoch_metrics, epoch_sofar+epoch)
        
        # Save best model
        if args.save:
            test_acc.append(100*test_acc_current)
            train_acc.append(100*train_acc_current)
            train_loss.append(train_loss_current)
            test_loss.append(test_loss_current)
            if test_correct > best:
                best = test_correct
                save_dic = {'model_state_dict': model.state_dict(), 'opt': optimizer.state_dict(),
                           'train_acc': train_acc, 'test_acc': test_acc, 
                           'best': best, 'epoch': epoch_sofar+epoch+1,
                           'train_loss': train_loss, 'test_loss': test_loss}
                save_dic['scheduler'] = scheduler.state_dict() if scheduler is not None else None
                torch.save(save_dic,  args.path + '/checkpoint.tar')
                torch.save(model, args.path + '/model.pt')
            
            # Save final checkpoint after every epoch (for crash recovery)
            print(f"Saving final checkpoint after epoch {epoch_sofar+epoch+1} to {args.path}")
            final_save_dic = {'model_state_dict': model.state_dict(), 'opt': optimizer.state_dict(),
                            'train_acc': train_acc, 'test_acc': test_acc, 
                            'best': best, 'epoch': epoch_sofar+epoch+1,
                            'train_loss': train_loss, 'test_loss': test_loss}
            final_save_dic['scheduler'] = scheduler.state_dict() if scheduler is not None else None
            torch.save(final_save_dic, args.path + '/final_checkpoint.tar')
            torch.save(model, args.path + '/final_model.pt')
            print(f"Final checkpoint saved successfully")
            
            plot_acc(train_acc, test_acc, args.path)
            plot_loss(train_loss, test_loss, args.path)

    ### FINAL MODEL ALREADY SAVED AFTER EACH EPOCH
    # Final checkpoint is now saved after every epoch for crash recovery
    
    return train_acc, test_acc, train_loss, test_loss






















            
def evaluate(model, loader, T, device, plot=False, return_velocities=False, criterion=None, noise_level=0.0):
    """
    Evaluate the model on a dataloader with T steps for the dynamics
    
    Parameters:
    - model: The neural network model
    - loader: DataLoader for evaluation data
    - T: Number of time steps for the dynamics
    - device: Device to run the model on
    - plot: Whether to plot phase dynamics
    - return_velocities: Whether to return phase velocities
    - criterion: Loss function to use
    - noise_level: Level of noise to add during phase dynamics (only for OIM_MLP models)
    """
    model.eval()
    correct = 0
    total_loss = 0.0
    phase = 'Train' if loader.dataset.train else 'Test'
    
    # Store neurons for metrics calculation
    final_neurons = None
    
    for idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        neurons = model.init_neurons(x.size(0), device)
        
        # Run dynamics with or without velocity tracking
        # Only track velocities for first batch to reduce overhead
        should_get_metrics = return_velocities and idx == 0
        should_plot = plot and idx == 0

        neurons, velocities = model(x, y, neurons, T, beta=0.0, criterion=criterion, plot=plot and idx == 0, 
                                  phase_type="Evaluate", return_velocities=return_velocities, noise_level=noise_level)
        
        # Save neurons from the first batch for metrics calculation
        if idx == 0:
            final_neurons = neurons
        
        # Calculate model outputs for both prediction and loss calculation
        if not model.softmax:
            # Check if this is an OIM model (which works with phases) or a standard model (which works with activations)
            if isinstance(model, OIM_MLP):
                # For OIM models, convert phases to activations
                model_output = model.activation(neurons[-1])
            else:
                # For standard models like P_MLP, neurons already contain activations
                model_output = neurons[-1]
        else: # prediction is done as a readout of the penultimate layer (output is not part of the system)
            # For models with softmax (CNN models only)
            model_output = F.softmax(model.synapses[-1](neurons[-1].view(x.size(0),-1)), dim=1)
            
        # Use the output for prediction
        pred = torch.argmax(model_output, dim=1).squeeze()
        correct += (y == pred).sum().item()
        
        # Calculate loss if criterion is provided
        with torch.no_grad():
            if criterion.__class__.__name__.find('MSE')!=-1: # i.e. if the loss is MSE
                y_one_hot = F.one_hot(y, num_classes=model.nc).float() 

                # Transform one-hot encoding from [0,1] to [-1,1] to match cosine output range if using OIM_MLP
                y_transformed = y_one_hot * 2 - 1  if isinstance(model, OIM_MLP) else y_one_hot

                batch_loss = 0.5*criterion(model_output.float(), y_transformed).sum(dim=1).mean().squeeze()
            else:
                if not model.softmax: # e.g. for cross entropy loss
                    batch_loss = criterion(model_output.float(), y).mean().squeeze()
                else: # i.e. apply extra linear layer to the output layer of neurons (only for CNN models)
                    batch_loss = criterion(model.synapses[-1](model_output.view(x.size(0),-1)).float(), y).mean().squeeze()
            total_loss += batch_loss.item() * x.size(0)  # Accumulate weighted by batch size

    acc = correct/len(loader.dataset) 
    avg_loss = total_loss/len(loader.dataset) if criterion is not None else None
    
    print(phase+' accuracy :\t', acc)
    print(phase+' loss :\t', avg_loss)
    
    return correct, avg_loss, velocities, final_neurons



            










### GDU CHECK ###

## GDU CHECK HELPER FUNCTIONS ##
def grad_or_zero(x):
    if x.grad is None:
        return torch.zeros_like(x).to(x.device)
    else:
        return x.grad
    
def neurons_zero_grad(neurons):
    for idx in range(len(neurons)):
        if neurons[idx].grad is not None:
            neurons[idx].grad.zero_()

## ##

def check_gdu(model, x, y, args, criterion, betas=None):
    # This function returns EP gradients and BPTT gradients for one training iteration
    #  given some labelled data (x, y), time steps for both phases and the loss
    
    ### EXTRACT PARAMETERS FROM ARGS ###
    T1 = args.T1
    T2 = args.T2
    plot = args.plot
    ### ###


    ### INITIALISE DICTIONARIES ###
    # that will contain BPTT INSTANTANEOUS GRADIENTS and EP INSTANTANEOUS UPDATES
    BPTT, EP = {}, {}

    for name, p in model.named_parameters(): # i.e. parameter gradients/updates
        BPTT[name], EP[name] = [], []

    neurons = model.init_neurons(x.size(0), x.device)
    for idx in range(len(neurons)): # i.e. neuron updates
        BPTT['neurons_'+str(idx)], EP['neurons_'+str(idx)] = [], []
    
    ### ###




    
    ### COMPUTE BPTT INSTANTANEOUS GRADIENTS ###

    beta_1, beta_2 = betas # Note beta_2 will be -beta_2 in check_gdu function call for third phase


    # First phase up to T1-T2
    result = model(x, y, neurons, T1-T2, beta=beta_1, criterion=criterion, plot=True, phase_type="Free Phase to T1-T2", noise_level=args.noise_level)
    neurons = result[0] if isinstance(result, tuple) else result  # Extract just the neurons if result is a tuple (the forward method now returns (neurons, _))
    ref_neurons = copy(neurons)
    
    
    # Last steps of the first phase
    # Note we start with K=0 (so end up running up to T2, and end with K=T2 (so end up running up to T1))
    for K in range(T2+1):
        print(f"Free Phase, T1-T2 to T1-T2+{K}, K={K}")

        # Handle the case where forward returns a tuple
        result = model(x, y, neurons, K, beta=beta_1, criterion=criterion, plot=False, phase_type=f"Free Phase, T1-T2 to T1-T2+K, K={K}", noise_level=args.noise_level) 
        neurons = result[0] if isinstance(result, tuple) else result  # Extract just the neurons if result is a tuple (the forward method now returns (neurons, _))

        # detach data and neurons from the graph
        x = x.detach() # (x is the input data)
        x.requires_grad = True
        leaf_neurons = [] # i.e. neurons that will be used to compute the gradient of the loss with respect to
        for idx in range(len(neurons)):
            neurons[idx] = neurons[idx].detach()
            neurons[idx].requires_grad = True
            leaf_neurons.append(neurons[idx])

        # Run T2-K steps of the dynamics (i.e. up to T1)
        result = model(x, y, neurons, T2-K, beta=beta_1, criterion=criterion, check_thm=True, plot=False, phase_type=f"Free Phase, T1-T2+{K} to T1", noise_level=args.noise_level)
        neurons = result[0] if isinstance(result, tuple) else result  # Extract just the neurons if result is a tuple (the forward method now returns (neurons, _))
        

        # final loss
        model_output = model.activation(neurons[-1]) if isinstance(model, OIM_MLP) else neurons[-1] # Check if this is an OIM model which works with phases


        if criterion.__class__.__name__.find('MSE')!=-1:
            y_one_hot = F.one_hot(y, num_classes=model.nc).float() 

            # Transform one-hot encoding from [0,1] to [-1,1] to match cosine output range if using OIM_MLP
            y_transformed = y_one_hot * 2 - 1  if isinstance(model, OIM_MLP) else y_one_hot

            loss = 0.5*criterion(model_output.float(), y_transformed).sum(dim=1).mean().squeeze()
        else:
            if not model.softmax: # TODO will need to change these for OIM eventually
                loss = criterion(model_output.float(), y).mean().squeeze()
            else:
                loss = criterion(model.synapses[-1](neurons[-1].view(x.size(0),-1)).float(), y).mean().squeeze()


        # setting gradients field to zero before backward
        neurons_zero_grad(leaf_neurons)
        model.zero_grad()
        
        # Backpropagation through time
        # Note that since we detached the neurons from the computational graph at time T1+K, this loss is only including (T2-K) time from T1-T2+K to T1
        # Note loss.backward() for will assign the gradient to the parameters equal to the sum of the gradients across all of these T2-K time steps
        loss.backward() 


        # Collecting BPTT gradients : 
        # for parameters they are partial sums over T2-K time steps (hence we will do a subtraction later to get the instantaneous gradients) because we use same parameters for all time steps
        # for neurons they are direct gradients (i.e. we use different leaf neurons for each time step)
        # Note the loop we are in runs from K=0 to K=T2 (so backwards in time)

        # For parameters we want K instantaneous gradients from time step T1 (K=0) backwards, so we don't do this for last time step
        if K!=T2: 
            for name, p in model.named_parameters():
                update = torch.empty_like(p).copy_(grad_or_zero(p))
                BPTT[name].append( update.unsqueeze(0) )  # unsqueeze for time dimension
                neurons = copy(ref_neurons) # Resetting the neurons to T1-T2 step

         # For neurons we don't want neuron gradients at time step T1 (K=0) because the loss at this time step is only dependent on the neurons BEFORE this time step 
         # (we define s_t+1 = F(x,s_t, \theta_t+1 = \theta))
        if K!=0:
            for idx in range(len(leaf_neurons)):
                update = torch.empty_like(leaf_neurons[idx]).copy_(grad_or_zero(leaf_neurons[idx]))
                BPTT['neurons_'+str(idx)].append( update.mul(-x.size(0)).unsqueeze(0) )  # unsqueeze for time dimension
                # Note we multiply by -1 (so we get update not gradient, and x.size(0) so we get the step to reduce total loss not the mean loss - and so we are consistent with EP updates)

    # Differentiating partial sums to get elementary parameter gradients
    # Since loss.backward() gave the sum of the gradients across all of T2-K time steps,
    # e.g. the first element of BPTT[name] is the sum of the gradients across all of T2 time steps (where K=0) in the loop above
    # and the second element of BPTT[name] is the sum of the gradients across all of T2-1 time steps (where K=1) in the loop above
    # Therefore by subtracting BPTT[name][idx+1] from BPTT[name][idx] we get the instantaneous gradient at time step T1-T2+idx
    for name, p in model.named_parameters():
        for idx in range(len(BPTT[name]) - 1):
            BPTT[name][idx] = BPTT[name][idx] - BPTT[name][idx+1] 
        
    # Reverse the time
    # BECAUSE NOTE THAT THE THEOREM WANTS TO ANALYSE TIME STEPS K DEFINED AS T1-K
    # BUT ABOVE SUBTRACTION GIVES US T1-T2+K
    # SO REVERSING THE ORDER OF BPTT GRADIENTS GIVES US THE CORRECT TIME STEPS IN THE CORRECT ORDER FOR COMPARISON IN THE THEOREM
    for key in BPTT.keys():
        BPTT[key].reverse()
            

    ### ###





    ### COMPUTE EP INSTANTANEOUS UPDATES ###

    # Second phase done step by step
    for t in range(T2):
        print(f"Nudged Phase from T1+{t} to T1+{t+1}")

        neurons_pre = copy(neurons)                                          # neurons at time step t
        
        result = model(x, y, neurons, 1, beta=beta_2, criterion=criterion, plot=False, phase_type=f"Nudged Phase from T1+{t} to T1+{t+1}", noise_level=args.noise_level)  # neurons at time step t+1
        neurons = result[0] if isinstance(result, tuple) else result  # Extract just the neurons if result is a tuple (the forward method now returns (neurons, _))
     
        # Compute the instantaneous parameter update using check_thm=True which uses beta_2 for total energy with neurons_pre too (as both in nudged phase)
        # But still does division by |beta_1 - beta_2| in denominator of update
        model.compute_syn_grads(x, y, neurons_pre, neurons, betas, criterion, check_thm=True)  # compute the EP parameter update


        # Collect the EP updates forward in time
        for name, p in model.named_parameters():
            # Note by implementing energy rather than a primitive, we have dynamics with -epsilon* dF/ds, rather than just a dPhi/ds, therefore we have already essentially factored out epsilon from our equivalent Phi
            # Therefore we do not need to divide by epsilon here to make consistent with primitive dynamics case
            update = torch.empty_like(p).copy_(grad_or_zero(p))
            EP[name].append( update.unsqueeze(0) ) # unsqueeze for time dimension
            
        for idx in range(len(neurons)):
            # For neuron updates, we need to account for the beta difference in the denominator explicitly
            # Note whereas for primitive case we have s(t) - s(t-1) = dPhi/ds_t - dPhi/ds_t-1,
            # for our energy-based model DYNAMICS IMPLEMENTATIONwe have s(t) - s(t-1) = -epsilon*(dF/ds_t - dF/ds_t-1)
            # Therefore we have to divide by epsilon to cancel this factor.
            # Equivalently if we had used a 0.5|s|^2 term in a Phi function and multiplied everything else by epsilon we would also have to divide by epsilon here.
            update = (neurons[idx] - neurons_pre[idx])/(beta_2 - beta_1) 
            update = update/model.epsilon if hasattr(model, 'epsilon') else update 
            EP['neurons_'+str(idx)].append( update.unsqueeze(0) )  # unsqueeze for time dimension
        

    ### ###



    ### CONCATENATE GRADIENTS AND UPDATES ###
    for key in BPTT.keys(): # i.e. parameters and neurons
        BPTT[key] = torch.cat(BPTT[key], dim=0).detach()
        EP[key] = torch.cat(EP[key], dim=0).detach()
    
    
    return BPTT, EP
    
