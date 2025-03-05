import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn.functional as F

import os
from datetime import datetime
import time
import math
from data_utils import *

from itertools import repeat
from torch.nn.parameter import Parameter
import collections
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
         



def train(model, optimizer, train_loader, test_loader, args, device, criterion, checkpoint=None, scheduler=None):
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
    """
    
    mbs = train_loader.batch_size
    start = time.time()
    iter_per_epochs = math.ceil(len(train_loader.dataset)/mbs)
    beta_1, beta_2 = args.betas

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
        


        for idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)




            ### FREE PHASE
            neurons = model.init_neurons(x.size(0), device) # Initialize the neurons to zero, x.size(0) is the batch size
            
            if args.alg=='EP' or args.alg=='CEP':
                # First phase - determine if we should track velocities
                should_get_metrics = (args.debug or args.wandb_mode != "disabled") and idx == 0
                should_plot = args.plot and idx == 0
                
                # Run free phase with or without velocity tracking
                neurons, velocities_free = model(x, y, neurons, args.T1, beta=beta_1, criterion=criterion, 
                                               plot=should_plot, phase_type="Free", 
                                               return_velocities=should_get_metrics)
                
                # Calculate metrics if tracking is enabled
                if should_get_metrics:
                    # Calculate free phase metrics
                    free_phase_convergence_metrics = get_convergence_metrics(velocities_free, phase_type="free")
                    free_phase_binarization_metrics = get_binarization_metrics(model, neurons, phase_type="free")
                
                neurons_1 = copy(neurons)

            elif args.alg=='BPTT': # TODO understand this bit
                neurons = model(x, y, neurons, args.T1-args.T2, beta=0.0, criterion=criterion) # First run for T1-T2 time steps without tracking gradients
                # detach data and neurons from the graph
                x = x.detach()
                x.requires_grad = True
                for k in range(len(neurons)):
                    neurons[k] = neurons[k].detach()
                    neurons[k].requires_grad = True

                neurons = model(x, y, neurons, args.T2, beta=0.0, criterion=criterion, check_thm=True) # T2 time steps with gradient tracking



            # Predictions for running accuracy
            with torch.no_grad():
                # Calculate model outputs for both prediction and loss calculation
                if not model.softmax:
                    # Check if this is an OIM model (which works with phases) or a standard model (which works with activations)
                    if isinstance(model, OIM_MLP):
                        # For OIM models, convert phases to activations
                        model_output = model.activation(neurons[-1])
                    else:
                        # For standard models like P_MLP, neurons already contain activations
                        model_output = neurons[-1]
                else:
                    # WATCH OUT: prediction is different when softmax == True
                    # Note we have an extra linear layer after the last layer of neurons
                    # But note that this is only defined for CNN models!
                    # For models with softmax
                    model_output = F.softmax(model.synapses[-1](neurons[-1].view(x.size(0),-1)), dim=1)
                
                # Use the output for prediction
                pred = torch.argmax(model_output, dim=1).squeeze()
                run_correct += (y == pred).sum().item()
                run_total += x.size(0)
                
                # Calculate and accumulate the free phase loss using the same output
                if criterion.__class__.__name__.find('MSE')!=-1: # i.e. if the loss is MSE
                    batch_loss = 0.5*criterion(model_output.float(), F.one_hot(y, num_classes=model.nc).float()).sum(dim=1).mean().squeeze()
                else:
                    if not model.softmax: # e.g. for cross entropy loss
                        batch_loss = criterion(model_output.float(), y).mean().squeeze()
                    else: # i.e. apply extra linear layer to the output layer of neurons (only for CNN models)
                        batch_loss = criterion(model.synapses[-1](model_output.view(x.size(0),-1)).float(), y).mean().squeeze() # i.e. apply extra linear layer to the output layer of neurons
                run_loss += batch_loss.item() * x.size(0)  # Accumulate weighted by batch size (to undo division by batch size in the loss function)
                
                if ((idx%(iter_per_epochs//10)==0) or (idx==iter_per_epochs-1)) and args.save:
                    plot_neural_activity(neurons, args.path)
            







            ### NUDGE PHASE(S)

            if args.alg=='EP':

                ## SECOND PHASE
                if args.random_sign and (beta_1==0.0):
                    rnd_sgn = 2*np.random.randint(2) - 1
                    betas = beta_1, rnd_sgn*beta_2
                    beta_1, beta_2 = betas
                

                # Determine if should reinitialize neurons or not
                # (Currently neurons are from end of first phase)
                if args.reinitialise_neurons:
                    neurons = model.init_neurons(x.size(0), device)


                # Determine if we should track velocities for nudged phase
                should_get_metrics = (args.debug or args.wandb_mode != "disabled") and idx == 0
                should_plot = args.plot and idx == 0
                
                # Run nudged phase with or without velocity tracking
                neurons, velocities_positive = model(x, y, neurons, args.T2, beta=beta_2, criterion=criterion, 
                                                 plot=should_plot, phase_type="Positive", 
                                                 return_velocities=should_get_metrics)
                
                # Calculate metrics if tracking is enabled
                if should_get_metrics:
                    # Calculate nudged phase metrics
                    positive_phase_convergence_metrics = get_convergence_metrics(velocities_positive, phase_type="positive")
                    positive_phase_binarization_metrics = get_binarization_metrics(model, neurons, phase_type="positive")
                
                neurons_2 = copy(neurons)




                ## THIRD PHASE 
                # (if we approximate f' as f'(x) = (f(x+h) - f(x-h))/2h)
                if args.thirdphase:

                    # Determine if should reinitialize neurons or not
                    if args.reinitialise_neurons:
                        neurons = model.init_neurons(x.size(0), device)
                    else:
                        #come back to the first equilibrium
                        neurons = copy(neurons_1)
                    
                    # Determine if we should track velocities for negative phase
                    should_get_metrics = (args.debug or args.wandb_mode != "disabled") and idx == 0
                    should_plot = args.plot and idx==0
                    
                    # Run negative phase with or without velocity tracking
                    neurons, velocities_negative = model(x, y, neurons, args.T2, beta=-beta_2, criterion=criterion, 
                                                       plot=should_plot, phase_type="Negative", 
                                                       return_velocities=should_get_metrics)
                    
                    # Calculate metrics if tracking is enabled
                    if should_get_metrics:
                        # Calculate negative phase metrics
                        negative_phase_convergence_metrics = get_convergence_metrics(velocities_negative, phase_type="negative")
                        negative_phase_binarization_metrics = get_binarization_metrics(model, neurons, phase_type="negative")
                    
                    neurons_3 = copy(neurons)
                    
                    model.compute_syn_grads(x, y, neurons_2, neurons_3, (beta_2, -beta_2), criterion) # Note use of neurons_2 and neurons_3 for the update

                else:
                    model.compute_syn_grads(x, y, neurons_1, neurons_2, betas, criterion)



                # DEBUGGING GRADIENT SIZES
                if (args.debug or args.wandb_mode != "disabled") and idx == 0:
                    network_metrics = get_network_metrics(model)
                    gradient_metrics = get_gradient_metrics(model)

                    if args.debug: 
                        print()
                        print("### DEBUGGING INFORMATION AT FIRST BATCH OF EPOCH", epoch_sofar+epoch, "###")
                        print_network_metrics(network_metrics)
                        print_gradient_metrics(gradient_metrics)
                        print()
                        print("### ###")
                        print()

                    

                # UPDATE WEIGHTS AND BIASES
                optimizer.step() # Update the weights and biases of the model

            # elif alg=='CEP':
            #     if random_sign and (beta_1==0.0):
            #         rnd_sgn = 2*np.random.randint(2) - 1
            #         betas = beta_1, rnd_sgn*beta_2
            #         beta_1, beta_2 = betas

            #     # second phase
            #     if cep_debug:
            #         prev_p = {}
            #         for (n, p) in model.named_parameters():
            #             prev_p[n] = p.clone().detach()
            #         for i in range(len(model.synapses)):
            #             prev_p['lrs'+str(i)] = optimizer.param_groups[i]['lr']
            #             prev_p['wds'+str(i)] = optimizer.param_groups[i]['weight_decay']
            #             optimizer.param_groups[i]['lr'] *= 6e-5
            #             #optimizer.param_groups[i]['weight_decay'] = 0.0
                                        
            #     for k in range(T2):
            #         neurons = model(x, y, neurons, 1, beta = beta_2, criterion=criterion)   # one step
            #         neurons_2  = copy(neurons)
            #         model.compute_syn_grads(x, y, neurons_1, neurons_2, betas, criterion)   # compute cep update between 2 consecutive steps 
            #         for (n, p) in model.named_parameters():
            #             p.grad.data.div_( (1 - optimizer.param_groups[int(n[9])]['lr']*optimizer.param_groups[int(n[9])]['weight_decay'])**(T2-1-k)  ) 
            #         optimizer.step()                                                        # update weights 
            #         neurons_1 = copy(neurons)  
               
            #     if cep_debug:
            #         debug(model, prev_p, optimizer)
 
            #     if thirdphase:    
            #         neurons = model(x, y, neurons, T2, beta = 0.0, criterion=criterion)     # come back to s*
            #         neurons_2 = copy(neurons)
            #         for k in range(T2):
            #             neurons = model(x, y, neurons, 1, beta = -beta_2, criterion=criterion)
            #             neurons_3 = copy(neurons)
            #             model.compute_syn_grads(x, y, neurons_2, neurons_3, (beta_2, -beta_2), criterion)
            #             optimizer.step()
            #             neurons_2 = copy(neurons)

            elif args.alg=='BPTT': # Simply calculate loss and do backprop
         
                # final loss
                if criterion.__class__.__name__.find('MSE')!=-1: # i.e. if the loss is MSE
                    loss = 0.5*criterion(neurons[-1].float(), F.one_hot(y, num_classes=model.nc).float()).sum(dim=1).mean().squeeze()
                else:
                    if not model.softmax: # e.g. for cross entropy loss
                        loss = criterion(neurons[-1].float(), y).mean().squeeze()
                    else:
                        loss = criterion(model.synapses[-1](neurons[-1].view(x.size(0),-1)).float(), y).mean().squeeze() # i.e. apply extra linear layer to the output layer of neurons
                # setting gradients field to zero before backward
                model.zero_grad()

                # Backpropagation through time
                loss.backward()
                optimizer.step() # Update the weights and biases of the model



            
            ## LOGGING
            if ((idx%(iter_per_epochs//10)==0) or (idx==iter_per_epochs-1)):
                train_acc_current = run_correct/run_total
                avg_loss = run_loss/run_total
                print('##### Epoch :', round(epoch_sofar+epoch+(idx/iter_per_epochs), 2), ' #####',
                      '\tRun train acc :', round(train_acc_current,3),'\t('+str(run_correct)+'/'+str(run_total)+')\t',
                      '\tRun train loss:', round(avg_loss,3),
                      timeSince(start, ((idx+1)+epoch*iter_per_epochs)/(args.epochs*iter_per_epochs)))

                # TODO understand this bit
                if args.check_thm and args.alg!='BPTT':
                    BPTT, EP = check_gdu(model, x[0:5,:], y[0:5], args, criterion, betas=betas, plot=args.plot)
                    RMSE(BPTT, EP)
    

        ### LEARNING RATE DECAY (after each epoch)
        if scheduler is not None: # learning rate decay step
            if epoch+epoch_sofar < scheduler.T_max:
                scheduler.step()






        ### TESTING PHASE
        # Determine if we should track velocities for testing
        # For testing, we track velocities for the entire test set, not just the first batch
        should_get_metrics = (args.debug or args.wandb_mode != "disabled")
        should_plot = args.plot
        
        test_correct, test_loss_current, test_velocities = evaluate(model, test_loader, args.T1, device, 
                                                plot=should_plot, 
                                                return_velocities=should_get_metrics,
                                                criterion=criterion)
        test_acc_current = test_correct/(len(test_loader.dataset))
        
        # Calculate metrics if tracking is enabled
        if should_get_metrics:
            test_phase_convergence_metrics = get_convergence_metrics(test_velocities, phase_type="test")
            test_phase_binarization_metrics = get_binarization_metrics(model, neurons, phase_type="test")






        ### METRICS CALCULATION AND LOGGING SECTION
        if args.debug or args.wandb_mode != "disabled":
                        
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
            
            # Log metrics to wandb if enabled
            if args.wandb_mode != "disabled":
                log_metrics_to_wandb(all_convergence_metrics, network_metrics, all_binarization_metrics, gradient_metrics)
                
                # Extra wandb logging
                wandb.log({"train/accuracy": 100*train_acc_current,
                          "test/accuracy": 100*test_acc_current,
                          "train/loss": avg_loss,
                          "test/loss": test_loss_current,
                          "epoch": epoch_sofar+epoch+1})
            
            # Print metrics if debug is enabled
            if args.debug:
                print("\n### DEBUGGING INFORMATION AFTER EPOCH", epoch_sofar+epoch, "###")
                
                # Print metrics
                print_network_metrics(network_metrics)
                print_binarization_metrics(all_binarization_metrics)
                print_convergence_metrics(all_convergence_metrics)
                
                # Print train and test accuracy 
                print(f"\nTrain accuracy: {100*train_acc_current:.4f} ({run_correct}/{run_total})")
                print(f"\nTest accuracy: {100*test_acc_current:.4f} ({test_correct}/{len(test_loader.dataset)})")
                print(f"\nTrain loss: {avg_loss:.4f}")
                print(f"\nTest loss: {test_loss_current:.4f}")

                print()
                print("### ###")
                print()

        ### SAVING BEST MODEL
        if args.save:
            test_acc.append(100*test_acc_current)
            train_acc.append(100*train_acc_current)
            train_loss.append(avg_loss)  # Add current train loss to the list
            test_loss.append(test_loss_current)  # Add current test loss to the list
            if test_correct > best:
                best = test_correct
                save_dic = {'model_state_dict': model.state_dict(), 'opt': optimizer.state_dict(),
                            'train_acc': train_acc, 'test_acc': test_acc, 
                            'best': best, 'epoch': epoch_sofar+epoch+1,
                            'train_loss': train_loss, 'test_loss': test_loss}
                save_dic['scheduler'] = scheduler.state_dict() if scheduler is not None else None
                torch.save(save_dic,  args.path + '/checkpoint.tar')
                torch.save(model, args.path + '/model.pt')
            plot_acc(train_acc, test_acc, args.path)
            plot_loss(train_loss, test_loss, args.path)        
    

    ### SAVING FINAL MODEL
    if args.save:
        save_dic = {'model_state_dict': model.state_dict(), 'opt': optimizer.state_dict(),
                    'train_acc': train_acc, 'test_acc': test_acc, 
                    'best': best, 'epoch': args.epochs,
                    'train_loss': train_loss, 'test_loss': test_loss}
        save_dic['scheduler'] = scheduler.state_dict() if scheduler is not None else None
        torch.save(save_dic,  args.path + '/final_checkpoint.tar')
        torch.save(model, args.path + '/final_model.pt')
 






















            
def evaluate(model, loader, T, device, plot=False, return_velocities=False, criterion=None):
    # Evaluate the model on a dataloader with T steps for the dynamics
    model.eval()
    correct = 0
    total_loss = 0.0
    phase = 'Train' if loader.dataset.train else 'Test'
    
    for idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        neurons = model.init_neurons(x.size(0), device)
        
        # Run dynamics with or without velocity tracking
        # Only track velocities for first batch to reduce overhead
        should_get_metrics = return_velocities and idx == 0
        should_plot = plot and idx == 0

        neurons, velocities = model(x, y, neurons, T, plot=should_plot, 
                                       phase_type=f"Evaluate", return_velocities=should_get_metrics)
        
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
                batch_loss = 0.5*criterion(model_output.float(), F.one_hot(y, num_classes=model.nc).float()).sum(dim=1).mean().squeeze()
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
    
    return correct, avg_loss, velocities



            










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

def check_gdu(model, x, y, args, criterion, betas=None, plot=None):
    # This function returns EP gradients and BPTT gradients for one training iteration
    #  given some labelled data (x, y), time steps for both phases and the loss
    
    # Extract parameters from args
    T1 = args.T1
    T2 = args.T2
    
    # Use provided betas if specified, otherwise use the ones from args
    if betas is None:
        betas = args.betas[0], args.betas[1]
    
    alg = args.alg
    
    # Use args.plot if plot is not explicitly provided
    if plot is None:
        plot = args.plot
    
    # Initialize dictionaries that will contain BPTT gradients and EP updates
    BPTT, EP = {}, {}
    # if alg=='CEP':
    #     prev_p = {}

    # Initialize the dictionaries that will contain BPTT gradients and EP updates for each parameter
    for name, p in model.named_parameters():
        BPTT[name], EP[name] = [], []
        # if alg=='CEP':
        #     prev_p[name] = p

    # Initialize the dictionaries that will contain BPTT gradients and EP updates for each layer of neurons
    neurons = model.init_neurons(x.size(0), x.device)
    for idx in range(len(neurons)): # i.e. for each layer of neurons
        BPTT['neurons_'+str(idx)], EP['neurons_'+str(idx)] = [], []
    
    # We first compute BPTT gradients
    # First phase up to T1-T2
    beta_1, beta_2 = betas
    neurons = model(x, y, neurons, T1-T2, beta=beta_1, criterion=criterion, plot=plot, phase_type="Free")
    
    # Create a deep copy of neurons that may contain nested lists
    ref_neurons = copy(neurons)
    
    
    # Last steps of the first phase
    for K in range(T2+1):

        neurons = model(x, y, neurons, K, beta=beta_1, criterion=criterion, plot=False) # Running K time step 

        # detach data and neurons from the graph
        x = x.detach()
        x.requires_grad = True
        leaf_neurons = []
        for idx in range(len(neurons)):
            neurons[idx] = neurons[idx].detach()
            neurons[idx].requires_grad = True
            leaf_neurons.append(neurons[idx])

        neurons = model(x, y, neurons, T2-K, beta=beta_1, criterion=criterion, check_thm=True, plot=False) # T2-K time step
        
        # final loss
        if criterion.__class__.__name__.find('MSE')!=-1:
            loss = (1/(2.0*x.size(0)))*criterion(neurons[-1].float(), F.one_hot(y, num_classes=model.nc).float()).sum(dim=1).squeeze()
        else:
            if not model.softmax:
                loss = (1/(x.size(0)))*criterion(neurons[-1].float(), y).squeeze()
            else:
                loss = (1/(x.size(0)))*criterion(model.synapses[-1](neurons[-1].view(x.size(0),-1)).float(), y).squeeze()

        # setting gradients field to zero before backward
        neurons_zero_grad(leaf_neurons)
        model.zero_grad()

        # Backpropagation through time
        loss.backward(torch.tensor([1 for i in range(x.size(0))], dtype=torch.float, device=x.device, requires_grad=True))

        # Collecting BPTT gradients : for parameters they are partial sums over T2-K time steps
        if K!=T2:
            for name, p in model.named_parameters():
                update = torch.empty_like(p).copy_(grad_or_zero(p))
                BPTT[name].append( update.unsqueeze(0) )  # unsqueeze for time dimension
                neurons = copy(ref_neurons) # Resetting the neurons to T1-T2 step
        if K!=0:
            for idx in range(len(leaf_neurons)):
                update = torch.empty_like(leaf_neurons[idx]).copy_(grad_or_zero(leaf_neurons[idx]))
                BPTT['neurons_'+str(idx)].append( update.mul(-x.size(0)).unsqueeze(0) )  # unsqueeze for time dimension
                                
    # Differentiating partial sums to get elementary parameter gradients
    for name, p in model.named_parameters():
        for idx in range(len(BPTT[name]) - 1):
            BPTT[name][idx] = BPTT[name][idx] - BPTT[name][idx+1]
        
    # Reverse the time
    for key in BPTT.keys():
        BPTT[key].reverse()




            
    # Now we compute EP gradients forward in time
    # Second phase done step by step
    for t in range(T2):
        neurons_pre = copy(neurons)                                          # neurons at time step t
        neurons = model(x, y, neurons, 1, beta=beta_2, criterion=criterion, plot=plot and t==0, phase_type="Positive")  # neurons at time step t+1, only plot first step
        
        model.compute_syn_grads(x, y, neurons_pre, neurons, betas, criterion, check_thm=True)  # compute the EP parameter update

        # if alg=='CEP':
        #     for p in model.parameters():
        #         p.data.add_(-1e-5 * p.grad.data)
        
        # Collect the EP updates forward in time
        for n, p in model.named_parameters():
            update = torch.empty_like(p).copy_(grad_or_zero(p))
            EP[n].append( update.unsqueeze(0) )                    # unsqueeze for time dimension
        for idx in range(len(neurons)):
            update = (neurons[idx] - neurons_pre[idx])/(beta_2 - beta_1)
            EP['neurons_'+str(idx)].append( update.unsqueeze(0) )  # unsqueeze for time dimension
        
    # Concatenating with respect to time dimension
    for key in BPTT.keys():
        BPTT[key] = torch.cat(BPTT[key], dim=0).detach()
        EP[key] = torch.cat(EP[key], dim=0).detach()
    
    # if alg=='CEP':
    #     for name, p in model.named_parameters():
    #         p.data.copy_(prev_p[name])    

    return BPTT, EP
    

# TODO read this later
def RMSE(BPTT, EP):
    # print the root mean square error, and sign error between EP and BPTT gradients
    print('\nGDU check :')
    for key in BPTT.keys():
        K = BPTT[key].size(0)
        f_g = (EP[key] - BPTT[key]).pow(2).sum(dim=0).div(K).pow(0.5)
        f =  EP[key].pow(2).sum(dim=0).div(K).pow(0.5)
        g = BPTT[key].pow(2).sum(dim=0).div(K).pow(0.5)
        comp = f_g/(1e-10+torch.max(f,g))
        sign = torch.where(EP[key]*BPTT[key] < 0, torch.ones_like(EP[key]), torch.zeros_like(EP[key]))
        print(key.replace('.','_'), '\t RMSE =', round(comp.mean().item(), 4), '\t SIGN err =', round(sign.mean().item(), 4))
    print('\n')


# TODO read this later
def debug(model, prev_p, optimizer):
    optimizer.zero_grad()
    for (n, p) in model.named_parameters():
        idx = int(n[9]) 
        p.grad.data.copy_((prev_p[n] - p.data)/(optimizer.param_groups[idx]['lr']))
        p.data.copy_(prev_p[n])
    for i in range(len(model.synapses)):
        optimizer.param_groups[i]['lr'] = prev_p['lrs'+str(i)]
        #optimizer.param_groups[i]['weight_decay'] = prev_p['wds'+str(i)]
    optimizer.step()