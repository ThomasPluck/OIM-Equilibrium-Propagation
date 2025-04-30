import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')

import os
from datetime import datetime
import time
import math








### GENERAL DATA UTILS ###


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return 'elapsed time : %s \t (will finish in %s)' % (asMinutes(s), asMinutes(rs))




### HYPERPARAMETERS FILE ###

def createHyperparametersFile(path, args, model, command_line):
    """
    Create a hyperparameters file with all command line arguments
    
    Parameters:
    - path: Path to save the hyperparameters file
    - args: Command line arguments
    - model: Model to print
    - command_line: Original command line used to run the script
    """
    with open(path + r"/hyperparameters.txt", "w+") as hyperparameters:
        # Write command line
        hyperparameters.write(f"{command_line}\n\n")
        
        # Write all arguments dynamically
        hyperparameters.write("=== HYPERPARAMETERS ===\n")
        for arg_name, arg_value in sorted(vars(args).items()):
            hyperparameters.write(f"- {arg_name}: {arg_value}\n")
        
        # Write model architecture
        hyperparameters.write("\n=== MODEL ===\n")
        if hasattr(model, 'pools'):
            hyperparameters.write(f"Poolings: {model.pools}\n\n")
        hyperparameters.write(f"{model}\n")






### DATASET GENERATION UTILS ###

class PositiveNegativeRangeNormalize:
    """Transform that maps values from [0,1] to [-1,1] range"""
    def __call__(self, x):
        return 2.0 * x - 1.0

    def __repr__(self):
        return self.__class__.__name__ + '()'

def generate_mnist(args):
    '''
    Generate mnist dataloaders
    If input_positive_negative_remapping is True, remaps pixel values from [0,1] to [-1,1]
    '''

    # Use custom training and test data size
    N_data_train = args.N_data_train
    N_data_test = args.N_data_test
    N_class = 10
    
    if args.input_positive_negative_mapping:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,)), # Since ToTensor already maps [0,255] to [0,1] this line should be redundant
            PositiveNegativeRangeNormalize()
        ])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,)) # Since ToTensor already maps [0,255] to [0,1] this line should be redundant
        ])

    # TODO maybe add data augmentation here like in Laydevant if needed


    # Training data
    mnist_dset_train = torchvision.datasets.MNIST('./mnist_pytorch', train=True, transform=transform, target_transform=None, download=True)
    
    # If requesting exact full dataset size, skip the subsampling
    use_full_train = (N_data_train == 60000)
    use_full_test = (N_data_test == 10000)
    
    if not use_full_train:
        # Reduce training dataset size to N_data_train points, but keep the same number of data points per class
        indices = []
        comp = torch.zeros(N_class)
        for idx, target in enumerate(mnist_dset_train.targets):
            if comp[target] < N_data_train / N_class:
                indices.append(idx)
                comp[target] += 1
            if len(indices) == N_data_train:
                break

        mnist_dset_train.data = mnist_dset_train.data[indices]
        mnist_dset_train.targets = mnist_dset_train.targets[indices]


    # Testing data
    mnist_dset_test = torchvision.datasets.MNIST('./mnist_pytorch', train=False, transform=transform, target_transform=None, download=True)
    
    if not use_full_test:
        # Reduce test dataset size, but keep the same number of data points per class
        indices = []
        comp = torch.zeros(N_class)
        for idx, target in enumerate(mnist_dset_test.targets):
            if comp[target] < N_data_test / N_class:
                indices.append(idx)
                comp[target] += 1
            if len(indices) == N_data_test:
                break

        mnist_dset_test.data = mnist_dset_test.data[indices]
        mnist_dset_test.targets = mnist_dset_test.targets[indices]


    # Create the data loaders
    # Use multiple workers for asynchronous data loading
    print(f"Using num_workers={args.num_workers} for DataLoaders.")
    train_loader = torch.utils.data.DataLoader(mnist_dset_train, batch_size=args.mbs, shuffle=True, num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(mnist_dset_test, batch_size=200, shuffle=False, num_workers=args.num_workers, persistent_workers=True, pin_memory=True)


    if args.debug:
        # Verify data ranges
        for batch, _ in train_loader:
            print(f"Data range verification:")
            print(f"Min pixel value: {batch.min():.4f}")
            print(f"Max pixel value: {batch.max():.4f}")

            break

        # Verify data size
        print(f"Train data size: {len(mnist_dset_train)}")
        print(f"Test data size: {len(mnist_dset_test)}")

    return train_loader, test_loader








### PLOT FUNCTIONS ###


def plot_neural_activity(neurons, path):   
    N = len(neurons)
    fig = plt.figure(figsize=(3*N,6))
    for idx in range(N):
        fig.add_subplot(2, N//2+1, idx+1)
        nrn = neurons[idx].cpu().detach().numpy().flatten()
        plt.hist(nrn, 50)
        plt.title('neurons of layer '+str(idx+1))
    fig.savefig(path + '/neural_activity.png')
    plt.close()


def plot_acc(train_acc, test_acc, path):
    fig = plt.figure(figsize=(16,9))
    x_axis = [i for i in range(len(train_acc))]
    plt.plot(x_axis, train_acc, label='train')
    plt.plot(x_axis, test_acc, label='test')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    fig.savefig(path + '/train-test_acc.png')
    plt.close()


def plot_loss(train_loss, test_loss, path):
    fig = plt.figure(figsize=(16,9))
    x_axis = [i for i in range(len(train_loss))]
    plt.plot(x_axis, train_loss, label='train')
    plt.plot(x_axis, test_loss, label='test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    fig.savefig(path + '/train-test_loss.png')
    plt.close()










### GDU CHECK UTILS ###
def get_estimate(dic): # Get estimate does INTEGRATION of (instantaneous) WEIGHT GRADIENTS (i.e. not s gradients) get actual weight update up to each time step of nudged phase for EP / backwards in BPTT 
    estimates = {}
    for key in dic.keys():
        if key.find('weight')!=-1: # i.e. if it's a weight (not a bias)
            estimate = integrate(dic[key])
            estimates[key] = estimate[-1,:]
    return estimates



def integrate(x): # Integrates (sums) the weight gradients up to each time step to get the actual weight update at end of nudged phase for EP / end of all BPTT backward steps
    y = torch.empty_like(x)
    with torch.no_grad():
        for j in reversed(range(x.shape[0])): # i.e. for final time steps T2 down to t=0
            integ=0.0
            for i in range(j): # i.e. for all previous time steps up to final time step j
                integ += x[i]
            y[j] = integ
    return y
    

def compare_estimate(bptt, ep_1, ep_2, path):
    heights=[]
    abscisse=[]
    plt.figure(figsize=(16,9))
    for key in bptt.keys(): # i.e. for each parameter
        
        ep_3 = (ep_1[key]+ep_2[key])/2 # i.e. EP symmetric
        
        ep1_bptt = (ep_1[key] - bptt[key]).abs()
        ep2_bptt = (ep_2[key] - bptt[key]).abs()
        ep3_bptt = (ep_3 - bptt[key]).abs()

        # Calculate relative improvement of symmetric EP
        # Formula: (2 * symmetric_EP_distance) / (positive_EP_distance + negative_EP_distance)
        # If denominator is 0, use 1 instead to avoid division by zero
        comp = torch.where( (ep1_bptt + ep2_bptt)==0, torch.ones_like(ep1_bptt), (2*ep3_bptt)/(ep1_bptt + ep2_bptt) )
        comp = comp.mean().item()

        # Only consider weight parameters (i.e. not biases)
        if key.find('weight')!=-1:
            heights.append(comp)  # Store the relative distance
            abscisse.append(int(key[9])+1) # Extract layer number from parameter name # key[9] gets the layer number from format "weight_X_Y"

    plt.bar(abscisse, heights)
    plt.ylim((0.,1.))
    plt.title('Euclidian distance between EP symmetric and BPTT, divided by mean distance between EP one-sided and BPTT\n 1.0 means EP symmetric is as close to BPTT as EP one-sided, 0.5 means EP symmetric twice closer to BPTT than EP one-sided')
    plt.ylabel('Relative distance to BPTT')
    plt.xlabel('Layer index')
    plt.savefig(path+'/bars.png', dpi=300)
    plt.close()


def plot_gdu(BPTT, EP, path, EP_2=None, alg='EP', pdf=False):
    # Get matplotlib's default color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    # Check if Times New Roman is available
    import matplotlib.font_manager as fm
    font_names = [f.name for f in fm.fontManager.ttflist]
    times_available = any('Times' in font_name for font_name in font_names)

    if not times_available:
        print("Warning: Times New Roman font not found. Using default serif font.")
        # Try to use a reasonable fallback that maintains IEEE style
        plt.rcParams['font.serif'] = ['DejaVu Serif', 'Bitstream Vera Serif', 'Computer Modern Roman', 
                                     'New Century Schoolbook', 'Century Schoolbook L', 'Utopia', 
                                     'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Times New Roman', 
                                     'Times', 'Palatino', 'Charter', 'serif']
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 8
    else:
        # Set font to Times New Roman size 8
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']
        plt.rcParams['font.size'] = 8
    
    # Set additional font parameters
    plt.rcParams['mathtext.fontset'] = 'dejavuserif' if not times_available else 'stix'
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    for key in EP.keys(): # i.e. for each parameter
        fig = plt.figure(figsize=(16,9))

        # Get 3 random samples from each parameter type 
        for idx in range(3):

            # For 3D tensors (i.e. weights)
            if len(EP[key].size())==3:
                # Get ep and bptt instantaneous gradient estimates at for random weight in this 3D tensor
                i, j = np.random.randint(EP[key].size(1)), np.random.randint(EP[key].size(2))
                ep = EP[key][:,i,j].cpu().detach()
                if EP_2 is not None:
                    ep_2 = EP_2[key][:,i,j].cpu().detach()
                bptt = BPTT[key][:,i,j].cpu().detach()

            # For 2D tensors (i.e. biases)
            elif len(EP[key].size())==2:
                # Get ep and bptt instantaneous gradient estimates at for random bias in this 2D tensor
                i = np.random.randint(EP[key].size(1))
                ep = EP[key][:,i].cpu().detach()
                if EP_2 is not None:
                    ep_2 = EP_2[key][:,i].cpu().detach()
                bptt = BPTT[key][:,i].cpu().detach()

            # For 5D tensors (i.e. weights of convolutional kernels)
            elif len(EP[key].size())==5:
                # Get ep and bptt instantaneous gradient estimates at for random convolutional kernel in this 5D tensor
                i, j = np.random.randint(EP[key].size(1)), np.random.randint(EP[key].size(2))
                k, l = np.random.randint(EP[key].size(3)), np.random.randint(EP[key].size(4))
                ep = EP[key][:,i,j,k,l].cpu().detach()
                if EP_2 is not None:
                    ep_2 = EP_2[key][:,i,j,k,l].cpu().detach()
                bptt = BPTT[key][:,i,j,k,l].cpu().detach()


            # INTEGRATE to get summed instantaneous gradient estimates (= actual parameter updates) up to each time step
            ep, bptt = integrate(ep), integrate(bptt)


            ### Format and plot
            ep, bptt = ep.numpy().flatten(), bptt.numpy().flatten()
            
            # Changed order of plots to match requested legend order
            plt.plot(bptt, color=colors[idx], linewidth=2, alpha=0.7, label='BPTT')
            plt.plot(ep, linestyle=':', linewidth=2, color=colors[idx], alpha=0.7, label='EP (positive $\\beta$ only)')

            if EP_2 is not None:
                ep_2 = integrate(ep_2)
                ep_2 = ep_2.numpy().flatten()
                plt.plot(ep_2, linestyle=':', linewidth=2, color=colors[idx], alpha=0.7, label='EP (negative $\\beta$ only)')
                plt.plot((ep + ep_2)/2, linestyle='--', linewidth=2, color=colors[idx], alpha=0.7, label='EP (symmetric)')
            plt.title(key.replace('.',' '))

        plt.grid()
        plt.legend()
        plt.xlabel('Nudged Phase Time, $t$')
        plt.ylabel('Instantaneous Parameter Update, $\\hat{\\nabla}^{\\rm EP}(\\beta,t)$')
        
        # Save as PNG file
        file_name = path+'/'+key.replace('.','_')
        fig.savefig(file_name+'.png', dpi=300)
        
        # Also save as PDF if requested
        if pdf:
            fig.savefig(file_name+'.pdf', format='pdf', dpi=300)
            
        plt.close()

def plot_gdu_instantaneous(BPTT, EP, args, EP_2=None, path=None):
    """Print random parameter and neuron updates for EP and BPTT.
    
    Prints 3 random elements from each parameter/neuron to compare 
    EP and BPTT updates at each time step. Also plots these values
    and saves the plots to files.
    
    Args:
        BPTT: Dictionary of BPTT updates
        EP: Dictionary of EP updates
        args: Arguments object containing other parameters
        EP_2: Optional dictionary of second EP updates. If provided, EP and EP_2 will be averaged.
        path: Path to save plots. If None, args.path will be used.
    """

    
    print("\n### COMPARING EP AND BPTT UPDATES ###")
    
    # Use the provided path or fall back to args.path
    plots_dir = path if path is not None else args.path
    os.makedirs(plots_dir, exist_ok=True)
    
    # Process all keys in BPTT/EP
    for key in BPTT.keys():
        print(f"\n## {key} ##")
        
        tensor = BPTT[key]
        
        # Get 3 random samples from this tensor
        for idx in range(3):
            # Extract values based on tensor dimension (just like in plotting code)
            if len(tensor.shape) == 2:  # 2D tensors (biases, or time x batch for neurons)
                i = np.random.randint(tensor.size(1))
                bptt_values = tensor[:, i].cpu().detach()
                ep_values = EP[key][:, i].cpu().detach()
                if EP_2 is not None:
                    ep2_values = EP_2[key][:, i].cpu().detach()
                    ep_values = (ep_values + ep2_values) / 2
                index_str = f"[{i}]"
                
            elif len(tensor.shape) == 3:  # 3D tensors (weights, or time x batch x features for neurons)
                i, j = np.random.randint(tensor.size(1)), np.random.randint(tensor.size(2))
                bptt_values = tensor[:, i, j].cpu().detach()
                ep_values = EP[key][:, i, j].cpu().detach()
                if EP_2 is not None:
                    ep2_values = EP_2[key][:, i, j].cpu().detach()
                    ep_values = (ep_values + ep2_values) / 2
                index_str = f"[{i}, {j}]"
                
            elif len(tensor.shape) == 4:  # 4D tensors (time x batch x dim1 x dim2 for neurons)
                i, j, k = np.random.randint(tensor.size(1)), np.random.randint(tensor.size(2)), np.random.randint(tensor.size(3))
                bptt_values = tensor[:, i, j, k].cpu().detach()
                ep_values = EP[key][:, i, j, k].cpu().detach()
                if EP_2 is not None:
                    ep2_values = EP_2[key][:, i, j, k].cpu().detach()
                    ep_values = (ep_values + ep2_values) / 2
                index_str = f"[{i}, {j}, {k}]"
                
            elif len(tensor.shape) == 5:  # 5D tensors (conv kernels, or time x batch x dim1 x dim2 x dim3 for neurons)
                i, j = np.random.randint(tensor.size(1)), np.random.randint(tensor.size(2))
                k, l = np.random.randint(tensor.size(3)), np.random.randint(tensor.size(4))
                bptt_values = tensor[:, i, j, k, l].cpu().detach()
                ep_values = EP[key][:, i, j, k, l].cpu().detach()
                if EP_2 is not None:
                    ep2_values = EP_2[key][:, i, j, k, l].cpu().detach()
                    ep_values = (ep_values + ep2_values) / 2
                index_str = f"[{i}, {j}, {k}, {l}]"
                
            else:  # Skip other dimensions
                continue
            
            # Print time steps for this element
            print(f"Sample {idx+1}, Index {index_str}:")
            for t in range(bptt_values.size(0)):
                bptt_val = bptt_values[t].item()
                ep_val = ep_values[t].item()
                print(f"  Time step K={t}: BPTT={bptt_val:.6f}, EP={ep_val:.6f}")
            
            # Plot instantaneous BPTT and EP values
            plt.figure(figsize=(10, 6))
            time_steps = range(bptt_values.size(0))
            plt.plot(time_steps, bptt_values.numpy(), 'b-', label='BPTT')
            plt.plot(time_steps, ep_values.numpy(), 'r-', label='EP (symmetric)' if EP_2 is not None else 'EP')
            plt.xlabel('Time Step K')
            plt.ylabel('Update Value')
            
            # Modified title to be more intuitive
            if len(tensor.shape) == 2:  # 2D tensors
                plt.title(f'{key} neuron[{i}] Instantaneous Updates')
            elif len(tensor.shape) == 3:  # 3D tensors
                plt.title(f'{key} neuron[{j}] (batch {i}) Instantaneous Updates')
            elif len(tensor.shape) == 4:  # 4D tensors
                plt.title(f'{key} neuron[{j},{k}] (batch {i}) Instantaneous Updates')
            elif len(tensor.shape) == 5:  # 5D tensors
                plt.title(f'{key} neuron[{j},{k},{l}] (batch {i}) Instantaneous Updates')
            
            plt.legend()
            plt.grid(True)
            
            # Save the instantaneous plot
            sanitized_key = key.replace('.', '_').replace('/', '_')
            filename = f"{plots_dir}/{sanitized_key}_{idx}_instantaneous.png"
            plt.savefig(filename)
            plt.close()

            # Plot integrated BPTT and EP values
            plt.figure(figsize=(10, 6))
            bptt_integrated = integrate(bptt_values)
            ep_integrated = integrate(ep_values)
            plt.plot(time_steps, bptt_integrated.numpy(), 'b-', label='BPTT')
            plt.plot(time_steps, ep_integrated.numpy(), 'r-', label='EP (symmetric)' if EP_2 is not None else 'EP')
            plt.xlabel('Time Step K')
            plt.ylabel('Integrated Update Value')
            
            # Modified title to be more intuitive
            if len(tensor.shape) == 2:  # 2D tensors
                plt.title(f'{key} neuron[{i}] Integrated Updates')
            elif len(tensor.shape) == 3:  # 3D tensors
                plt.title(f'{key} neuron[{j}] (batch {i}) Integrated Updates')
            elif len(tensor.shape) == 4:  # 4D tensors
                plt.title(f'{key} neuron[{j},{k}] (batch {i}) Integrated Updates')
            elif len(tensor.shape) == 5:  # 5D tensors
                plt.title(f'{key} neuron[{j},{k},{l}] (batch {i}) Integrated Updates')
            
            plt.legend()
            plt.grid(True)
            
            # Save the integrated plot
            filename = f"{plots_dir}/{sanitized_key}_{idx}_integrated.png"
            plt.savefig(filename)
            plt.close()
    
    print("\n### END OF COMPARISON ###\n")


def RMSE(BPTT, EP):
    # print the root mean square error, and sign error between EP and BPTT gradients
    print('\nGDU check :')
    error_dict = {}
    for key in BPTT.keys():
        K = BPTT[key].size(0)
        f_g = (EP[key] - BPTT[key]).pow(2).sum(dim=0).div(K).pow(0.5)
        f =  EP[key].pow(2).sum(dim=0).div(K).pow(0.5)
        g = BPTT[key].pow(2).sum(dim=0).div(K).pow(0.5)
        comp = f_g/(1e-10+torch.max(f,g))
        sign = torch.where(EP[key]*BPTT[key] < 0, torch.ones_like(EP[key]), torch.zeros_like(EP[key]))
        rmse_value = comp.mean().item()
        sign_error = sign.mean().item()
        error_dict[key] = {'rmse': rmse_value, 'sign_error': sign_error}
        print(key.replace('.','_'), '\t RMSE =', round(rmse_value, 4), '\t SIGN err =', round(sign_error, 4))
    print('\n')
    return error_dict





 
