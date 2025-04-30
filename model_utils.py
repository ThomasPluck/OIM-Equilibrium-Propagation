import torch
import torch.nn.functional as F
import math
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt
import os

from metric_utils import *




### ACTIVATION FUNCTIONS ###
def my_sigmoid(x):
    return 1/(1+torch.exp(-4*(x-0.5)))

def hard_sigmoid(x):
    return (1+F.hardtanh(2*x-1))*0.5

def ctrd_hard_sig(x):
    return (F.hardtanh(2*x))*0.5

def my_hard_sig(x):
    return (1+F.hardtanh(x-1))*0.5

### ###






### INITIALIZATION ###

def my_init(weight_scales, bias_scales=None):
    """
    Initialize weights and biases with separate per-layer scaling.
    
    Parameters:
    - weight_scales: Float or list of floats for weight scaling factors
    - bias_scales: Float or list of floats for bias scaling factors (optional, defaults to weight_scales)
    """
    # Prepare scales list or single scalar for weights
    if isinstance(weight_scales, (float, int)):
        weight_scales_list = None
        weight_scalar = weight_scales
    else:
        weight_scales_list = list(weight_scales)
        weight_scalar = None
    
    # Prepare scales list or single scalar for biases
    # If bias_scales is None, use the weight_scales
    if bias_scales is None:
        bias_scales_list = weight_scales_list
        bias_scalar = weight_scalar
    elif isinstance(bias_scales, (float, int)):
        bias_scales_list = None
        bias_scalar = bias_scales
    else:
        bias_scales_list = list(bias_scales)
        bias_scalar = None

    # if isinstance(m, torch.nn.Conv2d):
    #     torch.nn.init.kaiming_uniform_(m.weight, math.sqrt(5))
    #     m.weight.data.mul_(scale)
    #     if m.bias is not None:
    #         fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
    #         bound = 1 / math.sqrt(fan_in)
    #         torch.nn.init.uniform_(m.bias, -bound, bound)
    #         m.bias.data.mul_(scale)
        
    weight_idx = 0

    def my_scaled_init(m):
        nonlocal weight_idx
        
        # Handle Linear modules (synapses)
        if isinstance(m, torch.nn.Linear):
            # Get scaling values for weight and bias
            weight_scale = weight_scalar if weight_scales_list is None else weight_scales_list[weight_idx] if weight_idx < len(weight_scales_list) else weight_scales_list[-1]
            bias_scale = bias_scalar if bias_scales_list is None else bias_scales_list[weight_idx] if weight_idx < len(bias_scales_list) else bias_scales_list[-1]
            
            # Weight initialization
            torch.nn.init.kaiming_uniform_(m.weight, math.sqrt(5))
            m.weight.data.mul_(weight_scale)
            
            # Bias initialization if present
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound)
                m.bias.data.mul_(bias_scale)
                
            weight_idx += 1

        # Handle OIM_MLP biases separately
        elif isinstance(m, OIM_MLP):
            # Initialize biases
            for idx, bias in enumerate(m.biases):
                # For each bias, the fan_in is the size of the previous layer
                # Note: biases start from the first hidden layer (index 1 in archi)
                # So we need to use idx to get the correct previous layer
                # i.e. prev_layer_idx = idx  # idx is 0-based, so for first hidden layer (idx=0), we want input layer (0)
                # Choose scale value for this bias layer
                bias_scale = bias_scalar if bias_scales_list is None else bias_scales_list[idx] if idx < len(bias_scales_list) else bias_scales_list[-1]
                
                fan_in = m.archi[idx]
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(bias, -bound, bound)
                bias.data.mul_(bias_scale)
            # Sync parameters remain initialized to zero

    return my_scaled_init

### ###

















### OIM MLP ###

class OIM_MLP(torch.nn.Module):
    def __init__(self, archi, activation=torch.cos, epsilon=0.1, random_phase_initialisation=False, path=None, intralayer_connections=False,
                 quantisation_bits=0, J_max=1.0, h_max=1.0, sync_max=1.0):
        """
        Initialize an Oscillator Ising Machine Multi-Layer Perceptron.
        
        Parameters:
        - archi: List defining the architecture [input_size, hidden_size_1, ..., output_size]
        - activation: Activation function to use (default: torch.cos)
        - epsilon: Step size for OIM dynamics (default: 0.1)
        - random_phase_initialisation: Whether to initialize phases randomly (default: False)
        - path: Path where plot images will be saved (default: None)
        - intralayer_connections: Whether to include trainable synaptic connections within each hidden layer (default: False)
        - quantisation_bits: Number of bits for parameter quantization (0 means no quantization)
        - J_max: Maximum absolute value for synaptic weights
        - h_max: Maximum absolute value for bias parameters
        - sync_max: Maximum absolute value for synchronization parameters
        """
        super(OIM_MLP, self).__init__()
        
        self.archi = archi
        self.n_layers = len(archi) - 1
        self.activation = activation
        self.epsilon = epsilon
        self.random_phase_initialisation = random_phase_initialisation
        self.path = path  # Store path for saving plots
        self.intralayer_connections = intralayer_connections
        
        # Store quantization parameters
        self.quantisation_bits = quantisation_bits
        self.J_max = J_max
        self.h_max = h_max
        self.sync_max = sync_max
        
        self.nc = self.archi[-1]  # Number of classes equals the last layer size
        
        self.softmax = False  # For compatibility with original code (Softmax readout is only defined for CNN and VFCNN)
        
        # Initialize synaptic weights between layers (J parameters in energy function)
        self.synapses = torch.nn.ModuleList()
        for idx in range(len(archi)-1):
            self.synapses.append(torch.nn.Linear(archi[idx], archi[idx+1], bias=False)) # Set bias to False as OIM do not use simple linear bias
        
        # Initialize intralayer synaptic weights if enabled
        if self.intralayer_connections:
            # Create intralayer connections for each hidden layer (not output layer)
            self.intralayer_synapses = torch.nn.ModuleList()
            for idx in range(1, len(archi)-1):
                # Create a square weight matrix for each hidden layer
                self.intralayer_synapses.append(torch.nn.Linear(archi[idx], archi[idx], bias=False))
        
        # Initialize biases (h parameters in energy function)
        self.biases = torch.nn.ParameterList()
        for idx in range(1, len(archi)): # Start from 1 as we do not have a bias for the input layer
            self.biases.append(torch.nn.Parameter(torch.zeros(archi[idx]))) # Initialize biases to zero
        
        # Initialize synchronization terms (K_s parameters in energy function)
        self.syncs = torch.nn.ParameterList()
        for idx in range(1, len(archi)):
            self.syncs.append(torch.nn.Parameter(torch.zeros(archi[idx]))) # Initialize synchronization terms to zero
            
        # Apply quantization after initialization if needed
        if self.quantisation_bits > 0:
            self.quantize_parameters()

    def quantize_parameters(self):
        """
        Quantize all model parameters based on quantisation_bits and max values.
        This function applies both value clamping and quantization to synaptic weights, biases, 
        and sync parameters, but only when quantisation_bits > 0.
        """
        # Skip if quantization is disabled
        if self.quantisation_bits <= 0:
            return
        
        # Calculate quantization step sizes based on bits and max values
        J_levels = 2 ** self.quantisation_bits - 1
        h_levels = 2 ** self.quantisation_bits - 1
        sync_levels = 2 ** self.quantisation_bits - 1
        
        J_step = 2 * self.J_max / J_levels
        h_step = 2 * self.h_max / h_levels
        sync_step = 2 * self.sync_max / sync_levels
        
        # Quantize synaptic weights
        for synapse in self.synapses:
            # Clip weights to [-J_max, J_max]
            synapse.weight.data.clamp_(-self.J_max, self.J_max)
            # Quantize to discrete levels
            synapse.weight.data = torch.round(synapse.weight.data / J_step) * J_step
        
        # Quantize intralayer synaptic weights if enabled
        if self.intralayer_connections:
            for synapse in self.intralayer_synapses:
                synapse.weight.data.clamp_(-self.J_max, self.J_max)
                synapse.weight.data = torch.round(synapse.weight.data / J_step) * J_step
        
        # Quantize biases
        for bias in self.biases:
            bias.data.clamp_(-self.h_max, self.h_max)
            bias.data = torch.round(bias.data / h_step) * h_step
        
        # Quantize synchronization terms
        for sync in self.syncs:
            sync.data.clamp_(-self.sync_max, self.sync_max)
            sync.data = torch.round(sync.data / sync_step) * sync_step

    def total_energy(self, x, phases, beta=0.0, y=None, criterion=None):
        """
        Compute the OIM 'total' energy function, F:

        F = E_OIM + beta * L, 
        - for loss function L
        - for OIM energy funcition E_OIM = E^J_OIM + E^h_OIM + E^K_s_OIM
        
        OIM dynamics are given by dφ_i/dt = -δF/δφ_i (i.e. the gradient of the energy function with respect to the PHASES) in both free (beta=0) and nudged (beta != 0) phases
        - They are computed here by automatic differentiation framework 

        Parameters:
        - x: Input data [batch_size, input_dim]
        - phases: List of phase variables for each layer EXCLUDING the input layer (which are always clamped)
        - beta: Nudging factor for the loss
        - y: Target labels (optional)
        - criterion: Loss function (optional)
        """

        batch_size = x.size(0)
        device = x.device
        energy = torch.zeros(batch_size, device=device) # i.e. we want a separate energy for each batch element so we can calculate energy gradient descent dynamics for each batch element independently
        
        # Flatten input
        x_flat = x.view(batch_size, -1)
        
        # E_OIM^J term for input to first hidden layer (technically this gets implemented as a bias h in the actual OIM)
        # -∑_{i∈L_0, j∈L_1} J^(0)_{ij} x_i cos(φ_j)
        # Note self.synapses[0](x_flat) runs the first linear layer (with no biases) with argument of x_flat
        # Note we use phases[0] for the first hidden layer, so this is k=1 layer
        # Note self.activation is cos by default but can be changed to other activation functions
        energy_J_input = -torch.sum(self.synapses[0](x_flat) * self.activation(phases[0]), dim=1)  # Elementwise multiplication, then sum over neurons (dim=1) but keep batch (dim=0) so we get [batch_size] of energies

        # E_OIM^J term for hidden to hidden and hidden to output connections
        # -∑_{k=1}^{N_L-1} ∑_{i∈L_k, j∈L_{k+1}} J^(k)_{ij} cos(φ_i - φ_j)
        energy_J_hidden = torch.zeros(batch_size, device=device) # i.e. we want a separate energy for each batch element
        for k in range(1, len(phases)):  # k starts at 1 (second hidden layer) and goes up to len(phases)-1 (which is the index of the output layer) but we incorporate weights 'from behind'
            # Process entire batch at once
            phi_k = phases[k-1].unsqueeze(2)    # Shape: [batch_size, layer_k_size, 1] # phases[k-1] is layer k
            phi_k_plus_1 = phases[k].unsqueeze(1)     # Shape: [batch_size, 1, layer_k+1_size] # phases[k] is layer k+1
            phase_diffs = phi_k - phi_k_plus_1        # Shape: [batch_size, layer_k_size, layer_k+1_size]
            cos_diffs = self.activation(phase_diffs)  # Shape: [batch_size, layer_k_size, layer_k+1_size] # Note self.activation is cos by default but can be changed to other activation functions
            
            # Multiply by weights and sum
            weights = self.synapses[k].weight   # Shape: [layer_k+1_size, layer_k_size] # synapses[k] connects layer k to k+1
            # Multiply and sum over both neuron dimensions to get per-batch energy
            energy_J_hidden -= torch.sum(cos_diffs * weights.T, dim=(1,2))  # Shape: [batch_size]
        
        # E_OIM^J_intra term for intralayer connections if enabled
        energy_J_intralayer = torch.zeros(batch_size, device=device)
        if self.intralayer_connections:
            for k in range(len(self.intralayer_synapses)):  # k iterates over hidden layers (not including output)
                # Get phases for the current hidden layer (k+1 because phases[0] is the first hidden layer)
                phi_layer = phases[k]  # Shape: [batch_size, layer_size]
                
                # For each batch, compute a matrix of all pairwise phase differences
                # Reshape to [batch_size, layer_size, 1] and [batch_size, 1, layer_size]
                phi_i = phi_layer.unsqueeze(2)  # Shape: [batch_size, layer_size, 1]
                phi_j = phi_layer.unsqueeze(1)  # Shape: [batch_size, 1, layer_size]
                
                # Compute phase differences for all pairs
                phase_diffs_intra = phi_i - phi_j  # Shape: [batch_size, layer_size, layer_size]
                
                # Apply activation function (cosine by default)
                cos_diffs_intra = self.activation(phase_diffs_intra)  # Shape: [batch_size, layer_size, layer_size]
                
                # Multiply by intralayer weights and sum
                weights_intra = self.intralayer_synapses[k].weight  # Shape: [layer_size, layer_size]
                
                # We need to be careful with symmetry here - the energy term should only include each pair once
                # Using the upper triangular part of the matrix (excluding diagonal i.e. diagonal=1 is 1 step above the diagonal)
                # e.g. note for phase_diffs in energy_J_hidden we have phi(1,1) - phi(2,3) but not phi(2,3) - phi(1,1) etc (hence we don't use factor 0.5 in energy)
                # but for phase_diffs_intra we have phi(1,1) - phi(1,3) and phi(1,3) - phi(1,1) (hence we must use mask first)
                mask = torch.triu(torch.ones_like(weights_intra), diagonal=1)
                masked_weights = weights_intra * mask # elementwise multiplication
                
                # Multiply and sum over both neuron dimensions
                energy_J_intralayer -= torch.sum(cos_diffs_intra * masked_weights, dim=(1,2))  # Shape: [batch_size]
        
        # E_OIM^h term (bias energy)
        # ∑_{k=1}^{N_L} ∑_{i∈L_k} h^(k)_i cos(φ_i)
        energy_h = torch.zeros(batch_size, device=device) # i.e. we want a separate energy for each batch element
        for k in range(len(phases)): # k indexes into phases, so add 1 for true layer number
            cos_phi_k = self.activation(phases[k]) # Note self.activation is cos by default but can be changed to other activation functions
            energy_h -= torch.sum(cos_phi_k * self.biases[k].unsqueeze(0), dim=1) # Note we use biases[k].unsqueeze(0) to ensure the bias is broadcasted to the size of cos_phi_k not the batch size
        
        # E_OIM^K_s term (synchronization energy)
        # ∑_{k=1}^{N_L} ∑_{i∈L_k} K^(k)_{s i} cos(2φ_i)
        energy_K_s = torch.zeros(batch_size, device=device)
        for k in range(len(phases)): # k indexes into phases, so add 1 for true layer number
            cos_2phi_k = self.activation(2 * phases[k]) # Note self.activation is cos by default but can be changed to other activation functions
            energy_K_s -= 0.5*torch.sum(cos_2phi_k * self.syncs[k].unsqueeze(0), dim=1) # Note we use syncs[k].unsqueeze(0) to ensure the sync is broadcasted to the size of cos_2phi_k not the batch size # Also note 0.5 is from E_OIM definition, but could be absorbed into syncs anyway
        
        # OIM energy per batch element = sum of all energy terms
        energy = energy_J_input + energy_J_hidden + energy_h + energy_K_s # Size [batch_size]
        
        # Add intralayer energy if enabled
        if self.intralayer_connections:
            energy = energy + energy_J_intralayer



        # Add loss term if in nudged phase
        if beta != 0.0 and y is not None and criterion is not None:
            output_activations = self.activation(phases[-1]) # Note self.activation is cos by default but can be changed to other activation functions, note it is in the range [-1,1] !!
            

            if criterion.__class__.__name__.find('MSE') != -1: # i.e. if criterion is MSE
                y_one_hot = F.one_hot(y, num_classes=self.nc).float() 
                # Transform one-hot encoding from [0,1] to [-1,1] to match cosine output range
                y_transformed = y_one_hot * 2 - 1  # Scale from [0,1] to [-1,1]
                loss = 0.5 * criterion(output_activations, y_transformed).sum(dim=1)  # Shape [batch_size]

            else: # i.e. if criterion is not MSE e.g. CrossEntropyLoss # TODO will need to change this for OIM eventually
                # Note this works fine with output activations in the range [-1,1] but maybe rescaling before will be better (make softmax sharper or not)
                loss = criterion(output_activations, y)  # Shape [batch_size]
            
            energy = energy + beta * loss # Note + beta * loss is the loss term in the energy function but - beta * loss is the loss term in the primitive function # Also note energy is size [batch_size]
        
        return energy


    def forward(self, x, y, phases, T, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False, plot=False, phase_type="Free", return_velocities=False, plot_idx=0, noise_level=0.0): 
        # Note reduction='none' is important for OIM as we want to keep the loss as a vector of size [batch_size] to compute energy gradient descent dynamcis independently for each batch element
        """
        Run T steps of gradient descent on the OIM energy
        
        Parameters:
        - x: Input data
        - y: Target labels
        - phases: Initial phases
        - T: Number of time steps
        - beta: Nudging factor
        - criterion: Loss function
        - check_thm: Whether to check theorem relating EP and BPTT dynamics at various timesteps (requires keeping gradients) 
        - plot: Whether to plot phase dynamics over time
        - phase_type: Type of phase for labeling plots (Free/Positive/Negative/Evaluate)
        - return_velocities: Whether to return velocities (phase changes) along with phases
        - plot_idx: Index of the batch element to plot (default: 0)
        - noise_level: Standard deviation of Gaussian noise to add to phases (default: 0.0)
        """
        
        # Store initial phases (for possible rerun with high velocity)
        if return_velocities:
            initial_phases = []
            for phase in phases:
                initial_phases.append(phase.clone().detach())
        
        # If plotting is enabled, track phases over time
        if plot:
            # Get the correct label for the specified batch element
            correct_label = y[plot_idx].item() if y is not None and len(y) > 0 else None
            
            # Store time steps and pre-allocate phase history array
            time_steps = list(range(T+1))  # +1 to include initial state
            # Pre-allocate phase_history as a list with None placeholders
            phase_history = [None] * (T+1)
            # Store initial state for the specified batch element
            phase_history[0] = torch.cat([phase[plot_idx].detach().cpu() for phase in phases])


        # Run T steps of gradient descent on the OIM energy
        for t in range(T):
            # Ensure all phases require gradients
            for idx in range(len(phases)):
                if not phases[idx].requires_grad:
                    phases[idx].requires_grad_(True)
            
            # COMPUTE PER BATCH ELEMENT ENERGIES
            energies = self.total_energy(x, phases, beta, y, criterion)  # [batch_size] i.e. we want a separate energy for each batch element so we can calculate energy gradient descent dynamics for each batch element independently
            

            # COMPUTE GRADIENTS OF ENERGIES WITH RESPECT TO PHASES (SEPARATE FOR EACH BATCH ELEMENT)
            # Note autograd.grad(y,x) computes the gradient of y with respect to x i.e. grads[i] = dy/dx_i
            # In other words it computes a vector Jacobian product v^T J, where J = dy_i/dphi_j and v = grad_outputs = [1,1,1,...,1] by default
            # Therefore grads[i] = [1,1,1,...,1] * [dy_1/dphi_1, dy_2/dphi_1, ..., dy_batch_size/dphi_1; dy_1/dphi_2, dy_2/dphi_2, ..., dy_batch_size/dphi_2;] 
            # = [dy_1/dphi_1 + dy_2/dphi_1 + ... + dy_batch_size/dphi_1, dy_1/dphi_2 + dy_2/dphi_2 + ... + dy_batch_size/dphi_2, ...]
            # i.e. grads[i] = sum_{j=1}^{batch_size} dy_j/dphi_i
            # So equivalently we are computing the gradient of the summed energy over all batch elements with respect to each phi_i
            # However note that here phases has size [batch_size, layer_size] (i.e. each phase in each batch element is treated as a separate variable)
            # Therefore grads[i] = dy/dphi_i is a vector of size [batch_size, layer_size] so each phi_i is really phi_i^j for j = 1 to batch_size
            grads = torch.autograd.grad(energies, phases, 
                                    grad_outputs=torch.ones_like(energies), # This is the default anyway, but note autograd.grad(y,x)...
                                    create_graph=check_thm) # i.e. keep graph so we can computer derivatives of derivatives (if check_thm=True)
            
            
            
            # UPDATE PHASES USING GRADIENT DESCENT ON THE ENERGY
            
            # HIDDEN LAYERS
            for idx in range(len(phases)-1):
                # Use out-of-place operation so we don't modify the original phases tensor and mess up gradient computation stuff
                phases[idx] = phases[idx] - self.epsilon * grads[idx]
                
                # Add noise if noise_level > 0
                if noise_level > 0.0:
                    noise = torch.randn_like(phases[idx]) * noise_level
                    phases[idx] = phases[idx] + self.epsilon * noise
                
                if check_thm:
                    phases[idx].retain_grad()
                else:
                    phases[idx].requires_grad_(True)



            
            # OUTPUT LAYER 
            # (no distinction between MSE and non-MSE here), we don't need to do activation function for either
            phases[-1] = phases[-1] - self.epsilon * grads[-1]
            
            # Add noise if noise_level > 0
            if noise_level > 0.0:
                noise = torch.randn_like(phases[-1]) * noise_level
                phases[-1] = phases[-1] + self.epsilon * noise

            if check_thm:
                phases[-1].retain_grad()
            else:
                phases[-1].requires_grad_(True)




                
            # If plotting is enabled, store the current phases
            if plot:
                # Store the specified batch element's phases at index t+1
                phase_history[t+1] = torch.cat([phase[plot_idx].detach().cpu() for phase in phases])   
        
        # Create and save plot if requested after all time steps have been computed
        if plot:
            self._plot_phases(time_steps, phase_history, phase_type, correct_label)


        if return_velocities:
            # Store raw gradients as velocities (scaled by epsilon)
            velocities = []
            for idx in range(len(phases)):
                # Velocity is directly proportional to gradient magnitude
                velocity = self.epsilon * grads[idx].abs()
                velocities.append(velocity)



            ### HIGH VELOCITY CHECK ###
               
            # Check if any velocity is above threshold at end of dynamics
            max_velocity_threshold = 1.0
            high_velocity_detected = False
            problem_batch_idx = 0
            
            # First scan for any batch element with high velocity
            for vel in velocities:
                # vel shape is [batch_size, neuron_dim]
                # Find maximum velocity per batch element
                batch_max_velocities = vel.max(dim=1)[0]  # Returns max values and indices

                # Check if any batch element has velocity > threshold
                if torch.any(batch_max_velocities > max_velocity_threshold):
                    high_velocity_detected = True
                    # Find which batch element has the highest velocity
                    problem_batch_idx = torch.argmax(batch_max_velocities).item()
                    max_velocity = batch_max_velocities[problem_batch_idx].item()
                    break
                    
            # If high velocity detected and we're not already plotting that specific element,
            # run forward again with plotting for the problematic element
            if high_velocity_detected and (not plot or plot_idx != problem_batch_idx):
                print(f"WARNING: High velocity ({max_velocity:.4f}) detected in batch element {problem_batch_idx} at end of {phase_type} phase. Generating debug plot...")
                
                
                # Run forward again with plotting for the problematic example (via plot_idx)
                # Note: We don't need the output of this call, we just want the plot
                # DO NOT use torch.no_grad() here as we need gradients for the dynamics
                _ = self.forward(
                    x, 
                    y, 
                    initial_phases, 
                    T, 
                    beta=beta, 
                    criterion=criterion, 
                    check_thm=check_thm, 
                    plot=True,  # Enable plotting
                    phase_type=f"{phase_type}_high_velocity",  # Mark as high velocity case
                    return_velocities=False,  # No need to return velocities
                    plot_idx=problem_batch_idx  # Plot the problematic batch element
                )

            ### ###



            return phases, velocities
        else:
            return phases, None
    
    def _plot_phases(self, time_steps, phase_history, phase_type="Free", correct_label=None):
        """
        Plot phase evolution over time and save to file.
        
        Parameters:
        - time_steps: List of time steps
        - phase_history: List of phase tensors at each time step
        - phase_type: Type of phase for labeling (Free/Positive/Negative/Evaluate)
        - correct_label: Correct label for the plot
        """
        # Convert phase history to numpy array
        # No need to check for None values as we've pre-allocated and filled all positions
        phase_history = torch.stack(phase_history).numpy()
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        
        # Determine where output layer starts
        hidden_sizes = self.archi[1:-1]
        output_size = self.archi[-1]
        output_start_idx = sum(hidden_sizes)
        
        # Plot hidden layer phases with light gray color
        for i in range(output_start_idx):
            plt.plot(time_steps, phase_history[:, i], color='gray', alpha=0.3, linewidth=0.8)
        
        # Plot output layer phases with distinct styles and add labels
        # Define neon green color for correct label
        neon_green = '#00FF00'  # Hex code for neon green
        
        # Color palette for output neurons, ensuring we don't use neon green for wrong labels
        output_colors = ['red', 'blue', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'black', 'gold']
        
        for i in range(output_size):
            idx = output_start_idx + i
            
            # Determine if this is the correct label and plot with appropriate styling
            is_correct_label = (i == correct_label)
            
            # All conditional logic directly in the plot function
            plt.plot(time_steps, phase_history[:, idx], 
                     color=neon_green if is_correct_label else output_colors[i % len(output_colors)], 
                     linewidth=3.5 if is_correct_label else 2.5,
                     linestyle='-' if is_correct_label else '--',
                     alpha=1.0,
                     label=f"Output {i}{' (Correct Label)' if is_correct_label else ''}")
        
        # Set plot attributes
        title_text = f"{phase_type} Phase Dynamics"
        if "high_velocity" in phase_type.lower():
            title_text = f"High Velocity Detected in {phase_type.replace('_high_velocity', '')} Phase"
            
        plt.title(title_text)
        plt.xlabel("Time Step")
        plt.ylabel("Phase (radians)")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Save the plot with phase type information
        filename = f"dynamics_{phase_type.lower()}.png"
        
        # Use self.path if available, otherwise save to current directory
        if self.path:
            save_path = os.path.join(self.path, filename)
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        else:
            save_path = filename
                
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        if "high_velocity" in phase_type.lower():
            print(f"High velocity debug plot saved as {save_path}")
        # else:
        #     print(f"Phase dynamics plot saved as {save_path}")

    def init_neurons(self, batch_size, device):
        """
        Initialize phase variables. If random_phase_initialisation is True, 
        initialize randomly between 0 and 2π with a fixed seed, same phases for all batch elements.
        Otherwise, initialize all phases to π/2.
        """
        phases = []
        if self.random_phase_initialisation:
            # Set a fixed seed for reproducibility
            torch.manual_seed(42)
            for size in self.archi[1:]:
                # Generate random phases for one instance
                single_phase = 2 * math.pi * torch.rand((size,), device=device)
                # Repeat for batch size
                phase = single_phase.expand(batch_size, -1)
                phase.requires_grad_(True)
                phases.append(phase)
        else:
            for size in self.archi[1:]:
                phase = torch.full((batch_size, size), math.pi/2, device=device)
                phase.requires_grad_(True)
                phases.append(phase)
        return phases
    
    def compute_syn_grads(self, x, y, phases_1, phases_2, betas, criterion, check_thm=False):
        """
        Compute the EP update for synaptic weights based on the difference between
        the free fixed point and the weakly clamped fixed point

        Note if we use third phase, this function is called such that beta_1 is the beta_2, and beta_2 is -beta_2
       and neurons_1 = neurons_2 and neurons_2 = neurons_3 (see train function)

        
        Parameters:
        - x: Input data
        - y: Target labels
        - phases_1: Phases at first equilibrium (free phase) 
        - phases_2: Phases at second equilibrium (nudged phase)
        - betas: Tuple of (beta_1, beta_2)
        - criterion: Loss function
        - check_thm: Whether we're checking the EP theorem
        """


        beta_1, beta_2 = betas
        
        self.zero_grad() # p.grad is zero i.e. zero all gradients in the parameters (weights and biases and syncs) of the model
        
        # Compute energy at free fixed point
        if not check_thm:
            energy_1 = self.total_energy(x, phases_1, beta_1, y, criterion)
        else:
            energy_1 = self.total_energy(x, phases_1, beta_2, y, criterion) # For check_thm=True we use beta_2 because we are comparing to previous nudged phase step not to free phase
        energy_1 = energy_1.mean() # Get mean as want to use optimiser to minimise loss over sum of batch elements

        # Compute energy at nudged fixed point
        energy_2 = self.total_energy(x, phases_2, beta_2, y, criterion)
        energy_2 = energy_2.mean() # Get mean as want to use optimiser to minimise loss over sum of batch elements
        
        # Compute delta energy (now working with scalars)
        # Delta p = -(1/beta) (dF/dp(beta) - dF/dp(0)) for two phases 
        # Delta p = -(1/2beta) (dF/dp(beta) - dF/dp(-beta)) for three phases
        # We want delta_energy to be the 'loss function' i.e. we want optimiser to take negative gradient step -d/dp
        # Therefore delta_energy = -(energy_2 - energy_1) / (beta_1 - beta_2)
        # So for two phases with betas = 0, +beta: delta_energy = (energy(beta) - energy(0)) / beta, and optimiser step = -(1/beta) *(dF/dp(beta) - dF/dp(0))
        # and for three phases with betas = beta, -beta: delta_energy = -(energy(-beta) - energy(beta)) / (2beta), and optimiser step = -(1/(2beta)) *(dF/dp(beta) - dF/dp(-beta))
        delta_energy = -(energy_2 - energy_1) / (beta_1 - beta_2)
        
        # Backward pass to get gradients
        delta_energy.backward()

### ###





        




### PRIMITIVE MLP ###

class P_MLP(torch.nn.Module):
    def __init__(self, archi, activation=torch.tanh, path=None, intralayer_connections=False):
        """
        Initialize a Primitive-function MLP for Equilibrium Propagation.
        
        Parameters:
        - archi: List defining the architecture [input_size, hidden_size_1, ..., output_size]
        - activation: Activation function to use (default: torch.tanh)
        - path: Path where plot images will be saved (default: None)
        - intralayer_connections: Whether to include trainable synaptic connections within each hidden layer (default: False)
        """
        super(P_MLP, self).__init__()
        
        self.activation = activation
        self.archi = archi
        self.softmax = False    # Softmax readout is only defined for CNN and VFCNN       
        self.nc = self.archi[-1] # Number of classes equals the last layer size
        self.path = path  # Store path for saving plots
        self.intralayer_connections = intralayer_connections

        # Synapses between layers
        self.synapses = torch.nn.ModuleList()
        for idx in range(len(archi)-1):
            self.synapses.append(torch.nn.Linear(archi[idx], archi[idx+1], bias=True)) # Note we use bias=True as we want to include the bias in the primitive function
            
        # Intralayer synapses if enabled
        if self.intralayer_connections:
            # Create intralayer connections for each hidden layer (not output layer)
            self.intralayer_synapses = torch.nn.ModuleList()
            for idx in range(1, len(archi)-1):
                # Create a square weight matrix for each hidden layer
                self.intralayer_synapses.append(torch.nn.Linear(archi[idx], archi[idx], bias=False))
            
    def Phi(self, x, y, neurons, beta, criterion):
        # Computes the primitive function given static input x, label y, neurons is the sequence of (non-input) variable neurons
        # criterion is the loss 
        x = x.view(x.size(0),-1) # flattening the input
        
        layers = [x] + neurons  # concatenate the input to other layers i.e. [x, neurons[0], neurons[1], ...]
        
        # Primitive function computation for interlayer connections
        phi = 0.0
        for idx in range(len(self.synapses)):
            phi += torch.sum(self.synapses[idx](layers[idx]) * layers[idx+1], dim=1).squeeze() # Scalar product s_n.W.s_n-1 and automatically includes bias
        
        # Add intralayer connection terms if enabled
        if self.intralayer_connections:
            for idx, layer_idx in enumerate(range(1, len(layers)-1)):
                layer = layers[layer_idx]  # Current hidden layer activations
                
                # Get original contribution with self-synapses
                full_contribution = self.intralayer_synapses[idx](layer)
                
                # Create a mask that zeros out the diagonal connections
                # Creating a matrix with zeros on diagonal and ones elsewhere
                diag_mask = 1.0 - torch.eye(layer.shape[1], device=layer.device)
                
                # Mask the full contribution to remove self-synapses
                masked_contribution = full_contribution * diag_mask.unsqueeze(0)
                
                # Add masked contribution to primitive function
                phi += torch.sum(masked_contribution * layer, dim=1).squeeze()

        # Add the loss term to the primitive function
        if beta!=0.0: # Nudging the output layer when beta is non zero 
            if criterion.__class__.__name__.find('MSE')!=-1: # i.e. if the loss is MSE
                y = F.one_hot(y, num_classes=self.nc)
                L = 0.5*criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()   
            else: # i.e. if the loss is not MSE
                L = criterion(layers[-1].float(), y).squeeze()     
            phi -= beta*L
        
        return phi
    
    

    def forward(self, x, y, neurons, T, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False, return_velocities=False, plot=False, phase_type="Free", noise_level=0.0):
        # Run T steps of the dynamics for static input x, label y, neurons and nudging factor beta.
        # noise_level parameter is not used in P_MLP but included for consistent interface with OIM_MLP
        not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = x.size(0)
        device = x.device
        
        # If plotting is enabled, track neurons over time
        if plot:
            # Extract the correct label for the first example in batch
            correct_label = y[0].item()
            
            # Store time steps and pre-allocate neuron history array
            time_steps = list(range(T+1))  # +1 to include initial state
            # Pre-allocate neuron_history as a list with None placeholders
            neuron_history = [None] * (T+1)
            # Store initial state
            neuron_history[0] = torch.cat([neuron[0].detach().cpu() for neuron in neurons])
            
        for t in range(T):
            phi = self.Phi(x, y, neurons, beta, criterion) # Computing Phi
            init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True) # Initializing gradients
            grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=check_thm) # dPhi/ds

            # Store previous neurons if we need to calculate velocities
            if return_velocities and t == T-1:  # Only need to store for the last iteration
                previous_neurons = []
                for idx in range(len(neurons)):
                    previous_neurons.append(neurons[idx].clone().detach())

            # HIDDEN LAYERS
            for idx in range(len(neurons)-1): 
                neurons[idx] = self.activation(grads[idx])  # s_(t+1) = sigma(dPhi/ds)
                if check_thm:
                    neurons[idx].retain_grad()
                else:
                    neurons[idx].requires_grad_(True)
             
            # OUTPUT LAYER
            if not_mse:
                neurons[-1] = grads[-1] # i.e. just use logits if loss is not MSE, i.e. CEL as it wants logits
            else:
                neurons[-1] = self.activation(grads[-1]) # Otherwise do activation function as usual for MSE


            if check_thm:
                neurons[-1].retain_grad()
            else:
                neurons[-1].requires_grad_(True)
                
            # If plotting is enabled, store the current neurons
            if plot:
                # Store the first batch element's neurons at index t+1
                neuron_history[t+1] = torch.cat([neuron[0].detach().cpu() for neuron in neurons])

        # Create and save plot if requested after all time steps have been computed
        if plot:
            self._plot_neurons(time_steps, neuron_history, phase_type, correct_label)
            
        # Calculate velocities after all iterations if requested
        if return_velocities:
            velocities = []
            for idx in range(len(neurons)):
                # For all layers: velocity = current neuron value - previous neuron value
                velocity = (neurons[idx] - previous_neurons[idx]).abs()
                velocities.append(velocity)
            
            return neurons, velocities
        else:
            return neurons, None
            
    def _plot_neurons(self, time_steps, neuron_history, phase_type="Free", correct_label=None):
        """
        Plot neuron activation evolution over time and save to file.
        
        Parameters:
        - time_steps: List of time steps
        - neuron_history: List of neuron activations at each time step
        - phase_type: Type of phase for labeling (Free/Positive/Negative/Evaluate)
        - correct_label: Correct label for the plot
        """
        # Convert neuron history to numpy array
        neuron_history = torch.stack(neuron_history).numpy()
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        
        # Determine where output layer starts
        hidden_sizes = self.archi[1:-1]
        output_size = self.archi[-1]
        output_start_idx = sum(hidden_sizes)
        
        # Plot hidden layer neurons with light gray color
        for i in range(output_start_idx):
            plt.plot(time_steps, neuron_history[:, i], color='gray', alpha=0.3, linewidth=0.8)
        
        # Plot output layer neurons with distinct styles and add labels
        # Define neon green color for correct label
        neon_green = '#00FF00'  # Hex code for neon green
        
        # Color palette for output neurons, ensuring we don't use neon green for wrong labels
        output_colors = ['red', 'blue', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'black', 'gold']
        
        for i in range(output_size):
            idx = output_start_idx + i
            
            # Determine if this is the correct label and plot with appropriate styling
            is_correct_label = (i == correct_label)
            
            # All conditional logic directly in the plot function
            plt.plot(time_steps, neuron_history[:, idx], 
                     color=neon_green if is_correct_label else output_colors[i % len(output_colors)], 
                     linewidth=3.5 if is_correct_label else 2.5,
                     linestyle='-' if is_correct_label else '--',
                     alpha=1.0,
                     label=f"Output {i}{' (Correct Label)' if is_correct_label else ''}")
        
        # Set plot attributes
        plt.title(f"{phase_type} Phase Neuron Dynamics")
        plt.xlabel("Time Step")
        plt.ylabel("Neuron Activation")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Save the plot with phase type information
        filename = f"dynamics_{phase_type.lower()}_neuron.png"
        
        # Use self.path if available, otherwise save to current directory
        if self.path:
            save_path = os.path.join(self.path, filename)
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        else:
            save_path = filename
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        # print(f"Neuron dynamics plot saved as {save_path}")

    def init_neurons(self, mbs, device):
        # Initializing the neurons
        neurons = []
        append = neurons.append
        for size in self.archi[1:]:  
            append(torch.zeros((mbs, size), requires_grad=True, device=device)) # i.e. initialize all neurons to zero 
        return neurons


    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion, check_thm=False):
        # Computing the EP update given two steady states neurons_1 and neurons_2, static input x, label y
        # Note if we use third phase, this function is called such that beta_1 is the beta_2 and beta_2 is -beta_2
        # and neurons_1 = neurons_2 and neurons_2 = neurons_3 (see train function)

        beta_1, beta_2 = betas 
        
        self.zero_grad()            # p.grad is zero i.e. zero all gradients in the parameters (weights and biases) of the model
        if not(check_thm):
            phi_1 = self.Phi(x, y, neurons_1, beta_1, criterion)
        else:
            phi_1 = self.Phi(x, y, neurons_1, beta_2, criterion) # For check_thm=True we use beta_2 because we are comparing to previous nudged phase step not to free phase
        phi_1 = phi_1.mean()
        
        phi_2 = self.Phi(x, y, neurons_2, beta_2, criterion)
        phi_2 = phi_2.mean()
        
        # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1) ----> dL/dp  by the theorem
        # i.e. the automatic differentiation framework computes the gradient of the loss with respect to the parameters (weights and biases)

        # Compute delta phi
        # Delta p = (1/beta) (dphi/dp(beta) - dphi/dp(0)) for two phases 
        # Delta p = (1/2beta) (dF/dp(beta) - dF/dp(-beta)) for three phases
        # We want delta_phi to be the 'loss function' i.e. we want optimiser to take negative gradient step -d/dp
        # Therefore delta_phi = (phi_2 - phi_1) / (beta_1 - beta_2)
        # So for two phases with betas = 0, +beta: delta_phi = (phi(beta) - phi(0)) / -beta, and optimiser step = (1/beta) *(dphi/dp(beta) - dphi/dp(0))
        # and for three phases with betas = beta, -beta: delta_phi = (phi(-beta) - phi(+beta)) / (2beta), and optimiser step = (1/(2beta)) *(dphi/dp(beta) - dphi/dp(-beta))
       
        delta_phi = (phi_2 - phi_1)/(beta_1 - beta_2)        
        delta_phi.backward()         
### ###
    





















































    
    
# # Convolutional Neural Network

# def make_pools(letters):
#     pools = []
#     for p in range(len(letters)):
#         if letters[p]=='m':
#             pools.append( torch.nn.MaxPool2d(2, stride=2) )
#         elif letters[p]=='a':
#             pools.append( torch.nn.AvgPool2d(2, stride=2) )
#         elif letters[p]=='i':
#             pools.append( torch.nn.Identity() )
#     return pools
        


# class P_CNN(torch.nn.Module):
#     def __init__(self, in_size, channels, kernels, strides, fc, pools, paddings, activation=hard_sigmoid, softmax=False):
#         super(P_CNN, self).__init__()

#         # Dimensions used to initialize neurons
#         self.in_size = in_size
#         self.channels = channels
#         self.kernels = kernels
#         self.strides = strides
#         self.paddings = paddings
#         self.fc = fc
#         self.nc = fc[-1]        

#         self.activation = activation
#         self.pools = pools
        
#         self.synapses = torch.nn.ModuleList()
        
#         self.softmax = softmax # whether to use softmax readout or not

#         size = in_size # size of the input : 32 for cifar10

#         for idx in range(len(channels)-1): 
#             self.synapses.append(torch.nn.Conv2d(channels[idx], channels[idx+1], kernels[idx], 
#                                                  stride=strides[idx], padding=paddings[idx], bias=True))
                
#             size = int( (size + 2*paddings[idx] - kernels[idx])/strides[idx] + 1 )          # size after conv
#             if self.pools[idx].__class__.__name__.find('Pool')!=-1:
#                 size = int( (size - pools[idx].kernel_size)/pools[idx].stride + 1 )   # size after Pool

#         size = size * size * channels[-1]        
#         fc_layers = [size] + fc

#         for idx in range(len(fc)):
#             self.synapses.append(torch.nn.Linear(fc_layers[idx], fc_layers[idx+1], bias=True))
        


#     def Phi(self, x, y, neurons, beta, criterion):

#         mbs = x.size(0)       
#         conv_len = len(self.kernels)
#         tot_len = len(self.synapses)

#         layers = [x] + neurons        
#         phi = 0.0

#         #Phi computation changes depending on softmax == True or not
#         if not self.softmax:
#             for idx in range(conv_len):    
#                 phi += torch.sum( self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx+1], dim=(1,2,3)).squeeze()     
#             for idx in range(conv_len, tot_len):
#                 phi += torch.sum( self.synapses[idx](layers[idx].view(mbs,-1)) * layers[idx+1], dim=1).squeeze()
             
#             if beta!=0.0:
#                 if criterion.__class__.__name__.find('MSE')!=-1:
#                     y = F.one_hot(y, num_classes=self.nc)
#                     L = 0.5*criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()   
#                 else:
#                     L = criterion(layers[-1].float(), y).squeeze()             
#                 phi -= beta*L

#         else:
#             # the output layer used for the prediction is no longer part of the system ! Summing until len(self.synapses) - 1 only
#             for idx in range(conv_len):
#                 phi += torch.sum( self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx+1], dim=(1,2,3)).squeeze()     
#             for idx in range(conv_len, tot_len-1):
#                 phi += torch.sum( self.synapses[idx](layers[idx].view(mbs,-1)) * layers[idx+1], dim=1).squeeze()
             
#             # the prediction is made with softmax[last weights[penultimate layer]]
#             if beta!=0.0:
#                 L = criterion(self.synapses[-1](layers[-1].view(mbs,-1)).float(), y).squeeze()             
#                 phi -= beta*L            
        
#         return phi
    

#     def forward(self, x, y, neurons, T, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False):
 
#         not_mse = (criterion.__class__.__name__.find('MSE')==-1)
#         mbs = x.size(0)
#         device = x.device     
        
#         if check_thm:
#             for t in range(T):
#                 phi = self.Phi(x, y, neurons, beta, criterion)
#                 init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
#                 grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=True)

#                 for idx in range(len(neurons)-1):
#                     neurons[idx] = self.activation( grads[idx] )
#                     neurons[idx].retain_grad()
             
#                 if not_mse and not(self.softmax):
#                     neurons[-1] = grads[-1]
#                 else:
#                     neurons[-1] = self.activation( grads[-1] )

#                 neurons[-1].retain_grad()
#         else:
#              for t in range(T):
#                 phi = self.Phi(x, y, neurons, beta, criterion)
#                 init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
#                 grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=False)

#                 for idx in range(len(neurons)-1):
#                     neurons[idx] = self.activation( grads[idx] )
#                     neurons[idx].requires_grad_(True)
             
#                 if not_mse and not(self.softmax):
#                     neurons[-1] = grads[-1]
#                 else:
#                     neurons[-1] = self.activation( grads[-1] )

#                 neurons[-1].requires_grad_(True)

#         return neurons
       

#     def init_neurons(self, mbs, device):
        
#         neurons = []
#         append = neurons.append
#         size = self.in_size
#         for idx in range(len(self.channels)-1): 
#             size = int( (size + 2*self.paddings[idx] - self.kernels[idx])/self.strides[idx] + 1 )   # size after conv
#             if self.pools[idx].__class__.__name__.find('Pool')!=-1:
#                 size = int( (size - self.pools[idx].kernel_size)/self.pools[idx].stride + 1 )  # size after Pool
#             append(torch.zeros((mbs, self.channels[idx+1], size, size), requires_grad=True, device=device))

#         size = size * size * self.channels[-1]
        
#         if not self.softmax:
#             for idx in range(len(self.fc)):
#                 append(torch.zeros((mbs, self.fc[idx]), requires_grad=True, device=device))
#         else:
#             # we *REMOVE* the output layer from the system
#             for idx in range(len(self.fc) - 1):
#                 append(torch.zeros((mbs, self.fc[idx]), requires_grad=True, device=device))            
            
#         return neurons

#     def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion, check_thm=False):
        
#         beta_1, beta_2 = betas
        
#         self.zero_grad()            # p.grad is zero
#         if not(check_thm):
#             phi_1 = self.Phi(x, y, neurons_1, beta_1, criterion)
#         else:
#             phi_1 = self.Phi(x, y, neurons_1, beta_2, criterion)
#         phi_1 = phi_1.mean()
        
#         phi_2 = self.Phi(x, y, neurons_2, beta_2, criterion)
#         phi_2 = phi_2.mean()
        
#         delta_phi = (phi_2 - phi_1)/(beta_1 - beta_2)        
#         delta_phi.backward() # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1) ----> dL/dp  by the theorem
