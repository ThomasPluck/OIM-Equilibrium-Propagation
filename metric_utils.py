import torch
import numpy as np
import wandb


def get_network_metrics(model):
    """Calculate metrics for network parameters."""
    network_metrics = {}
    
    # Get metrics for synapses (weights)
    for i, syn in enumerate(model.synapses):
        network_metrics[f'weights_{i}'] = {
            'max': syn.weight.data.max().item(),
            'min': syn.weight.data.min().item(),
            'mean': syn.weight.data.mean().item(),
            'std': syn.weight.data.std().item()
        }
    
    # Get metrics for biases (if they exist in the model)
    if hasattr(model, 'biases'):
        for i, bias in enumerate(model.biases):
            network_metrics[f'bias_{i}'] = {
                'max': bias.data.max().item(),
                'min': bias.data.min().item(),
                'mean': bias.data.mean().item(),
                'std': bias.data.std().item()
            }
        
    # Get metrics for syncs (if they exist in the model)
    if hasattr(model, 'syncs'):
        for i, sync in enumerate(model.syncs):
            network_metrics[f'sync_{i}'] = {
                'max': sync.data.max().item(),
                'min': sync.data.min().item(),
                'mean': sync.data.mean().item(),
                'std': sync.data.std().item()
            }
    
    return network_metrics

def get_gradient_metrics(model):
    """Extract gradient metrics from the model parameters.
    
    This captures the same information printed in the debugging section of train_evaluate.py.
    
    Args:
        model: The network model
        
    Returns:
        Dictionary containing gradient metrics
    """
    gradient_metrics = {}
    
    # Collect weight gradient metrics
    for i, weight in enumerate(model.synapses):
        if weight.weight.grad is not None:
            gradient_metrics[f'weight_{i}'] = {
                'min': weight.weight.grad.min().item(),
                'max': weight.weight.grad.max().item()
            }
            
            # Add relative metrics if weights are not zero
            if weight.weight.data.mean().item() != 0:
                mean_weight = weight.weight.data.mean().item()
                gradient_metrics[f'weight_{i}']['rel_min'] = weight.weight.grad.min().item() / mean_weight
                gradient_metrics[f'weight_{i}']['rel_max'] = weight.weight.grad.max().item() / mean_weight
    
    # Collect bias gradient metrics if they exist
    if hasattr(model, 'biases'):
        for i, bias in enumerate(model.biases):
            if bias.grad is not None:
                gradient_metrics[f'bias_{i}'] = {
                    'min': bias.grad.min().item(),
                    'max': bias.grad.max().item()
                }
                
                # Add relative metrics if biases are not zero
                if bias.data.mean().item() != 0:
                    mean_bias = bias.data.mean().item()
                    gradient_metrics[f'bias_{i}']['rel_min'] = bias.grad.min().item() / mean_bias
                    gradient_metrics[f'bias_{i}']['rel_max'] = bias.grad.max().item() / mean_bias
    
    # Collect sync gradient metrics if they exist
    if hasattr(model, 'syncs'):
        for i, sync in enumerate(model.syncs):
            if sync.grad is not None:
                gradient_metrics[f'sync_{i}'] = {
                    'min': sync.grad.min().item(),
                    'max': sync.grad.max().item()
                }
                
                # Add relative metrics if syncs are not zero
                if sync.data.mean().item() != 0:
                    mean_sync = sync.data.mean().item()
                    gradient_metrics[f'sync_{i}']['rel_min'] = sync.grad.min().item() / mean_sync
                    gradient_metrics[f'sync_{i}']['rel_max'] = sync.grad.max().item() / mean_sync
    
    return gradient_metrics

def get_binarization_metrics(model, phases, phase_type="free"):
    """Calculate binarization metrics for all layers.
    
    Args:
        model: The network model containing the activation function
        phases: Dictionary of phase tensors for each layer
        phase_type: Type of phase (free, clamped, etc.)
    
    Returns:
        Dictionary of binarization metrics for each layer organized by phase_type
    """
    binarization_metrics = {phase_type: {}}
    
    # Calculate metrics for each layer's phases
    for i in range(len(phases)):
        layer_phase = phases[i]
        
        # Convert phases to activations using the model's activation function
        activations = model.activation(layer_phase)
        
        # Calculate metrics
        mean_distance = torch.mean(torch.abs(torch.abs(activations) - 1)).item()
        max_distance = torch.max(torch.abs(torch.abs(activations) - 1)).item()
        std_dev = torch.std(activations).item()
        strongly_binary_fraction = torch.mean((torch.abs(activations) > 0.9).float()).item()
        
        # Store metrics for this layer
        binarization_metrics[phase_type][f'layer_{i}'] = {
            'mean_distance_from_binary': mean_distance,
            'max_distance_from_binary': max_distance,
            'std_dev': std_dev,
            'strongly_binary_fraction': strongly_binary_fraction
        }
    
    return binarization_metrics

def get_convergence_metrics(velocities, phase_type="free"):
    """Calculate convergence metrics from velocities.
    
    Args:
        velocities: List of tensors containing velocity values for each layer
        phase_type: Type of phase (free, clamped, positive, negative, etc.)
    
    Returns:
        Dictionary of convergence metrics organized by phase_type
    """
    
    # Convert list of tensors to a single numpy array
    all_velocities = []
    for layer_velocity in velocities:
        # Convert tensor to numpy and flatten
        np_velocity = layer_velocity.detach().cpu().numpy()
        all_velocities.append(np_velocity.flatten())
    
    # Concatenate into a single array
    velocities_np = np.concatenate(all_velocities)
    
    # Calculate metrics
    abs_velocities = np.abs(velocities_np)
    squared_velocities = velocities_np ** 2
    
    metrics = {
        "max_velocity": np.max(abs_velocities),
        "mean_velocity": np.mean(abs_velocities),
        "rms_velocity": np.sqrt(np.mean(squared_velocities))
    }
    
    # Return metrics organized by phase
    return {phase_type: metrics}

def log_metrics_to_wandb(convergence_metrics, network_metrics, binarization_metrics, gradient_metrics):
    """Log all metrics to wandb.
    
    Args:
        convergence_metrics: Dictionary of convergence metrics for each phase
        network_metrics: Dictionary of network parameter metrics
        binarization_metrics: Dictionary of binarization metrics for each phase and layer
        gradient_metrics: Dictionary of gradient metrics
    """
    wandb_metrics = {}
    
    # Log convergence metrics
    for phase, metrics in convergence_metrics.items():
        for metric_name, values in metrics.items():
            wandb_metrics[f'convergence/{phase}/{metric_name}'] = values
    
    # Log network parameters
    for param_name, param_metrics in network_metrics.items():
        for metric_type, value in param_metrics.items():
            wandb_metrics[f'network/{param_name}_{metric_type}'] = value
    
    # Log binarization metrics
    for phase, layers in binarization_metrics.items():
        for layer, metrics in layers.items():
            for metric_name, value in metrics.items():
                wandb_metrics[f'binarization/{phase}/{layer}/{metric_name}'] = value
    
    # Log gradient metrics
    for key, metrics in gradient_metrics.items():
        # Parse the key to get parameter type and layer index
        parts = key.split('_')
        param_type = parts[0]  # 'weight', 'bias', or 'sync'
        layer_idx = parts[1]
        
        # Log each metric
        for metric_name, value in metrics.items():
            wandb_metrics[f'gradients/{param_type}/{layer_idx}/{metric_name}'] = value
    
    # Log to wandb
    wandb.log(wandb_metrics)

def print_network_metrics(network_metrics):
    """Print network metrics in a clean format."""
    print("\n # Network Parameters (min/max/mean/std):")
    
    # Identify unique layer indices from the keys
    layer_indices = set()
    for key in network_metrics.keys():
        # Extract layer index from keys like 'weights_0', 'bias_1', etc.
        parts = key.split('_')
        if len(parts) >= 2 and parts[-1].isdigit():
            layer_indices.add(int(parts[-1]))
    
    # Sort layer indices for consistent output
    layer_indices = sorted(layer_indices)
    
    # Print metrics for each layer
    for layer_idx in layer_indices:
        print(f"\nLayer {layer_idx}:")
        
        # Print weights metrics if they exist
        if f'weights_{layer_idx}' in network_metrics:
            w_metrics = network_metrics[f'weights_{layer_idx}']
            print(f"  Weights:  Min: {w_metrics['min']:.6f}, Max: {w_metrics['max']:.6f}, Mean: {w_metrics['mean']:.6f}, Std: {w_metrics['std']:.6f}")
        
        # Print bias metrics if they exist
        if f'bias_{layer_idx}' in network_metrics:
            b_metrics = network_metrics[f'bias_{layer_idx}']
            print(f"  Biases:   Min: {b_metrics['min']:.6f}, Max: {b_metrics['max']:.6f}, Mean: {b_metrics['mean']:.6f}, Std: {b_metrics['std']:.6f}")
        
        # Print sync metrics if they exist
        if f'sync_{layer_idx}' in network_metrics:
            s_metrics = network_metrics[f'sync_{layer_idx}']
            print(f"  Syncs:    Min: {s_metrics['min']:.6f}, Max: {s_metrics['max']:.6f}, Mean: {s_metrics['mean']:.6f}, Std: {s_metrics['std']:.6f}")


def print_convergence_metrics(convergence_metrics):
    """Print convergence metrics in a clean format for all phases."""
    print("\n # Convergence Metrics:")
    
    # Print metrics for each phase type
    for phase_type, metrics in convergence_metrics.items():
        print(f"\n{phase_type.capitalize()} Phase:")
        for metric_name, value in metrics.items():
            # Format the metric name for better readability
            formatted_name = metric_name.replace('_', ' ').title()
            print(f"  {formatted_name}: {value:.6f}")


def print_binarization_metrics(all_binarization_metrics):
    """Print binarization metrics in a clean format for all phases and layers."""
    print("\n # Binarization Metrics:")
    
    # Print metrics for each phase
    for phase_type, phase_metrics in all_binarization_metrics.items():
        print(f"\n{phase_type.capitalize()} Phase:")
        
        # Identify layer indices from the metrics dictionary
        layer_indices = set()
        for layer_name in phase_metrics.keys():
            # Extract layer index from keys like 'layer_0', 'layer_1', etc.
            parts = layer_name.split('_')
            if len(parts) >= 2 and parts[-1].isdigit():
                layer_indices.add(int(parts[-1]))
        
        # Sort layer indices for consistent output
        layer_indices = sorted(layer_indices)
        
        # Print metrics for each layer
        for layer_idx in layer_indices:
            layer_key = f'layer_{layer_idx}'
            if layer_key in phase_metrics:
                print(f"  Layer {layer_idx}:")
                metrics = phase_metrics[layer_key]
                for metric_name, value in metrics.items():
                    print(f"    {metric_name}: {value:.6f}")

def print_gradient_metrics(gradient_metrics):
    """Print gradient metrics in a clean format.
    
    Args:
        gradient_metrics: Dictionary containing gradient metrics information
    """
    print("\n # Gradient Metrics:")
    
    # Identify categories and indices
    categories = {
        'weight': [],
        'bias': [],
        'sync': []
    }
    
    for key in gradient_metrics.keys():
        parts = key.split('_')
        category = parts[0]
        idx = int(parts[1])
        if category in categories:
            categories[category].append(idx)
    
    # Print metrics for each category and layer
    for category, indices in categories.items():
        if indices:
            print(f"\n  {category.capitalize()} gradients:")
            for i in sorted(indices):
                metrics = gradient_metrics[f'{category}_{i}']
                
                # Basic gradient info
                print(f"    Layer {i+1}: Min: {metrics['min']:.6f}, Max: {metrics['max']:.6f}")
                
                # Relative metrics if available
                if 'rel_min' in metrics and 'rel_max' in metrics:
                    print(f"    Layer {i+1} (relative to mean): Min: {metrics['rel_min']:.6f}, Max: {metrics['rel_max']:.6f}")