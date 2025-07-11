import torch
import numpy as np
from Top_k_Dom_search import search_top_k_dominant_space
from utilities import get_hessian_eigenvalues 


def compute_and_analyze_hessian(model, loss_function, dataset_for_hessian, neigs, device, config, step):
    """
    Computes and analyzes Hessian eigenvalues and eigenvectors, and related metrics.
    This function is designed to be called within the training loop.

    Args:
        model (torch.nn.Module): The neural network model.
        loss_function (torch.nn.Module): The loss function.
        dataset_for_hessian (torch.utils.data.Dataset): The full dataset or a subset used for Hessian computation.
        neigs (int): The number of top Hessian eigenvalues to compute.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') where the model and data reside.
        config (dict): A dictionary containing configuration items like 'physical_batch_size' and 'method'.
        step (int): The current training step, used for logging.

    Returns:
        tuple: (eigenvalues, eigenvectors_dominant, dominant_dim, gaps, success_flag)
               If computation is successful, success_flag is True, otherwise False.
               eigenvalues: All computed eigenvalues (CPU Tensor).
               eigenvectors_dominant: Eigenvectors spanning the dominant subspace (CPU Tensor).
               dominant_dim: The identified dominant dimension.
               gaps: The eigenvalue gaps.
    """
    
    # Ensure the model is on the correct device (though Hessian computation handles data device internally)
    model.to(device)

    try:
        eigenvalues, evec = get_hessian_eigenvalues(
            network=model,
            loss_fn=loss_function,
            dataset=dataset_for_hessian, # Pass the Dataset object
            neigs=neigs, # Use the provided neigs parameter
            # Get physical_batch_size from config, providing a default value
            physical_batch_size=config.get("physical_batch_size", 1000) 
        )
        
        print(f"📊 Step {step}: Hessian Computation Results Verification")
        print(f"   Eigenvalues Data Type: {type(eigenvalues)}, Shape: {eigenvalues.shape}")
        print(f"   Eigenvectors Data Type: {type(evec)}, Shape: {evec.shape}")
        print(f"   Eigenvalues Device: {eigenvalues.device}") 
        print(f"   Eigenvectors Device: {evec.device}") # Note: evec is also on CPU
        
        # lanczos from utilities.py already ensures descending order
        is_descending = torch.all(eigenvalues[:-1] >= eigenvalues[1:])
        print(f"   Is Descending: {is_descending}")
        print(f"   Top 5 Eigenvalues: {eigenvalues[:5].tolist()}") # Convert directly to list for printing

        # Output from utilities.py is already a CPU tensor, can convert directly to list()
        eigenvalues_cpu_list = eigenvalues.tolist()
        print(f"🔧 Converted to CPU list (for dominant space search): {eigenvalues_cpu_list[:5]}")
        
        # Compute the dominant dimension
        # Use config["method"] directly as it's confirmed to exist
        dominant_dim, gaps = search_top_k_dominant_space(eigenvalues_cpu_list, method=config["method"])
        
        # Take the first dominant_dim eigenvectors
        # Note: evec is a (params_dim, neigs) Tensor
        eigenvectors_dominant = evec[:, :dominant_dim]
        # eigenvalues_dominant_for_print is just for printing, actual return is full eigenvalues
        eigenvalues_dominant_for_print = eigenvalues[:dominant_dim] 
        
        print(f"📊 Step {step}: Computation Successful, Dominant Dimension: {dominant_dim}")
        print(f"📊 Step {step}: Top {dominant_dim} Dominant Eigenvalues: {eigenvalues_dominant_for_print.tolist()}")
        print(f"📊 Step {step}: Dominant Eigenvectors Matrix Shape: {eigenvectors_dominant.shape}")
        print(f"📊 Step {step}: Top 5 Gaps: {gaps[:5] if len(gaps) >= 5 else gaps}")
        
        return eigenvalues, eigenvectors_dominant, dominant_dim, gaps, True
        
    except Exception as e:
        print(f"⚠️  Step {step}: Failed to compute Hessian eigenvalues: {e}")
        import traceback
        print(f"Error Details: {traceback.format_exc()}")
        return None, None, None, None, False


def collect_eigenvalue_data(eigenvalue_history, eigenvalues):
    """Collects eigenvalue data."""
    for i, eigenval in enumerate(eigenvalues):
        # Output from utilities.py is already a CPU tensor, convert directly to item()
        raw_eigenval = eigenval.item()
        eigenvalue_history[f"top_{i+1}"].append(raw_eigenval)

def prepare_wandb_log(eigenvalues, dominant_dim, gaps):
    """Prepares data for Weights & Biases logging."""
    hessian_eigenvals = {}
    for i, eigenval in enumerate(eigenvalues):
        # Output from utilities.py is already a CPU tensor, convert directly to item()
        raw_eigenval = eigenval.item()
        hessian_eigenvals[f"Raw_Hessian_Eigenvalues/Top_{i+1}"] = raw_eigenval
    
    # Gaps come from search_top_k_dominant_space, typically a Python list or numpy array
    if isinstance(gaps, np.ndarray):
        gaps_list = gaps.tolist()
    elif isinstance(gaps, list):
        gaps_list = gaps
    else: # Attempt to convert to list of floats, just in case
        try:
            gaps_list = [float(g) for g in gaps]
        except TypeError: 
            gaps_list = [] # If not iterable or convertible, set to empty list
    
    wandb_data = {
        "Dominant Dimension": dominant_dim,
        "Gaps": gaps_list[:5] if len(gaps_list) >= 5 else gaps_list
    }
    wandb_data.update(hessian_eigenvals)
    
    return wandb_data