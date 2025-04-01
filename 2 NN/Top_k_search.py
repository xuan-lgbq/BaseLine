import numpy as np
# import wandb

def Top_K_Search(recorded_steps_top_eigenvectors):
    
    sorted_steps = sorted(recorded_steps_top_eigenvectors.keys())  # Epoch
    num_steps = len(sorted_steps)
    
    if num_steps < 2:
        raise ValueError("At least two recorded steps are required for Top-K search.")
    
    # Initialize k
    k = recorded_steps_top_eigenvectors[sorted_steps[0]].shape[1]  # Number of columns
    # k = k_0
    tau = 1.95 # Tolerance
    
    for i in range(num_steps):
        current_step = sorted_steps[i]
        current = recorded_steps_top_eigenvectors[current_step][:, :k]   # Current top k eigenvectors
        
        if i > 0:
            previous_step = sorted_steps[i - 1]
            previous = recorded_steps_top_eigenvectors[previous_step][:,:k]  # Previous top k eigenvectors
            combined_vectors = np.concatenate((np.real(current), np.real(previous)), axis=1)  
            matrix = np.matmul(combined_vectors.T, combined_vectors)  
            eigenvalues = np.linalg.eigvalsh(matrix)
            sorted_eigenvalues = np.sort(eigenvalues)[::-1]   # Decreasing order, not bigger than 2
            
            count = 0 # Count the number of eigenvalues that are bigger than tau
            
            # Search next k
            for i in range(k):
                if sorted_eigenvalues[i] > tau:
                    count +=1
                else:
                    break
            
            k = count
        
    return k