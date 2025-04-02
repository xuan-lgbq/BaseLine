import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import svd
import torch.nn.functional as F
from MNIST_data_utils import generate_target_matrix
import wandb

"""
def balanced_init(input_dim, hidden_dim, output_dim, device):
    # using SVD method
    dims = [input_dim, hidden_dim, output_dim]
    d0, dN = dims[0], dims[-1]
    min_d = min(d0, dN)

    # Step 1: 采样 A
    A = np.random.randn(dN, d0)

    # Step 2: SVD 分解
    U, Sigma, Vt = svd(A, full_matrices=False)

    Sigma_power = np.power(np.diag(Sigma[:min_d]), 1 / (len(dims) - 1))

    # Step 4: 计算权重
    W1 = torch.from_numpy(Sigma_power @ Vt[:min_d, :]).float().to(device) # W1 ≃ Σ^(1/N) V^T
    W2 = torch.from_numpy(U[:, :min_d] @ Sigma_power).float().to(device)
    
    return W1, W2
"""

class LinearNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device):
        super(LinearNetwork, self).__init__()

        self.W1 = nn.Parameter(torch.randn(hidden_dim, input_dim).float().to(device))
        self.W2 = nn.Parameter(torch.randn(output_dim, hidden_dim).float().to(device))
        nn.init.xavier_normal_(self.W1)
        nn.init.xavier_normal_(self.W2)


    def forward(self, x = None):
        x = torch.matmul(self.W1, x.T)
        #x = F.relu(x)
        x = torch.matmul(self.W2, x)
        return x.T

def test_model(model, X_test, Y_test, loss_fn, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        y_predict = model.forward(X_test)
        _, predicted = torch.max(y_predict, 1)
        predicted = predicted.to(device) 
        loss = loss_fn(y_predict, Y_test)
        total_loss += loss.item() * X_test.size(0)
        total_samples += X_test.size(0)

        
        # Compare predictions with true labels
        correct_predictions += (predicted == Y_test).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    wandb.log({"test_loss": avg_loss, "test_accuracy": accuracy})
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
