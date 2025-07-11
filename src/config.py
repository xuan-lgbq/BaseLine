import torch
import numpy as np
import os

os.environ["SWANLAB_API_KEY"] = "RqRRX6B5cHPegbienmzua"

# 定义设备
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# 随机种子
np_seed = 12138
torch_seed = 12138

np.random.seed(np_seed)
torch.manual_seed(torch_seed)

num_layer = 2

top_k = 160

training_config = {
    # Dataset and Architecture
    "dataset": "mnist",         # Which dataset to train on (e.g., "cifar10", "mnist")
    "arch_id": f"fc-relu-depth{num_layer-1}",       

    "loss": "mse",                 # Which loss function to use ("ce" for cross-entropy, "mse" for mean squared error, "logtanh")
    "opt": "sgd",                 # Which optimization method to use ("sgd", "sam", "adam")
    "lr": 0.01,                   # The learning rate

    "num_layer": num_layer,
    "top_k": top_k,
            # Training Parameters
    "max_steps": 10000,           # The maximum number of gradient steps to train for
    "batch_size": 1000,           # Batch size for SGD
    "physical_batch_size": 1000,  # The maximum number of examples to fit on the GPU at once (for gradient computation)
    "seed": 12138,      
    "method": "log_gap",
    "device": str(device),
    "k_subclasses": 10,

    "beta": 0,               
    "rho": 0,                

    "loss_goal": None,             
    "acc_goal": None,          

    # Hessian Eigenvalue Computation
    "neigs": top_k,                  # The number of top Hessian eigenvalues to compute
    "neigs_dom": top_k,               # The number of dominant top eigenvalues (used for effective LR calculation)
    "eig_freq": 10,              # The frequency (in steps) at which to compute top Hessian eigenvalues (-1 means never)

    # Saving and Output
    "save_freq": 10,          
    "save_model": True,          
    "gpu_id": 0,   

     # Swanlab configuration
    "swanlab_project_name": "MNIST_Baseline",
    "swanlab_run_name": f" {num_layer}NN-topk{top_k}-layer{num_layer}",
    "swanlab_api_key": "RqRRX6B5cHPegbienmzua"             
}