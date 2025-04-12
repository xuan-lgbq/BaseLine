# config.py
import torch
import numpy as np
import wandb

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


# 超参数
config = {
    "np_seed": 12138,
    "torch_seed": 12138,
    "learning_rate": 0.1,
    "batch_size": 50, # 添加 Batch Size [Source: 1452 in 2405.16002v3.pdf]
    "steps": 20000, 
    #调整记录步数以适应更长的训练
    "record_steps": [100, 300, 500, 1000, 1500, 2500, 4000, 5000],
    "input_dim": 784,
    "hidden_dim": 200,
    "output_dim": 10,
    "train_samples": 5000,
    "test_samples": 5000,
    "top_k_pca_number": 10, # 论文中通常取 k=类别数=10 或 2，这里保持 5 或改为 10
    "wandb_project_name": "MNIST",
    # Updated run name to reflect setup
    "wandb_run_name": "3 NN MLP+Tanh+SGD+MSE+MNIST-5k(tiny space set up)(every 50 points)"
}