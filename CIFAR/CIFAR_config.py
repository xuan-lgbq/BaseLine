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
np_seed = 0
torch_seed = 0

np.random.seed(np_seed)
torch.manual_seed(torch_seed)


# 超参数
config = {
    "np_seed": 42,
    "torch_seed": 42,
    "learning_rate": 0.001,
    "batch_size": 50, # 添加 Batch Size [Source: 1452 in 2405.16002v3.pdf]
    "steps": 25000,  #此处先暂定
    #调整记录步数以适应更长的训练
    "record_steps": [5000,10000,15000,20000],
    "input_dim": 3072, # CIFAR-10 images are 32x32x3 = 3072
    "flatten_dim": 2048, #32 × 8 × 8
    "output_dim": 10,
    "train_samples": 5000,
    "test_samples": 5000,
    "top_k_pca_number": 10, 
    "wandb_project_name": "CIFAR",
    # Updated run name to reflect setup
    "wandb_run_name": "batch-CNN+ReLU+SGD+MSE+CIFAR10-5K(tiny space set up)"
}
