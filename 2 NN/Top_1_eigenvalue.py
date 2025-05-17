import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import wandb
from traditional_config import config, device

from traditional_model import LinearNetwork
from traditional_data_utils import generate_low_rank_identity, generative_dataset
from sam import SAM
import matplotlib.pyplot as plt  # Add matplotlib for plotting
import sys
import os
import argparse
from hessian_utils import compute_hessian_eigenvalues_pyhessian, compute_hessian_eigen, compute_max_eigenvalue
from torch.utils.data import TensorDataset, DataLoader
from pyhessian.hessian import hessian

# Set arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with different optimizers")
    
    # Use standard optimizer (SGD, Adam, etc.)
    parser.add_argument('--use_optimizer', action='store_true', help='使用标准优化器')
    parser.add_argument('--use_sam', action='store_true', help='使用SAM优化器')
    parser.add_argument('--use_adam', action='store_true', help='使用Adam优化器')
    parser.add_argument('--threshold', type=float, default=0.98, help='子空间相似度阈值')
    parser.add_argument('--rho', type=float, default=0.05, help='SAM的rho参数')
    return parser.parse_args()

# Parse arguments
args = parse_args()

# 初始化模型和优化器
model = LinearNetwork(config["input_dim"], config["hidden_dim"], config["output_dim"], device).to(device)

# Print the number of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# Choose optimizer based on arguments
if args.use_sam:
    # Use SAM optimizer
    base_optimizer = torch.optim.SGD # Defined for updating "sharpness-aware" 
    optimizer = SAM(model.parameters(), base_optimizer, lr=config["learning_rate"], momentum=0.9, rho=args.rho, adaptive=False)
    method = "SAM"
    print("Using SAM optimizer")
elif args.use_adam:
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    method = "Adam"
    print("Using Adam optimizer")
elif args.use_optimizer:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    method = "SGD"
    print("Using SGD optimizer")
else:
    # Default optimizer (you can choose a fallback)
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
    


wandb.login(key="19f26ee33b3dd19e282387aa75e310e4b07df17a")
# 初始化 wandb
wandb.init(project=config["wandb_project_name"], name=config["wandb_run_name"])

# 使用 config 字典更新 wandb.config
wandb.config.update(config)

loss_function = nn.MSELoss(reduction='mean')

# 训练过程
steps = config["steps"]
selected_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500]
max_hessian_eigenvalues = {}  # New dictionary for high-precision Hessian max eigenvalue
# max_hessian_eiggenvalues_manual = {}  # New dictionary for high-precision Hessian max eigenvalue
max_eigenvalues_weight = {}
loss_values = {}

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seeds = [42, 100, 500, 2000, 12138]
variance = 5.3045
seed = 100
# for seed in seeds:
# 设置随机种子
# set_seed(seed)

# Generate fake data
data, label = generative_dataset(config["input_dim"], config["output_dim"])
# data, label 的原始形状就是 (input_dim, input_dim) 和 (output_dim, input_dim)
# 只在最外层加一个 batch 维度
data = data.unsqueeze(0).to(device)   # 变为 (1, input_dim, input_dim)
label = label.unsqueeze(0).to(device) # 变为 (1, output_dim, input_dim)

single_loader = DataLoader(
        TensorDataset(data, label),
        batch_size=1,
        shuffle=False
    )

# 获取每个参数的名称和值
param_list = list(model.named_parameters())  # named_parameters() 返回的是 (name, parameter) 元组


for step in range(steps + 1):
    output = model.forward(data)
    loss = loss_function(output, label)
    loss_values[step] = loss.item()  # Store the loss value for this step
    wandb.log({"loss": loss.item()}, step=step)
    
    if args.use_sam:
        # First forward-backward pass
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # Second forward-backward pass
        output = model(data)
        loss = loss_function(output, label)
        loss.backward()

        optimizer.second_step(zero_grad=True)

    elif args.use_adam:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    elif args.use_optimizer:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    else:
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
        with torch.no_grad():
            for param, grad in zip(model.parameters(), grads):
                param.data.copy_(param - 0.01 * grad)  # 避免 in-place 修改
    
    print(f"Step {step}: Loss = {loss.item()}")
    
    if step in selected_steps:
        # New: Compute high-precision Hessian maximum eigenvalue at selected steps
        hessian_eigenvalues = compute_hessian_eigenvalues_pyhessian(
            model=model,
            criterion=loss_function,   # 这是计算 loss 的函数
            data_loader=single_loader, # 这是一个 DataLoader
            top_k=1,
            device=device
        )
        max_ev = hessian_eigenvalues[0].item()  # assuming the first eigenvalue is the maximum
        max_hessian_eigenvalues[step] = max_ev
        
        for name, param in param_list:
            # print(f"Parameter name: {name}, Parameter value: {param}")
            top_eigenvalue = compute_max_eigenvalue(param)
            # 确保字典结构正确，避免 KeyError
            if name not in max_eigenvalues_weight:
                max_eigenvalues_weight[name] = {}  # 如果没有该权重名，则初始化为一个空字典
            max_eigenvalues_weight[name][step] = top_eigenvalue # assuming the first eigenvalue is the maximum
        
        # Manual computation of Hessian eigenvalues
        # output = model.forward(data)
        # loss = loss_function(output, label)
        # eigenvalues, _ = compute_hessian_eigen(
        #     loss,
        #     model.parameters(),
        #     top_k=1
        # )
        # max_hessian_eiggenvalues_manual[step] = eigenvalues[0].item()  # assuming the first eigenvalue is the maximum
        
        log_dict = {
            f"max_hessian_auto_{method}_{seed}_{variance}": max_hessian_eigenvalues[step],
            
        }
    
        for name, _ in param_list:
            log_dict[f"max_egenvalue_{name}_{method}_{seed}_{variance}"] = max_eigenvalues_weight[name][step]
        wandb.log(log_dict, step=step)
            

    # After training loop, plot and log the Hessian max eigenvalue curve
    # steps_sorted = sorted(max_hessian_eigenvalues.keys())
    # values = [max_hessian_eigenvalues[s] for s in steps_sorted]
    # fig, ax = plt.subplots()
    # ax.plot(steps_sorted, values, marker='o')
    # ax.set_xlabel("Step")
    # ax.set_ylabel("Max Hessian Eigenvalue")
    # ax.set_title(f"Hessian Maximum Eigenvalue over Steps using {method}")
    # ax.grid(True)
    # wandb.log({f"Hessian_max_eigenvalue_curve_{method}": wandb.Image(fig)})
    # plt.close(fig)

    # steps_sorted = sorted(max_hessian_eiggenvalues_manual.keys())
    # values = [max_hessian_eiggenvalues_manual[s] for s in steps_sorted]
    # fig, ax = plt.subplots()
    # ax.plot(steps_sorted, values, marker='o')
    # ax.set_xlabel("Step")
    # ax.set_ylabel("Max Hessian Eigenvalue Manual")
    # ax.set_title(f"Hessian Maximum Eigenvalue over Steps using {method} manual")
    # ax.grid(True)
    # wandb.log({f"Hessian_max_eigenvalue_curve_{method}_manual": wandb.Image(fig)})
    # plt.close(fig)

# 完成 wandb 运行
wandb.finish()