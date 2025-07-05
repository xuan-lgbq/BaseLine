import torch
import torch.nn as nn
import numpy as np
import swanlab as wandb
import time
from datetime import datetime
from tqdm import tqdm
import os
from torch.utils.data import TensorDataset, DataLoader

# Configuration imports
from config_linear import training_config as config
from config_linear import device

# Model imports
from model_linear import LinearNetwork
from generate_low_rank import generate_low_rank_matrix, generative_dataset

# Custom modules
from training_utils import (
    set_seed, format_time, generate_eigenvalue_steps, 
    prepare_data_dimensions, print_training_info
)
from eigenvalue_analysis import (
    compute_and_analyze_eigenvalues, collect_eigenvalue_data, 
    prepare_wandb_log
)
from plotting import (
    plot_training_loss, plot_top_k_eigenvalues, 
    plot_dominant_dims, save_eigenvalue_csv
)

# 创建图片保存文件夹
IMAGE_SAVE_DIR = "/data/jiangjunmin/ouyangzhuoli/My-Baseline/Experiment_11_visualization/images"
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
print(f"📁 图片将保存到: {IMAGE_SAVE_DIR}")

def main():
    # 初始化模型
    model = LinearNetwork(config["input_dim"], config["hidden_dim"], config["output_dim"], config["num_layer"], config["variance"], device).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    # Setup SGD optimizer (simplified)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
    method = "SGD"
    print("Using SGD optimizer")
    
    # Get configuration parameters
    lr = config["learning_rate"]
    var = model.var
    top_k = config["top_k_pca_number"]
    rank = config.get("rank", 5)
    num_layer = config.get("num_layer", 3)
    output_dim = config.get("output_dim", 10)
    eigenvalue_interval = config.get("eigenvalue_interval", 25)
    steps = config["steps"]
    
    print(f"Learning rate: {lr}")
    print(f"Variance: {var}")
    print(f"Computing top {top_k} Hessian eigenvalues")
    print(f"Rank: {rank}")
    print(f"特征值计算间隔: 每 {eigenvalue_interval} 步")
    
    # Generate eigenvalue computation steps
    selected_steps = generate_eigenvalue_steps(steps, eigenvalue_interval)
    print(f"特征值计算步骤: {len(selected_steps)} 个计算点")
    
    # Initialize wandb
    gap_method = config["method"]
    wandb.init(project=config["swanlab_project_name"], 
               name=f"Linear{num_layer}+{gap_method}+{method}+outputdim{output_dim}+lr{lr}+var{var:.6f}_rank{rank}_top{top_k}", 
               api_key="zrVzavwSxtY7Gs0GWo9xV")
    
    wandb.config.update(config)
    wandb.config.update({
        "optimizer": method,
        "eigenvalue_interval": eigenvalue_interval,
        "target_rank": rank
    })
    
    loss_function = nn.MSELoss(reduction='mean')
    
    # Training setup
    seed = 12138
    total_start_time = time.time()
    print_training_info(steps, [seed], selected_steps)
    
    # Set seed
    set_seed(seed)
    print(f"\n🌱 Using seed: {seed}")
    
    # Initialize data collection variables
    eigenvalue_history = {f"top_{i+1}": [] for i in range(top_k)}
    step_history = []
    dominant_dim_history = []
    eigenvalue_step_history = []
    loss_history = []
    
    # Generate and prepare data
    data, label = generative_dataset(config["input_dim"], config["output_dim"], use_custom_rank=True)
    data, label = prepare_data_dimensions(data, label, device)
    
    # Print low rank matrix info
    projection_matrix = generate_low_rank_matrix(config["input_dim"], config["output_dim"])
    print(f"\n📊 Low Rank Matrix (rank={config['rank']}):")
    print(f"矩阵形状: {projection_matrix.shape}")
    print(f"实际秩: {torch.linalg.matrix_rank(projection_matrix)}")
    
    single_loader = DataLoader(TensorDataset(data, label), batch_size=1, shuffle=False)
    
    # Training loop
    progress_bar = tqdm(range(steps + 1), 
                       desc="Training", 
                       ncols=100)

    for step in progress_bar:
        step_start_time = time.time()
        
        # Forward pass
        output = model.forward(data)
        
        # Ensure dimension matching
        if output.shape != label.shape:
            if output.dim() == 2 and label.dim() == 3:
                output = output.unsqueeze(0)
            elif output.dim() == 3 and label.dim() == 2:
                label = label.unsqueeze(0)
            if output.shape != label.shape:
                output = output.view(label.shape)
        
        loss = loss_function(output, label)
        
        # Collect loss data
        step_history.append(step)
        loss_history.append(loss.item())
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Prepare log dictionary
        log_dict = {"Training Loss": loss.item()}
        
        # Eigenvalue computation
        hessian_time = 0
        if step in selected_steps:
            hessian_start = time.time()
            eigenvalue_step_history.append(step)
            
            eigenvalues, dominant_dim, gaps, success = compute_and_analyze_eigenvalues(
                model, loss_function, single_loader, top_k, device, config, step
            )
            
            if success:
                # Collect data
                dominant_dim_history.append(dominant_dim)
                collect_eigenvalue_data(eigenvalue_history, eigenvalues)
                
                # Prepare wandb log
                wandb_data = prepare_wandb_log(eigenvalues, dominant_dim, gaps)
                log_dict.update(wandb_data)
                log_dict[f"max_hessian_{method}"] = eigenvalues[0].item()
            
            hessian_time = time.time() - hessian_start
        
        # Log to wandb
        wandb.log(log_dict, step=step)
        
        # Update progress bar
        step_time = time.time() - step_start_time
        postfix_dict = {'Loss': f'{loss.item():.4f}'}
        if step in selected_steps:
            postfix_dict['EigenTime'] = f'{hessian_time:.1f}s'
        progress_bar.set_postfix(postfix_dict)

    progress_bar.close()
    print(f"\n✅ 训练完成!")
    
    # Generate visualizations
    print(f"\n🎨 生成可视化图表...")
    
    try:
        if len(loss_history) > 0:
            plot_training_loss(loss_history, step_history, method, lr, var, seed, rank, num_layer, config, IMAGE_SAVE_DIR, wandb)
        
        if len(eigenvalue_step_history) > 0:
            plot_top_k_eigenvalues(eigenvalue_history, eigenvalue_step_history, top_k, method, lr, var, seed, rank, num_layer, config, IMAGE_SAVE_DIR, wandb)
        
        if len(dominant_dim_history) > 0:
            plot_dominant_dims(dominant_dim_history, eigenvalue_step_history, top_k, method, lr, var, seed, rank, num_layer, config, IMAGE_SAVE_DIR, wandb)
            
        if len(eigenvalue_step_history) > 0 and eigenvalue_history:
            save_eigenvalue_csv(eigenvalue_history, eigenvalue_step_history, method, lr, var, seed, rank, num_layer, config, IMAGE_SAVE_DIR)

        print(f"✅ 可视化完成!")
        
    except Exception as e:
        print(f"⚠️  生成图表时出错: {e}")

    # Complete training
    total_time = time.time() - total_start_time
    print(f"\n🎉 训练完成! 总耗时: {format_time(total_time)}")
    print(f"📁 图片保存到: {IMAGE_SAVE_DIR}")
    
    wandb.finish()

if __name__ == "__main__":
    main()