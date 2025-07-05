import os
from datetime import datetime

def create_experiment_directories(config):
    """创建详细的实验目录结构"""
    
    # 基础信息
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 从config提取关键参数
    num_layer = config["num_layer"]
    method = config["method"]  # log_gap
    learning_rate = config["learning_rate"]
    variance = config["variance"]
    rank = config["rank"]
    input_dim = config["input_dim"]
    hidden_dim = config["hidden_dim"]
    output_dim = config["output_dim"]
    steps = config["steps"]
    top_k = config["top_k_pca_number"]
    seed = config["torch_seed"]
    eigenvalue_interval = config["eigenvalue_interval"]
    
    # 构建详细的目录名
    experiment_name = (
        f"Linear{num_layer}L_{method}_"
        f"dim{input_dim}x{hidden_dim}x{output_dim}_"
        f"lr{learning_rate}_var{variance:.4f}_"
        f"rank{rank}_steps{steps}_"
        f"topk{top_k}_interval{eigenvalue_interval}_"
        f"seed{seed}"
    )
    
    # 主实验目录
    main_dir = f"./experiments/{experiment_name}_{timestamp}"
    
    # 子目录结构
    directories = {
        'main': main_dir,
        'images': os.path.join(main_dir, "images"),
        'data': os.path.join(main_dir, "data"),
        'eigenvectors': os.path.join(main_dir, "eigenvectors"),
        'hessian': os.path.join(main_dir, "hessian"),
        'logs': os.path.join(main_dir, "logs"),
        'models': os.path.join(main_dir, "models")
    }
    
    # 创建所有目录
    for dir_name, dir_path in directories.items():
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 创建目录: {dir_path}")
    
    # 保存实验配置
    config_file = os.path.join(directories['logs'], "experiment_config.txt")
    with open(config_file, 'w') as f:
        f.write(f"实验配置 - {timestamp}\n")
        f.write("=" * 50 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"实验名称: {experiment_name}\n")
        f.write(f"主目录: {main_dir}\n")
    
    print(f"📋 实验配置已保存: {config_file}")
    
    return directories, experiment_name

def create_step_specific_paths(base_dirs, step, dominant_dim, config):
    """为特定step创建路径"""
    step_identifier = f"step{step:03d}_dom{dominant_dim}"
    
    paths = {
        'eigenvalue_data': os.path.join(base_dirs['data'], f"eigenvalues_{step_identifier}.pt"),
        'eigenvector_analysis': os.path.join(base_dirs['eigenvectors'], f"analysis_{step_identifier}.json"),
        'eigenvector_bar_chart': os.path.join(base_dirs['images'], f"eigvec_bar_{step_identifier}.png"),
        'eigenvector_stacked_chart': os.path.join(base_dirs['images'], f"eigvec_stacked_{step_identifier}.png"),
        'eigenvector_heatmap': os.path.join(base_dirs['images'], f"eigvec_heatmap_{step_identifier}.png"),
        'hessian_matrix': os.path.join(base_dirs['hessian'], f"hessian_{step_identifier}.npz"),
    }
    
    return paths

def save_experiment_summary(directories, config, experiment_name, results_summary):
    """保存实验总结"""
    summary_file = os.path.join(directories['logs'], "experiment_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write(f"实验总结 - {experiment_name}\n")
        f.write("=" * 80 + "\n\n")
        
        # 实验参数
        f.write("📋 实验参数:\n")
        f.write("-" * 40 + "\n")
        key_params = [
            'num_layer', 'method', 'learning_rate', 'variance', 'rank',
            'input_dim', 'hidden_dim', 'output_dim', 'steps', 'top_k_pca_number',
            'eigenvalue_interval', 'torch_seed'
        ]
        for key in key_params:
            if key in config:
                f.write(f"{key}: {config[key]}\n")
        
        f.write(f"\n📁 目录结构:\n")
        f.write("-" * 40 + "\n")
        for dir_name, dir_path in directories.items():
            f.write(f"{dir_name}: {dir_path}\n")
        
        f.write(f"\n📊 实验结果:\n")
        f.write("-" * 40 + "\n")
        for key, value in results_summary.items():
            f.write(f"{key}: {value}\n")
    
    print(f"📄 实验总结已保存: {summary_file}")