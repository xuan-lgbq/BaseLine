import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pyhessian import hessian

def compute_full_hessian(model, criterion, data_loader, device):
    """计算完整的Hessian矩阵"""
    hessian_comp = hessian(model, criterion, data=data_loader, cuda=device.type=='cuda')
    H = hessian_comp.hessian()
    return H

def plot_hessian_heatmap(hessian_matrix, title, save_path, figsize=(12, 10)):
    """绘制Hessian矩阵的热力图"""
    # 取绝对值
    hessian_abs = torch.abs(hessian_matrix).cpu().numpy()
    
    plt.figure(figsize=figsize)
    
    # 使用对数尺度来更好地显示不同量级的值
    hessian_log = np.log10(hessian_abs + 1e-10)  # 避免log(0)
    
    sns.heatmap(hessian_log, 
                cmap='viridis', 
                cbar=True,
                cbar_kws={'label': 'log10(|Hessian|)'})
    
    plt.title(f'{title}\n矩阵大小: {hessian_matrix.shape}', fontsize=14)
    plt.xlabel('参数索引', fontsize=12)
    plt.ylabel('参数索引', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Hessian热力图已保存: {save_path}")
    plt.close()

def plot_layerwise_hessian(model, hessian_matrix, param_boundaries, save_dir, prefix=""):
    """绘制分层的Hessian热力图"""
    layer_names = []
    layer_ranges = []
    
    # 获取层信息
    start_idx = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            layer_names.append(name)
            layer_ranges.append((start_idx, start_idx + param_count))
            start_idx += param_count
    
    # 为每一层绘制对应的Hessian块
    for i, (layer_name, (start, end)) in enumerate(zip(layer_names, layer_ranges)):
        # 提取该层对应的Hessian子矩阵
        layer_hessian = hessian_matrix[start:end, start:end]
        
        save_path = os.path.join(save_dir, f"{prefix}hessian_layer_{i+1}_{layer_name.replace('.', '_')}.png")
        plot_hessian_heatmap(
            layer_hessian, 
            f"第{i+1}层 Hessian矩阵 ({layer_name})", 
            save_path,
            figsize=(8, 8)
        )

def visualize_hessian_structure(model, criterion, data_loader, device, save_dir, prefix=""):
    """完整的Hessian结构可视化"""
    print("🔍 开始计算完整Hessian矩阵...")
    
    try:
        # 计算完整Hessian
        H = compute_full_hessian(model, criterion, data_loader, device)
        
        # 整体Hessian热力图
        full_heatmap_path = os.path.join(save_dir, f"{prefix}hessian_full_matrix.png")
        plot_hessian_heatmap(H, "完整模型Hessian矩阵", full_heatmap_path, figsize=(15, 15))
        
        # 获取参数边界
        param_boundaries = get_parameter_boundaries(model)
        
        # 分层Hessian热力图
        plot_layerwise_hessian(model, H, param_boundaries, save_dir, prefix)
        
        print("✅ Hessian可视化完成!")
        return H
        
    except Exception as e:
        print(f"❌ Hessian计算失败: {e}")
        return None

def get_parameter_boundaries(model):
    """获取每层参数在展平向量中的边界"""
    boundaries = []
    start_idx = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            boundaries.append({
                'name': name,
                'start': start_idx,
                'end': start_idx + param_count,
                'shape': param.shape,
                'count': param_count
            })
            start_idx += param_count
    
    return boundaries