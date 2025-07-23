import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import time
from typing import Dict, List, Tuple, Optional

def create_analysis_directories(base_dir: str, step: int) -> Dict[str, str]:
    """
    创建分析结果的目录结构
    
    Args:
        base_dir: 基础目录
        step: 当前步数
    
    Returns:
        包含各种目录路径的字典
    """
    dirs = {
        'hessian_matrices': os.path.join(base_dir, 'hessian_matrices', f'step_{step}'),
        'eigenvalues': os.path.join(base_dir, 'eigenvalues'),
        'eigenvectors': os.path.join(base_dir, 'eigenvectors', f'step_{step}'),
        'heatmaps': os.path.join(base_dir, 'heatmaps', f'step_{step}'),
        'projections': os.path.join(base_dir, 'projections', f'step_{step}'),
        'analysis': os.path.join(base_dir, 'analysis')
    }
    
    # 创建所有目录
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def get_parameter_blocks(model) -> List[Tuple[str, torch.Tensor]]:
    """
    获取模型的参数分块信息
    
    Args:
        model: PyTorch模型
    
    Returns:
        参数块列表，每个元素包含(层名称, 参数张量)
    """
    blocks = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            blocks.append((name, param))
    return blocks

def compute_parameter_ranges(blocks: List[Tuple[str, torch.Tensor]]) -> Dict[str, Tuple[int, int]]:
    """
    计算每个参数块在展平参数向量中的范围
    
    Args:
        blocks: 参数块列表
    
    Returns:
        每个参数块的起始和结束位置
    """
    ranges = {}
    current_pos = 0
    
    for name, param in blocks:
        param_size = param.numel()
        ranges[name] = (current_pos, current_pos + param_size)
        current_pos += param_size
    
    return ranges

def compute_full_hessian_with_blocks(loss, params, device=None) -> Tuple[torch.Tensor, Dict]:
    """
    计算完整Hessian矩阵及其分块信息
    
    Args:
        loss: 损失函数值
        params: 模型参数
        device: 计算设备
    
    Returns:
        (完整Hessian矩阵, 分块信息字典)
    """
    if device is None:
        device = loss.device
    
    params = list(params)
    params = [p.to(device) if p.device != device else p for p in params]
    
    print(f"🔍 计算完整Hessian矩阵及分块信息...")
    
    # 获取参数块信息
    param_blocks = []
    param_ranges = {}
    current_pos = 0
    
    for i, p in enumerate(params):
        param_size = p.numel()
        block_name = f"layer_{i}"
        param_blocks.append((block_name, p))
        param_ranges[block_name] = (current_pos, current_pos + param_size)
        current_pos += param_size
    
    # 计算梯度
    grads_flat = []
    for p in params:
        g = torch.autograd.grad(loss, p, create_graph=True, retain_graph=True)[0]
        grads_flat.append(g.view(-1))
    
    grads_flat = torch.cat(grads_flat).to(device)
    total_params = grads_flat.size(0)
    
    # 构建Hessian矩阵
    hessian_matrix = torch.zeros((total_params, total_params), device=device, dtype=torch.float32)
    
    print(f"   总参数数: {total_params}")
    
    for i, g in enumerate(grads_flat):
        if i % max(1, total_params // 10) == 0:
            print(f"   进度: {i}/{total_params} ({100*i/total_params:.1f}%)")
        
        hessian_row = torch.autograd.grad(
            outputs=g, 
            inputs=params, 
            retain_graph=True, 
            allow_unused=True
        )
        
        hessian_row_flat = []
        for h, p in zip(hessian_row, params):
            if h is None:
                h_flat = torch.zeros_like(p, device=device).view(-1)
            else:
                h_flat = h.to(device).view(-1)
            hessian_row_flat.append(h_flat)
        
        hessian_matrix[i] = torch.cat(hessian_row_flat)
    
    print(f"✅ Hessian矩阵计算完成")
    
    return hessian_matrix, param_ranges

def extract_hessian_blocks(hessian_matrix: torch.Tensor, param_ranges: Dict[str, Tuple[int, int]]) -> Dict[str, torch.Tensor]:
    """
    从完整Hessian矩阵中提取各层的分块
    
    Args:
        hessian_matrix: 完整Hessian矩阵
        param_ranges: 参数范围字典
    
    Returns:
        各层Hessian分块的字典
    """
    blocks = {}
    layer_names = list(param_ranges.keys())
    
    print(f"📊 提取Hessian分块矩阵...")
    
    # 提取对角块（每层内部的Hessian）
    for layer_name, (start, end) in param_ranges.items():
        blocks[f"{layer_name}_diagonal"] = hessian_matrix[start:end, start:end].clone()
        print(f"   {layer_name} 对角块: {blocks[f'{layer_name}_diagonal'].shape}")
    
    # 提取非对角块（层间交互的Hessian）
    for i, layer1 in enumerate(layer_names):
        for j, layer2 in enumerate(layer_names):
            if i < j:  # 只保存上三角部分，避免重复
                start1, end1 = param_ranges[layer1]
                start2, end2 = param_ranges[layer2]
                block_name = f"{layer1}_to_{layer2}"
                blocks[block_name] = hessian_matrix[start1:end1, start2:end2].clone()
                print(f"   {block_name} 交互块: {blocks[block_name].shape}")
    
    return blocks

def plot_hessian_heatmaps(hessian_blocks: Dict[str, torch.Tensor], save_dir: str, step: int, wandb_logger=None):
    """
    绘制Hessian分块的热力图
    """
    print(f"🎨 绘制Hessian分块热力图...")
    
    # 强制设置matplotlib后端和字体
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    
    # 清除所有字体设置，使用系统默认
    plt.rcdefaults()
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    for block_name, block_matrix in hessian_blocks.items():
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # 转换为CPU并取绝对值
            matrix_abs = torch.abs(block_matrix).cpu().numpy()
            
            # 如果矩阵太大，进行下采样
            max_size = 50  # 减小尺寸避免内存问题
            if matrix_abs.shape[0] > max_size or matrix_abs.shape[1] > max_size:
                step_size = max(1, max(matrix_abs.shape) // max_size)
                matrix_abs = matrix_abs[::step_size, ::step_size]
            
            # 直接使用imshow绘制，避免seaborn
            im = ax.imshow(matrix_abs, cmap='Blues', aspect='auto')
            plt.colorbar(im, ax=ax)
            
            # 使用英文标题
            ax.set_title(f'Hessian Block: {block_name} Step {step}')
            ax.set_xlabel('Parameter Index')
            ax.set_ylabel('Parameter Index')
            
            # 保存图片
            save_path = os.path.join(save_dir, f'hessian_heatmap_{block_name}_step_{step}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            
            # 添加这几行：上传到SwanLab
            if wandb_logger:
                wandb_logger.log({
                    f"Hessian_Heatmaps/Block_{block_name}_Step_{step}": wandb_logger.Image(save_path)
                })
            
            plt.close()
            
            # 验证文件是否真的保存了
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                print(f"   ✅ 热力图保存成功: {save_path} ({file_size} bytes)")
            else:
                print(f"   ❌ 热力图保存失败: {save_path}")
                
        except Exception as e:
            print(f"   ❌ 绘制 {block_name} 热力图失败: {e}")
            continue

def plot_global_hessian_heatmap(hessian_matrix: torch.Tensor, save_dir: str, step: int, wandb_logger=None):
    """
    绘制完整的全局Hessian矩阵热力图 - 线性刻度，绝对值
    """
    print(f"🎨 绘制全局Hessian热力图...")
    
    # 强制设置matplotlib后端和字体
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # 清除所有字体设置
    plt.rcdefaults()
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    try:
        # 转换为CPU并取绝对值
        matrix_abs = torch.abs(hessian_matrix).cpu().numpy()
        
        print(f"   全局Hessian矩阵大小: {matrix_abs.shape}")
        
        # 如果矩阵太大，进行下采样
        max_size = 200  # 全局矩阵可以稍大一些
        if matrix_abs.shape[0] > max_size or matrix_abs.shape[1] > max_size:
            step_size = max(1, max(matrix_abs.shape) // max_size)
            matrix_abs = matrix_abs[::step_size, ::step_size]
            print(f"   下采样后大小: {matrix_abs.shape}")
        
        # 1. 绘制完整热力图 - 线性刻度，蓝色配色
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(matrix_abs, cmap='Blues', aspect='auto')
        plt.colorbar(im, ax=ax, label='|Hessian|')
        
        ax.set_title(f'Global Hessian Matrix (Linear Scale) Step {step}')
        ax.set_xlabel('Parameter Index')
        ax.set_ylabel('Parameter Index')
        
        # 保存完整热力图
        save_path = os.path.join(save_dir, f'global_hessian_full_step_{step}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        
        # 上传到SwanLab
        if wandb_logger:
            wandb_logger.log({
                f"Global_Hessian/Full_Matrix_Step_{step}": wandb_logger.Image(save_path)
            })
        
        plt.close()
        
        # 验证文件
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path)
            print(f"   ✅ 全局热力图保存成功: {save_path} ({file_size} bytes)")
        else:
            print(f"   ❌ 全局热力图保存失败: {save_path}")
        
        # 2. 绘制对比图（全图 vs 对角线区域）
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 左图：完整矩阵
        im1 = ax1.imshow(matrix_abs, cmap='Blues', aspect='auto')
        plt.colorbar(im1, ax=ax1, label='|Hessian|')
        ax1.set_title(f'Full Hessian Matrix Step {step}')
        ax1.set_xlabel('Parameter Index')
        ax1.set_ylabel('Parameter Index')
        
        # 右图：对角线区域放大
        diag_size = min(100, matrix_abs.shape[0] // 2)
        center = matrix_abs.shape[0] // 2
        start_idx = max(0, center - diag_size // 2)
        end_idx = min(matrix_abs.shape[0], center + diag_size // 2)
        
        diag_region = matrix_abs[start_idx:end_idx, start_idx:end_idx]
        im2 = ax2.imshow(diag_region, cmap='Blues', aspect='auto')
        plt.colorbar(im2, ax=ax2, label='|Hessian|')
        ax2.set_title(f'Central Diagonal Region Step {step}')
        ax2.set_xlabel('Parameter Index')
        ax2.set_ylabel('Parameter Index')
        
        # 保存对比图
        save_path_combined = os.path.join(save_dir, f'global_hessian_combined_step_{step}.png')
        plt.savefig(save_path_combined, dpi=150, bbox_inches='tight', facecolor='white')
        
        # 上传到SwanLab
        if wandb_logger:
            wandb_logger.log({
                f"Global_Hessian/Combined_View_Step_{step}": wandb_logger.Image(save_path_combined)
            })
        
        plt.close()
        
        # 验证文件
        if os.path.exists(save_path_combined):
            file_size = os.path.getsize(save_path_combined)
            print(f"   ✅ 组合热力图保存成功: {save_path_combined} ({file_size} bytes)")
        else:
            print(f"   ❌ 组合热力图保存失败: {save_path_combined}")
            
    except Exception as e:
        print(f"   ❌ 绘制全局Hessian热力图失败: {e}")
        import traceback
        traceback.print_exc()

def compute_eigenvalue_projections(eigenvectors: torch.Tensor, param_ranges: Dict[str, Tuple[int, int]], dominant_dim: int) -> Dict[str, torch.Tensor]:
    """
    计算特征向量在各层参数上的投影范数
    
    Args:
        eigenvectors: 特征向量矩阵
        param_ranges: 参数范围字典
        dominant_dim: dominant空间维度
    
    Returns:
        各层投影范数的字典
    """
    print(f"📊 计算特征向量在各层的投影范数...")
    
    projections = {}
    
    # 只计算dominant space的特征向量
    # dominant_eigenvectors = eigenvectors[:, :dominant_dim]
    dominant_eigenvectors = eigenvectors
    
    for layer_name, (start, end) in param_ranges.items():
        # 提取该层对应的特征向量分量
        layer_components = dominant_eigenvectors[start:end, :]
        
        # 计算每个特征向量在该层的范数
        layer_norms = torch.norm(layer_components, dim=0)  # shape: (dominant_dim,)
        projections[layer_name] = layer_norms
        
        print(f"   {layer_name}: {layer_norms.shape}")
    
    return projections

def save_projections_to_csv(projections: Dict[str, torch.Tensor], save_dir: str, step: int):
    """
    将dominant eigenvector projections保存为CSV格式 - 只保存长格式
    
    Args:
        projections: 投影范数字典
        save_dir: 保存目录
        step: 当前步数
    """
    print(f"📊 保存投影数据为CSV格式...")
    
    # 提取层数据并排序
    layer_data = []
    for layer_name, projection_tensor in projections.items():
        try:
            if 'layer_' in layer_name:
                layer_num = int(layer_name.split('_')[1])
                layer_data.append((layer_num, layer_name, projection_tensor.cpu().numpy()))
        except:
            continue
    
    if len(layer_data) == 0:
        print("   ⚠️ 没有有效的层投影数据")
        return
    
    # 按层数排序
    layer_data.sort(key=lambda x: x[0])
    
    # 获取dominant特征向量数量
    dominant_dim = layer_data[0][2].shape[0]
    
    # 保存长格式CSV文件
    try:
        # 准备数据
        rows = []
        for layer_num, layer_name, projections_array in layer_data:
            for eigenvec_idx in range(dominant_dim):
                rows.append({
                    'step': step,
                    'layer_number': layer_num,
                    'layer_name': layer_name,
                    'eigenvector_index': eigenvec_idx + 1,
                    'projection_norm': projections_array[eigenvec_idx]
                })
        
        # 创建DataFrame并保存
        df = pd.DataFrame(rows)
        csv_path = os.path.join(save_dir, f'projections_step_{step}.csv')
        df.to_csv(csv_path, index=False)
        print(f"   ✅ 投影数据CSV保存成功: {csv_path}")
        
    except Exception as e:
        print(f"   ❌ 保存投影CSV失败: {e}")
        
def plot_eigenvalue_projections(projections: Dict[str, torch.Tensor], save_dir: str, step: int, eigenvalues: torch.Tensor = None, wandb_logger=None):
    """
    绘制特征向量投影范数图 - 为每个dominant特征向量单独绘制
    """
    print(f"🎨 绘制特征向量投影范数图...")
    
    # 强制设置matplotlib后端和字体
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # 清除所有字体设置
    plt.rcdefaults()
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    layer_names = list(projections.keys())
    if len(layer_names) == 0:
        print("   ⚠️ 没有投影数据")
        return
    
    # 提取层数并排序
    layer_data = []
    for layer_name in layer_names:
        try:
            if 'layer_' in layer_name:
                layer_num = int(layer_name.split('_')[1])
                layer_data.append((layer_num, layer_name, projections[layer_name]))
        except:
            continue
    
    if len(layer_data) == 0:
        print("   ⚠️ 没有有效的层数据")
        return
    
    # 按层数排序
    layer_data.sort(key=lambda x: x[0])
    layer_numbers = [x[0] for x in layer_data]
    dominant_dim = layer_data[0][2].shape[0]  # 获取dominant维度数
    
    print(f"   发现 {dominant_dim} 个dominant特征向量")
    
    # 1. 为每个特征向量单独绘制
    for eigenvec_idx in range(dominant_dim):
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 收集这个特征向量在各层的投影
            projection_values = []
            for layer_num, layer_name, norms in layer_data:
                projection_values.append(norms[eigenvec_idx].item())
            
            # 绘制柱状图
            bars = ax.bar(layer_numbers, projection_values, alpha=0.7, color='skyblue')
            
            # 添加数值标签
            for bar, value in zip(bars, projection_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(projection_values)*0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_xlabel('Layer Number', fontsize=12)
            ax.set_ylabel('Projection Norm', fontsize=12)
            
            # 标题包含特征值信息
            if eigenvalues is not None and eigenvec_idx < len(eigenvalues):
                eigenval = eigenvalues[eigenvec_idx].item()
                ax.set_title(f'Eigenvector {eigenvec_idx+1} Projections (λ={eigenval:.4e}) Step {step}', fontsize=14)
            else:
                ax.set_title(f'Eigenvector {eigenvec_idx+1} Projections Step {step}', fontsize=14)
            
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_xticks(layer_numbers)
            
            # 保存单个特征向量图片
            save_path = os.path.join(save_dir, f'projection_eigenvec_{eigenvec_idx+1}_step_{step}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            
            # 上传到SwanLab
            if wandb_logger:
                wandb_logger.log({
                    f"Eigenvalue_Projections/Eigenvec_{eigenvec_idx+1}_Step_{step}": wandb_logger.Image(save_path)
                })
            
            plt.close()
            
            # 验证文件
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                print(f"   ✅ 特征向量{eigenvec_idx+1}投影图保存成功: {save_path} ({file_size} bytes)")
            else:
                print(f"   ❌ 特征向量{eigenvec_idx+1}投影图保存失败: {save_path}")
                
        except Exception as e:
            print(f"   ❌ 绘制特征向量{eigenvec_idx+1}投影图失败: {e}")
            continue
    
    # 2. 绘制所有特征向量的组合图
    try:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 设置颜色
        colors = plt.cm.tab10(np.linspace(0, 1, dominant_dim))
        bar_width = 0.8 / dominant_dim
        
        # 为每个特征向量绘制柱状图
        for eigenvec_idx in range(dominant_dim):
            projection_values = []
            for layer_num, layer_name, norms in layer_data:
                projection_values.append(norms[eigenvec_idx].item())
            
            # 计算每个特征向量的柱子位置偏移
            offset = (eigenvec_idx - dominant_dim/2 + 0.5) * bar_width
            x_positions = [x + offset for x in layer_numbers]
            
            # 准备标签
            if eigenvalues is not None and eigenvec_idx < len(eigenvalues):
                eigenval = eigenvalues[eigenvec_idx].item()
                label = f'Eigenvec {eigenvec_idx+1} (λ={eigenval:.2e})'
            else:
                label = f'Eigenvector {eigenvec_idx+1}'
            
            ax.bar(x_positions, projection_values, 
                  width=bar_width, label=label, 
                  alpha=0.8, color=colors[eigenvec_idx])
        
        ax.set_xlabel('Layer Number', fontsize=12)
        ax.set_ylabel('Projection Norm', fontsize=12)
        ax.set_title(f'All Dominant Eigenvectors Projections Comparison (Step {step})', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(layer_numbers)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 保存组合图
        save_path_combined = os.path.join(save_dir, f'projection_all_eigenvecs_step_{step}.png')
        plt.savefig(save_path_combined, dpi=150, bbox_inches='tight', facecolor='white')
        
        # 上传到SwanLab
        if wandb_logger:
            wandb_logger.log({
                f"Eigenvalue_Projections/All_Eigenvecs_Step_{step}": wandb_logger.Image(save_path_combined)
            })
        
        plt.close()
        
        # 验证文件
        if os.path.exists(save_path_combined):
            file_size = os.path.getsize(save_path_combined)
            print(f"   ✅ 所有特征向量组合投影图保存成功: {save_path_combined} ({file_size} bytes)")
        else:
            print(f"   ❌ 所有特征向量组合投影图保存失败: {save_path_combined}")
            
    except Exception as e:
        print(f"   ❌ 绘制所有特征向量组合投影图失败: {e}")
        import traceback
        traceback.print_exc()

def compute_trace_ratio(eigenvalues: torch.Tensor, hessian_matrix: torch.Tensor, config: dict) -> Dict[str, float]:
    """
    计算dominant特征值之和与trace的比值，并与理论值比较
    
    Args:
        eigenvalues: 特征值
        hessian_matrix: Hessian矩阵
        config: 配置字典
    
    Returns:
        包含比值和理论值的字典
    """
    print(f"📊 计算trace比值...")
    
    # 计算trace
    trace = torch.trace(hessian_matrix).item()
    
    # 获取配置参数
    rank = config.get('rank', 5)
    num_layers = config.get('num_layer', 3)
    input_dim = config.get('input_dim', 10)
    hidden_dim = config.get('hidden_dim', 10)
    
    # 计算dominant特征值之和
    dominant_eigenvalues_sum = torch.sum(eigenvalues).item()
    
    # 计算比值
    ratio = dominant_eigenvalues_sum / trace if trace != 0 else 0
    
    # 计算理论值: r^L / (d₀ + d_L - 1 - r²)
    # 假设d_L = hidden_dim (最后一层的维度)
    theoretical_ratio = 0 # (rank ** num_layers) / (input_dim + hidden_dim - 1 - rank**2)
    
    results = {
        'trace': trace,
        'dominant_sum': dominant_eigenvalues_sum,
        'ratio': ratio,
        'theoretical_ratio': theoretical_ratio,
        'difference': abs(ratio - theoretical_ratio)
    }
    
    print(f"   Trace: {trace:.6e}")
    print(f"   Dominant特征值之和: {dominant_eigenvalues_sum:.6e}")
    print(f"   实际比值: {ratio:.6f}")
    print(f"   理论比值: {theoretical_ratio:.6f}")
    print(f"   差异: {results['difference']:.6f}")
    
    return results

def compute_energy_ratio(eigenvalues: torch.Tensor, fnorm_squared: float) -> Dict[str, float]:
    """
    计算dominant space特征值平方和与Hessian F-norm平方的比值
    
    Args:
        eigenvalues: 特征值
        fnorm_squared: Hessian矩阵的F-norm平方
    
    Returns:
        包含能量比值的字典
    """
    print(f"📊 计算能量比值...")
    
    # 计算dominant特征值平方和
    eigenvalues_squared_sum = torch.sum(eigenvalues ** 2).item()
    
    # 计算比值
    energy_ratio = eigenvalues_squared_sum / fnorm_squared if fnorm_squared != 0 else 0
    
    results = {
        'eigenvalues_squared_sum': eigenvalues_squared_sum,
        'fnorm_squared': fnorm_squared,
        'energy_ratio': energy_ratio
    }
    
    print(f"   Dominant特征值平方和: {eigenvalues_squared_sum:.6e}")
    print(f"   Hessian F-norm²: {fnorm_squared:.6e}")
    print(f"   能量比值: {energy_ratio:.6f}")
    
    return results

def save_analysis_data(save_dirs: Dict[str, str], step: int, 
                      # hessian_matrix: torch.Tensor, 
                      # hessian_blocks: Dict[str, torch.Tensor],
                      # eigenvalues: torch.Tensor, 
                      # eigenvectors: torch.Tensor,
                      projections: Dict[str, torch.Tensor],
                      # trace_analysis: Dict[str, float],
                      # energy_analysis: Dict[str, float]
                      ):
    """
    保存所有分析数据
    """
    print(f"💾 保存分析数据...")
    
    # 1. 保存完整Hessian矩阵
    # hessian_path = os.path.join(save_dirs['hessian_matrices'], f'hessian_full_step_{step}.pt')
    # torch.save(hessian_matrix.cpu(), hessian_path)
    # print(f"   保存完整Hessian: {hessian_path}")
    
    # 2. 保存Hessian分块
    #for block_name, block_matrix in hessian_blocks.items():
    #    block_path = os.path.join(save_dirs['hessian_matrices'], f'hessian_block_{block_name}_step_{step}.pt')
    #    torch.save(block_matrix.cpu(), block_path)
    #    print(f"   保存Hessian分块: {block_path}")
    
    # 3. 保存特征值
    #eigenvals_path = os.path.join(save_dirs['eigenvalues'], f'eigenvalues_step_{step}.csv')
    #pd.DataFrame({'eigenvalue': eigenvalues.cpu().numpy()}).to_csv(eigenvals_path, index=False)
    #print(f"   保存特征值: {eigenvals_path}")
    
    # 4. 保存特征向量
    #eigenvecs_path = os.path.join(save_dirs['eigenvectors'], f'eigenvectors_step_{step}.pt')
    #torch.save(eigenvectors.cpu(), eigenvecs_path)
    #print(f"   保存特征向量: {eigenvecs_path}")
    
    # 5. 保存投影数据 (PyTorch格式)
    # projections_path = os.path.join(save_dirs['projections'], f'projections_step_{step}.pt')
    #torch.save({k: v.cpu() for k, v in projections.items()}, projections_path)
    #print(f"   保存投影数据: {projections_path}")
    
    # 6. 新增：保存投影数据为CSV格式（长格式）
    save_projections_to_csv(projections, save_dirs['projections'], step)
    
    # 7. 保存分析结果
    # analysis_data = {
    #     'step': step,
    #    'trace_analysis': trace_analysis,
    #    'energy_analysis': energy_analysis
    #}
    #analysis_path = os.path.join(save_dirs['analysis'], f'analysis_step_{step}.pt')
    #torch.save(analysis_data, analysis_path)
    #print(f"   保存分析结果: {analysis_path}")

class LayerWiseLRScheduler:
    """分层学习率调度器"""
    def __init__(self, optimizer, layer_lr_config: Dict[str, float]):
        self.optimizer = optimizer
        self.layer_lr_config = layer_lr_config
        
        # 设置每个参数组的学习率
        for param_group, (layer_name, lr) in zip(optimizer.param_groups, layer_lr_config.items()):
            param_group['lr'] = lr
            print(f"   设置{layer_name}学习率: {lr}")
    
    def step(self):
        """执行优化步骤"""
        self.optimizer.step()
    
    def zero_grad(self):
        """清零梯度"""
        self.optimizer.zero_grad()

def create_layerwise_optimizer(model, config: dict):
    """创建分层学习率优化器"""
    print(f"🔧 创建分层学习率优化器...")
    
    # 获取分层学习率配置
    layer_lr_config = config.get('layer_learning_rates', {})
    
    if not layer_lr_config:
        # 如果没有配置，使用统一学习率
        base_lr = config.get('learning_rate', 0.01)
        layer_lr_config = {f'layer_{i}': base_lr for i in range(len(list(model.parameters())))}
        print(f"   使用统一学习率: {base_lr}")
    
    # 为每层参数创建参数组
    param_groups = []
    for i, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            layer_name = f'layer_{i}'
            lr = layer_lr_config.get(layer_name, config.get('learning_rate', 0.01))
            param_groups.append({'params': [param], 'lr': lr})
            print(f"   {name}: 学习率 = {lr}")
    
    # 创建优化器
    optimizer = torch.optim.SGD(param_groups, momentum=0.9)
    
    return LayerWiseLRScheduler(optimizer, layer_lr_config)

def run_complete_hessian_analysis(model, loss, config: dict, step: int, 
                                eigenvalues: torch.Tensor, eigenvectors: torch.Tensor, 
                                dominant_dim: int, fnorm_squared: float, 
                                base_save_dir: str, device=None, wandb_logger=None) -> Dict:
    """
    运行完整的Hessian分析流程
    
    Args:
        model: PyTorch模型
        loss: 损失值
        config: 配置字典
        step: 当前步数
        eigenvalues: 已计算的特征值
        eigenvectors: 已计算的特征向量
        dominant_dim: dominant空间维度
        fnorm_squared: Hessian F-norm平方
        base_save_dir: 基础保存目录
        device: 计算设备
    
    Returns:
        分析结果字典
    """
    print(f"\n🔍 开始完整Hessian分析 (Step {step})...")
    
    # 1. 创建目录结构
    save_dirs = create_analysis_directories(base_save_dir, step)
    
    # 2. 计算完整Hessian矩阵和分块
    hessian_matrix, param_ranges = compute_full_hessian_with_blocks(loss, model.parameters(), device)
    
    # 3. 绘制全局Hessian热力图（新增）
    # plot_global_hessian_heatmap(hessian_matrix, save_dirs['heatmaps'], step, wandb_logger)
    
    # 4. 提取Hessian分块
    # hessian_blocks = extract_hessian_blocks(hessian_matrix, param_ranges)
    
    # 5. 绘制Hessian分块热力图
    # plot_hessian_heatmaps(hessian_blocks, save_dirs['heatmaps'], step, wandb_logger)
    
    # 6. 计算特征向量投影
    projections = compute_eigenvalue_projections(eigenvectors, param_ranges, dominant_dim)
    
    # 7. 绘制投影图
    # plot_eigenvalue_projections(projections, save_dirs['projections'], step, 
    #                            eigenvalues[:dominant_dim], wandb_logger)
    
    # 8. 计算trace比值
    trace_analysis = compute_trace_ratio(eigenvalues[:dominant_dim], hessian_matrix, config)
    
    # 9. 计算能量比值
    energy_analysis = compute_energy_ratio(eigenvalues[:dominant_dim], fnorm_squared)
    
    # 10. 保存所有数据
    save_analysis_data(save_dirs, step, projections)
    
    # 10. 准备返回结果
    results = {
        # 'hessian_matrix': hessian_matrix,
        # 'hessian_blocks': hessian_blocks,
        'param_ranges': param_ranges,
        'projections': projections,
        'trace_analysis': trace_analysis,
        'energy_analysis': energy_analysis,
        'save_dirs': save_dirs
    }
    
    print(f"✅ 完整Hessian分析完成!")
    
    return results
