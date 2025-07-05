import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import json

def get_layer_parameter_info(model):
    """获取每层参数的详细信息"""
    layer_info = []
    start_idx = 0
    
    for i, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            param_count = param.numel()
            layer_info.append({
                'layer_idx': i,
                'name': name,
                'start_idx': start_idx,
                'end_idx': start_idx + param_count,
                'param_count': param_count,
                'shape': param.shape,
                'layer_type': 'weight' if 'weight' in name else 'bias'
            })
            start_idx += param_count
    
    return layer_info

def compute_layerwise_frobenius_norms(eigenvectors, layer_info, dominant_dim):
    """计算前dominant_dim个特征向量在各层的Frobenius范数"""
    # 确保只使用前dominant_dim个特征向量
    if eigenvectors.dim() == 1:
        eigenvectors = eigenvectors.unsqueeze(1)
    
    num_eigenvectors = min(dominant_dim, eigenvectors.shape[1])
    eigenvectors = eigenvectors[:, :num_eigenvectors]
    
    print(f"🔍 计算前{num_eigenvectors}个特征向量的分层F范数")
    
    # 存储结果
    layer_norms = {}  # {layer_idx: [norm_eigen1, norm_eigen2, ...]}
    
    for layer in layer_info:
        layer_idx = layer['layer_idx']
        start_idx = layer['start_idx']
        end_idx = layer['end_idx']
        
        # 提取该层对应的特征向量部分
        layer_eigenvectors = eigenvectors[start_idx:end_idx, :]
        
        # 计算每个特征向量在该层的Frobenius范数
        layer_norms[layer_idx] = []
        for k in range(num_eigenvectors):
            eigenve_layer_part = layer_eigenvectors[:, k]
            frobenius_norm = torch.norm(eigenve_layer_part, p='fro').item()
            layer_norms[layer_idx].append(frobenius_norm)
            
        print(f"  Layer {layer_idx} ({layer['name']}): F-norms = {layer_norms[layer_idx]}")
    
    return layer_norms

def plot_layerwise_frobenius_norms(layer_norms, layer_info, eigenvalues, dominant_dim, save_path, step):
    """可视化前dominant_dim个特征向量在各层的Frobenius范数"""
    num_layers = len(layer_info)
    num_eigenvectors = min(dominant_dim, len(next(iter(layer_norms.values()))))
    
    plt.figure(figsize=(14, 10))
    
    # 准备数据
    layer_indices = list(range(num_layers))
    layer_names = [layer['name'] for layer in layer_info]
    
    # 为每个特征向量绘制一条线
    colors = plt.cm.tab10(np.linspace(0, 1, min(num_eigenvectors, 10)))
    
    for k in range(num_eigenvectors):
        norms = [layer_norms[layer_idx][k] for layer_idx in range(num_layers)]
        
        eigenval = eigenvalues[k].item() if hasattr(eigenvalues[k], 'item') else eigenvalues[k]
        label = f'特征向量 {k+1} (λ={eigenval:.4f})'
        
        plt.plot(layer_indices, norms, 
                'o-', 
                color=colors[k % len(colors)], 
                linewidth=2, 
                markersize=6,
                label=label)
    
    plt.xlabel('层数', fontsize=14)
    plt.ylabel('Frobenius 范数', fontsize=14)
    plt.title(f'前{dominant_dim}个特征向量在各层的Frobenius范数分布 (Step {step})\nDominant Dimension = {dominant_dim}', fontsize=16)
    
    # 设置x轴标签
    plt.xticks(layer_indices, [f'Layer {i+1}\n{name.split(".")[-1]}' for i, name in enumerate(layer_names)], 
               rotation=45, ha='right')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 前{dominant_dim}个特征向量分层分析图已保存: {save_path}")
    plt.close()

def plot_layerwise_frobenius_norms(layer_norms, layer_info, eigenvalues, dominant_dim, save_path, step):
    """可视化前dominant_dim个特征向量在各层的Frobenius范数 - 柱状图版本"""
    num_layers = len(layer_info)
    num_eigenvectors = min(dominant_dim, len(next(iter(layer_norms.values()))))
    
    # 创建更大的图像以容纳更多信息
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 准备数据
    layer_names = [f"Layer {i}\n{layer['name'].split('.')[-1]}" for i, layer in enumerate(layer_info)]
    x = np.arange(num_layers)  # 层的位置
    
    # 计算柱子宽度
    width = 0.8 / num_eigenvectors if num_eigenvectors > 0 else 0.8
    
    # 颜色映射
    colors = plt.cm.Set3(np.linspace(0, 1, num_eigenvectors))
    
    # 为每个特征向量绘制柱状图
    for k in range(num_eigenvectors):
        # 获取该特征向量在各层的F范数
        norms = [layer_norms[layer_idx][k] for layer_idx in range(num_layers)]
        
        # 计算柱子位置（相对于中心偏移）
        offset = (k - (num_eigenvectors - 1) / 2) * width
        x_pos = x + offset
        
        # 获取特征值
        eigenval = eigenvalues[k].item() if hasattr(eigenvalues[k], 'item') else eigenvalues[k]
        label = f'Eigenvector {k+1} (λ={eigenval:.4f})'
        
        # 绘制柱状图
        bars = ax.bar(x_pos, norms, width, 
                     color=colors[k], 
                     alpha=0.8,
                     label=label,
                     edgecolor='black',
                     linewidth=0.5)
        
        # 在柱子顶部添加数值标签
        for i, (bar, norm) in enumerate(zip(bars, norms)):
            if norm > 0:  # 只在有值的柱子上添加标签
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(norms)*0.01,
                       f'{norm:.3f}', 
                       ha='center', va='bottom', 
                       fontsize=8, 
                       rotation=90 if num_eigenvectors > 3 else 0)
    
    # 设置坐标轴
    ax.set_xlabel('Network Layers', fontsize=14, fontweight='bold')  # 改为英文
    ax.set_ylabel('Frobenius Norm', fontsize=14, fontweight='bold')  # 改为英文
    ax.set_title(f'Frobenius Norms of Top {dominant_dim} Eigenvectors Across Layers (Step {step})\n'
                f'Dominant Dimension = {dominant_dim}', 
                fontsize=16, fontweight='bold', pad=20)  # 改为英文
    
    # 设置x轴刻度和标签
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, fontsize=12)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)
    
    # 设置图例
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 前{dominant_dim}个特征向量分层柱状图已保存: {save_path}")
    plt.close()

def plot_layerwise_frobenius_norms_stacked(layer_norms, layer_info, eigenvalues, dominant_dim, save_path, step):
    """堆叠柱状图版本 - 更适合比较不同特征向量的相对贡献"""
    num_layers = len(layer_info)
    num_eigenvectors = min(dominant_dim, len(next(iter(layer_norms.values()))))
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 准备数据
    layer_names = [f"Layer {i}\n{layer['name'].split('.')[-1]}" for i, layer in enumerate(layer_info)]
    
    # 构建数据矩阵 (num_eigenvectors x num_layers)
    data_matrix = np.zeros((num_eigenvectors, num_layers))
    labels = []
    
    for k in range(num_eigenvectors):
        for layer_idx in range(num_layers):
            data_matrix[k, layer_idx] = layer_norms[layer_idx][k]
        
        eigenval = eigenvalues[k].item() if hasattr(eigenvalues[k], 'item') else eigenvalues[k]
        labels.append(f'Eigenvector {k+1} (λ={eigenval:.4f})')
    
    # 颜色映射
    colors = plt.cm.Set3(np.linspace(0, 1, num_eigenvectors))
    
    # 绘制堆叠柱状图
    x = np.arange(num_layers)
    bottom = np.zeros(num_layers)
    
    for k in range(num_eigenvectors):
        ax.bar(x, data_matrix[k], 
               color=colors[k], 
               alpha=0.8,
               label=labels[k],
               bottom=bottom,
               edgecolor='black',
               linewidth=0.5)
        bottom += data_matrix[k]
    
    # 设置坐标轴
    ax.set_xlabel('Network Layers', fontsize=14, fontweight='bold')  # 改为英文
    ax.set_ylabel('Frobenius Norm', fontsize=14, fontweight='bold')  # 改为英文
    ax.set_title(f'Frobenius Norms of Top {dominant_dim} Eigenvectors Across Layers (Step {step})\n'
                f'Dominant Dimension = {dominant_dim}', 
                fontsize=16, fontweight='bold', pad=20)  # 改为英文
    
    # 设置x轴
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, fontsize=12)
    
    # 添加网格和图例
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)
    
    # 设置图例
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '_stacked.png'), dpi=300, bbox_inches='tight')
    print(f"📊 堆叠柱状图已保存: {save_path.replace('.png', '_stacked.png')}")
    plt.close()

def plot_layerwise_frobenius_norms_combined(layer_norms, layer_info, eigenvalues, dominant_dim, save_path, step):
    """组合版本：同时生成普通柱状图和堆叠柱状图"""
    # 生成普通柱状图
    plot_layerwise_frobenius_norms(layer_norms, layer_info, eigenvalues, dominant_dim, save_path, step)
    
    # 生成堆叠柱状图
    plot_layerwise_frobenius_norms_stacked(layer_norms, layer_info, eigenvalues, dominant_dim, save_path, step)
    
    # 生成热力图版本
    plot_layerwise_frobenius_heatmap(layer_norms, layer_info, eigenvalues, dominant_dim, save_path, step)

def plot_layerwise_frobenius_heatmap(layer_norms, layer_info, eigenvalues, dominant_dim, save_path, step):
    """热力图版本 - 适合查看模式"""
    num_layers = len(layer_info)
    num_eigenvectors = min(dominant_dim, len(next(iter(layer_norms.values()))))
    
    # 构建数据矩阵
    data_matrix = np.zeros((num_eigenvectors, num_layers))
    
    for k in range(num_eigenvectors):
        for layer_idx in range(num_layers):
            data_matrix[k, layer_idx] = layer_norms[layer_idx][k]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制热力图
    im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')
    
    # 设置坐标轴
    layer_names = [f"Layer {i}\n{layer['name'].split('.')[-1]}" for i, layer in enumerate(layer_info)]
    eigenvector_names = []
    for k in range(num_eigenvectors):
        eigenval = eigenvalues[k].item() if hasattr(eigenvalues[k], 'item') else eigenvalues[k]
        eigenvector_names.append(f'EigenVec {k+1}\n(λ={eigenval:.3f})')
    
    ax.set_xticks(np.arange(num_layers))
    ax.set_yticks(np.arange(num_eigenvectors))
    ax.set_xticklabels(layer_names)
    ax.set_yticklabels(eigenvector_names)
    
    # 添加数值标签
    for i in range(num_eigenvectors):
        for j in range(num_layers):
            text = ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                         ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title(f'Eigenvector Layerwise Frobenius Norm Heatmap (Step {step})\n'
                f'Dominant Dimension = {dominant_dim}', 
                fontsize=16, fontweight='bold', pad=20)  # 改为英文
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Frobenius Norm', fontsize=12)  # 改为英文
    
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '_heatmap.png'), dpi=300, bbox_inches='tight')
    print(f"📊 热力图已保存: {save_path.replace('.png', '_heatmap.png')}")
    plt.close()
    
def save_eigenvector_data(analysis_data, save_dir, step, method, lr, var, seed, rank, num_layer):
    """保存特征向量相关数据"""
    if analysis_data is None:
        return
    
    # 创建保存文件名
    base_filename = f"eigenvectors_step_{step}_{method}_lr{lr}_var{var:.6f}_seed{seed}_rank{rank}_layer{num_layer}"
    
    # 1. 保存为PyTorch格式 (.pt文件)
    torch_save_path = os.path.join(save_dir, f"{base_filename}.pt")
    torch_data = {
        'step': step,
        'eigenvalues': analysis_data['eigenvalues'],
        'eigenvectors': analysis_data['eigenvectors'],
        'dominant_dim': analysis_data['dominant_dim'],
        'layer_norms': analysis_data['layer_norms'],
        'layer_info': analysis_data['layer_info'],
        'metadata': {
            'method': method,
            'learning_rate': lr,
            'variance': var,
            'seed': seed,
            'rank': rank,
            'num_layer': num_layer,
            'timestamp': step
        }
    }
    torch.save(torch_data, torch_save_path)
    print(f"💾 特征向量数据已保存: {torch_save_path}")
    
    # 2. 保存分层分析结果为JSON格式
    json_save_path = os.path.join(save_dir, f"{base_filename}_layer_analysis.json")
    json_data = {
        'step': step,
        'dominant_dim': analysis_data['dominant_dim'],
        'eigenvalues': [float(ev) for ev in analysis_data['eigenvalues']],
        'layer_norms': {str(k): [float(norm) for norm in v] for k, v in analysis_data['layer_norms'].items()},
        'layer_info': [
            {
                'layer_idx': layer['layer_idx'],
                'name': layer['name'],
                'start_idx': layer['start_idx'],
                'end_idx': layer['end_idx'],
                'param_count': layer['param_count'],
                'shape': list(layer['shape']),
                'layer_type': layer['layer_type']
            }
            for layer in analysis_data['layer_info']
        ],
        'parameter_ranges': analysis_data.get('parameter_ranges', {}),
        'metadata': {
            'method': method,
            'learning_rate': lr,
            'variance': var,
            'seed': seed,
            'rank': rank,
            'num_layer': num_layer
        }
    }
    
    with open(json_save_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"💾 分层分析数据已保存: {json_save_path}")
    
    # 3. 保存特征向量为numpy格式 (便于其他工具读取)
    numpy_save_path = os.path.join(save_dir, f"{base_filename}_eigenvectors.npz")
    np.savez(numpy_save_path,
             eigenvalues=analysis_data['eigenvalues'].cpu().numpy(),
             eigenvectors=analysis_data['eigenvectors'].cpu().numpy(),
             dominant_dim=analysis_data['dominant_dim'],
             step=step)
    print(f"💾 特征向量numpy数据已保存: {numpy_save_path}")
    
    return {
        'torch_path': torch_save_path,
        'json_path': json_save_path,
        'numpy_path': numpy_save_path
    }

def load_eigenvector_data(file_path):
    """加载已保存的特征向量数据"""
    if file_path.endswith('.pt'):
        data = torch.load(file_path)
        print(f"📂 已加载特征向量数据: {file_path}")
        return data
    elif file_path.endswith('.npz'):
        data = np.load(file_path)
        print(f"📂 已加载numpy特征向量数据: {file_path}")
        return data
    else:
        raise ValueError("不支持的文件格式，请使用 .pt 或 .npz 文件")

def analyze_dominant_space_layerwise(model, eigenvectors, eigenvalues, dominant_dim, save_dir, step):
    """分析前dominant_dim个特征向量的分层分布"""
    print(f"🎨 Step {step}: 开始分析前{dominant_dim}个特征向量的分层分布...")
    
    if eigenvectors is None or eigenvalues is None:
        print("❌ 特征向量或特征值为空，跳过分析")
        return None
    
    # 获取层信息
    layer_info = get_layer_parameter_info(model)
    
    # 计算分层Frobenius范数
    layer_norms = compute_layerwise_frobenius_norms(eigenvectors, layer_info, dominant_dim)
    
    # 可视化 - 生成多种图表
    base_save_path = os.path.join(save_dir, f"eigenvector_layerwise_step_{step}_dom{dominant_dim}")
    
    # 1. 普通柱状图
    bar_save_path = f"{base_save_path}_bar.png"
    plot_layerwise_frobenius_norms(layer_norms, layer_info, eigenvalues, dominant_dim, bar_save_path, step)
    
    # 2. 堆叠柱状图
    stacked_save_path = f"{base_save_path}_stacked.png"
    plot_layerwise_frobenius_norms_stacked(layer_norms, layer_info, eigenvalues, dominant_dim, stacked_save_path, step)
    
    # 3. 热力图
    heatmap_save_path = f"{base_save_path}_heatmap.png"
    plot_layerwise_frobenius_heatmap(layer_norms, layer_info, eigenvalues, dominant_dim, heatmap_save_path, step)
    
    # 保存数据
    analysis_data = {
        'eigenvalues': eigenvalues[:dominant_dim],
        'eigenvectors': eigenvectors[:, :dominant_dim] if eigenvectors.dim() > 1 else eigenvectors,
        'layer_norms': layer_norms,
        'layer_info': layer_info,
        'dominant_dim': dominant_dim,
        'parameter_ranges': {f"Layer_{i}": f"[{info['start_idx']}:{info['end_idx']})" 
                            for i, info in enumerate(layer_info)}
    }
    
    print(f"✅ Step {step}: 前{dominant_dim}个特征向量分层分析完成")
    print(f"📊 生成了3种图表: 柱状图、堆叠柱状图、热力图")
    
    return analysis_data