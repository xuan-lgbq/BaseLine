import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter
import swanlab 

def get_activation_suffix(config):
    """根据配置生成激活函数后缀"""
    use_activation = config.get("use_activation", False)
    if use_activation:
        activation_type = config.get("activation_type", "relu")
        return f"_act_{activation_type}"
    else:
        return "_no_act"

def format_number(x, pos):
    """格式化数字，避免科学计数法"""
    if x == 0:
        return '0'
    elif abs(x) >= 1:
        return f'{x:.4f}'
    elif abs(x) >= 0.0001:
        return f'{x:.6f}'
    else:
        return f'{x:.2e}'

def plot_training_loss(loss_history, step_history, config, IMAGE_SAVE_DIR, swanlab_logger):
    """绘制训练损失图 - 使用新的简洁风格"""
    if len(loss_history) == 0:
        print("⚠️  没有损失数据用于绘图")
        return
    
    # 准备绘图 - 使用新的简洁风格
    plt.figure(figsize=(12, 8), dpi=100)
    
    # 绘制训练损失曲线 - 蓝色，线性刻度，无网格
    plt.plot(step_history, loss_history, color='#1f77b4', linewidth=2.5, alpha=0.9)
    
    # 设置图形属性 - 移除标题、图例和网格
    plt.xlabel("Training Steps", fontsize=16)
    plt.ylabel("", fontsize=12)  # 移除y轴标签
    
    plt.tight_layout()
    
    # 生成文件名 - 保持原有命名规则
    seed = config["seed"]
    activation_suffix = get_activation_suffix(config)
    
    plot_filename = os.path.join(IMAGE_SAVE_DIR, f"linear_training_loss_seed{seed}_{activation_suffix}.png")
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"📉 训练损失图已保存: {plot_filename}")
    
    # 上传到swanlab - 保持原有接口
    if swanlab_logger and os.path.exists(plot_filename):
        swanlab_logger.log({"Training_Loss_Plot": swanlab.Image(plot_filename)})
        print(f"📤 训练损失图已上传到swanlab")
    
    plt.close()

def plot_top_k_eigenvalues(eigenvalue_history, step_history, config, IMAGE_SAVE_DIR, swanlab_logger):
    """绘制前top k个原始特征值演化图 - 使用dominant/bulk space风格"""
    if len(step_history) == 0:
        print("⚠️  没有特征值数据用于绘图")
        return
    
    # 准备数据
    steps = np.array(step_history)
    eigenvalues_matrix = []
    top_k = 20
    
    # 构建特征值矩阵
    max_data_len = max(len(eigenvalue_history[f"top_{i+1}"]) for i in range(top_k) if f"top_{i+1}" in eigenvalue_history)
    
    for step_idx in range(max_data_len):
        eigenvals_at_step = []
        for i in range(top_k):
            key = f"top_{i+1}"
            if key in eigenvalue_history and step_idx < len(eigenvalue_history[key]):
                eigenvals_at_step.append(eigenvalue_history[key][step_idx])
            else:
                eigenvals_at_step.append(0.0)  # 填充缺失值
        eigenvalues_matrix.append(eigenvals_at_step)
    
    eigenvalues_matrix = np.array(eigenvalues_matrix)
    
    if eigenvalues_matrix.size == 0:
        print("⚠️  特征值数据为空")
        return
    
    # 使用最后一步的特征值来确定dominant space
    final_eigenvalues = eigenvalues_matrix[-1, :].tolist()
    print(f"\n🔍 分析最后一步的特征值分布...")
    
    # 搜索dominant space (使用gap方法)
    try:
        from Top_k_Dom_search import search_top_k_dominant_bulk_space
        result = search_top_k_dominant_bulk_space(final_eigenvalues, method='gap')
        dominant_k = result['dominant_k']
        bulk_start = result['bulk_start'] 
        bulk_end = min(result['bulk_end'], top_k)  # 不超过总数
    except:
        # 如果导入失败，使用简单的分割方法
        dominant_k = min(6, top_k // 3)  # 前1/3作为dominant
        bulk_start = dominant_k
        bulk_end = top_k
    
    print(f"   Dominant space: 前{dominant_k}个特征值")
    print(f"   Bulk space: 第{bulk_start+1}到第{bulk_end}个特征值")
    
    # 准备绘图 - 使用新的风格
    plt.figure(figsize=(12, 8), dpi=100)
    
    # 绘制dominant space (蓝色系) - 只标注范围
    dominant_added_to_legend = False
    if dominant_k > 0:
        cmap_blue = plt.cm.get_cmap("Blues", max(dominant_k, 3))
        for i in range(dominant_k):
            if i < eigenvalues_matrix.shape[1]:
                eig_i = eigenvalues_matrix[:, i]
                # 根据特征值大小确定颜色深浅
                color_intensity = 0.3 + 0.7 * (dominant_k - i) / dominant_k  # 越大越深
                color = cmap_blue(color_intensity)
                # 只为第一条线添加图例标签
                if not dominant_added_to_legend:
                    plt.plot(steps[:len(eig_i)], eig_i, color=color, linewidth=2.5, 
                            label=f'Dominant space (λ1-λ{dominant_k})', alpha=0.9)
                    dominant_added_to_legend = True
                else:
                    plt.plot(steps[:len(eig_i)], eig_i, color=color, linewidth=2.5, alpha=0.9)
    
    # 绘制bulk space (红色系) - 不标注
    bulk_added_to_legend = False
    if bulk_start < bulk_end:
        bulk_size = bulk_end - bulk_start
        cmap_red = plt.cm.get_cmap("Reds", max(bulk_size, 3))
        for i in range(bulk_start, bulk_end):
            if i < eigenvalues_matrix.shape[1]:
                eig_i = eigenvalues_matrix[:, i]
                # 根据在bulk space中的位置确定颜色深浅
                bulk_idx = i - bulk_start
                color_intensity = 0.3 + 0.7 * (bulk_size - bulk_idx) / bulk_size
                color = cmap_red(color_intensity)
                # 只为第一条线添加图例标签
                if not bulk_added_to_legend:
                    plt.plot(steps[:len(eig_i)], eig_i, color=color, linewidth=1.0, 
                            label=f'Bulk space', alpha=0.6)
                    bulk_added_to_legend = True
                else:
                    plt.plot(steps[:len(eig_i)], eig_i, color=color, linewidth=1.0, alpha=0.6)
    
    # 设置图形属性 - 移除标题，使用新风格
    plt.xlabel("Training Steps", fontsize=16)
    plt.ylabel("", fontsize=12)  # 移除y轴标签
    
    # 把图例移到图内右上角 - 正方形黑色框，无阴影
    plt.legend(loc='upper right', frameon=True, fancybox=False, shadow=False, 
              framealpha=1.0, facecolor='white', edgecolor='black', fontsize=14)
    
    plt.tight_layout()
    
    # 生成文件名 - 保持原有命名规则
    # activation_suffix = get_activation_suffix(config)
    
    plot_filename = os.path.join(IMAGE_SAVE_DIR, 
                                f"linear_eigenvalues.png")
    
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"📊 Top-{top_k}特征值演化图已保存: {plot_filename}")
    
    # 上传到swanlab - 保持原有接口
    if swanlab_logger and os.path.exists(plot_filename):
        swanlab_logger.log({"Top_K_Eigenvalues_Plot": swanlab.Image(plot_filename)})
        print(f"📤 Top-{top_k}特征值演化图已上传到swanlab")
    
    plt.close()
    
    # 统计信息 - 简化版本
    print(f"📊 特征值统计信息:")
    for i in range(min(5, dominant_k)):  # 只打印前5个dominant特征值
        if i < eigenvalues_matrix.shape[1]:
            eigenvals = eigenvalues_matrix[:, i]
            print(f"   λ{i+1}: {eigenvals[0]:.6f} → {eigenvals[-1]:.6f}")

def plot_dominant_dims(dominant_dim_history, step_history, top_k, method, lr, var, seed, rank, num_layer, config, IMAGE_SAVE_DIR, swanlab_logger):
    """绘制主导维度演化图 - 使用新的简洁风格"""
    if len(dominant_dim_history) == 0:
        print("⚠️  没有主导维度数据用于绘图")
        return
    
    # 准备绘图 - 使用新的简洁风格
    plt.figure(figsize=(12, 8), dpi=100)
    
    # 绘制dominant dimension演化曲线 - 蓝色，线性刻度，无网格
    plt.plot(step_history, dominant_dim_history, color='#1f77b4', linewidth=2.5, alpha=0.9)
    
    # 设置图形属性 - 移除标题和网格
    plt.xlabel("Training Steps", fontsize=16)
    plt.ylabel("", fontsize=12)  # 移除y轴标签
    
    # 主导维度通常是整数
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
    
    plt.tight_layout()

    # 生成文件名 - 保持原有命名规则
    activation_suffix = get_activation_suffix(config)
    
    plot_filename = os.path.join(IMAGE_SAVE_DIR, 
                                f"linear{num_layer}_top{top_k}_dominant_dims_{method}_lr{lr}_var{var:.6f}_rank{rank}_seed{seed}{activation_suffix}.png")
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"📊 主导维度演化图已保存: {plot_filename}")
    
    # 上传到swanlab - 保持原有接口
    if swanlab_logger and os.path.exists(plot_filename):
        swanlab_logger.log({"Dominant_Dimension_Plot": swanlab.Image(plot_filename)})
        print(f"📤 主导维度演化图已上传到swanlab")
    
    plt.close()

def save_eigenvalue_csv(eigenvalue_history, eigenvalue_step_history, config, IMAGE_SAVE_DIR):
    """保存特征值数据到CSV文件"""
    if len(eigenvalue_step_history) == 0:
        print("⚠️  没有步骤数据，跳过CSV保存")
        return None
    
    try:
        data = {'step': eigenvalue_step_history}
        valid_count = 0
        
        # 处理特征值数据
        for key, values in eigenvalue_history.items():
            if len(values) > 0:
                # 确保长度匹配
                target_length = len(eigenvalue_step_history)
                if len(values) == target_length:
                    data[f'lambda_{key.split("_")[1]}'] = values
                    valid_count += 1
                elif len(values) < target_length:
                    # 用最后一个值填充
                    padded_values = values + [values[-1]] * (target_length - len(values))
                    data[f'lambda_{key.split("_")[1]}'] = padded_values
                    valid_count += 1
                else:
                    # 截取
                    data[f'lambda_{key.split("_")[1]}'] = values[:target_length]
                    valid_count += 1
        
        if valid_count == 0:
            print("⚠️  没有有效的特征值数据")
            return None
        
        # 创建DataFrame并保存
        df = pd.DataFrame(data)
        
        # 生成文件名
        activation_suffix = get_activation_suffix(config)
        
        csv_filename = os.path.join(IMAGE_SAVE_DIR, 
                                   f"linear{activation_suffix}.csv")
        
        df.to_csv(csv_filename, index=False)
        print(f"✅ 特征值数据已保存: {csv_filename}")
        print(f"📊 数据形状: {df.shape}")
        
        return csv_filename
        
    except Exception as e:
        print(f"❌ 保存CSV文件时出错: {e}")
        return None

def save_eigenvalue_csv_safe(eigenvalue_history, eigenvalue_step_history, method, lr, var, seed, rank, num_layer, config, IMAGE_SAVE_DIR):
    """安全版本的CSV保存函数"""
    # 过滤空数据
    cleaned_history = {k: v for k, v in eigenvalue_history.items() if len(v) > 0}
    
    if len(cleaned_history) == 0:
        print("⚠️  没有有效的特征值数据")
        return None
    
    return save_eigenvalue_csv(cleaned_history, eigenvalue_step_history, method, lr, var, seed, rank, num_layer, config, IMAGE_SAVE_DIR)

def plot_space_analysis_evolution(space_history, eigenvalue_step_history, config, save_dir, swanlab_logger):
    """绘制空间分析演化图"""
    print("🎨 绘制空间分析演化图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 绘制dominant和bulk维度演化
    axes[0, 0].plot(eigenvalue_step_history, space_history['dominant_dims'], 'o-', label='Dominant Dimension', linewidth=2, markersize=6)
    axes[0, 0].plot(eigenvalue_step_history, space_history['bulk_sizes'], 's-', label='Bulk Dimension', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Dimension')
    axes[0, 0].set_title('Space Dimensions Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 绘制gap值演化
    axes[0, 1].plot(eigenvalue_step_history, space_history['max_gap_values'], 'o-', label='Max Gap', linewidth=2, markersize=6)
    valid_second_gaps = [v for v in space_history['second_max_gap_values'] if v is not None]
    valid_steps = [eigenvalue_step_history[i] for i, v in enumerate(space_history['second_max_gap_values']) if v is not None]
    if valid_second_gaps:
        axes[0, 1].plot(valid_steps, valid_second_gaps, 's-', label='Second Max Gap', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Gap Value')
    axes[0, 1].set_title('Gap Values Evolution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 绘制bulk空间位置
    axes[1, 0].plot(eigenvalue_step_history, space_history['bulk_starts'], 'o-', label='Bulk Start', linewidth=2, markersize=6)
    axes[1, 0].plot(eigenvalue_step_history, space_history['bulk_ends'], 's-', label='Bulk End', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('Index')
    axes[1, 0].set_title('Bulk Space Position Evolution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 绘制gap位置
    axes[1, 1].plot(eigenvalue_step_history, space_history['max_gap_indices'], 'o-', label='Max Gap Index', linewidth=2, markersize=6)
    valid_second_indices = [v for v in space_history['second_max_gap_indices'] if v is not None]
    if valid_second_indices:
        axes[1, 1].plot(valid_steps, valid_second_indices, 's-', label='Second Max Gap Index', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Gap Index')
    axes[1, 1].set_title('Gap Indices Evolution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 添加激活函数信息
    activation_info = ""
    if config.get("use_activation", False):
        activation_info = f", Activation: {config.get('activation_type', 'relu').upper()}"
    else:
        activation_info = ", No Activation"
    
    plt.suptitle(f'Space Analysis Evolution{activation_info}', fontsize=16)
    plt.tight_layout()
    
    # 保存图表
    activation_suffix = get_activation_suffix(config)
    save_path = os.path.join(save_dir, f"space_analysis_evolution{activation_suffix}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 空间分析演化图已保存: {save_path}")
    
    # 上传到
    if swanlab_logger and os.path.exists(save_path):
        swanlab_logger.log({"Space_Analysis_Evolution": swanlab.Image(save_path)})
        print(f"📤 空间分析演化图已上传到swanlab")
    
    plt.close()

def plot_hessian_stats_evolution(hessian_stats_history, hessian_step_history, config, save_dir, swanlab_logger):
    """绘制Hessian统计信息的演化图"""
    print("🎨 绘制Hessian统计信息演化图...")
    
    # 提取统计信息
    stats_keys = ['max_absolute_value', 'mean_absolute_value', 'trace', 'frobenius_norm', 'spectral_norm']
    stats_data = {key: [] for key in stats_keys}
    
    for stats in hessian_stats_history:
        for key in stats_keys:
            if key in stats and isinstance(stats[key], (int, float)):
                stats_data[key].append(stats[key])
            else:
                stats_data[key].append(0)
    
    # 绘制图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (key, values) in enumerate(stats_data.items()):
        if i < len(axes):
            axes[i].plot(hessian_step_history, values, 'o-', linewidth=2, markersize=6)
            axes[i].set_xlabel('Training Step')
            axes[i].set_ylabel(key.replace('_', ' ').title())
            axes[i].grid(True, alpha=0.3)
            axes[i].set_title(f'Hessian {key.replace("_", " ").title()} Evolution')
    
    # 隐藏多余的子图
    for i in range(len(stats_data), len(axes)):
        axes[i].set_visible(False)
    
    # 添加激活函数信息
    activation_info = ""
    if config.get("use_activation", False):
        activation_info = f", Activation: {config.get('activation_type', 'relu').upper()}"
    else:
        activation_info = ", No Activation"
    
    plt.suptitle(f'Hessian Statistics Evolution{activation_info}', fontsize=16)
    plt.tight_layout()
    
    # 保存图表
    activation_suffix = get_activation_suffix(config)
    save_path = os.path.join(save_dir, f"hessian_stats_evolution{activation_suffix}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Hessian统计信息演化图已保存: {save_path}")
    
    # 上传到swanlab
    if swanlab_logger and os.path.exists(save_path):
        swanlab_logger.log({"Hessian_Stats_Evolution": swanlab.Image(save_path)})
        print(f"📤 Hessian统计信息演化图已上传到swanlab")
    
    plt.close()
