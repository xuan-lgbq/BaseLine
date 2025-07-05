import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

def plot_training_loss(loss_history, step_history, method, lr, var, seed, rank, num_layer, config, IMAGE_SAVE_DIR, wandb):
    """绘制训练损失图"""
    if len(loss_history) == 0:
        print("⚠️  没有损失数据用于绘图")
        return
    
    plt.figure(figsize=(12, 8))
    plt.plot(step_history, loss_history, 'b-', linewidth=2, label='Training Loss')
    
    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title(f'Training Loss Evolution\n{method} Optimizer, LR={lr}, Variance={var:.6f}, Rank={rank}, Seed={seed}', 
          fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 添加统计信息
    final_loss = loss_history[-1]
    min_loss = min(loss_history)
    initial_loss = loss_history[0]
    
    plt.text(0.02, 0.98, 
             f'Initial Loss: {initial_loss:.6f}\nFinal Loss: {final_loss:.6f}\nMin Loss: {min_loss:.6f}\nReduction: {(initial_loss-final_loss)/initial_loss*100:.2f}%', 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    gap_method = config["method"]
    output_dim = config["output_dim"]
    
    plot_filename = os.path.join(IMAGE_SAVE_DIR, f"linear{num_layer}_{gap_method}training_loss_{method}_outputdim{output_dim}_lr{lr}_var{var:.6f}_rank{rank}_seed{seed}_raw.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📉 训练损失图已保存: {plot_filename}")
    
    wandb.log({"Training_Loss_Plot": wandb.Image(plot_filename)})
    plt.close()

def plot_top_k_eigenvalues(eigenvalue_history, step_history, top_k, method, lr, var, seed, rank, num_layer, config, IMAGE_SAVE_DIR, wandb):
    """绘制前top k个原始特征值演化图（不归一化）"""
    if len(step_history) == 0:
        print("⚠️  没有特征值数据用于绘图")
        return
        
    plt.figure(figsize=(16, 12))
    
    # 设置颜色映射
    if top_k <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, top_k))
    elif top_k <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, top_k))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, top_k))
    
    valid_lines = 0
    eigenvalue_stats = {}
    
    # 计算特征值范围
    all_eigenvals = []
    for i in range(top_k):
        key = f"top_{i+1}"
        if key in eigenvalue_history and len(eigenvalue_history[key]) > 0:
            all_eigenvals.extend(eigenvalue_history[key])
    
    if all_eigenvals:
        max_eigenval = max(all_eigenvals)
        min_eigenval = min(all_eigenvals)
        eigenval_range = max_eigenval - min_eigenval
        print(f"📊 特征值范围: [{min_eigenval:.6f}, {max_eigenval:.6f}], 跨度: {eigenval_range:.6f}")
    
    for i in range(top_k):
        key = f"top_{i+1}"
        if key in eigenvalue_history and len(eigenvalue_history[key]) > 0:
            data_len = len(eigenvalue_history[key])
            steps_for_this_data = step_history[:data_len]
            eigenvals = eigenvalue_history[key]
            
            eigenvalue_stats[f'λ{i+1}'] = {
                'initial': eigenvals[0],
                'final': eigenvals[-1],
                'max': max(eigenvals),
                'min': min(eigenvals),
                'range': max(eigenvals) - min(eigenvals)
            }
            
            plt.plot(steps_for_this_data, 
                    eigenvals, 
                    color=colors[i], 
                    linewidth=3 if i < 3 else (2.5 if i < 10 else 2),
                    alpha=0.9 if i < 10 else 0.7,
                    label=f'λ{i+1} (原始)',
                    marker='o' if len(steps_for_this_data) < 30 and i < 5 else None,
                    markersize=6 if i < 3 else (5 if i < 10 else 4))
            valid_lines += 1
    
    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel('Raw Hessian Eigenvalue (未归一化)', fontsize=14)
    plt.title(f'Evolution of Top {top_k} Raw Hessian Eigenvalues\n{method} Optimizer, LR={lr}, Variance={var:.6f}, Rank={rank}, Seed={seed}', 
          fontsize=16)
    
    # 图例处理
    if valid_lines <= 15:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    else:
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[:15], labels[:15], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.grid(True, alpha=0.3)
    
    # 尺度调整
    if all_eigenvals and max(all_eigenvals) > 0:
        max_val = max(all_eigenvals)
        min_val = min([v for v in all_eigenvals if v > 0])
        if max_val / min_val > 1000:
            plt.yscale('log')
            print("📊 使用对数尺度显示特征值（跨度较大）")
        else:
            print("📊 使用线性尺度显示特征值")
    
    # 统计信息
    if valid_lines > 0:
        stats_text = f'Rank: {rank}\nEigenvalue Count: {valid_lines}\nComputations: {len(step_history)}'
        if valid_lines >= 1:
            first_ev = eigenvalue_stats.get('λ1', {})
            if first_ev:
                stats_text += f'\nλ1: {first_ev["initial"]:.6f} → {first_ev["final"]:.6f}'
                stats_text += f'\nλ1 Range: {first_ev["range"]:.6f}'
        
        if valid_lines >= 2:
            second_ev = eigenvalue_stats.get('λ2', {})
            if second_ev:
                stats_text += f'\nλ2: {second_ev["initial"]:.6f} → {second_ev["final"]:.6f}'
        
        plt.text(0.02, 0.02, stats_text, 
                transform=plt.gca().transAxes, fontsize=11, 
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    gap_method = config["method"]
    output_dim = config["output_dim"]
    
    plot_filename = os.path.join(IMAGE_SAVE_DIR, f"linear{num_layer}_{gap_method}_top{top_k}_eigenvalues_raw_{method}_outputdim{output_dim}_lr{lr}_var{var:.6f}_rank{rank}_seed{seed}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Top-{top_k}原始特征值演化图已保存: {plot_filename}")
    
    wandb.log({"Top_K_Raw_Eigenvalues_Plot": wandb.Image(plot_filename)})
    plt.close()
    
    # 打印统计信息
    print(f"\n📊 特征值统计信息:")
    for i, (name, stats) in enumerate(eigenvalue_stats.items()):
        if i < 5:
            print(f"   {name}: 初始={stats['initial']:.6f}, 最终={stats['final']:.6f}, 变化={stats['range']:.6f}")

def plot_dominant_dims(dominant_dim_history, step_history, top_k, method, lr, var, seed, rank, num_layer, config, IMAGE_SAVE_DIR, wandb):
    """绘制主导维度演化图"""
    if len(dominant_dim_history) == 0:
        print("⚠️  没有主导维度数据用于绘图")
        return
    
    plt.figure(figsize=(12, 8))
    plt.plot(step_history, dominant_dim_history, 'g-', linewidth=2, label='Dominant Dimension')
    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel('Dominant Dimension', fontsize=14)
    plt.title('Evolution of Dominant Dimension', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    gap_method = config["method"]
    output_dim = config["output_dim"]
    
    plot_filename = os.path.join(IMAGE_SAVE_DIR, f"linear{num_layer}_{gap_method}_top{top_k}_dominant_space_dimension_{method}_outputdim{output_dim}_lr{lr}_var{var:.6f}_rank{rank}_seed{seed}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 主导维度演化图已保存: {plot_filename}")
    
    wandb.log({f"Dominant_Dimension_Plot_{num_layer}": wandb.Image(plot_filename)})
    plt.close()

def save_eigenvalue_csv(eigenvalue_history, eigenvalue_step_history, method, lr, var, seed, rank, num_layer, config, IMAGE_SAVE_DIR):
    """保存特征值数据到CSV文件 - 修正版本"""
    try:
        print(f"🔍 开始保存CSV数据...")
        print(f"   eigenvalue_step_history 长度: {len(eigenvalue_step_history)}")
        print(f"   eigenvalue_history 键数量: {len(eigenvalue_history)}")
        
        # 检查数据完整性
        target_length = len(eigenvalue_step_history)
        if target_length == 0:
            print("⚠️  没有步骤数据，跳过CSV保存")
            return None
        
        data = {'step': eigenvalue_step_history}
        valid_eigenvalues = 0
        
        # 只处理非空的特征值序列
        for key, values in eigenvalue_history.items():
            print(f"   检查 {key}: 长度={len(values)}")
            
            # 只处理非空的数据
            if len(values) > 0:
                valid_eigenvalues += 1
                
                # 确保数据长度一致
                if len(values) == target_length:
                    # 数据长度正确，直接使用
                    data[f'lambda_{key.split("_")[1]}'] = values
                    print(f"   ✅ {key}: 长度匹配 ({len(values)})")
                    
                elif len(values) < target_length:
                    # 数据不足，用最后一个值填充
                    if len(values) > 0:
                        padded_values = values + [values[-1]] * (target_length - len(values))
                        data[f'lambda_{key.split("_")[1]}'] = padded_values
                        print(f"   🔧 {key}: 填充到 {target_length} (原长度: {len(values)})")
                    else:
                        # 如果是空列表，跳过
                        print(f"   ⚠️  {key}: 空列表，跳过")
                        continue
                        
                else:
                    # 数据超出，截取前面的部分
                    data[f'lambda_{key.split("_")[1]}'] = values[:target_length]
                    print(f"   ✂️  {key}: 截取到 {target_length} (原长度: {len(values)})")
            else:
                print(f"   ❌ {key}: 空数据，跳过")
        
        if valid_eigenvalues == 0:
            print("⚠️  没有有效的特征值数据，跳过CSV保存")
            return None
        
        print(f"🔍 有效特征值数量: {valid_eigenvalues}")
        
        # 最终验证：确保所有列的长度都一致
        print(f"🔍 最终数据验证:")
        data_lengths = {}
        for col_name, col_data in data.items():
            data_lengths[col_name] = len(col_data)
            print(f"   {col_name}: {len(col_data)}")
        
        # 检查是否所有长度都一致
        unique_lengths = set(data_lengths.values())
        if len(unique_lengths) != 1:
            print(f"❌ 数据长度不一致: {unique_lengths}")
            print("🔧 尝试修复...")
            
            # 找到最短的长度作为标准
            min_length = min(data_lengths.values())
            print(f"   使用最短长度: {min_length}")
            
            # 截取所有数据到相同长度
            for col_name in data.keys():
                if len(data[col_name]) > min_length:
                    data[col_name] = data[col_name][:min_length]
                    print(f"   截取 {col_name} 到 {min_length}")
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        print(f"✅ DataFrame创建成功: {df.shape}")
        
        # 生成文件名
        gap_method = config["method"]
        output_dim = config["output_dim"]
        
        csv_filename = os.path.join(IMAGE_SAVE_DIR, 
                                   f"linear{num_layer}_{gap_method}_{method}_eigenvalues_outputdim{output_dim}_lr{lr}_var{var:.6f}_rank{rank}_seed{seed}.csv")
        
        # 保存文件
        df.to_csv(csv_filename, index=False)
        print(f"✅ 特征值数据已保存到: {csv_filename}")
        print(f"📊 CSV文件包含 {len(df)} 行, {len(df.columns)} 列")
        print(f"📊 列名: {list(df.columns)}")
        
        # 显示前几行数据作为验证
        print(f"📊 前3行数据预览:")
        print(df.head(3))
        
        return csv_filename
        
    except Exception as e:
        print(f"❌ 保存CSV文件时出错: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        return None

def clean_eigenvalue_history(eigenvalue_history):
    """清理特征值历史数据，移除空序列"""
    cleaned_history = {}
    
    print(f"🧹 清理特征值历史数据...")
    
    for key, values in eigenvalue_history.items():
        if len(values) > 0:
            cleaned_history[key] = values
            print(f"   保留 {key}: {len(values)} 个值")
        else:
            print(f"   移除 {key}: 空序列")
    
    print(f"✅ 清理完成: {len(eigenvalue_history)} → {len(cleaned_history)} 个有效序列")
    return cleaned_history

def save_eigenvalue_csv_safe(eigenvalue_history, eigenvalue_step_history, method, lr, var, seed, rank, num_layer, config, IMAGE_SAVE_DIR):
    """安全版本的CSV保存函数"""
    # 先清理数据
    cleaned_history = clean_eigenvalue_history(eigenvalue_history)
    
    if len(cleaned_history) == 0:
        print("⚠️  没有有效的特征值数据，跳过CSV保存")
        return None
    
    # 使用清理后的数据保存
    return save_eigenvalue_csv(cleaned_history, eigenvalue_step_history, method, lr, var, seed, rank, num_layer, config, IMAGE_SAVE_DIR)