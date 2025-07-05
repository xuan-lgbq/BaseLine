import torch
from hessian_utils import compute_hessian_eigenvalues_pyhessian_fixed  # 使用修正版本
from Top_k_Dom_search import search_top_k_dominant_space

def compute_and_analyze_eigenvalues(model, loss_function, single_loader, top_k, device, config, step):
    """计算并分析特征值和特征向量"""
    try:
        # 使用修正版本计算 Hessian 特征值和特征向量
        eigenvalues, eigenvectors = compute_hessian_eigenvalues_pyhessian_fixed(
            model=model,
            criterion=loss_function,
            data_loader=single_loader,
            top_k=top_k,
            device=device
        )
        
        # 验证数据
        print(f"📊 Step {step}: 验证计算结果")
        print(f"   特征值数据类型: {type(eigenvalues)}, 形状: {eigenvalues.shape}")
        print(f"   特征向量数据类型: {type(eigenvectors)}, 形状: {eigenvectors.shape}")
        print(f"   特征值设备: {eigenvalues.device}")
        print(f"   特征向量设备: {eigenvectors.device}")
        
        # 特征值排序验证
        is_descending = torch.all(eigenvalues[:-1] >= eigenvalues[1:])
        print(f"   是否降序: {is_descending}")
        print(f"   前5个特征值: {eigenvalues[:5]}")

        # 🔧 修正：将CUDA tensor转换为CPU上的Python列表
        eigenvalues_cpu_list = eigenvalues.cpu().tolist()
        print(f"🔧 转换为CPU列表: {eigenvalues_cpu_list[:5]}")
        
        # 计算dominant dimension
        dominant_dim, gaps = search_top_k_dominant_space(eigenvalues_cpu_list, method=config["method"])
        
        # 取前 dominant_dim 个特征向量
        eigenvectors_dominant = eigenvectors[:, :dominant_dim]
        eigenvalues_dominant = eigenvalues[:dominant_dim]
        
        print(f"📊 Step {step}: 计算成功, Dominant Dimension: {dominant_dim}")
        print(f"📊 Step {step}: 前{dominant_dim}个特征值: {eigenvalues_dominant}")
        print(f"📊 Step {step}: 特征向量矩阵形状: {eigenvectors_dominant.shape}")
        print(f"📊 Step {step}: Gaps前5个: {gaps[:5] if len(gaps) >= 5 else gaps}")
        
        return eigenvalues, eigenvectors_dominant, dominant_dim, gaps, True
        
    except Exception as e:
        print(f"⚠️  Step {step}: 计算 Hessian 特征值失败: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")
        return None, None, None, None, False

def collect_eigenvalue_data(eigenvalue_history, eigenvalues):
    """收集特征值数据"""
    for i, eigenval in enumerate(eigenvalues):
        # 确保eigenval是tensor，然后转换为Python标量
        if isinstance(eigenval, torch.Tensor):
            raw_eigenval = eigenval.cpu().item()
        else:
            raw_eigenval = float(eigenval)
        eigenvalue_history[f"top_{i+1}"].append(raw_eigenval)

def prepare_wandb_log(eigenvalues, dominant_dim, gaps):
    """准备wandb记录数据"""
    hessian_eigenvals = {}
    for i, eigenval in enumerate(eigenvalues):
        # 确保eigenval是tensor，然后转换为Python标量
        if isinstance(eigenval, torch.Tensor):
            raw_eigenval = eigenval.cpu().item()
        else:
            raw_eigenval = float(eigenval)
        hessian_eigenvals[f"Raw_Hessian_Eigenvalues/Top_{i+1}"] = raw_eigenval
    
    # 确保gaps是Python列表
    if isinstance(gaps, torch.Tensor):
        gaps_list = gaps.cpu().tolist()
    elif isinstance(gaps, list):
        gaps_list = gaps
    else:
        gaps_list = [float(g) for g in gaps]
    
    wandb_data = {
        "Dominant Dimension": dominant_dim,
        "Gaps": gaps_list[:5] if len(gaps_list) >= 5 else gaps_list
    }
    wandb_data.update(hessian_eigenvals)
    
    return wandb_data