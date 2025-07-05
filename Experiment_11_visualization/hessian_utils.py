import torch
import numpy as np
from config_linear import device
from pyhessian import hessian

# 计算 Hessian 矩阵的特征值和特征向量
def compute_hessian_eigen(loss, params, top_k = 832):  # previously top_k = 5
    params = list(params)
    
    # 获取所有参数的展平梯度
    grads_flat = []
    for p in params:
        g = torch.autograd.grad(loss, p, create_graph=True, retain_graph=True)[0]
        grads_flat.append(g.view(-1))  # 每个参数展平为向量
    grads_flat = torch.cat(grads_flat)  # 形状: (total_params,)
    total_params = grads_flat.size(0)   # 总参数量 p = 75
    
    # 计算Hessian矩阵
    hessian_rows = []
    for g in grads_flat:  # 遍历每个梯度元素
        # 计算二阶导数（Hessian行）
        hessian_row = torch.autograd.grad(
            outputs=g, 
            inputs=params, 
            retain_graph=True, 
            allow_unused=True
        )
        # 处理未使用的参数梯度（填充零）
        hessian_row_flat = []
        for h, p in zip(hessian_row, params):
            if h is None:
                h_flat = torch.zeros_like(p).view(-1)
            else:
                h_flat = h.view(-1)
            hessian_row_flat.append(h_flat)
        hessian_row_flat = torch.cat(hessian_row_flat)  # 形状: (total_params,)
        hessian_rows.append(hessian_row_flat)
    
    # 构建Hessian矩阵
    hessian_matrix = torch.stack(hessian_rows)  # 形状: (total_params, total_params)
    
    # 转换为NumPy并计算特征值/向量
    hessian_numpy = hessian_matrix.detach().cpu().numpy()
    eigenvalues, eigenvectors = np.linalg.eigh(hessian_numpy)
    sorted_indices = np.argsort(-eigenvalues)  # 降序排列
    return eigenvalues[sorted_indices], eigenvectors[:, sorted_indices][:, :top_k]

def flatten_eigenvector_list(eigenvector_list):
    """
    将PyHessian返回的嵌套tensor列表展平为单个tensor
    
    Args:
        eigenvector_list: 形如 [tensor1, tensor2, tensor3] 的列表
    
    Returns:
        torch.Tensor: 展平后的特征向量
    """
    flattened_parts = []
    for tensor_part in eigenvector_list:
        if isinstance(tensor_part, torch.Tensor):
            flattened_parts.append(tensor_part.view(-1))
        else:
            flattened_parts.append(torch.tensor(tensor_part).view(-1))
    
    return torch.cat(flattened_parts)

def process_pyhessian_eigenvectors(eigenvectors_matrix, eigenvalues_list):
    """
    处理PyHessian返回的特征向量格式
    
    Args:
        eigenvectors_matrix: PyHessian返回的特征向量（嵌套列表格式）
        eigenvalues_list: 对应的特征值
    
    Returns:
        tuple: (eigenvalues_tensor, eigenvectors_tensor)
    """
    print(f"🔍 处理PyHessian特征向量...")
    print(f"   eigenvectors_matrix 类型: {type(eigenvectors_matrix)}")
    print(f"   eigenvectors_matrix 长度: {len(eigenvectors_matrix)}")
    
    if isinstance(eigenvectors_matrix, list) and len(eigenvectors_matrix) > 0:
        # 检查第一个元素的结构
        first_eigenvector = eigenvectors_matrix[0]
        print(f"   第一个特征向量类型: {type(first_eigenvector)}")
        
        if isinstance(first_eigenvector, list):
            print(f"   第一个特征向量包含 {len(first_eigenvector)} 个部分")
            for i, part in enumerate(first_eigenvector):
                print(f"     部分 {i}: 类型={type(part)}, 形状={getattr(part, 'shape', 'N/A')}")
            
            # 处理嵌套列表格式
            num_eigenvectors = len(eigenvectors_matrix)
            flattened_eigenvectors = []
            
            for i, eigenvector_list in enumerate(eigenvectors_matrix):
                flattened_vec = flatten_eigenvector_list(eigenvector_list)
                flattened_eigenvectors.append(flattened_vec)
                print(f"   特征向量 {i+1}: 展平后形状 {flattened_vec.shape}")
            
            # 堆叠为矩阵 (total_params, num_eigenvectors)
            eigenvectors_tensor = torch.stack(flattened_eigenvectors, dim=1)
            
        elif isinstance(first_eigenvector, torch.Tensor):
            # 如果是tensor列表，直接堆叠
            eigenvectors_tensor = torch.stack(eigenvectors_matrix, dim=1)
        else:
            # 其他格式，尝试转换
            eigenvectors_tensor = torch.tensor(eigenvectors_matrix, dtype=torch.float32)
    
    else:
        # 如果不是列表格式，按原来的方式处理
        if isinstance(eigenvectors_matrix, torch.Tensor):
            eigenvectors_tensor = eigenvectors_matrix.clone().detach()
        elif isinstance(eigenvectors_matrix, np.ndarray):
            eigenvectors_tensor = torch.from_numpy(eigenvectors_matrix)
        else:
            eigenvectors_tensor = torch.tensor(eigenvectors_matrix, dtype=torch.float32)
    
    # 处理特征值
    if isinstance(eigenvalues_list, torch.Tensor):
        eigenvalues_tensor = eigenvalues_list.clone().detach()
    elif isinstance(eigenvalues_list, (list, np.ndarray)):
        eigenvalues_tensor = torch.tensor(eigenvalues_list, dtype=torch.float32)
    else:
        eigenvalues_tensor = torch.tensor(eigenvalues_list, dtype=torch.float32)
    
    print(f"✅ 处理完成:")
    print(f"   eigenvalues_tensor: {eigenvalues_tensor.shape}")
    print(f"   eigenvectors_tensor: {eigenvectors_tensor.shape}")
    
    return eigenvalues_tensor, eigenvectors_tensor

# This a function when running in remote, please check the input agruments before run.
# Function to compute the eigenvalues and eigenvectors of the Hessian matrix (using pyhessian)
def compute_hessian_eigen_pyhessian(model, criterion, data_loader, top_k=5, device=device):
    """
    Computes the top eigenvalues and eigenvectors of the Hessian matrix using the pyhessian library.

    Args:
        model: PyTorch model.
        criterion: Loss function.
        data_loader: Data loader used for computing the Hessian.
        top_k: The number of top eigenvalues and eigenvectors to return.
        device: The computation device (CPU or CUDA).

    Returns:
        tuple: Contains two NumPy arrays:
            - eigenvalues: Top k eigenvalues in descending order (shape: (top_k,)).
            - eigenvectors: Corresponding eigenvectors (shape: (total_params, top_k)).
    """
    hessian_computer = hessian(model=model, criterion=criterion, data=data_loader, cuda='cuda')
    hessian_eigen = hessian_computer.eigenvalues(top_n=top_k)
    eigenvalues = np.array(hessian_eigen[0])
    eigenvectors = np.array(hessian_eigen[1])
    return eigenvalues, eigenvectors

def compute_layer_weight_eigenvalues(model, top_k=None):
    """
    计算模型每一层权重矩阵的特征值和特征向量
    
    Args:
        model: PyTorch模型
        top_k: 返回前k个特征值，如果为None则返回所有特征值
    
    Returns:
        dict: 包含每层特征值和特征向量的字典
    """
    layer_eigenvalues = {}
    layer_eigenvectors = {}
    
    for name, param in model.named_parameters():
        # 修改条件：匹配 W1, W2, W3 参数名，并且是2维矩阵
        if name in ['W1', 'W2', 'W3'] and param.dim() == 2:
            weight_matrix = param.detach().cpu().numpy()
            
            # 如果不是方阵，使用 W @ W.T 来计算特征值
            if weight_matrix.shape[0] != weight_matrix.shape[1]:
                # 对于非方阵，计算 W @ W.T 的特征值
                gram_matrix = np.dot(weight_matrix, weight_matrix.T)
                eigenvals, eigenvecs = np.linalg.eigh(gram_matrix)
            else:
                # 对于方阵，直接计算特征值
                eigenvals, eigenvecs = np.linalg.eigh(weight_matrix)
            
            # 按降序排列
            sorted_indices = np.argsort(-eigenvals)
            eigenvals = eigenvals[sorted_indices]
            eigenvecs = eigenvecs[:, sorted_indices]
            
            # 如果指定了top_k，只返回前k个
            if top_k is not None:
                eigenvals = eigenvals[:top_k]
                eigenvecs = eigenvecs[:, :top_k]
            
            layer_eigenvalues[name] = eigenvals
            layer_eigenvectors[name] = eigenvecs
    
    return layer_eigenvalues, layer_eigenvectors

def compute_hessian_eigenvalues_pyhessian_fixed(model, criterion, data_loader, top_k=5, device='cuda'):
    """
    修正版本：处理PyHessian的嵌套tensor格式
    """
    try:
        # 计算模型参数总数
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"🔍 模型参数总数: {total_params}")
        
        # 自动调整top_k
        actual_top_k = min(top_k, total_params)
        if actual_top_k != top_k:
            print(f"⚠️  警告: 请求的top_k={top_k} 超过参数总数={total_params}")
            print(f"🔧 自动调整为 top_k={actual_top_k}")
        
        hessian_computer = hessian(model=model, 
                                  criterion=criterion, 
                                  dataloader=data_loader,
                                  cuda=device.type=='cuda' if hasattr(device, 'type') else 'cuda' in str(device))

        print(f"🔍 开始计算前{actual_top_k}个特征值/特征向量...")
        
        # 使用调整后的top_k
        result = hessian_computer.eigenvalues(
            maxIter=200, 
            tol=1e-6, 
            top_n=actual_top_k
        )
        
        if isinstance(result, tuple) and len(result) == 2:
            eigenvalues_list, eigenvectors_matrix = result
        else:
            raise ValueError(f"PyHessian返回格式异常: {type(result)}")
        
        print(f"🔍 PyHessian原始返回:")
        print(f"   eigenvalues_list: {type(eigenvalues_list)}")
        print(f"   eigenvectors_matrix: {type(eigenvectors_matrix)}")
        
        # 使用新的处理函数
        eigenvalues_tensor, eigenvectors_tensor = process_pyhessian_eigenvectors(
            eigenvectors_matrix, eigenvalues_list
        )
        
        # 确保在正确的设备上
        eigenvalues_tensor = eigenvalues_tensor.to(device)
        eigenvectors_tensor = eigenvectors_tensor.to(device)
        
        # 确保特征值从大到小排序
        eigenvalues_sorted, sort_indices = torch.sort(eigenvalues_tensor, descending=True)
        eigenvectors_sorted = eigenvectors_tensor[:, sort_indices]
        
        print(f"✅ 最终结果:")
        print(f"   特征值形状: {eigenvalues_sorted.shape}")
        print(f"   特征向量形状: {eigenvectors_sorted.shape}")
        print(f"   前3个特征值: {eigenvalues_sorted[:3]}")
        
        return eigenvalues_sorted, eigenvectors_sorted
        
    except Exception as e:
        print(f"❌ PyHessian计算失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        raise e

def debug_model_parameters(model):
    """调试函数：检查模型参数详情"""
    print("🔍 模型参数详情:")
    total_params = 0
    for i, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            param_count = param.numel()
            print(f"  {i}: {name} - 形状: {param.shape}, 参数数: {param_count}")
            total_params += param_count
    
    print(f"📊 总参数数: {total_params}")
    return total_params

def compute_dominant_projection_matrix(top_eigenvectors, k):
    """
    根据特征向量构造投影矩阵 P_k = Σ u_i u_i^T
    Args:
        top_eigenvectors: 形状为 (p, k) 的矩阵，每列是特征向量
    Returns:
        P_k: 形状为 (p, p) 的投影矩阵
    """
    p = top_eigenvectors.shape[0]
    P_k = np.zeros((p, p))
    for i in range(k):
        u_i = top_eigenvectors[:, i].reshape(-1, 1)
        P_k += np.dot(u_i, u_i.T)  # 累加外积
    return torch.from_numpy(P_k).float().to(device)  # 转为Tensor并指定设备