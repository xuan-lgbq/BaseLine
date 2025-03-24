import torch
import numpy as np
from config import config, device

# 计算 Hessian 矩阵的特征值和特征向量
def compute_hessian_eigen(loss, params, top_k=5):
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