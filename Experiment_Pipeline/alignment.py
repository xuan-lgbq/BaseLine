import torch
import torch.nn.functional as F

def grad_dominant_alignment(eigenvectors: torch.Tensor, gradient: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    计算向量 v 在由 eigvecs 张成的主子空间上的投影比例 chi_k.

    Args:
        eigvecs: Tensor of shape (p, k), 每列是一个归一化特征向量 u_i.
        v:        Tensor of shape (p,), 待测向量（如梯度向量）.
        eps:      防止除零的小常数.

    Returns:
        chi_k:    标量 Tensor, 等于 ||P_k v|| / ||v||.
    """
    # 1) 在 U^T v 上的投影系数
    coeffs = torch.matmul(eigenvectors.t(), gradient)             # shape: (k,)

    # 2) 重构主子空间分量 U (U^T v)
    v_dom = torch.matmul(eigenvectors, coeffs)             # shape: (p,)

    # 3) 归一化，得投影比例
    norm_v_dom = torch.norm(v_dom, p=2)
    norm_v     = torch.norm(gradient,     p=2).clamp(min=eps)
    chi_k      = norm_v_dom / norm_v

    return chi_k

def grad_dominant_alignment_cosine(eigvecs: torch.Tensor,
                                      v: torch.Tensor,
                                      eps: float = 1e-12) -> torch.Tensor:
    """
    用余弦相似度计算向量 v 在由 eigvecs 张成的主子空间上的投影比例 chi_k.

    Args:
        eigvecs: Tensor of shape (p, k)，每列是一个归一化的特征向量 u_i。
        v:        Tensor of shape (p,)，待测向量（如梯度向量）。
        eps:      防止除零的小数。

    Returns:
        chi_k:    标量 Tensor，等于 ||P_k v|| / ||v||。
    """
    # 1. 把 eigvecs 转置为 (k, p)，方便一次性和 v 计算 cosine
    U_t = eigvecs.t()                     # shape (k, p)

    # 2. 扩展 v 到 (k, p)，以便按行计算
    V = v.unsqueeze(0).expand_as(U_t)     # shape (k, p)

    # 3. 计算余弦相似度——得到 shape (k,)
    cosines = F.cosine_similarity(U_t, V, dim=1, eps=eps)

    # 4. 根据余弦平方和求 chi_k
    chi_k = torch.sqrt(torch.clamp(torch.sum(cosines**2), min=0.0))

    return chi_k
