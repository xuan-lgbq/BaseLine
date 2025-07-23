import torch
import torch.distributed as dist
from config_mnist import training_config as config
import torch.nn.functional as F
num_classes = config.get("num_classes", 10)


def hvp(model, loss, vec):
    """
    Compute Hessian-vector product ∇²L · v
    """
    params = [p for p in model.parameters() if p.requires_grad]
    grad1 = torch.autograd.grad(loss, params, create_graph=True)
    flat_grad1 = torch.cat([g.reshape(-1) for g in grad1])
    grad_v = torch.dot(flat_grad1, vec)
    grad2 = torch.autograd.grad(grad_v, params, retain_graph=True)
    flat_grad2 = torch.cat([g.reshape(-1) for g in grad2])

    dist.all_reduce(flat_grad2, op=dist.ReduceOp.SUM)
    return flat_grad2

def lanczos_distributed(model, data_loader, criterion, k, max_iter=3000, batch_size=10):
    """
    Distributed Lanczos method to compute top-k eigenvalues/eigenvectors of the Hessian.
    Only rank 0 returns results.
    """
    rank = dist.get_rank()
    device = next(model.parameters()).device
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 1. 初始化 v0
    if rank == 0:
        v0 = torch.randn(num_params, device=device)
        v0 /= v0.norm()
    else:
        v0 = torch.zeros(num_params, device=device)

    dist.broadcast(v0, src=0)

    V = [v0]
    alphas, betas = [], []
    beta = torch.tensor(0.0, device=device)

    for i in range(max_iter):
        vi = V[-1]

        # 2. 累加 loss over multiple batches
        loss = 0.0
        data_iter = iter(data_loader)
        for _ in range(batch_size):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            one_hot_labels = F.one_hot(targets, num_classes=num_classes).float()
            loss += criterion(outputs, one_hot_labels)
        loss = loss / float(batch_size)

        loss = loss / dist.get_world_size()

        # 3. HVP
        w = hvp(model, loss, vi)

        # 4. Lanczos 迭代
        alpha = torch.dot(vi, w)
        alphas.append(alpha.item())
        w = w - alpha * vi
        if i > 0:
            w -= beta * V[-2]
        beta = w.norm()
        betas.append(beta.item())

        if beta < 1e-6 or len(alphas) >= k:
            break

        V.append(w / beta)

    # 5. 仅 rank0 构建 T 并计算谱
    if rank == 0:
        dtype = torch.float32
        T = torch.zeros(len(alphas), len(alphas), dtype=dtype, device=device)
        for j in range(len(alphas)):
            T[j, j] = alphas[j]
            if j < len(betas) - 1:
                T[j, j + 1] = T[j + 1, j] = betas[j + 1]

        eigvals, eigvecs = torch.linalg.eigh(T)
        topk = min(k, eigvals.shape[0])
        indices = eigvals.argsort(descending=True)[:topk]
        return eigvals[indices], eigvecs[:, indices]
    else:
        return None, None


def distributed_power_iteration(model, data, dataloader, criterion,
                                top_n=1, max_iter=100, tol=1e-3, hessian_type="full",  # or batch
                                batch_size=None):
    """
    分布式 Power-Iteration：
      - 用 hvp(model, loss, v) 计算 H·v（含 all_reduce）
      - deflation（正交消去前面找到的特征向量）
      - 只在 rank 0 返回 eigenvalues/eigenvectors
    """
    rank = dist.get_rank()
    device = next(model.parameters()).device
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    eigenvalues = []
    eigenvectors = []

    for _ in range(top_n):
        # 随机初始向量
        v = torch.randn(num_params, device=device)
        v /= v.norm()
        prev_lambda = None

        for it in range(max_iter):
            # —— 构造平均 loss —— #
            if hessian_type == "batch" and data is not None:  # 已经 one-hot
                x, y = data
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y.to(device))
            elif hessian_type == "full":   # 原始数据，没有one-hot，需要追加
                total_loss, cnt = 0.0, 0
                for x,y in dataloader:
                    x, y = x.to(device), y.to(device)
                    b = x.size(0)
                    l = criterion(model(x), y)
                    if getattr(criterion, "reduction", "sum") == "mean":
                        total_loss += l * b
                    else:  # sum
                        total_loss += l
                    cnt    += b
                loss = total_loss / cnt

            # 如果你用 mean，再除 world_size
            loss = loss / dist.get_world_size()

            # —— 计算 H·v —— #
            Hv = hvp(model, loss, v)

            # —— Deflation：正交消去已算出的 eigenvectors —— #
            if eigenvectors:
                for u in eigenvectors:
                    proj = torch.dot(Hv, u)
                    Hv -= proj * u

            # —— 更新 λ 和 v —— #
            lamb = torch.dot(v, Hv).item()
            v_new = Hv / (Hv.norm() + 1e-12)

            # 收敛判断
            if prev_lambda is not None and abs(lamb - prev_lambda) / (abs(prev_lambda) + 1e-6) < tol:
                break
            v, prev_lambda = v_new, lamb

        eigenvalues.append(prev_lambda)
        eigenvectors.append(v)

    if rank == 0:
        # 转为 tensor
        eigenvalues = torch.tensor(eigenvalues, device=device)
        eigenvectors = torch.stack(eigenvectors, dim=1)  # [p, top_n]

        # 按特征值降序排列
        sorted_idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        return eigenvalues, eigenvectors
    else:
        return None, None

def trace_distributed(model, data, dataloader, criterion, max_iter=100, tol=1e-3, hessian_type="full", batch_size=None):
    """
    分布式 Hutchinson 方法估算 Hessian trace
    model: nn.Module
    dataloader: DataLoader
    criterion: 损失函数
    max_iter: 最大采样次数
    tol: 收敛容忍度
    """
    rank = dist.get_rank()
    device = next(model.parameters()).device

    params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in params)
    
    trace_vhv = []  # List[float]
    trace = 0.0     # float

    for i in range(max_iter):
        model.zero_grad()
        # 1) 生成扁平 Rademacher 向量
        v = torch.randint(0, 2, (num_params,), device=device, dtype=torch.float32)
        v[v == 0] = -1

        
        # —— 构造平均 loss —— #
        if hessian_type == "batch" and data is not None:  # 已经 one-hot
            x, y = data
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y.to(device))
        elif hessian_type == "full":   # 已经 one-hot
            total_loss, cnt = 0.0, 0
            for x,y in dataloader:
                x, y = x.to(device), y.to(device)
                b = x.size(0)
                l = criterion(model(x), y)
                if getattr(criterion, "reduction", "sum") == "mean":
                    total_loss += l * b
                else:  # sum
                    total_loss += l
                cnt += b
            loss = total_loss/cnt 

        # 如果你用 mean，再除 world_size
        loss = loss / dist.get_world_size()

        # —— 计算 H·v —— #
        Hv = hvp(model, loss, v)  

        trace_vhv.append(torch.dot(v, Hv).item())
        if abs(torch.mean(torch.tensor(trace_vhv)) - trace) / (trace + 1e-6) < tol:
            return torch.tensor(trace_vhv)       
        else:
            trace = torch.mean(torch.tensor(trace_vhv))
    
    return torch.tensor(trace_vhv)

        


    



    
