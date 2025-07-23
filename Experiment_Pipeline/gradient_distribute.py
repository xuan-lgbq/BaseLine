import torch
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector
from config_mnist import training_config as config  
def batch_gradient_distribute(model, criterion, data, batch_size):
    rank = dist.get_rank()
    device = next(model.parameters()).device

    model.zero_grad()

    inputs, targets = data
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    loss = loss / dist.get_world_size()
    
    loss.backward()
    grad = torch.nn.utils.parameters_to_vector([p.grad for p in model.parameters() if p.requires_grad])

    dist.all_reduce(grad, op=dist.ReduceOp.SUM)
    return grad
    
def global_gradient_distribute(model, criterion, dataloader, world_size):
    """
    计算全局平均梯度（全数据集loss对参数的梯度），分布式实现。
    Returns:
        grad: 全局平均梯度向量（torch.Tensor）
    """
    device = next(model.parameters()).device
    model.zero_grad()
    total_samples = 0
    grad_accum = None

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        batch_size = x.size(0)
        loss = criterion(model(x), y)
        if hasattr(criterion, "reduction") and criterion.reduction == 'mean':
            loss = loss * batch_size  # 转为sum
        grads = torch.autograd.grad(loss, model.parameters(), retain_graph=False)
        grad_vec = parameters_to_vector(grads)
        if grad_accum is None:
            grad_accum = torch.zeros_like(grad_vec)
        grad_accum += grad_vec
        total_samples += batch_size

    # 所有进程的 grad_accum 和 total_samples 求和
    dist.all_reduce(grad_accum, op=dist.ReduceOp.SUM)
    total_samples_tensor = torch.tensor(total_samples, device=device)
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
    grad_accum /= total_samples_tensor
    return grad_accum


"""
grad = batch_gradient_distribute(
                model=ddp_model.module,
                data=(images, labels),
                criterion=criterion,
                batch_size=batch_size
            )
"""

        



