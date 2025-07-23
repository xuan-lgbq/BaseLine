import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, Subset
import random
import numpy as np

def setup(rank, world_size):
    """
    初始化分布式环境
    :param rank: 当前进程的编号
    :param world_size: 总进程数
    """
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def gather_full_batch(local_images, local_labels):
    world_size = dist.get_world_size()
    
    # 为收集准备 tensor list（需相同 shape）
    gathered_images = [torch.zeros_like(local_images) for _ in range(world_size)]
    gathered_labels = [torch.zeros_like(local_labels) for _ in range(world_size)]
    
    dist.all_gather(gathered_images, local_images)
    dist.all_gather(gathered_labels, local_labels)
    
    full_images = torch.cat(gathered_images, dim=0)
    full_labels = torch.cat(gathered_labels, dim=0)
    return full_images, full_labels

def collect_eigenvalue_data(eigenvalue_history, eigenvalues):
    for i, eigenval in enumerate(eigenvalues):
        if isinstance(eigenval, torch.Tensor):
            raw_eigenval = eigenval.cpu().item()
        else:
            raw_eigenval = float(eigenval)
        eigenvalue_history[f"top_{i+1}"].append(raw_eigenval)

def evaluate_and_log_test_accuracy(ddp_model, test_set, rank, epoch, accuracy_history, accuracy_step_history, swanlab):
    ddp_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for test_images, test_labels in DataLoader(test_set, batch_size=256):
            test_images = test_images.to(rank)
            test_labels = test_labels.to(rank)
            outputs = ddp_model(test_images)
            _, predicted = torch.max(outputs.data, 1)
            # 如果 test_labels 是 one-hot，需要转为类别索引
            if test_labels.dim() > 1:
                test_labels = torch.argmax(test_labels, dim=1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()
    test_acc = correct / total
    accuracy_history.append(test_acc)
    accuracy_step_history.append(epoch)
    swanlab.log({"Test Accuracy": test_acc}, step=epoch)
    print(f"[Rank {rank}]: Epoch {epoch + 1}, Test Accuracy: {test_acc:.4f}")
    ddp_model.train()
