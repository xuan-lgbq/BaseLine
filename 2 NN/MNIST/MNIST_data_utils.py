# data_utils.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from MNIST_config import config, device

def load_mnist_data(config, device):
    """加载MNIST数据集并进行规范化预处理
    
    Args:
        config (dict): 包含以下键值：
            data_root (str): 数据存储路径
            train_samples (int): 训练集采样数量
            test_samples (int): 测试集采样数量
            output_dim (int): 输出维度（类别数）
        device (torch.device): 目标计算设备
    
    Returns:
        Tuple: (X_train, Y_train, X_test, Y_test, A_matrix)
    """
    # 数据预处理管道
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST标准归一化参数
        transforms.Lambda(lambda x: x.view(-1))      # 展平为784维向量
    ])
    
    # 加载完整数据集
    train_set = datasets.MNIST(
        root="/jumbo/yaoqingyang/boyao/2 NN/MNIST", 
        train=True, 
        download=True, 
        transform=transform_pipeline
    )
    test_set = datasets.MNIST(
        root="/jumbo/yaoqingyang/boyao/2 NN/MNIST", 
        train=False, 
        download=True, 
        transform=transform_pipeline
    )
    
    # 创建子集采样
    train_subset = Subset(train_set, indices=range(5000))
    test_subset = Subset(test_set, indices=range(500))
    
    # 批量加载全部数据
    train_loader = DataLoader(train_subset, batch_size=len(train_subset), shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=len(test_subset), shuffle=False)
    
    X_train, Y_train = next(iter(train_loader))
    X_test, Y_test = next(iter(test_loader))
    
    # 转换设备并生成one-hot标签
    Y_train_onehot = _to_one_hot(Y_train, config["output_dim"], device)
    Y_test_onehot = _to_one_hot(Y_test, config["output_dim"], device)
    Y_train = Y_train.to(device)
    Y_test = Y_test.to(device)
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    
    # 计算目标矩阵A
    target_matrix = generate_target_matrix(X_train, Y_train_onehot)
    test_target_matrix = generate_target_matrix(X_test, Y_test_onehot)
    return X_train, Y_train, X_test, Y_test, target_matrix, test_target_matrix


def _to_one_hot(y, num_classes, device):
    """内部函数:生成one-hot编码"""
    return torch.eye(num_classes, device=device)[y.squeeze().long()]

def generate_target_matrix(X, Y_onehot):
    """生成目标矩阵 A = (1/m)Y^TX
    
    Args:
        X (Tensor): 输入特征矩阵 (m x d)
        Y_onehot (Tensor): one-hot标签矩阵 (m x c)
    
    Returns:
        Tensor: 目标矩阵 A (c x d)
    """
    m = X.size(0)
    return (1/m) * torch.mm(Y_onehot.T, X)

