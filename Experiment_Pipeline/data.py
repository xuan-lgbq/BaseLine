import torch
import numpy as np
from typing import Tuple
from torch.utils.data import TensorDataset
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from cifar import load_cifar
from mnist import load_mnist
from sst import load_sst2

DATASETS = ["cifar10-5k", "mnist-5k", "sst2-1k"]

def flatten(arr: np.ndarray):
    return arr.reshape(arr.shape[0], -1)

def unflatten(arr: np.ndarray, shape: Tuple):
    return arr.reshape(arr.shape[0], *shape)

def num_input_channels(dataset_name: str) -> int:
    if dataset_name.startswith("cifar"):
        return 3
    elif dataset_name.startswith("mnist"):
        return 1

def image_size(dataset_name: str) -> int:
    if dataset_name.startswith("cifar"):
        return 32
    elif dataset_name.startswith("mnist"):
        return 28

def num_classes(dataset_name: str) -> int:
    if dataset_name.startswith('cifar10'):
        return 10
    elif dataset_name.startswith("mnist"):
        return 10
    elif dataset_name.startswith("sst2"):
        return 2
    
def get_pooling(pooling: str):
    if pooling == 'max':
        return torch.nn.MaxPool2d((2, 2))
    elif pooling == 'average':
        return torch.nn.AvgPool2d((2, 2))
    else:
        raise NotImplementedError("unknown pooling: {}".format(pooling))

def num_pixels(dataset_name: str) -> int:
    return num_input_channels(dataset_name) * image_size(dataset_name)**2

def take_first(dataset: TensorDataset, num_to_keep: int):
    return TensorDataset(dataset.tensors[0][0:num_to_keep], dataset.tensors[1][0:num_to_keep])

def load_dataset(dataset_name: str, loss: str) -> (TensorDataset, TensorDataset):
    if dataset_name == "mnist-5k":
        train, test = load_mnist(loss)
        return take_first(train, 5000), test
    elif dataset_name == "cifar10-5k":
        train, test = load_cifar(loss)
        return take_first(train, 5000), test
    elif dataset_name == "sst2-1k":
        train, test = load_sst2(loss)
        return take_first(train, 1000), test







class OneHotDataset(Dataset):
    def __init__(self, base_dataset, num_classes):
        self.base = base_dataset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        y_onehot = torch.nn.functional.one_hot(torch.tensor(y), num_classes=self.num_classes).float()
        return x, y_onehot
    

def get_train_and_test_set(config):
    dataset_name = config.get("dataset", "MNIST").lower()
    data_path = '/jumbo/yaoqingyang/ouyangzhuoli/Low_rank_identity/data'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    if dataset_name.startswith("mnist"):
        train_set = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
        train_set = Subset(train_set, list(range(config.get("train_size", 5000))))
        test_set = Subset(test_set, list(range(config.get("test_size", 1000))))
    elif dataset_name.startswith("cifar"):
        train_set = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
        train_set = Subset(train_set, list(range(config.get("train_size", 5000))))
        test_set = Subset(test_set, list(range(config.get("test_size", 1000))))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return train_set, test_set
