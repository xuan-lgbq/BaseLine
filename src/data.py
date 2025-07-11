import torch
import numpy as np
from typing import Tuple
from torch.utils.data import TensorDataset
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
