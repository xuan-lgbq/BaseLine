import numpy as np
from torchvision.datasets import MNIST
from typing import Tuple
from torch.utils.data.dataset import TensorDataset
import os
import torch
from torch import Tensor
import torch.nn.functional as F

# DATASETS_FOLDER = os.environ["DATASETS"]

def center(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(0)
    return X_train - mean, X_test - mean

def standardize(X_train: np.ndarray, X_test: np.ndarray):
    std = X_train.std(0)
    return (X_train / std, X_test / std)

def flatten(arr: np.ndarray):
    return arr.reshape(arr.shape[0], -1)

def unflatten(arr: np.ndarray, shape: Tuple):
    return arr.reshape(arr.shape[0], *shape)

def _one_hot(tensor: Tensor, num_classes: int, default=0):
    M = F.one_hot(tensor, num_classes)
    M[M == 0] = default
    return M.float()

def make_labels(y, loss, num_classes = 10):
    if loss == "ce":
        return y
    elif loss == "mse":
        return _one_hot(y, num_classes, 0)


def load_mnist(loss: str) -> (TensorDataset, TensorDataset):
    DATASETS_FOLDER = os.environ["DATASETS"]
    mnist_train = MNIST(root=DATASETS_FOLDER, download=True, train=True)
    mnist_test = MNIST(root=DATASETS_FOLDER, download=True, train=False)
    X_train, X_test = flatten(np.array(mnist_train.data) / 255), flatten(np.array(mnist_test.data) / 255)
    y_train, y_test = make_labels(torch.tensor(np.array(mnist_train.targets)), loss), make_labels(torch.tensor(np.array(mnist_test.targets)), loss)
    train = TensorDataset(torch.from_numpy(unflatten(X_train, (28, 28, 1)).transpose((0, 3, 1, 2))).float(), y_train)
    test = TensorDataset(torch.from_numpy(unflatten(X_test, (28, 28, 1)).transpose((0, 3, 1, 2))).float(), y_test)

    print("Loaded MNIST dataset")
    return train, test
