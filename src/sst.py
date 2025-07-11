import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _one_hot(tensor: Tensor, num_classes: int, default=0):
    M = F.one_hot(tensor, num_classes)
    M[M == 0] = default
    return M.float()

def make_labels(y, loss, num_classes = 10):
    if loss == "ce":
        return y
    elif loss == "mse":
        return _one_hot(y, num_classes, 0)


def load_sst2(loss: str) -> (TensorDataset, TensorDataset):
    sst2_train = load_dataset('sst2')['train']
    sst2_test = load_dataset('sst2')['validation']
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    tokenized_train = tokenizer(sst2_train['sentence'], truncation=True, padding=True)
    tokenized_test = tokenizer(sst2_test['sentence'], truncation=True, padding=True)
    X_train, X_test = torch.stack([torch.tensor(tokenized_train['input_ids'], device=device), torch.tensor(tokenized_train['attention_mask'], device=device)], dim=1), torch.stack([torch.tensor(tokenized_test['input_ids'], device=device), torch.tensor(tokenized_test['attention_mask'], device=device)], dim=1)
    y_train, y_test = torch.tensor(sst2_train['label'], device=device), torch.tensor(sst2_test['label'], device=device)
    y_train, y_test = make_labels(y_train, loss, num_classes=2), make_labels(y_test, loss, num_classes=2)
    train = TensorDataset(X_train, y_train)
    test = TensorDataset(X_test, y_test)

    print("Loaded SST2 dataset")
    return train, test
