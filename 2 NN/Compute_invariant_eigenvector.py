import numpy as np
import torch
from config import config
import wandb

def compute_invariant_eigenvector(W1, W2):
    return 