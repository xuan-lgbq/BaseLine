from typing import List

import torch
import torch.nn as nn
import math

from config import training_config
from data import num_classes, num_input_channels, image_size, num_pixels 
from resnet import resnet8
from vgg import vgg11_nodropout

_CONV_OPTIONS = {"kernel_size": 3, "padding": 1, "stride": 1}

def get_activation(activation: str):
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'hardtanh':
        return torch.nn.Hardtanh()
    elif activation == 'leaky_relu':
        return torch.nn.LeakyReLU()
    elif activation == 'selu':
        return torch.nn.SELU()
    elif activation == 'elu':
        return torch.nn.ELU()
    elif activation == "tanh":
        return torch.nn.Tanh()
    elif activation == "softplus":
        return torch.nn.Softplus()
    elif activation == "sigmoid":
        return torch.nn.Sigmoid()
    else:
        raise NotImplementedError("unknown activation function: {}".format(activation))

def get_pooling(pooling: str):
    if pooling == 'max':
        return torch.nn.MaxPool2d((2, 2))
    elif pooling == 'average':
        return torch.nn.AvgPool2d((2, 2))

# 1. 修改 fully_connected_net 以支持自定义 output_dim
def fully_connected_net(dataset_name: str, widths: List[int], activation: str, bias: bool = True, output_dim: int = None) -> nn.Module:
    modules = [nn.Flatten()]
    
    # 如果没有指定 output_dim，则使用默认的 num_classes
    final_output_dim = output_dim if output_dim is not None else num_classes(dataset_name)

    for l in range(len(widths)):
        prev_width = widths[l - 1] if l > 0 else num_pixels(dataset_name)
        modules.extend([
            nn.Linear(prev_width, widths[l], bias=bias),
            get_activation(activation),
        ])
    modules.append(nn.Linear(widths[-1], final_output_dim, bias=bias)) # 使用 final_output_dim
    return nn.Sequential(*modules)


# 1. 修改 fully_connected_net_bn 以支持自定义 output_dim
def fully_connected_net_bn(dataset_name: str, widths: List[int], activation: str, bias: bool = True, output_dim: int = None) -> nn.Module:
    modules = [nn.Flatten()]
    
    # 如果没有指定 output_dim，则使用默认的 num_classes
    final_output_dim = output_dim if output_dim is not None else num_classes(dataset_name)

    for l in range(len(widths)):
        prev_width = widths[l - 1] if l > 0 else num_pixels(dataset_name)
        modules.extend([
            nn.Linear(prev_width, widths[l], bias=bias),
            get_activation(activation),
            nn.BatchNorm1d(widths[l])
        ])
    modules.append(nn.Linear(widths[-1], final_output_dim, bias=bias)) # 使用 final_output_dim
    return nn.Sequential(*modules)


# 1. 修改 convnet 以支持自定义 output_dim
def convnet(dataset_name: str, widths: List[int], activation: str, pooling: str, bias: bool, output_dim: int = None) -> nn.Module:
    modules = []
    size = image_size(dataset_name)
    
    # 如果没有指定 output_dim，则使用默认的 num_classes
    final_output_dim = output_dim if output_dim is not None else num_classes(dataset_name)

    for l in range(len(widths)):
        prev_width = widths[l - 1] if l > 0 else num_input_channels(dataset_name)
        modules.extend([
            nn.Conv2d(prev_width, widths[l], bias=bias, **_CONV_OPTIONS),
            get_activation(activation),
            get_pooling(pooling),
        ])
        size //= 2
    modules.append(nn.Flatten())
    modules.append(nn.Linear(widths[-1]*size*size, final_output_dim)) # 使用 final_output_dim
    return nn.Sequential(*modules)


# 1. 修改 convnet_bn 以支持自定义 output_dim
def convnet_bn(dataset_name: str, widths: List[int], activation: str, pooling: str, bias: bool, output_dim: int = None) -> nn.Module:
    modules = []
    size = image_size(dataset_name)
    
    # 如果没有指定 output_dim，则使用默认的 num_classes
    final_output_dim = output_dim if output_dim is not None else num_classes(dataset_name)

    for l in range(len(widths)):
        prev_width = widths[l - 1] if l > 0 else num_input_channels(dataset_name)
        modules.extend([
            nn.Conv2d(prev_width, widths[l], bias=bias, **_CONV_OPTIONS),
            get_activation(activation),
            nn.BatchNorm2d(widths[l]),
            get_pooling(pooling),
        ])
        size //= 2
    modules.append(nn.Flatten())
    modules.append(nn.Linear(widths[-1]*size*size, final_output_dim)) # 使用 final_output_dim
    return nn.Sequential(*modules)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.0, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerClassifier(nn.Module):
    def __init__(self, input_size=30522, hidden_size=64, seed=0, num_layers=2, num_labels=2):
        super(TransformerClassifier, self).__init__()
        torch.manual_seed(seed)
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=2048, dropout=0.0),
            num_layers=num_layers,
        )
        self.fc = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x_inp = x[:, 0, :]
        x_mask = ~x[:, 1, :].bool()
        x = self.embedding(x_inp)
        x = self.positional_encoding(x)
        # Permute to (sequence_length, batch_size, hidden_size) for nn.TransformerEncoder
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x, src_key_padding_mask=x_mask)
        # Global average pooling
        x = torch.mean(x, dim=0)
        # Linear layer for classification
        x = self.fc(x)
        return x

def load_architecture(arch_id: str, dataset_name: str) -> nn.Module:
    #  ======   fully-connected networks =======
    if arch_id == 'fc-relu':
        return fully_connected_net(dataset_name, [200, 200], 'relu', bias=True)
    elif arch_id == 'fc-elu':
        return fully_connected_net(dataset_name, [200, 200], 'elu', bias=True)
    elif arch_id == 'fc-tanh':
        return fully_connected_net(dataset_name, [200, 200], 'tanh', bias=True)
    elif arch_id == 'fc-hardtanh':
        return fully_connected_net(dataset_name, [200, 200], 'hardtanh', bias=True)
    elif arch_id == 'fc-softplus':
        return fully_connected_net(dataset_name, [200, 200], 'softplus', bias=True)

    #  ======   convolutional networks =======
    elif arch_id == 'cnn-relu':
        return convnet(dataset_name, [32, 32], activation='relu', pooling='max', bias=True)
    elif arch_id == 'cnn-elu':
        return convnet(dataset_name, [32, 32], activation='elu', pooling='max', bias=True)
    elif arch_id == 'cnn-tanh':
        return convnet(dataset_name, [32, 32], activation='tanh', pooling='max', bias=True)
    elif arch_id == 'cnn-avgpool-relu':
        return convnet(dataset_name, [32, 32], activation='relu', pooling='average', bias=True)
    elif arch_id == 'cnn-avgpool-elu':
        return convnet(dataset_name, [32, 32], activation='elu', pooling='average', bias=True)
    elif arch_id == 'cnn-avgpool-tanh':
        return convnet(dataset_name, [32, 32], activation='tanh', pooling='average', bias=True)

    #  ======   convolutional networks with BN =======
    elif arch_id == 'cnn-bn-relu':
        return convnet_bn(dataset_name, [32, 32], activation='relu', pooling='max', bias=True)
    elif arch_id == 'cnn-bn-elu':
        return convnet_bn(dataset_name, [32, 32], activation='elu', pooling='max', bias=True)
    elif arch_id == 'cnn-bn-tanh':
        return convnet_bn(dataset_name, [32, 32], activation='tanh', pooling='max', bias=True)

    #  ======   real networks on CIFAR-10  =======
    elif arch_id == 'resnet8':
        return resnet8()
    elif arch_id == 'vgg11':
        return vgg11_nodropout()
    
    # ====== additional networks ========
    elif arch_id == 'transformer':
        # Note: TransformerClassifier currently has hardcoded num_labels=2.
        # You might want to make num_labels configurable based on dataset_name
        # or a specific 'output_dim' argument if you're using it for multi-class.
        # For now, it remains as is based on its __init__ signature.
        return TransformerClassifier()

    # ======= vary depth (already done) =======
    elif arch_id == 'fc-tanh-depth1':
        return fully_connected_net(dataset_name, [200], 'tanh', bias=True)
    elif arch_id == 'fc-tanh-depth2':
        return fully_connected_net(dataset_name, [200, 200], 'tanh', bias=True)
    elif arch_id == 'fc-tanh-depth3':
        return fully_connected_net(dataset_name, [200, 200, 200], 'tanh', bias=True)
    elif arch_id == 'fc-tanh-depth4':
        return fully_connected_net(dataset_name, [200, 200, 200, 200], 'tanh', bias=True)
    
    elif arch_id == 'fc-relu-depth1':
        return fully_connected_net(dataset_name, [200], 'relu', bias=True)
    elif arch_id == 'fc-relu-depth2':
        return fully_connected_net(dataset_name, [200, 200], 'relu', bias=True)
    elif arch_id == 'fc-relu-depth3':
        return fully_connected_net(dataset_name, [200, 200, 200], 'relu', bias=True)
    elif arch_id == 'fc-relu-depth4':
        return fully_connected_net(dataset_name, [200, 200, 200, 200], 'relu', bias=True)

    # ======= NEW: vary hidden layer width =======
    elif arch_id == 'fc-tanh-width100':
        return fully_connected_net(dataset_name, [100, 100], 'tanh', bias=True)
    elif arch_id == 'fc-tanh-width300':
        return fully_connected_net(dataset_name, [300, 300], 'tanh', bias=True)
    elif arch_id == 'fc-tanh-width500':
        return fully_connected_net(dataset_name, [500, 500], 'tanh', bias=True)
    
    elif arch_id == 'fc-relu-width100':
        return fully_connected_net(dataset_name, [100, 100], 'relu', bias=True)
    elif arch_id == 'fc-relu-width300':
        return fully_connected_net(dataset_name, [300, 300], 'relu', bias=True)
    elif arch_id == 'fc-relu-width500':
        return fully_connected_net(dataset_name, [500, 500], 'relu', bias=True)
        
   
    elif arch_id == 'fc-relu-out2': # Example: Binary classification
        return fully_connected_net(dataset_name, [200, 200], 'relu', bias=True, output_dim=2)
    elif arch_id == 'fc-tanh-out5': # Example: 5-class classification
        return fully_connected_net(dataset_name, [200, 200], 'tanh', bias=True, output_dim=5)
    elif arch_id == 'cnn-relu-out100': # Example: 100-class classification (e.g., ImageNet subset)
        return convnet(dataset_name, [32, 32], activation='relu', pooling='max', bias=True, output_dim=100)

    elif arch_id == 'fc-subclass-classifier':
        k_subclasses = training_config["k_subclasses"] 
        return fully_connected_net(dataset_name, [200, 200], 'relu', bias=True, output_dim=k_subclasses)
    elif arch_id == 'cnn-subclass-classifier':
        k_subclasses = training_config["k_subclasses"] 
        return convnet(dataset_name, [32, 32], activation='relu', pooling='max', bias=True, output_dim=k_subclasses)

    else:
        raise NotImplementedError(f"Architecture '{arch_id}' not implemented for dataset '{dataset_name}'")