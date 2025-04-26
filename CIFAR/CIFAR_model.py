# model.py

import torch
import torch.nn as nn
import wandb
from CIFAR_config import config, device

# CNN 网络结构
class ConvNet(nn.Module):
    def __init__(self, output_dim=config["output_dim"]):
        super(ConvNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32,bias=True,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32,track_running_stats=False), #在每个active layer后添加BN
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32,bias=True,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32,track_running_stats=False), 

            nn.MaxPool2d(2), #也可以替换成平均池化

            nn.Flatten(),
            nn.Linear(2048, output_dim, bias=True)
        )

    def forward(self, x):
        return self.model(x)  # 输出 shape: (N, output_dim)

# 测试函数
def test_model(model, X_test, Y_test_labels, Y_test_onehot, device):
    model.eval()
    loss_fn = nn.MSELoss()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        y_predict = model(X_test.to(device))  
        loss = loss_fn(y_predict, Y_test_onehot.to(device))
        total_loss += loss.item() * X_test.size(0)
        total_samples += X_test.size(0)

        _, predicted_indices = torch.max(y_predict, 1)
        correct_predictions += (predicted_indices == Y_test_labels.to(device)).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    wandb.log({"test_loss": avg_loss, "test_accuracy": accuracy})
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
