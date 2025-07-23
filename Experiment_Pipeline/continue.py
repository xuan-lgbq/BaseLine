import os
import re

def extract_step_from_ckpt(ckpt_path):
    # 匹配 model_step_数字.pth
    match = re.search(r'model_step_(\d+)\.pth', os.path.basename(ckpt_path))
    if match:
        return int(match.group(1))
    else:
        return 0  # 或 raise Exception("No step found")

# 用法
ckpt_path = "/jumbo/yaoqingyang/ouyangzhuoli/Low_rank_identity/MNIST/experiments/fc-tanh_mnist-5k_lr0.1_seed0/checkpoints/model_step_24501.pth"
step = extract_step_from_ckpt(ckpt_path)
print(step)  # 输出 24501
