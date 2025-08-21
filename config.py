'''
Description: 参数配置
Author: Damocles_lin
Date: 2025-08-19 16:01:56
LastEditTime: 2025-08-21 22:50:17
LastEditors: Damocles_lin
'''
import torch

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 实验超参数
config = {
    "batch_size": 120,
    "learning_rate": 0.001,
    "num_epochs": 15,
    "num_workers": 4,
    "optimizer": "Adam",
    "pretrained": True,
    "data_augmentation": True
}

# CIFAR-10类别名称
classes = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)

# 创建结果保存目录
def create_dirs():
    """
    创建结果保存目录
    
    Returns:
        None: 函数直接创建目录，不返回任何值
    """
    import os
    os.makedirs("results",exist_ok=True)
    os.makedirs("checkpoints",exist_ok=True)
    