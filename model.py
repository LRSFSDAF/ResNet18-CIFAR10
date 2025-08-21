'''
Description: 模型定义
Author: Damocles_lin
Date: 2025-08-19 20:20:54
LastEditTime: 2025-08-21 22:48:57
LastEditors: Damocles_lin
'''
import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights
from config import config

class CustomResNet18(nn.Module):
    """
    自定义ResNet-18模型，适配CIFAR-10数据集
    
    Attributes:
        model (torch.nn.Module): 基础的ResNet-18模型
    """
    def __init__(self, num_classes=10):
        """
        初始化自定义ResNet-18模型
        
        Args:
            num_classes (int, optional): 输出类别数，默认为10（CIFAR-10）
        """
        super(CustomResNet18, self).__init__()

        # 加载与训练模型ResNet-18
        self.model = torchvision.models.resnet18(
              weights=ResNet18_Weights.DEFAULT
        ) if config["pretrained"] else torchvision.models.resnet18(weights=None)

        # 修改第一层卷积：适配CIFAR-10的32*32输入
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=1, padding=1, bias=False)
        self.model.bn1 = nn.BatchNorm2d(64)

        # 修改最后的全连接层
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features,num_classes)

    def forward(self,x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入图像张量
            
        Returns:
            torch.Tensor: 模型输出
        """
        return self.model(x)
    
def get_model(device):
    """
    创建并返回自定义ResNet-18模型
    
    Args:
        device (torch.device): 模型运行的设备
        
    Returns:
        CustomResNet18: 自定义ResNet-18模型实例
    """
    model = CustomResNet18(num_classes=10).to(device)
    print("模型结构：")
    print(model)
    return model
