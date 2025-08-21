'''
Description: 数据加载
Author: Damocles_lin
Date: 2025-08-19 16:27:45
LastEditTime: 2025-08-21 22:48:41
LastEditors: Damocles_lin
'''
import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, random_split
from config import config

def get_data_loaders():
    """
    获取CIFAR-10数据集的数据加载器
    
    Returns:
        tuple: 包含数据加载器和数据集大小的元组
            - train_loader (torch.utils.data.DataLoader): 训练数据加载器
            - val_loader (torch.utils.data.DataLoader): 验证数据加载器
            - test_loader (torch.utils.data.DataLoader): 测试数据加载器
            - train_size (int): 训练集大小
            - val_size (int): 验证集大小
            - test_size (int): 测试集大小
    """
    # 数据增强和归一化
    transform_train = transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
    ]) if config["data_augmentation"] else transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
    ])

    # 下载并加载CIFAR-10数据集
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, transform=transform_train, download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, transform=transform_test, download=True
    )

    # 划分训练集和验证集（90% 训练， 10% 验证）
    generator0 = torch.Generator().manual_seed(0)
    train_subset, val_subset = random_split(train_dataset, lengths=[0.9, 0.1],generator=generator0)

    # 验证集使用测试集transform（无数据增强）
    val_subset.dataset.transform = transform_test

    train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    val_loader = DataLoader(val_subset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    
    return train_loader, val_loader, test_loader, len(train_subset), len(val_subset), len(test_dataset)