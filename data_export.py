'''
Description: 
Author: Damocles_lin
Date: 2025-08-09 17:04:41
LastEditTime: 2025-08-09 22:57:30
LastEditors: Damocles_lin
'''
import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_cifar_batch(file_path):
    """加载单个 CIFAR-10 批次文件"""
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch

def display_batch_info(batch):
    """显示批次文件信息并可视化图像"""
    # 解码字节键名
    data = batch[b'data']
    labels = batch[b'labels']
    filenames = [name.decode('utf-8') for name in batch[b'filenames']]
    
    print(f"数据维度: {data.shape}")
    print(f"标签数量: {len(labels)}")
    print(f"前5个文件名: {filenames[:5]}")
    print(f"前5个标签: {labels[:5]}")
    
    # 可视化前16张图像
    plt.figure(figsize=(10, 10))
    for i in range(16):
        # 重塑图像 (3072 = 32x32x3)
        img = data[i].reshape(3, 32, 32).transpose(1, 2, 0)
        # 反归一化 (原始值范围 0-255)
        img = img / 255.0
        
        plt.subplot(4, 4, i+1)
        plt.imshow(img)
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 示例：查看第一个训练批次
batch_path = './data/cifar-10-batches-py/data_batch_1'
batch = load_cifar_batch(batch_path)
display_batch_info(batch)

# 查看元数据
meta_path = './data/cifar-10-batches-py/batches.meta'
with open(meta_path, 'rb') as f:
    meta = pickle.load(f, encoding='bytes')
print("\n元数据:")
print(f"标签名称: {[name.decode('utf-8') for name in meta[b'label_names']]}")
print(f"每批次图像数: {meta[b'num_cases_per_batch']}")