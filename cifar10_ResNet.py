'''
Description: 分类神经网络的搭建和实验
Author: Damocles_lin
Date: 2025-08-07 13:19:56
LastEditTime: 2025-08-07 15:42:52
LastEditors: Damocles_lin
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import platform
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import time
import os

# ==============================
# 1. 环境配置和超参数设置
# ==============================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 实验超参数
config = {
    'batch_size': 128,
    'learning_rate': 0.001,
    'num_epochs': 15,
    'num_workers': 4,
    'optimizer': 'Adam',
    'pretrained': True,
    'data_augmentation': True
}

# 创建结果保存目录
os.makedirs('results', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# ==============================
# 2. 数据预处理和加载
# ==============================
# 数据增强和归一化
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
]) if config['data_augmentation'] else transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 下载并加载CIFAR-10数据集
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(
    train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
test_loader = DataLoader(
    test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

# CIFAR-10类别名称
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

# ==============================
# 3. 模型定义 - 修改ResNet-18
# ==============================
class CustomResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomResNet18, self).__init__()
        
        # 加载预训练ResNet-18
        self.model = torchvision.models.resnet18(pretrained=config['pretrained'])
        
        # 修改第一层卷积：适配CIFAR-10的32x32输入
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.bn1 = nn.BatchNorm2d(64)
        
        # 修改最后的全连接层
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
        # 残差学习思想：通过短路连接解决深层网络梯度消失问题
        # 基本块结构：输入 -> 卷积层 -> BN -> ReLU -> 卷积层 -> BN -> 残差连接 -> ReLU
    
    def forward(self, x):
        return self.model(x)

# 创建模型并移至设备
model = CustomResNet18(num_classes=10).to(device)

# 打印模型结构
print("模型结构:")
print(model)

# ==============================
# 4. 损失函数和优化器
# ==============================
criterion = nn.CrossEntropyLoss()  # 内部包含LogSoftmax和NLLLoss
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

# 学习率调度器 - 移除verbose参数以兼容旧版本
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2)

# ==============================
# 5. 训练和验证函数
# ==============================
def train(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计信息
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 99:
            print(f'Epoch: {epoch+1} | Batch: {batch_idx+1}/{len(train_loader)} | '
                  f'Loss: {running_loss/(batch_idx+1):.4f} | Acc: {100.*correct/total:.2f}%')
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

def validate():
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss /= len(test_loader)
    val_acc = 100. * correct / total
    
    print(f'Validation | Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%')
    return val_loss, val_acc

# ==============================
# 6. 训练循环和结果记录
# ==============================
# 初始化记录变量
train_losses = []
train_accs = []
val_losses = []
val_accs = []
best_acc = 0.0

start_time = time.time()

print("开始训练...")
for epoch in range(config['num_epochs']):
    train_loss, train_acc = train(epoch)
    val_loss, val_acc = validate()
    
    # 更新学习率
    scheduler.step(val_acc)
    
    # 保存训练过程
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), f'checkpoints/best_model.pth')
        print(f'保存最佳模型，准确率: {best_acc:.2f}%')

# 保存最终模型
torch.save(model.state_dict(), f'checkpoints/final_model.pth')
print(f"训练完成! 总耗时: {time.time()-start_time:.2f}秒")

# ==============================
# 7. 结果可视化和分析
# ==============================
# 绘制学习曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='训练损失')
plt.plot(val_losses, label='验证损失')
plt.title('损失曲线')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='训练准确率')
plt.plot(val_accs, label='验证准确率')
plt.title('准确率曲线')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig('results/learning_curves.png')
plt.close()

# 在测试集上评估最终模型
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
model.eval()

all_targets = []
all_preds = []
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_targets.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

test_acc = 100 * correct / total
print(f'测试集准确率: {test_acc:.2f}%')

# 生成混淆矩阵
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes)
plt.title('混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.savefig('results/confusion_matrix.png')
plt.close()

# 保存类别准确率
class_acc = {}
for i in range(10):
    class_acc[classes[i]] = cm[i, i] / cm[i].sum() * 100

# 可视化一些预测结果
def imshow(img):
    img = img / 2 + 0.5  # 反归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

# 获取一批测试图像
dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.cpu(), labels.cpu()

# 显示图像和预测
outputs = model(images.to(device))
_, predicted = torch.max(outputs, 1)

# 绘制预测结果
plt.figure(figsize=(15, 10))
for i in range(20):
    plt.subplot(4, 5, i+1)
    imshow(images[i])
    
    color = 'green' if predicted[i] == labels[i] else 'red'
    plt.title(f'{classes[predicted[i]]}\n({classes[labels[i]]})', color=color)
    
plt.tight_layout()
plt.savefig('results/predictions.png')
plt.close()

# ==============================
# 8. 生成实验报告
# ==============================
report = f"""
{'='*50}
实验报告: 基于ResNet-18的CIFAR-10图像分类
{'='*50}

1. 运行环境
   - 操作系统: {platform.system()}
   - Python版本: {platform.python_version()}
   - PyTorch版本: {torch.__version__}
   - CUDA版本: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}
   - GPU型号: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}

2. 网络结构
   - 基础模型: ResNet-18
   - 修改部分:
       1. 第一层卷积: kernel_size=3, stride=1, padding=1 (适应32x32输入)
       2. 添加BatchNorm层
       3. 最后一层全连接层: 输出维度10 (对应10个类别)
   - 残差学习: 通过短路连接解决深层网络梯度消失问题
   - 总参数量: {sum(p.numel() for p in model.parameters()):,}

3. 数据处理
   - 训练集增强: {'是' if config['data_augmentation'] else '否'}
        - 随机裁剪 (32x32, padding=4)
        - 随机水平翻转
   - 归一化: 使用CIFAR-10的均值和标准差

4. 损失函数
   - CrossEntropyLoss: 适用于多分类任务，结合了LogSoftmax和NLLLoss

5. 实验设置
   | 超参数         | 值          |
   |----------------|------------|
   | Batch Size     | {config['batch_size']} |
   | Learning Rate  | {config['learning_rate']} |
   | Optimizer      | {config['optimizer']} |
   | Epochs         | {config['num_epochs']} |
   | 预训练         | {'是' if config['pretrained'] else '否'} |
   | 数据增强       | {'是' if config['data_augmentation'] else '否'} |

6. 实验结果
   - 最佳验证准确率: {best_acc:.2f}%
   - 最终测试准确率: {test_acc:.2f}%
   - 类别准确率:
        {pd.Series(class_acc).to_string()}

7. 分析
   - 学习曲线: 已保存至 results/learning_curves.png
   - 混淆矩阵: 已保存至 results/confusion_matrix.png
   - 预测示例: 已保存至 results/predictions.png

8. 问题与解决方案
   - 输入尺寸不匹配: 通过修改第一层卷积解决
   - ReduceLROnPlateau兼容性问题: 移除verbose参数
   - 过拟合风险: 使用数据增强和预训练权重缓解
   - 学习率调整: 使用ReduceLROnPlateau动态调整学习率

{'='*50}
实验完成! 模型和结果已保存至 checkpoints/ 和 results/ 目录
{'='*50}
"""

print(report)
with open('results/experiment_report.txt', 'w') as f:
    f.write(report)