'''
Description: 
Author: Damocles_lin
Date: 2025-08-20 15:07:38
LastEditTime: 2025-08-22 20:47:48
LastEditors: Damocles_lin
'''
import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from config import classes, config

def plot_learning_curves(train_losses, val_losses, train_accs, val_accs):
    """
    绘制训练和验证的损失曲线及准确率曲线
    
    Args:
        train_losses (list): 训练损失值列表
        val_losses (list): 验证损失值列表
        train_accs (list): 训练准确率列表
        val_accs (list): 验证准确率列表
        
    Returns:
        None: 函数直接保存图像文件，不返回任何值
    """
    x1_coordinates = [i + 1 for i in range(len(train_losses))]
    x2_coordinates = [i + 1 for i in range(len(val_losses))]    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x1_coordinates, train_losses, label='Training loss')
    plt.plot(x2_coordinates, val_losses, label='Validation loss')
    plt.title('Loss curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    x1_coordinates = [i + 1 for i in range(len(train_accs))]
    x2_coordinates = [i + 1 for i in range(len(val_accs))] 
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training accuracy')
    plt.plot(val_accs, label='Validation accuracy')
    plt.title('Accuracy curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/learning_curves.png')
    plt.close()

def plot_confusion_matrix(model, test_loader):
    """
    绘制混淆矩阵并计算各类别准确率
    
    Args:
        model (torch.nn.Module): 训练好的模型
        test_loader (torch.utils.data.DataLoader): 测试数据加载器
        
    Returns:
        dict: 包含每个类别准确率的字典
    """    
    all_targets = []
    all_preds = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(model.device), labels.to(model.device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion matrix')
    plt.xlabel('Prediction label')
    plt.ylabel('Authentic labels')
    plt.savefig('results/confusion_matrix.png')
    plt.close()
    
    # 计算类别准确率
    class_acc = {}
    for i in range(10):
        class_acc[classes[i]] = cm[i, i] / cm[i].sum() * 100
    
    return class_acc

def imshow(img):
    """
    显示单张图像（反归一化处理）
    
    Args:
        img (torch.Tensor): 输入的图像张量
        
    Returns:
        None: 函数直接显示图像，不返回任何值
    """
    # CIFAR-10 官方的标准化参数
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])

    img = img * std[:, None, None] + mean[:, None, None]  # 反归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

def plot_predictions(model, test_loader):
    """
    绘制模型在测试集上的预测结果示例
    
    Args:
        model (torch.nn.Module): 训练好的模型
        test_loader (torch.utils.data.DataLoader): 测试数据加载器
        
    Returns:
        None: 函数直接保存图像文件，不返回任何值
    """    
    # 获取一批测试图像
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.cpu(), labels.cpu()

    # 显示图像和预测
    outputs = model(images.to(model.device))
    _, predicted = torch.max(outputs, 1)

    # 绘制预测结果
    plt.figure(figsize=(15, 10))
    for i in range(20):
        plt.subplot(4, 5, i+1)
        imshow(images[i])
        
        color = 'green' if predicted[i] == labels[i] else 'red'
        plt.title(f'pred: {classes[predicted[i]]}\ntrue: {classes[labels[i]]}', color=color)
        
    plt.tight_layout()
    plt.savefig('results/predictions.png')
    plt.close()

def plot_correct_predictions(model, test_loader, num_examples=20):
    """
    显示所有正确的预测图片
    
    Args:
        model (torch.nn.Module): 训练好的模型
        test_loader (torch.utils.data.DataLoader): 测试数据加载器
        num_examples (int, optional): 要显示的示例数量，默认为20
        
    Returns:
        None: 函数直接保存图像文件，不返回任何值
    """
    # 收集所有正确的预测
    correct_images = []
    correct_labels = []
    correct_preds = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(model.device), labels.to(model.device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # 找出正确的预测
            correct_mask = predicted == labels
            correct_indices = correct_mask.nonzero(as_tuple=True)[0]
            
            for idx in correct_indices:
                if len(correct_images) < num_examples:
                    correct_images.append(images[idx].cpu())
                    correct_labels.append(labels[idx].cpu())
                    correct_preds.append(predicted[idx].cpu())
                else:
                    break
            
            if len(correct_images) >= num_examples:
                break
    
    # 绘制正确的预测结果
    plt.figure(figsize=(15, 10))
    for i in range(min(num_examples, len(correct_images))):
        plt.subplot(4, 5, i+1)
        imshow(correct_images[i])
        
        plt.title(f'pred: {classes[correct_preds[i]]}\ntrue: {classes[correct_labels[i]]}', color='green')
        
    plt.tight_layout()
    plt.savefig('results/correct_predictions.png')
    plt.close()

def plot_incorrect_predictions(model, test_loader, num_examples=20):
    """
    显示所有错误的预测图片
    
    Args:
        model (torch.nn.Module): 训练好的模型
        test_loader (torch.utils.data.DataLoader): 测试数据加载器
        num_examples (int, optional): 要显示的示例数量，默认为20
        
    Returns:
        None: 函数直接保存图像文件，不返回任何值
    """
    # 收集所有错误的预测
    incorrect_images = []
    incorrect_labels = []
    incorrect_preds = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(model.device), labels.to(model.device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # 找出错误的预测
            incorrect_mask = predicted != labels
            incorrect_indices = incorrect_mask.nonzero(as_tuple=True)[0]
            
            for idx in incorrect_indices:
                if len(incorrect_images) < num_examples:
                    incorrect_images.append(images[idx].cpu())
                    incorrect_labels.append(labels[idx].cpu())
                    incorrect_preds.append(predicted[idx].cpu())
                else:
                    break
            
            if len(incorrect_images) >= num_examples:
                break
    
    # 绘制错误的预测结果
    plt.figure(figsize=(15, 10))
    for i in range(min(num_examples, len(incorrect_images))):
        plt.subplot(4, 5, i+1)
        imshow(incorrect_images[i])
        
        plt.title(f'pred: {classes[incorrect_preds[i]]}\ntrue: {classes[incorrect_labels[i]]}', color='red')
        
    plt.tight_layout()
    plt.savefig('results/incorrect_predictions.png')
    plt.close()

def generate_report(train_size, val_size, test_size, model, best_acc, test_acc, class_acc):
    """
    生成实验报告并保存为文本文件
    
    Args:
        train_size (int): 训练集大小
        val_size (int): 验证集大小
        test_size (int): 测试集大小
        model (torch.nn.Module): 训练好的模型
        best_acc (float): 最佳验证准确率
        test_acc (float): 测试准确率
        class_acc (dict): 各类别准确率字典
        
    Returns:
        None: 函数直接保存报告文件，不返回任何值
    """
    import platform
    
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
   - 数据集划分:
        - 训练集: {train_size} 样本
        - 验证集: {val_size} 样本
        - 测试集: {test_size} 样本

4. 损失函数
   - CrossEntropyLoss: 适用于多分类任务，结合了LogSoftmax和NLLLoss

5. 实验设置
   | 超参数             |  值        |
   |-------------------|------------|
   | Batch Size        | {config['batch_size']} |
   | Learning Rate     | {config['learning_rate']} |
   | Optimizer         | {config['optimizer']} |
   | Epochs            | {config['num_epochs']} |
   | pretrained        | {'是' if config['pretrained'] else '否'} |
   | data_augmentation | {'是' if config['data_augmentation'] else '否'} |

6. 实验结果
   - 最佳验证准确率: {best_acc:.2f}%
   - 测试准确率: {test_acc:.2f}%
   - 类别准确率:
{pd.Series(class_acc).to_string()}

7. 分析
   - 学习曲线: 已保存至 results/learning_curves.png
   - 混淆矩阵: 已保存至 results/confusion_matrix.png
   - 预测示例: 已保存至 results/predictions.png
   - 正确预测示例: 已保存至 results/correct_predictions.png
   - 错误预测示例: 已保存至 results/incorrect_predictions.png
   

{'='*50}
实验完成! 模型和结果已保存至 checkpoints/ 和 results/ 目录
{'='*50}
"""

    print(report)
    with open('results/experiment_report.txt', 'w') as f:
        f.write(report)