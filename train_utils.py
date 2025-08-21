'''
Description: 训练函数
Author: Damocles_lin
Date: 2025-08-19 21:15:56
LastEditTime: 2025-08-21 22:48:16
LastEditors: Damocles_lin
'''
import torch
import torch.optim as optim
import time
from config import config

def get_optimizer(model):
    """
    创建优化器和学习率调度器
    
    Args:
        model (torch.nn.Module): 需要优化的模型
        
    Returns:
        tuple: 包含优化器和学习率调度器的元组
            - optimizer (torch.optim.Optimizer): Adam优化器
            - scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): 学习率调度器
    """    
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",factor=0.5, patience=2)
    return optimizer, scheduler

def train(model, train_loader, criterion, optimizer,epoch):
    """
    训练模型一个epoch
    
    Args:
        model (torch.nn.Module): 要训练的模型
        train_loader (torch.utils.data.DataLoader): 训练数据加载器
        criterion (torch.nn.Module): 损失函数
        optimizer (torch.optim.Optimizer): 优化器
        epoch (int): 当前训练周期数
        
    Returns:
        tuple: 包含训练损失和准确率的元组
            - train_loss (float): 平均训练损失
            - train_acc (float): 训练准确率百分比
    """    
    model.train()
    running_loss =0.0
    correct = 0
    total = 0

    for batch_idx, (inputs,tragets) in enumerate(train_loader):
        inputs, tragets = inputs.to(model.device), tragets.to(model.device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, tragets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计信息
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += tragets.size(0)
        correct += predicted.eq(tragets).sum().item()

        if (batch_idx+1) % 100 == 0:
            print(
                f"Epoch:{epoch+1} | Batch:{batch_idx+1}/{len(train_loader)} | "
                f"Loss: {running_loss/(batch_idx+1):.4f} | Acc: {100.*correct/total:.2f}%"
            )
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

def validate(model, loader, criterion):
    """
    验证或测试模型性能
    
    Args:
        model (torch.nn.Module): 要验证的模型
        loader (torch.utils.data.DataLoader): 验证或测试数据加载器
        criterion (torch.nn.Module): 损失函数
        
    Returns:
        tuple: 包含验证损失和准确率的元组
            - val_loss (float): 平均验证损失
            - val_acc (float): 验证准确率百分比
    """
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs,tragets) in enumerate(loader):
            inputs, tragets = inputs.to(model.device), tragets.to(model.device)

            outputs = model(inputs)
            loss = criterion(outputs, tragets)

            # 统计信息
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += tragets.size(0)
            correct += predicted.eq(tragets).sum().item()

    val_loss /= len(loader)
    val_acc = 100. * correct / total

    print(f"Validation | Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
    return val_loss, val_acc

def train_loop(model, train_loader, val_loader, criterion, optimizer,scheduler):
    """
    完整的训练循环，包括多个epoch的训练和验证
    
    Args:
        model (torch.nn.Module): 要训练的模型
        train_loader (torch.utils.data.DataLoader): 训练数据加载器
        val_loader (torch.utils.data.DataLoader): 验证数据加载器
        criterion (torch.nn.Module): 损失函数
        optimizer (torch.optim.Optimizer): 优化器
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器
        
    Returns:
        tuple: 包含训练过程中各项指标的元组
            - train_losses (list): 每个epoch的训练损失列表
            - train_accs (list): 每个epoch的训练准确率列表
            - val_losses (list): 每个epoch的验证损失列表
            - val_accs (list): 每个epoch的验证准确率列表
            - best_acc (float): 最佳验证准确率
    """
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_acc = 0.0

    start_time = time.time()

    print("开始训练...")
    for epoch in range(config["num_epochs"]):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion)

        # 更新学习率
        scheduler.step(val_acc)

        # 保存训练过程
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_losses.append(val_loss)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"checkpoints/best_model.pth")
            print(f"保存最佳模型，准确率：{best_acc:.2f}%")
    
    # 保存最终的模型
    torch.save(model.state_dict(), f"checkpoints/final_model.pth")
    print(f"训练完成! 总耗时: {time.time()-start_time:.2f}秒")

    return train_losses, train_accs, val_losses, val_accs, best_acc