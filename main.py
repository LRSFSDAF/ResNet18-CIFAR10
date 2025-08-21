'''
Description: 主函数
Author: Damocles_lin
Date: 2025-08-19 16:10:51
LastEditTime: 2025-08-21 22:26:52
LastEditors: Damocles_lin
'''
import torch
import torch.nn as nn
from config import device, create_dirs
from data_loader import get_data_loaders
from model import get_model
from train_utils import get_optimizer, train_loop, validate
from visualization import (plot_learning_curves, plot_confusion_matrix, plot_predictions,
                           plot_correct_predictions, plot_incorrect_predictions, generate_report)
def main():
    # 初始化
    create_dirs()

    # 获取数据加载器
    train_loader, val_loader, test_loader, train_size, val_size, test_size = get_data_loaders()

    # 获取模型
    model = get_model(device)
    model.device = device

    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 优化器和学习率调度器
    optimizer, scheduler = get_optimizer(model)
    
    # 训练模型
    train_losses, train_accs, val_losses, val_accs, best_acc = train_loop(
        model, train_loader, val_loader, criterion, optimizer, scheduler
    )

    # 可视化学习曲线
    plot_learning_curves(train_losses, val_losses, train_accs, val_accs)
    
    # 加载最佳模型进行测试
    model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    test_loss, test_acc = validate(model, test_loader, criterion)
    
    # 可视化混淆矩阵
    class_acc = plot_confusion_matrix(model, test_loader)
    
    # 可视化预测结果
    plot_predictions(model, test_loader)

    # 可视化正确预测结果
    plot_correct_predictions(model, test_loader)
    
    # 可视化错误预测结果
    plot_incorrect_predictions(model, test_loader)

    # 生成实验报告
    generate_report(train_size, val_size, test_size, model, best_acc, test_acc, class_acc)


if __name__ == "__main__":
    main()
