<!--
 * @Description: 
 * @Author: Damocles_lin
 * @Date: 2025-08-22 14:16:42
 * @LastEditTime: 2025-08-22 15:03:07
 * @LastEditors: Damocles_lin
-->
# CIFAR-10 图像分类项目
基于 ResNet-18 的 CIFAR-10 数据集图像分类项目，包含完整的训练、验证、测试流程及结果可视化功能。模型基于ResNet-18，并针对CIFAR-10数据集进行了修改：修改第一层卷积层，适应32x32的输入尺寸，添加BatchNorm层修改全连接层输出为10类。

# 项目结构
```text
.
├── main.py               # 主程序入口
├── data_loader.py        # 数据加载与预处理
├── model.py              # 模型定义(自定义ResNet-18)
├── train_utils.py        # 训练相关工具函数
├── visualization.py      # 结果可视化函数
├── config.py             # 配置参数
├── checkpoints/          # 模型保存目录
├── results/              # 结果文件保存目录
├── data/                 # 数据集存放目录
├── requirements.txt      # 项目依赖
└── README.md             # 项目说明
```
# 安装依赖
```bash
pip install -r requirements.txt
```

# 使用方法
运行主程序开始训练和测试：
```bash
python main.py
```
程序会自动下载CIFAR-10数据集（如果尚未下载），进行训练，并生成以下结果：
- 学习曲线（results/learning_curves.png）
- 混淆矩阵（results/confusion_matrix.png）
- 预测示例（results/predictions.png）
- 正确预测示例（results/correct_predictions.png）
- 错误预测示例（results/incorrect_predictions.png）
- 实验报告（results/experiment_report.txt）

# 配置
在`config.py`中可以修改超参数，包括：
- 批量大小（batch_size）
- 学习率（learning_rate）
- 训练轮数（num_epochs）
- 是否使用预训练模型（pretrained）
- 是否使用数据增强（data_augmentation）

# 参考
[pytorch](https://pytorch.org/),[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)