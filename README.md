# 基于超材料的无源物联网感知系统复现

[cite_start]本项目是针对论文《超越后向散射通信:超材料无源物联网感知新范式》 [cite: 2] 的Python复现。项目完整地模拟了从物理环境建模、无线信号传输到最终通过深度学习进行环境状态感知的全过程。

## 项目简介

[cite_start]系统通过模拟一个由多个无源超材料标签和一对无线收发器组成的物联网感知系统 [cite: 39, 89][cite_start]，旨在实现对特定物理量（如湿度）空间分布的精确感知 [cite: 5][cite_start]。核心思想是，超材料标签的电磁散射特性会随环境物理量的变化而改变 [cite: 39, 47][cite_start]，接收器通过分析这些变化的散射信号，并利用深度神经网络来反演出环境状态的分布图 [cite: 27]。

本项目实现了以下关键模块：
* **物理环境仿真**: 随机生成待测空间的湿度分布图，并部署超材料标签。
* [cite_start]**电磁模型**: 基于等效电路理论，模拟超材料标签的阻抗和散射系数如何随湿度变化 [cite: 51, 52, 53, 54, 55, 58]。
* [cite_start]**信道传输模型**: 模拟包含目标信号、干扰信号、环境散射和噪声的复杂接收信号功率 [cite: 104, 105]。
* [cite_start]**深度学习感知模型**: 构建一个基于PyTorch的编解码神经网络，用于从复杂的接收信号中智能重构环境分布图 [cite: 45, 171]。

## 项目结构

```
.
├── dataset/              # 存放生成的训练和测试数据集 (.npz)
├── best_model.pth        # 训练后保存的最佳模型权重
│
├── config.py             # 所有仿真参数的配置文件
├── environment.py        # 环境和超材料标签的物理模型
├── model.py              # PyTorch实现的感知神经网络
├── transceiver.py        # 无线收发器和信号传输模型
│
├── generate_dataset.py   # 用于生成数据集的脚本
├── train.py              # 用于训练神经网络的脚本
│
├── requirements.txt      # 项目依赖库
└── README.md             # 本文档
```

## 安装与配置

本项目推荐使用 `conda` 进行环境管理。

**1. 克隆/下载项目**

将所有项目文件下载到您的本地计算机。

**2. 创建并激活Conda环境**

打开Anaconda Prompt或终端，运行以下命令：

```bash
# 创建一个名为 "meta_sensing" 的新环境，并指定Python版本
conda create -n meta_sensing python=3.9 -y

# 激活新创建的环境
conda activate meta_sensing
```

**3. 安装依赖**

使用项目提供的 `requirements.txt` 文件安装所有必要的库。

```bash
# 首先安装 PyTorch (根据您的CUDA情况选择相应命令，此处为CPU版本)
# 访问 [https://pytorch.org/](https://pytorch.org/) 获取适合您设备的安装命令
pip install torch==2.3.1

# 然后安装其余所有依赖
pip install -r requirements.txt
```

## 使用流程

请按照以下步骤运行整个复现流程。

**步骤 1: 生成数据集**

运行数据生成脚本，创建用于模型训练和测试的数据集。

```bash
python generate_dataset.py
```
该脚本会创建一个 `dataset/` 文件夹，并在其中生成 `train_dataset.npz` 和 `test_dataset.npz`。

**步骤 2: 训练感知模型**

运行训练脚本，它会自动加载上一步生成的数据，进行模型训练，并评估其性能。

```bash
python train.py
```
训练过程中，脚本会实时显示损失和准确率。训练结束后，性能最佳的模型权重将被保存为 `best_model.pth`。

**步骤 3: 预测与评估 (未来工作)**

下一步可以编写一个 `predict.py` 脚本：
1.  加载已训练好的模型权重 `best_model.pth`。
2.  生成一个新的、独立的测试样本。
3.  使用模型进行预测。
4.  将模型的预测结果与真实分布图进行可视化对比，以直观地评估模型性能。