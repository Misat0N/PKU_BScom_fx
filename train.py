# # train.py (修正版)

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# from tqdm import tqdm
# import os

# from config import SimulationConfig
# from model import SensingNetwork

# # 1. 创建自定义数据集类来加载 .npz 数据
# class SensingDataset(Dataset):
#     def __init__(self, filepath):
#         data = np.load(filepath)
#         self.X = torch.tensor(data['X'], dtype=torch.float32)
#         self.y = torch.tensor(data['y'], dtype=torch.long)
        
#     def __len__(self):
#         return len(self.X)
    
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]

# def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
#     """
#     模型训练主循环。
#     """
#     model.to(device)
#     best_val_loss = float('inf')

#     for epoch in range(num_epochs):
#         # --- 训练阶段 ---
#         model.train()
#         running_loss = 0.0
#         train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
        
#         for inputs, labels in train_pbar:
#             inputs, labels = inputs.to(device), labels.to(device)
            
#             optimizer.zero_grad()
#             outputs = model(inputs)
            
#             # 【【核心修正点】】
#             # 使用 reshape 解决内存不连续问题，并确保维度正确 (N, C)
#             loss = criterion(outputs.reshape(-1, len(model.config.HUMIDITY_CLASSES)), labels.reshape(-1))
            
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item()
#             train_pbar.set_postfix({'loss': running_loss / (train_pbar.n + 1)})
            
#         # --- 验证阶段 ---
#         model.eval()
#         val_loss = 0.0
#         correct_predictions = 0
#         total_predictions = 0
        
#         with torch.no_grad():
#             val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
#             for inputs, labels in val_pbar:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs)
                
#                 # 【【核心修正点】】
#                 loss = criterion(outputs.reshape(-1, len(model.config.HUMIDITY_CLASSES)), labels.reshape(-1))
#                 val_loss += loss.item()

#                 _, predicted = torch.max(outputs.data, 2)
#                 total_predictions += labels.numel()
#                 correct_predictions += (predicted == labels).sum().item()

#                 val_pbar.set_postfix({'val_loss': val_loss / (val_pbar.n + 1)})

#         avg_val_loss = val_loss / len(val_loader)
#         accuracy = 100 * correct_predictions / total_predictions
#         print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")

#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             torch.save(model.state_dict(), 'best_model.pth')
#             print(f"✓ Best model saved with validation loss: {best_val_loss:.4f}")

# # --- 主程序入口 ---
# if __name__ == '__main__':
#     LEARNING_RATE = 0.005
#     BATCH_SIZE = 32
#     NUM_EPOCHS = 100

#     config = SimulationConfig()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     dataset_dir = "dataset"
#     if not os.path.exists(os.path.join(dataset_dir, "train_dataset.npz")):
#         print(f"错误: 找不到数据集文件！请先运行 'python generate_dataset.py'。")
#     else:
#         train_dataset = SensingDataset(os.path.join(dataset_dir, "train_dataset.npz"))
#         test_dataset = SensingDataset(os.path.join(dataset_dir, "test_dataset.npz"))
        
#         train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#         test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#         model = SensingNetwork(config)
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#         print("\n--- 开始训练模型 ---")
#         train_model(model, train_loader, test_loader, criterion, optimizer, NUM_EPOCHS, device)
#         print("\n--- 训练完成 ---")
#         print("表现最好的模型权重已保存为 'best_model.pth'")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau # 导入学习率调度器
import numpy as np
from tqdm import tqdm
import os

from config import SimulationConfig
from model import SensingNetwork

class SensingDataset(Dataset):
    def __init__(self, filepath):
        data = np.load(filepath)
        self.X = torch.tensor(data['X'], dtype=torch.float32)
        self.y = torch.tensor(data['y'], dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    """
    模型训练主循环。
    """
    model.to(device)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
        
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.reshape(-1, len(model.config.HUMIDITY_CLASSES)), labels.reshape(-1))
            loss.backward()

            # 【【新增】】 进行梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{running_loss / (train_pbar.n + 1):.4f}'})
            
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.reshape(-1, len(model.config.HUMIDITY_CLASSES)), labels.reshape(-1))
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 2)
                total_predictions += labels.numel()
                correct_predictions += (predicted == labels).sum().item()
                val_pbar.set_postfix({'val_loss': f'{val_loss / (val_pbar.n + 1):.4f}'})

        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct_predictions / total_predictions
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%, LR: {current_lr:.6f}")

        # 【【新增】】 根据验证损失更新学习率
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ Best model saved with validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    LEARNING_RATE = 0.001 # 初始学习率
    BATCH_SIZE = 32
    NUM_EPOCHS = 30 # 保持或增加训练轮数

    config = SimulationConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_dir = "dataset"
    if not os.path.exists(os.path.join(dataset_dir, "train_dataset.npz")):
        print(f"错误: 找不到数据集文件！请先运行 'python generate_dataset.py'。")
    else:
        train_dataset = SensingDataset(os.path.join(dataset_dir, "train_dataset.npz"))
        test_dataset = SensingDataset(os.path.join(dataset_dir, "test_dataset.npz"))
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = SensingNetwork(config)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 【【新增】】 定义学习率调度器
        # 如果验证损失在3个epoch内没有改善，则将学习率乘以0.1 (factor)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

        print("\n--- 开始训练模型 ---")
        train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, NUM_EPOCHS, device)
        print("\n--- 训练完成 ---")
        print("表现最好的模型权重已保存为 'best_model.pth'")