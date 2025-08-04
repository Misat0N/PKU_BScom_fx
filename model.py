# # model.py

# import torch
# import torch.nn as nn
# from config import SimulationConfig # 导入我们的配置文件

# class SensingNetwork(nn.Module):
#     """
#     根据论文图3和项目配置适配后的神经网络模型。
#     """
#     def __init__(self, config: SimulationConfig):
#         """
#         初始化网络。所有参数从config对象中获取。
#         """
#         super(SensingNetwork, self).__init__()
        
#         # 保存维度信息
#         self.config = config
        
#         # ---- 定义网络各层 ----
        
#         # 1. 双向循环网络层 (LSTM)
#         # 提取测量矩阵中的时序/序列特征
#         rnn_hidden_size = 256 # 这是一个可以调整的超参数
#         self.rnn = nn.LSTM(
#             input_size=config.FREQ_POINTS,
#             hidden_size=rnn_hidden_size,
#             num_layers=2,
#             batch_first=True,
#             bidirectional=True
#         )
        
#         # 2. 全连接层
#         # 将RNN提取的特征映射到2D空间
#         # 【适配点】: 为了得到最终的 6x16 网格，我们设定一个中间尺寸。
#         # 选择 Y=3, X=8, 上采样因子n=2, 最终输出 (2*3)x(2*8) = 6x16，与仿真环境完全匹配。
#         self.intermediate_y = 3
#         self.intermediate_x = 8
        
#         self.fc = nn.Linear(
#             in_features=config.NUM_TAGS * rnn_hidden_size * 2, # *2 是因为双向
#             out_features=self.intermediate_y * self.intermediate_x * 16 # 输出16个通道给卷积层
#         )
        
#         # 3. 编码器：两个2D卷积层
#         # 使用 kernel_size=3, padding=1 来保持空间维度(Y, X)不变
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
        
#         # 4. 解码器：一个2D反卷积层
#         self.decoder = nn.ConvTranspose2d(
#             in_channels=64,
#             out_channels=len(config.HUMIDITY_CLASSES), # 输出通道数等于类别数
#             kernel_size=2, # 上采样因子n=2
#             stride=2
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         定义前向传播路径。
        
#         Args:
#             x (torch.Tensor): 输入张量，形状 (batch_size, N, L)。
        
#         Returns:
#             torch.Tensor: 输出的原始Logits，形状 (batch_size, M, K_n)。
#         """
#         batch_size = x.size(0)

#         # 1. 通过RNN层
#         rnn_out, _ = self.rnn(x)
        
#         # 2. 展平并通过FC层
#         rnn_out_flat = rnn_out.reshape(batch_size, -1)
#         fc_out = self.fc(rnn_out_flat)
        
#         # 3. Reshape为2D特征图，准备输入卷积层
#         # (batch, 16, Y, X)
#         conv_input = fc_out.view(batch_size, 16, self.intermediate_y, self.intermediate_x)
        
#         # 4. 通过卷积编码器
#         encoded_features = self.encoder(conv_input)
        
#         # 5. 通过反卷积解码器
#         # (batch, K_n, n*Y, n*X) -> (batch, K_n, 6, 16)
#         output_logits_2d = self.decoder(encoded_features)
        
#         # 6. 【重要改动】调整输出形状以匹配 (batch, M, K_n)
#         # PyTorch的CrossEntropyLoss期望的Logits形状是 (batch, K_n, M) 或 (batch, M, K_n)
#         # 我们调整为 (batch, M, K_n) 以便直观理解和使用
        
#         # (batch, K_n, 6, 16) -> (batch, K_n, 96)
#         output_logits_flat = output_logits_2d.view(batch_size, len(self.config.HUMIDITY_CLASSES), -1)
        
#         # (batch, K_n, 96) -> (batch, 96, K_n)
#         output_logits = output_logits_flat.permute(0, 2, 1)

#         return output_logits


# # --- 主程序入口，用于演示和测试 ---
# if __name__ == '__main__':
#     # 1. 从配置文件加载参数
#     config = SimulationConfig()

#     print("=" * 60)
#     print("适配项目配置后的神经网络模型")
#     print(f"输入维度 (Tags, Freqs): ({config.NUM_TAGS}, {config.FREQ_POINTS})")
#     print(f"输出维度 (Total Grids, Classes): ({config.TOTAL_GRIDS}, {len(config.HUMIDITY_CLASSES)})")
#     print("=" * 60)

#     # 2. 实例化模型
#     model = SensingNetwork(config)
#     print("\n模型结构:")
#     print(model)

#     # 3. 创建一个虚拟的输入张量 (模拟一批测量矩阵)
#     BATCH_SIZE = 4
#     dummy_input = torch.randn(BATCH_SIZE, config.NUM_TAGS, config.FREQ_POINTS)
#     print(f"\n输入张量 (P_R) 的形状: {dummy_input.shape}")

#     # 4. 通过模型进行前向传播
#     output = model(dummy_input)
#     print(f"最终输出Logits张量 (Q_n) 的形状: {output.shape}")
#     print(f"预期的输出形状: ({BATCH_SIZE}, {config.TOTAL_GRIDS}, {len(config.HUMIDITY_CLASSES)})")

#     # 5. 验证输出
#     assert output.shape == (BATCH_SIZE, config.TOTAL_GRIDS, len(config.HUMIDITY_CLASSES)), "输出形状不正确!"
#     print("\n✓ 输出形状验证成功！")
    
#     # 6. (可选) 检查经过softmax后的概率和是否为1
#     # 在实际训练中，softmax操作会由损失函数(nn.CrossEntropyLoss)隐式完成
#     output_probs = torch.softmax(output, dim=2)
#     prob_sum = output_probs.sum(dim=2) # 在K_n维度上求和
#     assert torch.allclose(prob_sum, torch.ones(BATCH_SIZE, config.TOTAL_GRIDS)), "每个网格的概率和不为1!"
#     print("✓ Softmax后的概率和验证成功！")

# model.py

# model.py (增强性能版)

import torch
import torch.nn as nn
from config import SimulationConfig # 导入我们的配置文件

class SensingNetwork(nn.Module):
    """
    【【增强性能版】】
    通过增加通道数来提升模型容量，旨在获得更高的准确率。
    通道数: FC(64) -> Conv1(128) -> Conv2(256)
    """
    def __init__(self, config: SimulationConfig):
        """
        初始化网络。所有参数从config对象中获取。
        """
        super(SensingNetwork, self).__init__()
        
        self.config = config
        
        # 1. 双向循环网络层 (LSTM) - 保持不变
        rnn_hidden_size = 256
        self.rnn = nn.LSTM(
            input_size=config.FREQ_POINTS,
            hidden_size=rnn_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # 2. 全连接层
        self.intermediate_y = 3
        self.intermediate_x = 8
        
        # 【【核心修正1】】 显著增加FC层的输出通道数，从4/16增加到64
        # 这为后续卷积层提供了更丰富的初始特征
        fc_out_channels = 64
        
        self.fc = nn.Linear(
            in_features=config.NUM_TAGS * rnn_hidden_size * 2, # *2 是因为双向
            out_features=self.intermediate_y * self.intermediate_x * fc_out_channels
        )
        
        # 3. 编码器：两个2D卷积层
        # 【【核心修正2】】 相应地增加卷积层的通道数
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=fc_out_channels, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # 4. 解码器：一个2D反卷积层
        # 【【核心修正3】】 反卷积层的输入通道数现在是256
        self.decoder = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=len(config.HUMIDITY_CLASSES), # 输出通道数等于类别数
            kernel_size=2, # 上采样因子n=2
            stride=2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        定义前向传播路径。
        """
        batch_size = x.size(0)

        # 1. 通过RNN层
        rnn_out, _ = self.rnn(x)
        
        # 2. 展平并通过FC层
        rnn_out_flat = rnn_out.reshape(batch_size, -1)
        fc_out = self.fc(rnn_out_flat)
        
        # 3. Reshape为2D特征图，准备输入卷积层
        # 【【核心修正4】】 Reshape时使用新的通道数 64
        conv_input = fc_out.view(batch_size, 64, self.intermediate_y, self.intermediate_x)
        
        # 4. 通过卷积编码器
        encoded_features = self.encoder(conv_input)
        
        # 5. 通过反卷积解码器
        output_logits_2d = self.decoder(encoded_features)
        
        # 6. 调整输出形状以匹配 (batch, M, K_n)
        output_logits_flat = output_logits_2d.view(batch_size, len(self.config.HUMIDITY_CLASSES), -1)
        output_logits = output_logits_flat.permute(0, 2, 1)

        return output_logits


# --- 主程序入口，用于演示和测试 ---
if __name__ == '__main__':
    # 1. 从配置文件加载参数
    config = SimulationConfig()

    print("=" * 60)
    print("【增强性能版】神经网络模型")
    print(f"输入维度 (Tags, Freqs): ({config.NUM_TAGS}, {config.FREQ_POINTS})")
    print(f"输出维度 (Total Grids, Classes): ({config.TOTAL_GRIDS}, {len(config.HUMIDITY_CLASSES)})")
    print("=" * 60)

    # 2. 实例化模型
    model = SensingNetwork(config)
    print("\n模型结构:")
    print(model)

    # 3. 创建一个虚拟的输入张量 (模拟一批测量矩阵)
    BATCH_SIZE = 4
    dummy_input = torch.randn(BATCH_SIZE, config.NUM_TAGS, config.FREQ_POINTS)
    print(f"\n输入张量 (P_R) 的形状: {dummy_input.shape}")

    # 4. 通过模型进行前向传播
    output = model(dummy_input)
    print(f"最终输出Logits张量 (Q_n) 的形状: {output.shape}")
    print(f"预期的输出形状: ({BATCH_SIZE}, {config.TOTAL_GRIDS}, {len(config.HUMIDITY_CLASSES)})")

    # 5. 验证输出
    assert output.shape == (BATCH_SIZE, config.TOTAL_GRIDS, len(config.HUMIDITY_CLASSES)), "输出形状不正确!"
    print("\n✓ 输出形状验证成功！")
    
    # 6. (可选) 检查经过softmax后的概率和是否为1
    output_probs = torch.softmax(output, dim=2)
    prob_sum = output_probs.sum(dim=2)
    assert torch.allclose(prob_sum, torch.ones(BATCH_SIZE, config.TOTAL_GRIDS)), "每个网格的概率和不为1!"
    print("✓ Softmax后的概率和验证成功！")