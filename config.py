# config.py (修正版 - 增强信号特征)

import numpy as np

class SimulationConfig:
    # --- 物理环境参数 ---
    SENSING_AREA_WIDTH = 8.0
    SENSING_AREA_HEIGHT = 3.0
    GRID_SIZE = 0.5
    
    # --- 无线通信参数 ---
    FREQ_START = 5.0e9
    FREQ_END = 6.0e9
    FREQ_POINTS = 101
    TX_POWER_W = 1.0
    TRANSCEIVER_DISTANCE = 3.0
    
    # --- 超材料标签参数 ---
    NUM_TAGS = 15
    ETA_0 = 377.0

    # 【【核心修正1】】 显著降低环境背景反射
    # 之前0.5的值过高，导致背景噪声太强。我们假设一个更真实的、较低的反射水平。
    WALL_REFLECTION_COEFFICIENT = 0.001 

    # 【【核心修正2】】 引入标签天线增益和波束成形旁瓣抑制
    # 这更符合原文中提到的定向天线和标签阵列能够集中能量的描述。
    TAG_ANTENNA_GAIN = 200.0            # 标签等效天线增益，用于放大标签的散射信号
    BEAMFORMING_SIDELOBE_SUPPRESSION = 0.05 # 波束成形对旁瓣（干扰标签）的抑制因子 (例如-13dB)

    # --- 仿真模型参数 (保持不变) ---
    R_L_C_VAL = {'R_n': 0.5, 'L_n': 1.5e-9, 'C_surf': 0.2e-12}
    SENSITIVE_RESISTANCE_MODEL = {'R_base': 10, 'k': 0.8}
    GAP_CAPACITANCE_MODEL = {'C_base': 0.1e-12, 'gap_width': 1e-3}

    # --- 噪声与干扰 ---
    SNR_DB = 25 # 适当提高信噪比，让信号更清晰

    # --- 湿度类别定义 (保持不变) ---
    HUMIDITY_CLASSES = {0: (20, 25), 1: (25, 30), 2: (30, 35), 3: (35, 40), 4: (40, 45)}

    def __init__(self):
        self.NUM_GRIDS_X = int(self.SENSING_AREA_WIDTH / self.GRID_SIZE)
        self.NUM_GRIDS_Y = int(self.SENSING_AREA_HEIGHT / self.GRID_SIZE)
        self.TOTAL_GRIDS = self.NUM_GRIDS_X * self.NUM_GRIDS_Y
        self.FREQUENCIES = np.linspace(self.FREQ_START, self.FREQ_END, self.FREQ_POINTS)
        self.TRANSCEIVER_POS = np.array([self.SENSING_AREA_WIDTH / 2, -self.TRANSCEIVER_DISTANCE, self.SENSING_AREA_HEIGHT / 2])