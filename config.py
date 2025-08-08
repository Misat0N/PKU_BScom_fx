import numpy as np

class SimulationConfig:
    """
    存放所有仿真参数。
    """
    # 【【核心修正】】
    # 在类的顶层声明实例属性及其类型
    # 这会告诉 Pylance 和其他开发者，每个 SimulationConfig 实例都会有一个名为 SNR_DB 的整数属性。
    SNR_DB: int

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

    # --- 信号特征参数 ---
    WALL_REFLECTION_COEFFICIENT = 0.001 
    TAG_ANTENNA_GAIN = 200.0
    BEAMFORMING_SIDELOBE_SUPPRESSION = 0.05

    # --- 湿度类别定义 ---
    HUMIDITY_CLASSES = {
        0: (20.0, 22.5), 1: (22.5, 25.0), 2: (25.0, 27.5),
        3: (27.5, 30.0), 4: (30.0, 32.5), 5: (32.5, 35.0),
        6: (35.0, 37.5), 7: (37.5, 40.0), 8: (40.0, 42.5),
        9: (42.5, 45.0)
    }

    def __init__(self):
        self.NUM_GRIDS_X = int(self.SENSING_AREA_WIDTH / self.GRID_SIZE)
        self.NUM_GRIDS_Y = int(self.SENSING_AREA_HEIGHT / self.GRID_SIZE)
        self.TOTAL_GRIDS = self.NUM_GRIDS_X * self.NUM_GRIDS_Y
        self.FREQUENCIES = np.linspace(self.FREQ_START, self.FREQ_END, self.FREQ_POINTS)
        self.TRANSCEIVER_POS = np.array([self.SENSING_AREA_WIDTH / 2, -self.TRANSCEIVER_DISTANCE, self.SENSING_AREA_HEIGHT / 2])
        
        # 在 __init__ 中为实例属性赋初始值
        self.SNR_DB = 25