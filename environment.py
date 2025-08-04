# # environment.py

# import numpy as np
# import noise
# from config import SimulationConfig
# from scipy.ndimage import gaussian_filter

# class MetamaterialTag:
#     def __init__(self, position, config: SimulationConfig):
#         self.position = np.array(position)
#         self.config = config

#     def get_scattering_profile(self, humidity):
#         min_h = min(c[0] for c in self.config.HUMIDITY_CLASSES.values())
#         max_h = max(c[1] for c in self.config.HUMIDITY_CLASSES.values())
#         humidity_clipped = np.clip(humidity, min_h, max_h)
#         resonant_freq = np.interp(
#             humidity_clipped,
#             [min_h, max_h],
#             [self.config.FREQ_START, self.config.FREQ_END]
#         )
#         peak_depth = 0.9
#         peak_width = 50e6
#         s_values = 1.0 - peak_depth * np.exp(-((self.config.FREQUENCIES - resonant_freq)**2) / (2 * peak_width**2))
#         return s_values

# class SensingEnvironment:
#     def __init__(self, config: SimulationConfig):
#         self.config = config
#         self.grid = np.zeros((config.NUM_GRIDS_Y, config.NUM_GRIDS_X))
#         self.tags = []
#         self.tag_positions = None
#         self._deploy_tags() # 首先部署固定的标签
#         self._generate_random_humidity_map() # 然后生成随机湿度图

#     def _generate_random_humidity_map(self):
#         shape = (self.config.NUM_GRIDS_Y, self.config.NUM_GRIDS_X)
#         scale = np.random.uniform(5.0, 20.0); octaves = np.random.randint(4, 7)
#         persistence = np.random.uniform(0.4, 0.6); lacunarity = np.random.uniform(1.8, 2.2)
#         seed = np.random.randint(0, 1000)
#         raw_noise_map = np.zeros(shape)
#         for y in range(shape[0]):
#             for x in range(shape[1]):
#                 nx = x / shape[1] - 0.5; ny = y / shape[0] - 0.5
#                 raw_noise_map[y, x] = noise.pnoise2(nx * scale, ny * scale, octaves=octaves, 
#                                                     persistence=persistence, lacunarity=lacunarity, 
#                                                     repeatx=shape[1], repeaty=shape[0], base=seed)
#         smoothed_noise_map = gaussian_filter(raw_noise_map, sigma=1.5)
#         flat_noise = smoothed_noise_map.flatten()
#         ranks = flat_noise.argsort().argsort()
#         uniform_map_flat = ranks / (len(ranks) - 1)
#         uniform_map_2d = uniform_map_flat.reshape(shape)
#         min_h = min(c[0] for c in self.config.HUMIDITY_CLASSES.values())
#         max_h = max(c[1] for c in self.config.HUMIDITY_CLASSES.values())
#         self.grid = np.interp(uniform_map_2d, [0, 1], [min_h, max_h])

#     def _deploy_tags(self):
#         """
#         【【核心修正】】
#         使用一组预定义的、均匀分布的网格坐标来部署15个标签，取代随机部署。
#         """
#         # 预定义的15个均匀分布的网格坐标 (行, 列)
#         # 坐标原点在左上角
#         fixed_grid_coords = [
#             (0, 1), (0, 7), (0, 13),  # 第1行 (Y=0)
#             (1, 4), (1, 10),           # 第2行 (Y=1)
#             (2, 1), (2, 7), (2, 13),  # 第3行 (Y=2)
#             (3, 4), (3, 10),           # 第4行 (Y=3)
#             (4, 1), (4, 7), (4, 13),  # 第5行 (Y=4)
#             (5, 4), (5, 10)            # 第6行 (Y=5)
#         ]

#         positions = []
#         for y_idx, x_idx in fixed_grid_coords:
#             # 根据网格坐标计算物理坐标 (中心点)
#             pos_x = (x_idx + 0.5) * self.config.GRID_SIZE
#             pos_y = 0 # 标签贴在y=0的墙面上
#             pos_z = (y_idx + 0.5) * self.config.GRID_SIZE
#             position_3d = [pos_x, pos_y, pos_z]
#             positions.append(position_3d)
#             self.tags.append(MetamaterialTag(position_3d, self.config))
        
#         self.tag_positions = np.array(positions)

#     def get_humidity_at_position(self, position):
#         x_idx = int(position[0] // self.config.GRID_SIZE); y_idx = int(position[2] // self.config.GRID_SIZE)
#         x_idx = min(x_idx, self.config.NUM_GRIDS_X - 1); y_idx = min(y_idx, self.config.NUM_GRIDS_Y - 1)
#         return self.grid[y_idx, x_idx]

#     def get_ground_truth_labels(self):
#         labels = np.zeros_like(self.grid, dtype=int)
#         for class_idx, (min_h, max_h) in self.config.HUMIDITY_CLASSES.items():
#             mask = (self.grid >= min_h) & (self.grid < max_h)
#             labels[mask] = class_idx
#         return labels.flatten()

# environment.py

import numpy as np
import noise
from config import SimulationConfig
from scipy.ndimage import gaussian_filter

class MetamaterialTag:
    def __init__(self, position, config: SimulationConfig):
        self.position = np.array(position)
        self.config = config

    def get_scattering_profile(self, humidity):
        min_h = min(c[0] for c in self.config.HUMIDITY_CLASSES.values())
        max_h = max(c[1] for c in self.config.HUMIDITY_CLASSES.values())
        humidity_clipped = np.clip(humidity, min_h, max_h)
        resonant_freq = np.interp(
            humidity_clipped,
            [min_h, max_h],
            [self.config.FREQ_START, self.config.FREQ_END]
        )
        peak_depth = 0.9
        peak_width = 50e6
        s_values = 1.0 - peak_depth * np.exp(-((self.config.FREQUENCIES - resonant_freq)**2) / (2 * peak_width**2))
        return s_values

class SensingEnvironment:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.grid = np.zeros((config.NUM_GRIDS_Y, config.NUM_GRIDS_X))
        self.tags = []
        self.tag_positions = None
        self._deploy_tags()
        self._generate_random_humidity_map()

    def _generate_random_humidity_map(self):
        """
        【【核心修正】】
        模拟单一源的扩散场景，生成更符合物理规律的湿度分布。
        """
        shape = (self.config.NUM_GRIDS_Y, self.config.NUM_GRIDS_X)
        
        # 1. 在四条边上随机选择一个源点 (y, x)
        edge = np.random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            source_pos = (0, np.random.randint(0, shape[1]))
        elif edge == 'bottom':
            source_pos = (shape[0] - 1, np.random.randint(0, shape[1]))
        elif edge == 'left':
            source_pos = (np.random.randint(0, shape[0]), 0)
        else: # right
            source_pos = (np.random.randint(0, shape[0]), shape[1] - 1)

        # 2. 创建网格坐标
        y_coords, x_coords = np.mgrid[0:shape[0], 0:shape[1]]
        
        # 3. 计算每个点到源点的距离
        distance = np.sqrt((y_coords - source_pos[0])**2 + (x_coords - source_pos[1])**2)
        
        # 4. 基于距离生成高斯衰减图
        # sigma控制衰减的速度（扩散范围），随机化sigma可以产生更多样的分布
        sigma = np.random.uniform(shape[1] / 4, shape[1] / 2)
        value_map = np.exp(-(distance**2) / (2 * sigma**2)) # 值域在(0, 1]
        
        # 5. 将高斯分布的值映射到湿度范围
        min_h = min(c[0] for c in self.config.HUMIDITY_CLASSES.values())
        max_h = max(c[1] for c in self.config.HUMIDITY_CLASSES.values())
        self.grid = np.interp(value_map, [0, 1], [min_h, max_h])

    def _deploy_tags(self):
        # 使用固定的、均匀分布的标签位置
        fixed_grid_coords = [
            (0, 1), (0, 7), (0, 13), (1, 4), (1, 10),
            (2, 1), (2, 7), (2, 13), (3, 4), (3, 10),
            (4, 1), (4, 7), (4, 13), (5, 4), (5, 10)
        ]
        positions = []
        for y_idx, x_idx in fixed_grid_coords:
            pos_x = (x_idx + 0.5) * self.config.GRID_SIZE
            pos_y = 0
            pos_z = (y_idx + 0.5) * self.config.GRID_SIZE
            position_3d = [pos_x, pos_y, pos_z]
            positions.append(position_3d)
            self.tags.append(MetamaterialTag(position_3d, self.config))
        self.tag_positions = np.array(positions)

    def get_humidity_at_position(self, position):
        x_idx = int(position[0] // self.config.GRID_SIZE); y_idx = int(position[2] // self.config.GRID_SIZE)
        x_idx = min(x_idx, self.config.NUM_GRIDS_X - 1); y_idx = min(y_idx, self.config.NUM_GRIDS_Y - 1)
        return self.grid[y_idx, x_idx]

    def get_ground_truth_labels(self):
        labels = np.zeros_like(self.grid, dtype=int)
        for class_idx, (min_h, max_h) in self.config.HUMIDITY_CLASSES.items():
            mask = (self.grid >= min_h) & (self.grid < max_h)
            labels[mask] = class_idx
        return labels.flatten()