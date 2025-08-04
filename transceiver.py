# # transceiver.py (修正版 - 应用新参数)

# import numpy as np
# from config import SimulationConfig
# from environment import SensingEnvironment

# class Transceiver:
#     def __init__(self, config: SimulationConfig, environment: SensingEnvironment):
#         self.config = config
#         self.env = environment
#         self.tx_pos = config.TRANSCEIVER_POS

#     def _calculate_path_loss(self, tag_pos):
#         distance = np.linalg.norm(self.tx_pos - tag_pos)
#         # 增加一个最小距离防止路径损耗为0或过小
#         distance = max(distance, 0.1)
#         return (4 * np.pi * distance)**2

#     def perform_measurement_sweep(self):
#         P_R = np.zeros((self.config.NUM_TAGS, self.config.FREQ_POINTS))
#         tx_power_linear = self.config.TX_POWER_W
        
#         for i, target_tag in enumerate(self.env.tags):
#             for j, freq in enumerate(self.config.FREQUENCIES):
#                 target_humidity = self.env.get_humidity_at_position(target_tag.position)
                
#                 # 1. 目标信号功率 (应用天线增益)
#                 s_target = target_tag.get_scattering_coefficient(freq, target_humidity)
#                 path_loss_target = self._calculate_path_loss(target_tag.position)
#                 # 【【核心修正】】 乘以标签天线增益，放大有用信号
#                 power_target_linear = tx_power_linear * self.config.TAG_ANTENNA_GAIN * (s_target**2) / path_loss_target
                
#                 # 2. 干扰信号功率 (应用天线增益和旁瓣抑制)
#                 power_interference_linear = 0
#                 for k, interfering_tag in enumerate(self.env.tags):
#                     if i == k: continue
#                     interfering_humidity = self.env.get_humidity_at_position(interfering_tag.position)
#                     s_interfering = interfering_tag.get_scattering_coefficient(freq, interfering_humidity)
#                     path_loss_interfering = self._calculate_path_loss(interfering_tag.position)
#                     # 【【核心修正】】 干扰信号同样有增益，但被旁瓣抑制因子削弱
#                     power_interference_linear += (tx_power_linear * self.config.TAG_ANTENNA_GAIN * self.config.BEAMFORMING_SIDELOBE_SUPPRESSION * (s_interfering**2) / path_loss_interfering)
                    
#                 # 3. 环境散射 (使用新的、更低的反射系数)
#                 power_env_scatter = tx_power_linear * self.config.WALL_REFLECTION_COEFFICIENT
                
#                 total_signal_power = power_target_linear + power_interference_linear + power_env_scatter
                
#                 # 4. 计算并添加噪声 (使用新的信噪比)
#                 signal_power_db = 10 * np.log10(max(1e-15, total_signal_power))
#                 noise_power_db = signal_power_db - self.config.SNR_DB
#                 noise_power_linear = 10**(noise_power_db / 10)
#                 noise = np.random.normal(0, np.sqrt(noise_power_linear))
                
#                 received_power_linear = total_signal_power + noise
#                 P_R[i, j] = 10 * np.log10(max(1e-15, received_power_linear))

#         return P_R

# transceiver.py (修正版 - 适配新的标签模型)

import numpy as np
from config import SimulationConfig
from environment import SensingEnvironment

class Transceiver:
    def __init__(self, config: SimulationConfig, environment: SensingEnvironment):
        self.config = config
        self.env = environment
        self.tx_pos = config.TRANSCEIVER_POS

    def _calculate_path_loss(self, tag_pos):
        distance = np.linalg.norm(self.tx_pos - tag_pos)
        distance = max(distance, 0.1)
        return (4 * np.pi * distance)**2

    def perform_measurement_sweep(self):
        P_R = np.zeros((self.config.NUM_TAGS, self.config.FREQ_POINTS))
        tx_power_linear = self.config.TX_POWER_W
        
        # 【【核心修正】】
        # 预先计算所有标签的散射频谱，提高效率
        all_scattering_profiles = []
        for tag in self.env.tags:
            humidity = self.env.get_humidity_at_position(tag.position)
            all_scattering_profiles.append(tag.get_scattering_profile(humidity))

        for i, target_tag in enumerate(self.env.tags):
            target_scattering_profile = all_scattering_profiles[i]
            
            for j, freq in enumerate(self.config.FREQUENCIES):
                # 从预计算的频谱中直接获取散射系数值
                s_target = target_scattering_profile[j]
                
                path_loss_target = self._calculate_path_loss(target_tag.position)
                power_target_linear = tx_power_linear * self.config.TAG_ANTENNA_GAIN * (s_target**2) / path_loss_target
                
                power_interference_linear = 0
                for k, interfering_tag in enumerate(self.env.tags):
                    if i == k: continue
                    interfering_scattering_profile = all_scattering_profiles[k]
                    s_interfering = interfering_scattering_profile[j]
                    
                    path_loss_interfering = self._calculate_path_loss(interfering_tag.position)
                    power_interference_linear += (tx_power_linear * self.config.TAG_ANTENNA_GAIN * self.config.BEAMFORMING_SIDELOBE_SUPPRESSION * (s_interfering**2) / path_loss_interfering)
                    
                power_env_scatter = tx_power_linear * self.config.WALL_REFLECTION_COEFFICIENT
                
                total_signal_power = power_target_linear + power_interference_linear + power_env_scatter
                
                signal_power_db = 10 * np.log10(max(1e-15, total_signal_power))
                noise_power_db = signal_power_db - self.config.SNR_DB
                noise_power_linear = 10**(noise_power_db / 10)
                noise = np.random.normal(0, np.sqrt(noise_power_linear))
                
                received_power_linear = total_signal_power + noise
                P_R[i, j] = 10 * np.log10(max(1e-15, received_power_linear))

        return P_R