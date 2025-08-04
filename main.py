# main.py

import matplotlib.pyplot as plt
import seaborn as sns
from config import SimulationConfig
from environment import SensingEnvironment
from transceiver import Transceiver

def main():
    """
    主函数：初始化、运行仿真并展示结果。
    """
    print("--- 初始化仿真配置 ---")
    config = SimulationConfig()

    print("--- 创建感知环境和真实湿度分布 ---")
    environment = SensingEnvironment(config)

    print("--- 模拟收发器进行测量扫描 ---")
    transceiver = Transceiver(config, environment)
    # P_R是最终提供给神经网络的输入数据
    measurement_matrix_Pr = transceiver.perform_measurement_sweep()

    print("\n--- 仿真完成 ---")
    print(f"生成的测量矩阵 P_R 的形状: {measurement_matrix_Pr.shape}")
    print("这个矩阵现在可以作为您神经网络的输入。")
    
    # -------------------------------------------------------------
    # 在这里，您可以插入您的神经网络代码
    # 
    # 示例:
    # my_network = YourNeuralNetwork()
    # my_network.train(training_data, training_labels)
    # predicted_humidity_map = my_network.predict(measurement_matrix_Pr)
    #
    # -------------------------------------------------------------

    # --- 可视化以供验证 ---
    # 1. 真实湿度分布图 (Ground Truth)
    plt.figure(figsize=(10, 4))
    sns.heatmap(environment.grid, annot=True, fmt=".1f", cmap="viridis_r", cbar_kws={'label': 'Humidity (%)'})
    plt.title("Ground Truth Humidity Distribution")
    plt.xlabel("X Grid")
    plt.ylabel("Y Grid")
    plt.show()

    # 2. 测量矩阵 P_R
    plt.figure(figsize=(10, 6))
    sns.heatmap(measurement_matrix_Pr, cmap="magma", cbar_kws={'label': 'Received Power (dB)'})
    plt.title("Generated Measurement Matrix (P_R)")
    plt.xlabel("Frequency Points (Index)")
    plt.ylabel("Tag Index")
    plt.show()


if __name__ == "__main__":
    main()