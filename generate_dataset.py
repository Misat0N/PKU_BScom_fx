# generate_dataset.py

import numpy as np
from tqdm import tqdm
from config import SimulationConfig
from environment import SensingEnvironment
from transceiver import Transceiver
import os

def generate_data_samples(num_samples: int, config: SimulationConfig):
    """
    生成指定数量的数据样本和标签。
    
    Args:
        num_samples (int): 要生成的样本数量。
        config (SimulationConfig): 仿真配置。
        
    Returns:
        tuple: (X_data, y_data)
               X_data是(num_samples, num_tags, num_freq_points)的numpy数组。
               y_data是(num_samples, total_grids)的numpy数组。
    """
    X_data = []
    y_data = []

    print(f"开始生成 {num_samples} 个数据样本...")
    for _ in tqdm(range(num_samples)):
        # 1. 每次循环都创建一个全新的、随机的环境
        environment = SensingEnvironment(config)
        
        # 2. 模拟收发器在该环境下进行测量
        transceiver = Transceiver(config, environment)
        measurement_matrix_Pr = transceiver.perform_measurement_sweep()
        
        # 3. 获取对应的真实标签 (已经是1D向量)
        ground_truth_labels = environment.get_ground_truth_labels()
        
        # 4. 存储数据和标签
        X_data.append(measurement_matrix_Pr)
        y_data.append(ground_truth_labels)
        
    return np.array(X_data), np.array(y_data)

def main():
    """
    主函数，用于生成并保存训练集和测试集。
    """
    # --- 配置 ---
    NUM_TRAIN_SAMPLES = 1000  # 您可以根据需要调整训练集大小
    NUM_TEST_SAMPLES = 200   # 您可以根据需要调整测试集大小
    OUTPUT_DIR = "dataset"     # 保存数据集的文件夹

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    config = SimulationConfig()

    # --- 生成训练集 ---
    X_train, y_train = generate_data_samples(NUM_TRAIN_SAMPLES, config)
    train_filepath = os.path.join(OUTPUT_DIR, "train_dataset.npz")
    np.savez_compressed(train_filepath, X=X_train, y=y_train)
    print(f"\n训练集已生成并保存到: {train_filepath}")
    print(f"  - X_train shape: {X_train.shape}")
    print(f"  - y_train shape: {y_train.shape}")

    # --- 生成测试集 ---
    X_test, y_test = generate_data_samples(NUM_TEST_SAMPLES, config)
    test_filepath = os.path.join(OUTPUT_DIR, "test_dataset.npz")
    np.savez_compressed(test_filepath, X=X_test, y=y_test)
    print(f"\n测试集已生成并保存到: {test_filepath}")
    print(f"  - X_test shape: {X_test.shape}")
    print(f"  - y_test shape: {y_test.shape}")

    # --- 如何加载数据示例 ---
    print("\n--- 如何加载和使用数据 ---")
    with np.load(train_filepath) as data:
        X_loaded = data['X']
        y_loaded = data['y']
    print("成功加载训练集数据！")
    print(f"加载的X形状: {X_loaded.shape}")
    print(f"加载的y形状: {y_loaded.shape}")

if __name__ == "__main__":
    main()