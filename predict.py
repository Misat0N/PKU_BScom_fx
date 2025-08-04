import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from config import SimulationConfig
from model import SensingNetwork
from environment import SensingEnvironment
from transceiver import Transceiver

def plot_input_matrix(Pr_matrix, config):
    """
    【【新增函数】】
    可视化输入到神经网络的测量矩阵 P_R。
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(Pr_matrix, cmap="magma", cbar_kws={'label': 'Received Power (dB)'})
    plt.title("Input Measurement Matrix (P_R) to Neural Network", fontsize=16)
    plt.xlabel("Frequency Points (Index)")
    plt.ylabel("Tag Index")
    plt.show()

def plot_results(ground_truth_map, predicted_map, accuracy, config):
    """
    可视化真实分布与模型预测结果的对比图。
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    num_classes = len(config.HUMIDITY_CLASSES)
    class_labels = [f"{v[0]}-{v[1]}%" for v in config.HUMIDITY_CLASSES.values()]
    vmin, vmax = 0, num_classes - 1
    
    sns.heatmap(ground_truth_map, ax=axes[0], annot=True, fmt="d", cmap="viridis", vmin=vmin, vmax=vmax,
                cbar_kws={'label': 'Humidity Class Index', 'ticks': np.arange(num_classes)})
    axes[0].set_title("Ground Truth Distribution")
    axes[0].set_xlabel("X Grid")
    axes[0].set_ylabel("Y Grid")
    cbar1 = axes[0].collections[0].colorbar
    cbar1.set_ticklabels(class_labels)

    sns.heatmap(predicted_map, ax=axes[1], annot=True, fmt="d", cmap="viridis", vmin=vmin, vmax=vmax,
                cbar_kws={'label': 'Humidity Class Index', 'ticks': np.arange(num_classes)})
    axes[1].set_title(f"Model Prediction (Accuracy: {accuracy:.2f}%)")
    axes[1].set_xlabel("X Grid")
    axes[1].set_ylabel("Y Grid")
    cbar2 = axes[1].collections[0].colorbar
    cbar2.set_ticklabels(class_labels)

    plt.suptitle("Sensing Result Comparison", fontsize=16)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()

def main():
    config = SimulationConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = 'best_model.pth'
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 '{model_path}'！请先运行 'python train.py' 来训练并保存模型。")
        return

    model = SensingNetwork(config)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    print("\n--- 正在生成一个新的随机环境用于测试... ---")
    test_env = SensingEnvironment(config)
    test_transceiver = Transceiver(config, test_env)
    
    Pr_matrix = test_transceiver.perform_measurement_sweep()
    ground_truth_labels = test_env.get_ground_truth_labels()

    # --- 【【新增步骤】】 可视化输入矩阵 ---
    print("--- 正在可视化输入矩阵 P_R... ---")
    plot_input_matrix(Pr_matrix, config)
    # ------------------------------------

    print("--- 测试样本已生成，开始进行预测... ---")

    with torch.no_grad():
        input_tensor = torch.tensor(Pr_matrix, dtype=torch.float32).unsqueeze(0).to(device)
        output_logits = model(input_tensor)
        _, predicted_labels = torch.max(output_logits.data, 2)
        predicted_labels = predicted_labels.squeeze(0).cpu().numpy()

    correct_predictions = (predicted_labels == ground_truth_labels).sum()
    total_predictions = len(ground_truth_labels)
    accuracy = 100 * correct_predictions / total_predictions
    
    print(f"\n--- 预测完成 ---")
    print(f"模型在此样本上的准确率: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")

    ground_truth_map = ground_truth_labels.reshape(config.NUM_GRIDS_Y, config.NUM_GRIDS_X)
    predicted_map = predicted_labels.reshape(config.NUM_GRIDS_Y, config.NUM_GRIDS_X)

    plot_results(ground_truth_map, predicted_map, accuracy, config)

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    main()