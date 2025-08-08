import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

# 从项目中导入所有必要的模块
from config import SimulationConfig
from model import SpatialSensingNetwork
from environment import SensingEnvironment
from transceiver import Transceiver

def plot_input_matrix(Pr_matrix, snr):
    """可视化单个输入矩阵 P_R。"""
    plt.figure(figsize=(12, 8))
    sns.heatmap(Pr_matrix, cmap="magma", cbar_kws={'label': 'Received Power (dB)'})
    plt.title(f"Input Measurement Matrix (P_R) at SNR = {snr} dB", fontsize=16)
    plt.xlabel("Frequency Points (Index)")
    plt.ylabel("Tag Index")
    plt.show()

def run_visual_analysis(model, config, tag_positions, device, snr_levels):
    """
    定性分析：对同一个真实场景，在不同SNR下进行预测并可视化。
    """
    print("\n--- Part 1: Starting Qualitative Visual Analysis ---")
    
    # 1. 生成一个固定的真实环境用于本次评估
    print("Generating a fixed ground truth map for visualization...")
    fixed_env = SensingEnvironment(config)
    fixed_env.tag_positions = tag_positions
    fixed_env.tags = [MetamaterialTag(pos, config) for pos in tag_positions]
    ground_truth_labels = fixed_env.get_ground_truth_labels()
    ground_truth_map = ground_truth_labels.reshape(config.NUM_GRIDS_Y, config.NUM_GRIDS_X)

    # 2. 准备绘图
    num_plots = len(snr_levels) + 1
    nrows, ncols = 2, 4 if num_plots > 4 else num_plots
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    
    # 可视化参数
    num_classes = len(config.HUMIDITY_CLASSES)
    class_labels = [f"{v[0]}-{v[1]}%" for v in config.HUMIDITY_CLASSES.values()]
    vmin, vmax = 0, num_classes - 1
    
    # 首先绘制真实分布图
    sns.heatmap(ground_truth_map, ax=axes[0], annot=True, fmt="d", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth Distribution")
    
    # 3. 循环评估并绘图
    plot_index = 1
    for snr in tqdm(snr_levels, desc="Visualizing SNR Impact"):
        temp_config = SimulationConfig(); temp_config.SNR_DB = snr
        transceiver = Transceiver(temp_config, fixed_env)
        Pr_matrix = transceiver.perform_measurement_sweep()
        
        # 在第一次循环时（最高SNR），可视化输入矩阵
        if plot_index == 1:
            print("\nDisplaying input matrix for the highest SNR case to validate features...")
            plot_input_matrix(Pr_matrix, snr)

        with torch.no_grad():
            input_tensor = torch.tensor(Pr_matrix, dtype=torch.float32).unsqueeze(0).to(device)
            output_logits = model(input_tensor)
            _, predicted_labels = torch.max(output_logits.data, 2)
            predicted_labels = predicted_labels.squeeze(0).cpu().numpy()
            
        accuracy = 100 * (predicted_labels == ground_truth_labels).sum() / len(ground_truth_labels)
        predicted_map = predicted_labels.reshape(config.NUM_GRIDS_Y, config.NUM_GRIDS_X)
        
        sns.heatmap(predicted_map, ax=axes[plot_index], annot=True, fmt="d", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[plot_index].set_title(f"Prediction (SNR={snr}dB, Acc={accuracy:.2f}%)")
        plot_index += 1
        
    for i in range(plot_index, len(axes)):
        axes[i].set_visible(False)
        
    fig.suptitle("Model Prediction under Different SNR Conditions", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def run_quantitative_analysis(model, config, tag_positions, device, snr_levels):
    """
    定量分析：在多个SNR下计算平均准确率并绘制折线图。
    """
    print("\n--- Part 2: Starting Quantitative Statistical Analysis ---")
    num_test_samples = 200 # 每个SNR点用于统计的样本数
    accuracies = []

    pbar = tqdm(snr_levels, desc="Calculating Average Accuracies")
    for snr in pbar:
        pbar.set_postfix_str(f"Current SNR: {snr} dB")
        temp_config = SimulationConfig(); temp_config.SNR_DB = snr
        
        correct_predictions, total_predictions = 0, 0
        
        env = SensingEnvironment(temp_config)
        env.tag_positions = tag_positions
        env.tags = [MetamaterialTag(pos, temp_config) for pos in tag_positions]

        for _ in range(num_test_samples):
            env._generate_random_humidity_map()
            transceiver = Transceiver(temp_config, env)
            Pr_matrix = transceiver.perform_measurement_sweep()
            ground_truth_labels = env.get_ground_truth_labels()

            with torch.no_grad():
                input_tensor = torch.tensor(Pr_matrix, dtype=torch.float32).unsqueeze(0).to(device)
                output_logits = model(input_tensor)
                _, predicted_labels = torch.max(output_logits.data, 2)
                predicted_labels = predicted_labels.squeeze(0).cpu().numpy()
            
            total_predictions += len(ground_truth_labels)
            correct_predictions += (predicted_labels == ground_truth_labels).sum()

        accuracy = 100 * correct_predictions / total_predictions
        accuracies.append(accuracy)
        print(f"Average Accuracy @ SNR = {snr} dB: {accuracy:.2f}%")

    # 绘制结果折线图
    print("\nEvaluation finished, plotting results...")
    plt.figure(figsize=(10, 6))
    plt.plot(snr_levels, accuracies, marker='o', linestyle='-', color='b')
    plt.title('Model Accuracy vs. Signal-to-Noise Ratio (SNR)', fontsize=16)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Detection Accuracy (%)', fontsize=12)
    plt.grid(True)
    plt.xticks(snr_levels)
    plt.ylim(0, 100)
    plt.gca().invert_xaxis() # 反转X轴，让高SNR在左侧
    plt.show()

def main():
    SNR_LEVELS_TO_TEST = [30, 25, 20, 15, 10, 5, 0]
    
    config = SimulationConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = 'best_model.pth'
    dataset_path = os.path.join("dataset", "train_dataset.npz")
    if not os.path.exists(model_path) or not os.path.exists(dataset_path):
        print(f"错误: 找不到模型 '{model_path}' 或数据集 '{dataset_path}'！")
        print("请确保您已成功运行 'generate_dataset.py' 和 'run_training.py'。")
        return

    print("Loading trained model and tag positions...")
    train_data = np.load(dataset_path)
    tag_positions = train_data['tag_positions']
    
    model = SpatialSensingNetwork(config, tag_positions)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # 依次执行定性分析和定量分析
    run_visual_analysis(model, config, tag_positions, device, SNR_LEVELS_TO_TEST)
    run_quantitative_analysis(model, config, tag_positions, device, SNR_LEVELS_TO_TEST)

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    main()