import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

from config import SimulationConfig
from model import SensingNetwork
from environment import SensingEnvironment
from transceiver import Transceiver

def generate_report_figures(results):
    """
    一次性绘制所有结果图表。
    """
    config = results['config']
    snr_levels = results['snr_levels']

    # --- 图1：可视化输入矩阵 ---
    print("\n--- Plotting Figure 1: Input Matrices (P_R) ---")
    num_snrs = len(snr_levels)
    fig1, axes1 = plt.subplots(2, (num_snrs + 1) // 2, figsize=(20, 10))
    axes1 = axes1.flatten()
    for i, snr in enumerate(snr_levels):
        sns.heatmap(results['input_matrices'][snr], ax=axes1[i], cmap="magma", cbar=False)
        axes1[i].set_title(f"Input Matrix (SNR={snr}dB)")
        axes1[i].set_xlabel("Frequency Points")
        axes1[i].set_ylabel("Tag Index")
    for i in range(num_snrs, len(axes1)): axes1[i].set_visible(False)
    fig1.suptitle("Input Measurement Matrix vs. SNR", fontsize=20)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()

    # --- 图2：可视化预测结果对比 ---
    print("--- Plotting Figure 2: Visual Prediction Comparison ---")
    fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))
    axes2 = axes2.flatten()
    
    num_classes = len(config.HUMIDITY_CLASSES)
    class_labels = [f"{v[0]}-{v[1]}%" for v in config.HUMIDITY_CLASSES.values()]
    vmin, vmax = 0, num_classes - 1

    sns.heatmap(results['ground_truth_map'], ax=axes2[0], annot=True, fmt="d", cmap="viridis", vmin=vmin, vmax=vmax)
    axes2[0].set_title("Ground Truth Distribution")

    for i, snr in enumerate(snr_levels):
        ax_idx = i + 1
        sns.heatmap(results['predicted_maps'][snr], ax=axes2[ax_idx], annot=True, fmt="d", cmap="viridis", vmin=vmin, vmax=vmax)
        acc = results['visual_accuracies'][snr]
        axes2[ax_idx].set_title(f"Prediction (SNR={snr}dB, Acc={acc:.2f}%)")
    
    for i in range(len(snr_levels) + 1, len(axes2)): axes2[i].set_visible(False)
    fig2.suptitle("Model Prediction under Different SNR Conditions", fontsize=20)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()

    # --- 图3：定量准确率折线图 ---
    print("--- Plotting Figure 3: Quantitative Accuracy Curve ---")
    plt.figure(figsize=(10, 6))
    plt.plot(snr_levels, results['avg_accuracies'], marker='o', linestyle='-', color='b')
    plt.title('Model Accuracy vs. Signal-to-Noise Ratio (SNR)', fontsize=16)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Average Detection Accuracy (%)', fontsize=12)
    plt.grid(True)
    plt.xticks(snr_levels)
    plt.ylim(0, 100)
    plt.gca().invert_xaxis()
    plt.show()

def main():
    # --- 配置 ---
    SNR_LEVELS_TO_TEST = [30, 25, 20, 15, 10, 5, 0]
    NUM_QUANTITATIVE_SAMPLES = 200 # 用于计算平均准确率的样本数

    config = SimulationConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 加载模型 ---
    model_path = 'best_model.pth'
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型 '{model_path}'！请确保您已成功运行 'train.py'。")
        return
    model = SensingNetwork(config)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # ==================================================================
    #
    #                     阶段一：纯计算
    #
    # ==================================================================
    print("\n--- Starting Stage 1: Pure Computation (No Plotting) ---")
    
    # 存储所有计算结果的字典
    results = {
        'config': config,
        'snr_levels': SNR_LEVELS_TO_TEST,
        'input_matrices': {},
        'predicted_maps': {},
        'visual_accuracies': {},
        'avg_accuracies': [],
        'ground_truth_map': None,
    }

    # --- 1. 定性分析所需的数据计算 ---
    print("Pre-computing data for visual analysis...")
    vis_env = SensingEnvironment(config)
    gt_labels = vis_env.get_ground_truth_labels()
    results['ground_truth_map'] = gt_labels.reshape(config.NUM_GRIDS_Y, config.NUM_GRIDS_X)

    for snr in tqdm(SNR_LEVELS_TO_TEST, desc="Computing for visual plots"):
        temp_config = SimulationConfig(); temp_config.SNR_DB = snr
        transceiver = Transceiver(temp_config, vis_env)
        Pr_matrix = transceiver.perform_measurement_sweep()
        results['input_matrices'][snr] = Pr_matrix

        with torch.no_grad():
            input_tensor = torch.tensor(Pr_matrix, dtype=torch.float32).unsqueeze(0).to(device)
            logits = model(input_tensor)
            _, pred_labels = torch.max(logits.data, 2)
            pred_labels = pred_labels.squeeze(0).cpu().numpy()
        
        results['predicted_maps'][snr] = pred_labels.reshape(config.NUM_GRIDS_Y, config.NUM_GRIDS_X)
        results['visual_accuracies'][snr] = 100 * (pred_labels == gt_labels).sum() / len(gt_labels)

    # --- 2. 定量分析所需的数据计算 ---
    print("\nPre-computing data for quantitative analysis...")
    pbar = tqdm(SNR_LEVELS_TO_TEST, desc="Calculating Average Accuracies")
    for snr in pbar:
        temp_config = SimulationConfig(); temp_config.SNR_DB = snr
        correct, total = 0, 0
        env = SensingEnvironment(temp_config)
        for _ in range(NUM_QUANTITATIVE_SAMPLES):
            env._generate_random_humidity_map()
            transceiver = Transceiver(temp_config, env)
            Pr_matrix = transceiver.perform_measurement_sweep()
            gt_labels = env.get_ground_truth_labels()
            with torch.no_grad():
                input_tensor = torch.tensor(Pr_matrix, dtype=torch.float32).unsqueeze(0).to(device)
                logits = model(input_tensor)
                _, pred_labels = torch.max(logits.data, 2)
                pred_labels = pred_labels.squeeze(0).cpu().numpy()
            total += len(gt_labels)
            correct += (pred_labels == gt_labels).sum()
        
        accuracy = 100 * correct / total
        results['avg_accuracies'].append(accuracy)

    # ==================================================================
    #
    #                     阶段二：纯绘图
    #
    # ==================================================================
    print("\n--- Starting Stage 2: Pure Plotting ---")
    generate_report_figures(results)

    print("\n--- Final Report Generation Finished ---")

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    main()