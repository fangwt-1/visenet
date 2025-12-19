import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os
import glob
import json
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === 引入模型 ===
from slope_net import LaneSlopeNet

# === 配置 ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "visenet2_best.pth"
DATASET_ROOT = "dataset_root"
INPUT_SIZE = (256, 256)
OUTPUT_DIR = "eval_results"
SCALER_PATH = "scaler_params.json"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# === 加载标准化参数 ===
if os.path.exists(SCALER_PATH):
    with open(SCALER_PATH, "r") as f:
        STATS = json.load(f)
else:
    STATS = {"speed_mean": 0, "speed_std": 1, "slope_mean": 0, "slope_std": 1}

# === Dataset 定义 (保持一致) ===
class EvalSlopeDataset(Dataset):
    def __init__(self, csv_files, mask_root, input_size=(256, 256), future_dt=5.0):
        self.mask_root = mask_root
        self.input_size = input_size
        self.samples = []
        
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            df = df.dropna(subset=['slope_rad', 'speed_mps', 'timestamp', 'left_img'])
            if len(df) < 2: continue
            
            df = df.sort_values('timestamp')
            timestamps = df['timestamp'].values
            slopes = df['slope_rad'].values
            speeds = df['speed_mps'].values
            img_paths = df['left_img'].values
            
            target_times = timestamps + future_dt
            idx_candidates = np.searchsorted(timestamps, target_times)
            
            for i in range(len(timestamps)):
                target_idx = idx_candidates[i]
                if target_idx >= len(timestamps): continue
                if abs((timestamps[target_idx] - timestamps[i]) - future_dt) < 0.5:
                    self.samples.append({
                        'mask_rel_path': img_paths[i].replace(".jpg", ".png").replace(".jpeg", ".png"),
                        'speed': speeds[i],
                        'current_slope': slopes[i],
                        'label_slope': slopes[target_idx],
                        'timestamp': timestamps[i],
                        'seq_id': os.path.basename(csv_file)
                    })

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        mask_path = os.path.join(self.mask_root, item['mask_rel_path'])
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None: mask = np.zeros(self.input_size, dtype=np.uint8)
            elif mask.shape != self.input_size: mask = cv2.resize(mask, self.input_size)
        else:
            mask = np.zeros(self.input_size, dtype=np.uint8)
        
        mask_tensor = torch.from_numpy(mask).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0)
        
        norm_speed = (item['speed'] - STATS['speed_mean']) / STATS['speed_std']
        norm_curr_slope = (item['current_slope'] - STATS['slope_mean']) / STATS['slope_std']
        
        return (mask_tensor, 
                torch.tensor([norm_speed], dtype=torch.float32), 
                torch.tensor([norm_curr_slope], dtype=torch.float32), 
                torch.tensor([item['label_slope']], dtype=torch.float32), 
                torch.tensor([item['current_slope']], dtype=torch.float32),
                item)

# === 模块 1: 权重可视化分析 ===
def analyze_weights(model):
    print("\n" + "="*40)
    print(" Feature Importance Analysis")
    print("="*40)
    
    # 提取全连接层第一层权重
    # self.fc[0] shape: [128, 1024 + 2]
    # 前1024是Mask，倒数第2是Speed，倒数第1是CurrentSlope
    fc_weights = model.fc[0].weight.detach().cpu().numpy()
    
    # 分离权重
    w_mask = fc_weights[:, :1024]      # [128, 1024]
    w_speed = fc_weights[:, 1024]      # [128]
    w_curr = fc_weights[:, 1025]       # [128]
    
    # 计算绝对值重要性
    abs_mask = np.abs(w_mask)
    abs_speed = np.abs(w_speed)
    abs_curr = np.abs(w_curr)
    
    # 统计指标
    # Mask 我们取平均值（因为有1024个特征，单个肯定小，要看整体水平）
    mean_mask_imp = np.mean(abs_mask)
    mean_speed_imp = np.mean(abs_speed)
    mean_curr_imp = np.mean(abs_curr)
    
    # 归一化用于绘图 (以 Mask 为基准)
    base = mean_mask_imp
    ratio_speed = mean_speed_imp / base
    ratio_curr = mean_curr_imp / base
    
    print(f"1. Mask Features (Avg): {mean_mask_imp:.6f} (Baseline 1.0x)")
    print(f"2. Speed Feature      : {mean_speed_imp:.6f} ({ratio_speed:.2f}x importance)")
    print(f"3. Current Slope      : {mean_curr_imp:.6f} ({ratio_curr:.2f}x importance)")
    
    # 绘图：特征重要性对比
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    features = ['Mask (Avg)', 'Speed', 'Current Slope']
    values = [mean_mask_imp, mean_speed_imp, mean_curr_imp]
    colors = ['skyblue', 'orange', 'green']
    bars = plt.bar(features, values, color=colors, alpha=0.8)
    plt.title('Feature Importance (Mean Absolute Weight)')
    plt.ylabel('Weight Magnitude')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', ha='center', va='bottom')
    
    # 绘图：神经元敏感度分布
    # 展示128个神经元分别更关注谁
    plt.subplot(1, 2, 2)
    # 每个神经元对 Mask 的平均关注度
    neuron_mask = np.mean(abs_mask, axis=1)
    x = np.arange(128)
    
    plt.plot(x, neuron_mask, label='Mask Attention', color='skyblue', linewidth=1.5)
    plt.plot(x, abs_speed, label='Speed Attention', color='orange', alpha=0.7)
    plt.plot(x, abs_curr, label='CurrSlope Attention', color='green', alpha=0.7)
    plt.title('Per-Neuron Sensitivity')
    plt.xlabel('Neuron Index (0-127)')
    plt.ylabel('Abs Weight')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
    plt.close()
    print(f"Saved feature importance plot to {OUTPUT_DIR}/feature_importance.png")

# === 模块 2: 预测性能评估 ===
def evaluate_and_plot():
    # 准备数据
    all_csvs = glob.glob(os.path.join(DATASET_ROOT, "**", "*.csv"), recursive=True)
    all_csvs.sort()
    test_csvs = all_csvs[int(0.8 * len(all_csvs)):] # 后20%作为测试集
    
    dataset = EvalSlopeDataset(test_csvs, os.path.join(DATASET_ROOT, "masks"))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # 加载模型
    model = LaneSlopeNet().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print("Model not found!")
        return
    
    # === 运行权重分析 ===
    analyze_weights(model)
    
    model.eval()
    all_preds, all_labels, all_currs = [], [], []
    seq_data = {'id': [], 'ts': [], 'gt': [], 'pred': [], 'curr': []}
    
    print("\nRunning inference...")
    with torch.no_grad():
        for masks, speeds, currs, labels_raw, currs_raw, meta in tqdm(dataloader):
            masks, speeds, currs = masks.to(DEVICE), speeds.to(DEVICE), currs.to(DEVICE)
            
            # 模型预测 (输出的是 Normalized Delta 或 Normalized Slope，取决于你的模型实现)
            # 假设你用了 Residual Connection: output = curr + delta
            # 但注意：模型内部是在 Normalized 空间运作的。
            # 如果你的 forward 返回的是 out = curr_norm + delta
            # 那么真实 output = out * std + mean
            
            preds_norm = model(masks, speeds, currs)
            
            # 反标准化
            preds_raw = preds_norm.cpu().numpy().flatten() * STATS['slope_std'] + STATS['slope_mean']
            
            all_preds.extend(preds_raw)
            all_labels.extend(labels_raw.numpy().flatten())
            all_currs.extend(currs_raw.numpy().flatten())
            
            seq_data['id'].extend(meta['seq_id'])
            seq_data['ts'].extend(meta['timestamp'])
            seq_data['gt'].extend(labels_raw.numpy().flatten())
            seq_data['pred'].extend(preds_raw)
            seq_data['curr'].extend(currs_raw.numpy().flatten())

    # 指标计算
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_currs = np.array(all_currs)
    
    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    r2 = r2_score(all_labels, all_preds)
    baseline_rmse = np.sqrt(np.mean((all_currs - all_labels)**2))
    
    print("\n" + "="*40)
    print(f" Performance Metrics")
    print("="*40)
    print(f"MAE          : {mae:.6f} rad")
    print(f"RMSE         : {rmse:.6f} rad")
    print(f"Baseline RMSE: {baseline_rmse:.6f} rad")
    if rmse < baseline_rmse:
        print(f"Improvement  : {(baseline_rmse - rmse)/baseline_rmse*100:.2f}% better than baseline")
    else:
        print(f"Warning      : Model relies too much on visual noise?")
    
    # 绘图：序列跟踪
    df = pd.DataFrame(seq_data)
    unique_seqs = df['id'].unique()[:5] # 只画前5个序列
    
    for seq in unique_seqs:
        sub = df[df['id'] == seq].sort_values('ts')
        plt.figure(figsize=(12, 6))
        plt.plot(sub['ts'], sub['gt'], 'g-', label='Ground Truth (Future)', linewidth=2)
        plt.plot(sub['ts'], sub['curr'], 'k:', label='Input (Current)', alpha=0.5)
        plt.plot(sub['ts'], sub['pred'], 'r--', label='Prediction', linewidth=1.5)
        plt.title(f'Sequence Prediction: {seq}')
        plt.xlabel('Time (s)')
        plt.ylabel('Slope (rad)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, f"seq_{seq.replace('.csv','')}.png"))
        plt.close()
    
    print(f"\nEvaluation complete. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    evaluate_and_plot()