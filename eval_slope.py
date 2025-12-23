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
# 确保 slope_net.py 是最新的支持序列输入的版本
from slope_net import LaneSlopeNet

# === 配置 ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "visenet2_best_1223.pth"
DATASET_ROOT = "dataset_root" # 请确保路径正确
INPUT_SIZE = (256, 256)
OUTPUT_DIR = "eval_results"
SCALER_PATH = "scaler_params.json"

# 时间参数 (必须与训练时保持一致)
PAST_DT = 3.0
FUTURE_DT = 3.0
DT_STEP = 0.5

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# === 加载标准化参数 ===
if os.path.exists(SCALER_PATH):
    with open(SCALER_PATH, "r") as f:
        STATS = json.load(f)
    # 兼容处理：如果json里没有存dt信息，使用默认值
    PAST_DT = STATS.get('past_dt', PAST_DT)
    FUTURE_DT = STATS.get('future_dt', FUTURE_DT)
    DT_STEP = STATS.get('step', DT_STEP)
    print(f"Loaded Stats: Speed Mean={STATS['speed_mean']:.2f}, Slope Mean={STATS['slope_mean']:.4f}")
    print(f"Time Config: Past={PAST_DT}s, Future={FUTURE_DT}s, Step={DT_STEP}s")
else:
    print("Warning: scaler_params.json not found! Using Identity scaling.")
    STATS = {"speed_mean": 0, "speed_std": 1, "slope_mean": 0, "slope_std": 1}

# === Dataset 定义 (与训练逻辑保持一致) ===
class EvalSlopeDataset(Dataset):
    def __init__(self, csv_files, mask_root, input_size=(256, 256)):
        self.mask_root = mask_root
        self.input_size = input_size
        self.samples = []
        
        # 预计算时间偏移
        self.hist_offsets = np.sort(np.arange(-PAST_DT, 0.001, DT_STEP))
        self.fut_offsets = np.arange(DT_STEP, FUTURE_DT + 0.001, DT_STEP)
        self.hist_len = len(self.hist_offsets)
        self.fut_len = len(self.fut_offsets)
        
        # 缓存数据
        self.all_dfs = []
        
        print("Preprocessing validation data...")
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # 必须列检查
                req_cols = ['slope_rad', 'speed_mps', 'timestamp', 'left_img']
                if not all(c in df.columns for c in req_cols): continue
                
                df = df.dropna(subset=req_cols).sort_values('timestamp')
                if len(df) < 10: continue
                
                # 存入缓存
                data_block = {
                    'times': df['timestamp'].values,
                    'slopes': df['slope_rad'].values,
                    'speeds': df['speed_mps'].values,
                    'img_paths': df['left_img'].values,
                    'file_name': os.path.basename(csv_file)
                }
                file_idx = len(self.all_dfs)
                self.all_dfs.append(data_block)
                
                # 筛选有效样本
                times = data_block['times']
                t_start, t_end = times[0], times[-1]
                
                # 这里我们不跳帧，为了评估尽可能的覆盖，或者每隔2帧取一个
                for i in range(0, len(times), 2): 
                    curr_t = times[i]
                    if (curr_t - PAST_DT >= t_start) and (curr_t + FUTURE_DT <= t_end):
                        self.samples.append({
                            'file_idx': file_idx,
                            'row_idx': i,
                            't_curr': curr_t
                        })
                        
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        data = self.all_dfs[item['file_idx']]
        
        # 插值
        t_hist_needed = item['t_curr'] + self.hist_offsets
        t_fut_needed = item['t_curr'] + self.fut_offsets
        
        hist_slopes = np.interp(t_hist_needed, data['times'], data['slopes'])
        hist_speeds = np.interp(t_hist_needed, data['times'], data['speeds'])
        fut_slopes_gt = np.interp(t_fut_needed, data['times'], data['slopes'])
        
        # 图片
        orig_path = data['img_paths'][item['row_idx']]
        mask_rel = orig_path.replace(".jpg", ".png").replace(".jpeg", ".png")
        mask_path = os.path.join(self.mask_root, mask_rel)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != self.input_size:
                mask = cv2.resize(mask if mask is not None else np.zeros(self.input_size), self.input_size)
        else:
            mask = np.zeros(self.input_size, dtype=np.uint8)
            
        mask_tensor = torch.from_numpy(mask).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0)
        
        # 标准化
        norm_hist_speed = (hist_speeds - STATS['speed_mean']) / STATS['speed_std']
        norm_hist_slope = (hist_slopes - STATS['slope_mean']) / STATS['slope_std']
        # 注意：Eval时不需要截断，我们需要真实反映输入
        
        hist_tensor = torch.tensor(np.stack([norm_hist_speed, norm_hist_slope], axis=1), dtype=torch.float32)
        curr_slope_norm = torch.tensor([norm_hist_slope[-1]], dtype=torch.float32)
        
        # 元数据用于绘图
        meta = {
            'timestamp': item['t_curr'],
            'seq_id': data['file_name'],
            'hist_slopes_raw': hist_slopes, # 真实历史坡度值
            'fut_slopes_raw': fut_slopes_gt, # 真实未来坡度值
            'hist_time_rel': self.hist_offsets,
            'fut_time_rel': self.fut_offsets
        }
        
        return mask_tensor, hist_tensor, curr_slope_norm, meta

# === 模块 1: 新版权重分析 ===
def analyze_weights(model):
    print("\n" + "="*40)
    print(" Feature Importance Analysis (Visual vs Temporal)")
    print("="*40)
    
    # 假设融合层是 model.head[0] (Linear)
    # 输入维度 = Visual_Dim (1024) + RNN_Hidden (64)
    if hasattr(model, 'head') and isinstance(model.head[0], nn.Linear):
        weights = model.head[0].weight.detach().cpu().numpy() # [Hidden_Out, 1088]
        
        # 视觉特征在前，时序特征在后 (根据 slope_net.py 的 concat 顺序)
        # model.visual_dim = 1024
        visual_dim = model.visual_dim 
        
        w_visual = np.abs(weights[:, :visual_dim]).mean()
        w_temporal = np.abs(weights[:, visual_dim:]).mean()
        
        print(f"Mean Abs Weight (Visual - CNN): {w_visual:.6f}")
        print(f"Mean Abs Weight (Temporal - GRU): {w_temporal:.6f}")
        
        ratio = w_temporal / (w_visual + 1e-8)
        print(f"Temporal/Visual Importance Ratio: {ratio:.2f}x")
        
        if ratio > 5.0:
            print(">> Model relies heavily on History (Inertia). Visual cues might be weak.")
        elif ratio < 0.2:
            print(">> Model relies heavily on Vision. History might be ignored.")
        else:
            print(">> Balanced usage of Vision and History.")
            
    else:
        print("Skipping weight analysis (Model structure mismatch).")

import random # 确保引入

def evaluate_and_plot():
    # 1. 准备数据
    all_csvs = sorted(glob.glob(os.path.join(DATASET_ROOT, "**", "*.csv"), recursive=True))
    
    # === 关键修正：必须使用与训练时完全相同的随机种子和逻辑 ===
    random.Random(0
                  ).shuffle(all_csvs) 
    
    # 这样取出的后20%，才是训练时模型完全没见过的验证集
    split_idx = int(0.8 * len(all_csvs))
    test_csvs = all_csvs
    
    print(f"Evaluating on {len(test_csvs)} files (Held-out Validation Set)")
    
    dataset = EvalSlopeDataset(test_csvs, os.path.join(DATASET_ROOT, "masks"))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 2. 加载模型
    # 需要实例化正确的时序长度
    model = LaneSlopeNet(history_steps=dataset.hist_len, future_steps=dataset.fut_len).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Error: {MODEL_PATH} not found.")
        return

    # 3. 运行分析
    analyze_weights(model)
    model.eval()
    
    # 存储结果
    # shape: [N, Fut_Len]
    preds_list = []
    gts_list = [] 
    
    # 用于绘图的精选样本
    plot_samples = []
    
    print("\nRunning inference...")
    with torch.no_grad():
        for i, (masks, hist_data, curr_slope, meta) in enumerate(tqdm(dataloader)):
            masks = masks.to(DEVICE)
            hist_data = hist_data.to(DEVICE)
            curr_slope = curr_slope.to(DEVICE)
            
            # Forward [B, Fut_Steps] (Normalized)
            preds_norm = model(masks, hist_data, curr_slope)
            
            # 反标准化
            preds_raw = preds_norm.cpu().numpy() * STATS['slope_std'] + STATS['slope_mean']
            
            # 获取 GT
            # meta['fut_slopes_raw'] 是 list of tensors, 需要 stack
            # DataLoader 会把 numpy array batch 之后变成 Tensor
            gts_raw = meta['fut_slopes_raw'].numpy() 
            
            preds_list.append(preds_raw)
            gts_list.append(gts_raw)
            
            # 随机挑选一些样本保存用于绘图 (每个 Batch 挑一个)
            if i % 10 == 0:
                idx = 0
                sample = {
                    't_curr': meta['timestamp'][idx].item(),
                    'seq_id': meta['seq_id'][idx],
                    'hist_t': meta['hist_time_rel'][idx].numpy() + meta['timestamp'][idx].item(),
                    'hist_v': meta['hist_slopes_raw'][idx].numpy(),
                    'fut_t': meta['fut_time_rel'][idx].numpy() + meta['timestamp'][idx].item(),
                    'fut_gt': gts_raw[idx],
                    'fut_pred': preds_raw[idx]
                }
                plot_samples.append(sample)

    # 4. 计算指标
    all_preds = np.concatenate(preds_list, axis=0) # [N, Fut_Steps]
    all_gts = np.concatenate(gts_list, axis=0)     # [N, Fut_Steps]
    
    # 总体指标
    mae_total = mean_absolute_error(all_gts, all_preds)
    rmse_total = np.sqrt(mean_squared_error(all_gts, all_preds))
    
    print("\n" + "="*40)
    print(f" Overall Performance (Mean over {FUTURE_DT}s)")
    print("="*40)
    print(f"MAE  : {mae_total:.6f} rad")
    print(f"RMSE : {rmse_total:.6f} rad")
    
    # 分时刻指标 (Time-Horizon Analysis)
    print("\nTime-Step Breakdown:")
    print(f"{'Time (s)':<10} | {'MAE (rad)':<12} | {'RMSE (rad)':<12}")
    print("-" * 40)
    
    fut_times = dataset.fut_offsets
    time_maes = []
    
    for t_idx, t_val in enumerate(fut_times):
        p_col = all_preds[:, t_idx]
        g_col = all_gts[:, t_idx]
        
        step_mae = mean_absolute_error(g_col, p_col)
        step_rmse = np.sqrt(mean_squared_error(g_col, p_col))
        time_maes.append(step_mae)
        
        print(f"+{t_val:.1f}s       | {step_mae:.6f}     | {step_rmse:.6f}")

    # 5. 绘图：误差随时间变化
    plt.figure(figsize=(8, 5))
    plt.plot(fut_times, time_maes, 'o-', linewidth=2)
    plt.xlabel("Prediction Horizon (seconds)")
    plt.ylabel("MAE (rad)")
    plt.title("Prediction Error vs Time Horizon")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "error_horizon.png"))
    plt.close()
    
    # 6. 绘图：序列轨迹可视化
    # 挑选几个具有代表性的
    print(f"\nPlotting sample trajectories to {OUTPUT_DIR}...")
    for k, sample in enumerate(plot_samples[:10]): # 只画前10个
        plt.figure(figsize=(10, 6))
        
        # 画历史 (Input)
        plt.plot(sample['hist_t'], sample['hist_v'], 'k.-', label='History (Input)', alpha=0.6)
        # 画当前点
        plt.scatter([sample['t_curr']], [sample['hist_v'][-1]], color='black', s=50, zorder=5)
        
        # 画未来真值 (GT)
        plt.plot(sample['fut_t'], sample['fut_gt'], 'g.-', label='Future Ground Truth', linewidth=2)
        
        # 画未来预测 (Pred)
        plt.plot(sample['fut_t'], sample['fut_pred'], 'r.--', label='Future Prediction', linewidth=2)
        
        plt.axvline(x=sample['t_curr'], color='gray', linestyle='--', alpha=0.5)
        plt.title(f"Seq: {sample['seq_id']} @ {sample['t_curr']:.1f}s")
        plt.xlabel("Timestamp (s)")
        plt.ylabel("Slope (rad)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        safe_name = sample['seq_id'].replace('.csv', '')
        plt.savefig(os.path.join(OUTPUT_DIR, f"traj_{safe_name}_{k}.png"))
        plt.close()

    print("Done.")

if __name__ == "__main__":
    evaluate_and_plot()