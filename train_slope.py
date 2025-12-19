import os
import sys

# === 强制屏蔽系统 CUDA 路径，防止版本冲突 ===
if 'LD_LIBRARY_PATH' in os.environ:
    # 过滤掉包含 /usr/local/cuda 的路径
    paths = os.environ['LD_LIBRARY_PATH'].split(':')
    new_paths = [p for p in paths if 'cuda' not in p.lower()]
    os.environ['LD_LIBRARY_PATH'] = ':'.join(new_paths)
    
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import pandas as pd
import numpy as np
import cv2
import glob
import json
from tqdm import tqdm

# === 引入你的模型定义 ===
from slope_net import LaneSlopeNet

# === 数据集定义 ===
class SlopeDataset(Dataset):
    def __init__(self, dataset_root, input_size=(256, 256), future_dt=5.0):
        self.dataset_root = dataset_root
        self.mask_root = os.path.join(dataset_root, "masks")
        self.input_size = input_size
        self.samples = []
        
        csv_files = glob.glob(os.path.join(dataset_root, "**", "*.csv"), recursive=True)
        all_speeds = []
        all_slopes = [] # 用于计算坡度的统计量

        print("Scanning dataset and calculating stats...")
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
                
                actual_dt = timestamps[target_idx] - timestamps[i]
                if abs(actual_dt - future_dt) < 0.5:
                    curr_s = slopes[i]
                    target_s = slopes[target_idx]
                    spd_val = speeds[i]
                    
                    if np.isnan(curr_s) or np.isnan(target_s) or np.isnan(spd_val):
                        continue
                        
                    self.samples.append({
                        'mask_rel_path': img_paths[i].replace(".jpg", ".png").replace(".jpeg", ".png"),
                        'speed': spd_val,
                        'current_slope': curr_s,
                        'label_slope': target_s
                    })
                    all_speeds.append(spd_val)
                    all_slopes.append(curr_s)
                    # 注意：label_slope 分布通常和 current_slope 一致，共用统计量即可
        
        # === 计算并保存统计量 ===
        if len(all_speeds) > 0:
            self.speed_mean = float(np.mean(all_speeds))
            self.speed_std = float(np.std(all_speeds)) + 1e-6
            
            self.slope_mean = float(np.mean(all_slopes))
            self.slope_std = float(np.std(all_slopes)) + 1e-6
        else:
            self.speed_mean = 0.0; self.speed_std = 1.0
            self.slope_mean = 0.0; self.slope_std = 1.0
            
        print(f"Stats -> Speed: {self.speed_mean:.2f}+/-{self.speed_std:.2f}, Slope: {self.slope_mean:.4f}+/-{self.slope_std:.4f}")

    def get_stats(self):
        return {
            "speed_mean": self.speed_mean,
            "speed_std": self.speed_std,
            "slope_mean": self.slope_mean,
            "slope_std": self.slope_std
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        mask_path = os.path.join(self.mask_root, item['mask_rel_path'])
        
        if not os.path.exists(mask_path):
            mask = np.zeros(self.input_size, dtype=np.uint8)
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None: mask = np.zeros(self.input_size, dtype=np.uint8)
            else:
                if mask.shape != self.input_size:
                    mask = cv2.resize(mask, self.input_size)
        
        # Mask 归一化 (0-1)
        mask_tensor = torch.from_numpy(mask).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0)
        
        # === 核心：输入输出标准化 (Standardization) ===
        # 1. Speed 标准化
        norm_speed = (item['speed'] - self.speed_mean) / self.speed_std
        norm_speed = np.clip(norm_speed, -5.0, 5.0) # 截断离群值
        speed_tensor = torch.tensor([norm_speed], dtype=torch.float32)
        
        # 2. Current Slope 标准化
        norm_curr_slope = (item['current_slope'] - self.slope_mean) / self.slope_std
        # 坡度不需要截断太狠，但也防止极值
        curr_slope_tensor = torch.tensor([norm_curr_slope], dtype=torch.float32)
        
        # 3. Label Slope 标准化
        # 既然输入都是标准化的，为了让 Loss 稳定，Target 最好也是标准化的
        norm_label_slope = (item['label_slope'] - self.slope_mean) / self.slope_std
        label_tensor = torch.tensor([norm_label_slope], dtype=torch.float32)
        
        if torch.isnan(mask_tensor).any() or torch.isnan(speed_tensor).any() or torch.isnan(curr_slope_tensor).any():
             return torch.zeros_like(mask_tensor), torch.zeros_like(speed_tensor), torch.zeros_like(curr_slope_tensor), torch.zeros_like(label_tensor)

        return mask_tensor, speed_tensor, curr_slope_tensor, label_tensor

# === DDP 设置保持不变 ===
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_worker(rank, world_size):
    setup(rank, world_size)
    
    DATASET_ROOT = "dataset_root"
    BATCH_SIZE = 256
    EPOCHS = 500
    LR = 1e-4
    
    dataset = SlopeDataset(DATASET_ROOT)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    
    model = LaneSlopeNet().to(rank)
    model = DDP(model, device_ids=[rank])
    if os.path.exists("visenet2_best.pth"):
        state_dict = torch.load("visenet2_best.pth", map_location=f'cuda:{rank}')
        model.load_state_dict(state_dict)
        print(f"Rank {rank}: Loaded pretrained weights.")
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # 主进程保存统计量
    if rank == 0:
        print(f"Start Distributed Training on {world_size} GPUs...")
        stats = dataset.get_stats()
        with open("scaler_params.json", "w") as f:
            json.dump(stats, f, indent=4)
        print("Saved scaler_params.json")
    
    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        valid_batches = 0
        
        iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}") if rank == 0 else dataloader
        
        for masks, speeds, curr_slopes, labels in iterator:
            masks = masks.to(rank)
            speeds = speeds.to(rank)
            curr_slopes = curr_slopes.to(rank)
            labels = labels.to(rank)
            
            optimizer.zero_grad()
            preds = model(masks, speeds, curr_slopes)
            loss = criterion(preds, labels)
            
            if torch.isnan(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            valid_batches += 1
            
        if rank == 0:
            avg_loss = train_loss / max(valid_batches, 1)
            print(f"Epoch [{epoch+1}/{EPOCHS}] Avg Loss (Normalized): {avg_loss:.6f}")
            if (epoch + 1) % 50 == 0:
                torch.save(model.module.state_dict(), f"visenet2_ddp_ep{epoch+1}.pth")
                torch.save(model.module.state_dict(), "visenet2_best.pth")

    cleanup()

def main():
    world_size = 2
    mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    main()