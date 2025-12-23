import os
import sys

# === 强制屏蔽系统 CUDA 路径 ===
if 'LD_LIBRARY_PATH' in os.environ:
    paths = os.environ['LD_LIBRARY_PATH'].split(':')
    new_paths = [p for p in paths if 'cuda' not in p.lower()]
    os.environ['LD_LIBRARY_PATH'] = ':'.join(new_paths)
    
import torch
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
import random
from tqdm import tqdm

# === 引入你的模型 ===
from slope_net import LaneSlopeNet

# === 数据集定义 ===
class SlopeDataset(Dataset):
    def __init__(self, dataset_root, file_list=None, input_size=(256, 256), 
                 past_dt=3.0, future_dt=3.0, step=0.5, is_train=True):
        self.dataset_root = dataset_root
        self.mask_root = os.path.join(dataset_root, "masks")
        self.input_size = input_size
        self.samples = []
        
        # 时间参数
        self.hist_offsets = np.sort(np.arange(-past_dt, 0.001, step))
        self.fut_offsets = np.arange(step, future_dt + 0.001, step)
        self.hist_len = len(self.hist_offsets)
        self.fut_len = len(self.fut_offsets)
        self.all_dfs = []

        if file_list is None:
            file_list = glob.glob(os.path.join(dataset_root, "**", "*.csv"), recursive=True)
            file_list.sort() 
        
        if len(file_list) == 0:
            print(f"Error: No CSV files found in {dataset_root}")
            return

        print(f"[{'Train' if is_train else 'Valid'}] Loading {len(file_list)} CSV files...")
        
        all_speeds = []
        all_slopes = [] 

        for csv_file in file_list:
            try:
                df = pd.read_csv(csv_file)
            except: continue
            
            req_cols = ['slope_rad', 'speed_mps', 'timestamp', 'left_img']
            if not all(c in df.columns for c in req_cols): continue
            
            df = df.dropna(subset=req_cols).sort_values('timestamp')
            if len(df) < 10: continue
            
            timestamps = df['timestamp'].values
            slopes = df['slope_rad'].values
            speeds = df['speed_mps'].values
            img_paths = df['left_img'].values
            
            file_idx = len(self.all_dfs)
            self.all_dfs.append({
                'times': timestamps, 'slopes': slopes, 'speeds': speeds, 
                'img_paths': img_paths
            })
            
            t_start = timestamps[0]
            t_end = timestamps[-1]
            
            skip_step = 2
            
            for i in range(0, len(timestamps), skip_step):
                curr_t = timestamps[i]
                if (curr_t - past_dt < t_start) or (curr_t + future_dt > t_end):
                    continue
                
                self.samples.append({
                    'file_idx': file_idx, 'row_idx': i, 't_curr': curr_t
                })
                
                if is_train:
                    all_speeds.append(speeds[i])
                    all_slopes.append(slopes[i])
        
        # 统计量初始化
        self.speed_mean = 0.0; self.speed_std = 1.0
        self.slope_mean = 0.0; self.slope_std = 1.0
        
        if is_train and len(all_speeds) > 0:
            self.speed_mean = float(np.mean(all_speeds))
            self.speed_std = float(np.std(all_speeds)) + 1e-6
            self.slope_mean = float(np.mean(all_slopes))
            self.slope_std = float(np.std(all_slopes)) + 1e-6
            print(f"Stats Computed -> Speed: {self.speed_mean:.2f}, Slope: {self.slope_mean:.4f}")

    def set_stats(self, stats):
        self.speed_mean = stats['speed_mean']
        self.speed_std = stats['speed_std']
        self.slope_mean = stats['slope_mean']
        self.slope_std = stats['slope_std']

    def get_stats(self):
        return {
            "speed_mean": self.speed_mean, "speed_std": self.speed_std,
            "slope_mean": self.slope_mean, "slope_std": self.slope_std
        }

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        data = self.all_dfs[item['file_idx']]
        
        t_hist = item['t_curr'] + self.hist_offsets
        t_fut = item['t_curr'] + self.fut_offsets
        
        hist_slope = np.interp(t_hist, data['times'], data['slopes'])
        hist_speed = np.interp(t_hist, data['times'], data['speeds'])
        fut_slope = np.interp(t_fut, data['times'], data['slopes'])
        
        orig_path = data['img_paths'][item['row_idx']]
        mask_rel = orig_path.replace(".jpg", ".png").replace(".jpeg", ".png")
        mask_path = os.path.join(self.mask_root, mask_rel)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None: mask = np.zeros(self.input_size, dtype=np.uint8)
            elif mask.shape != self.input_size: mask = cv2.resize(mask, self.input_size)
        else:
            mask = np.zeros(self.input_size, dtype=np.uint8)
            
        mask_tensor = torch.from_numpy(mask).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0)
        
        # Standardization
        norm_hist_speed = (hist_speed - self.speed_mean) / self.speed_std
        norm_hist_slope = (hist_slope - self.slope_mean) / self.slope_std
        norm_fut_slope = (fut_slope - self.slope_mean) / self.slope_std
        
        norm_hist_speed = np.clip(norm_hist_speed, -5.0, 5.0)
        
        hist_tensor = torch.tensor(np.stack([norm_hist_speed, norm_hist_slope], axis=1), dtype=torch.float32)
        label_tensor = torch.tensor(norm_fut_slope, dtype=torch.float32)
        curr_slope_tensor = torch.tensor([norm_hist_slope[-1]], dtype=torch.float32)
        
        if torch.isnan(mask_tensor).any() or torch.isnan(hist_tensor).any() or torch.isnan(label_tensor).any():
             return torch.zeros_like(mask_tensor), torch.zeros_like(hist_tensor), torch.zeros_like(curr_slope_tensor), torch.zeros_like(label_tensor)

        return mask_tensor, hist_tensor, curr_slope_tensor, label_tensor

# === DDP Setup ===
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_worker(rank, world_size):
    setup(rank, world_size)
    
    DATASET_ROOT = "dataset_root" # 请确保路径正确
    BATCH_SIZE = 128
    EPOCHS = 2500
    LR = 1e-4
    
    # 1. 划分数据集
    all_csvs = sorted(glob.glob(os.path.join(DATASET_ROOT, "**", "*.csv"), recursive=True))
    random.Random(42).shuffle(all_csvs)
    
    split_idx = int(0.8 * len(all_csvs))
    train_files = all_csvs
    val_files = all_csvs
    
    # 2. Dataset & Loader
    train_dataset = SlopeDataset(DATASET_ROOT, file_list=train_files, is_train=True)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4, pin_memory=True)
    
    val_dataset = SlopeDataset(DATASET_ROOT, file_list=val_files, is_train=False)
    val_dataset.set_stats(train_dataset.get_stats()) # 共享统计量
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=2, pin_memory=True)
    
    # 3. Model
    model = LaneSlopeNet(history_steps=train_dataset.hist_len, future_steps=train_dataset.fut_len).to(rank)
    model = DDP(model, device_ids=[rank])
    
    if os.path.exists("visenet2_best.pth"):
        try:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            state_dict = torch.load("visenet2_best.pth", map_location=map_location)
            model.load_state_dict(state_dict, strict=False)
            if rank == 0: print("Resumed from visenet2_best.pth")
        except: pass

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    
    # === 改进点：加入 LR Scheduler ===
    # 当 val_loss 5个 epoch 不下降时，LR 减半
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=50, verbose=True
    )
    
    criterion = nn.MSELoss()
    
    if rank == 0:
        stats = train_dataset.get_stats()
        stats.update({'past_dt': 3.0, 'future_dt': 3.0, 'step': 0.5})
        with open("scaler_params.json", "w") as f:
            json.dump(stats, f, indent=4)
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        model.train()
        
        train_loss_sum = 0.0
        train_batches = 0
        
        iterator = tqdm(train_loader, desc=f"Ep {epoch+1} Train") if rank == 0 else train_loader
        
        for masks, hist, curr, labels in iterator:
            masks, hist, curr, labels = masks.to(rank), hist.to(rank), curr.to(rank), labels.to(rank)
            
            optimizer.zero_grad()
            preds = model(masks, hist, curr)
            loss = criterion(preds, labels)
            
            if torch.isnan(loss): continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_batches += 1
            
        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for masks, hist, curr, labels in val_loader:
                masks, hist, curr, labels = masks.to(rank), hist.to(rank), curr.to(rank), labels.to(rank)
                preds = model(masks, hist, curr)
                loss = criterion(preds, labels)
                if not torch.isnan(loss):
                    val_loss_sum += loss.item()
                    val_batches += 1
        
        # Reduce Results
        tr_tensor = torch.tensor([train_loss_sum, train_batches], device=rank)
        val_tensor = torch.tensor([val_loss_sum, val_batches], device=rank)
        
        dist.all_reduce(tr_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
        
        if rank == 0:
            avg_train_loss = tr_tensor[0] / max(tr_tensor[1], 1)
            avg_val_loss = val_tensor[0] / max(val_tensor[1], 1)
            
            # 获取当前 LR
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}: Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} | LR {current_lr:.2e}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.module.state_dict(), "visenet2_best.pth")
                print(f">>> New Best Saved (Val: {best_val_loss:.4f})")
                
            if (epoch + 1) % 50 == 0:
                torch.save(model.module.state_dict(), f"visenet2_ep{epoch+1}.pth")

        # === Scheduler Step (所有进程都要执行) ===
        # avg_val_loss 在所有进程上应该是一样的 (因为做了 all_reduce)
        # 重新计算一下 avg_val_loss 确保每个 rank 都有值
        avg_val_loss_local = val_tensor[0] / max(val_tensor[1], 1)
        scheduler.step(avg_val_loss_local)

    cleanup()

def main():
    world_size = torch.cuda.device_count()
    if world_size < 1: return
    mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    main()