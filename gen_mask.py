import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob

# 引入你的 YOLOP 类
from standalone_yolop import YOLOPTester

def generate_masks():
    # 配置路径
    DATASET_ROOT = "dataset_root"
    IMG_ROOT = os.path.join(DATASET_ROOT, "images")
    MASK_ROOT = os.path.join(DATASET_ROOT, "masks") # 生成的 mask 存放位置
    
    # 查找所有 CSV 文件 (假设 CSV 在 images 目录下或其子目录)
    # 根据你的描述 "images同目录下有对应的csv"，可能是 dataset_root/*.csv 或者 dataset_root/images/*.csv
    # 这里我们搜索 dataset_root 下所有的 csv
    csv_files = glob.glob(os.path.join(DATASET_ROOT, "**", "*.csv"), recursive=True)
    
    if not csv_files:
        print("Error: No CSV files found in dataset_root!")
        return

    print(f"Found {len(csv_files)} CSV files. Initializing YOLOP...")
    
    # 初始化 YOLOP
    tester = YOLOPTester()
    
    # 只需要车道线分割头
    tester.model.eval()
    
    for csv_path in csv_files:
        print(f"Processing {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # 确保有 left_img 列
        if 'left_img' not in df.columns:
            print(f"Skipping {csv_path}: 'left_img' column not found.")
            continue
            
        # 遍历每一行
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            rel_path = row['left_img'] # 例如: record_.../left/xxx.jpg
            
            # 拼接完整图片路径
            # 假设 CSV 里的路径是相对于 dataset_root/images 的
            img_path = os.path.join(IMG_ROOT, rel_path)
            
            if not os.path.exists(img_path):
                # 尝试另一种路径拼接（如果 CSV 在子文件夹里）
                img_path = os.path.join(os.path.dirname(csv_path), rel_path)
                if not os.path.exists(img_path):
                    continue

            # 定义 Mask 保存路径 (保持相同的目录结构)
            save_rel_path = rel_path.replace(".jpg", ".png").replace(".jpeg", ".png")
            save_path = os.path.join(MASK_ROOT, save_rel_path)
            
            # 如果已经存在，跳过（断点续传）
            if os.path.exists(save_path):
                continue
                
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # --- YOLOP 推理 ---
            img0 = cv2.imread(img_path)
            if img0 is None: continue
            
            # 预处理 (Ref: standalone_yolop.py)
            img = cv2.resize(img0, (640, 640))
            img = img.astype(np.float32) / 255.0
            img = (img - tester.normalize_mean) / tester.normalize_std
            img = img.transpose(2, 0, 1)
            img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(tester.device)
            
            with torch.no_grad():
                # det_out, da_seg_out, ll_seg_out
                _, _, ll_seg_out = tester.model(img_tensor)
                
                # ll_seg_out: [1, 2, 640, 640]
                # 取通道 1 (车道线), sigmoid 归一化
                mask_prob = torch.sigmoid(ll_seg_out[0, 1])
                
                # 缩放到 256x256 (减少存储空间，且对回归任务够用了)
                # 使用 interpolate
                mask_prob = torch.nn.functional.interpolate(
                    mask_prob.unsqueeze(0).unsqueeze(0), 
                    size=(256, 256), 
                    mode='bilinear'
                ).squeeze()
                
                # 转为 0-255 图片保存
                mask_np = (mask_prob.cpu().numpy() * 255).astype(np.uint8)
                
                cv2.imwrite(save_path, mask_np)

    print("Mask generation complete!")

if __name__ == "__main__":
    generate_masks()