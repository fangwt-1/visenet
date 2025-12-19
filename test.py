import pandas as pd
import numpy as np
import glob
import os

def check_data():
    # 替换为你的 CSV 路径模式
    csv_files = glob.glob(os.path.join("dataset_root", "**", "*.csv"), recursive=True)
    
    print(f"Checking {len(csv_files)} CSV files...")
    
    has_nan = False
    for f in csv_files:
        df = pd.read_csv(f)
        
        # 检查关键列
        cols = ['slope_rad', 'speed_mps', 'timestamp']
        for c in cols:
            if c not in df.columns: continue
            
            # 检查 NaN
            nans = df[c].isna().sum()
            # 检查 Inf
            infs = np.isinf(df[c]).sum()
            
            if nans > 0 or infs > 0:
                print(f"[BAD DATA] File: {f}")
                print(f"  Column '{c}': NaNs={nans}, Infs={infs}")
                has_nan = True
                
                # 打印出坏数据的几行看看
                print(df[df[c].isna() | np.isinf(df[c])].head())

    if not has_nan:
        print("✅ 数据检查通过：没有发现 NaN 或 Inf。")
    else:
        print("❌ 发现坏数据！请在 Dataset 加载时过滤掉这些行。")

if __name__ == "__main__":
    check_data()