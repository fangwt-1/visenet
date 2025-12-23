import torch
import cv2
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from collections import deque
from tqdm import tqdm
import glob

# === 引入模型 ===
# 确保 slope_net.py 在当前目录下
from slope_net import LaneSlopeNet

# === 配置 ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "visenet2_best.pth"
SCALER_PATH = "scaler_params.json"
INPUT_SIZE = (256, 256)
MASK_ROOT = "dataset_root/masks"  # 请修改为实际 mask 根目录

class HistoryBuffer:
    def __init__(self, past_dt=3.0, step=0.5):
        self.past_dt = past_dt
        
        # === 修改点：把容量改大 ===
        # 即使是 100fps 的数据，3秒也只需要 300 帧。
        # 设为 1000 足够涵盖绝大多数车端采样频率，且这点内存在现在的机器上忽略不计。
        self.max_len = 1000 
        
        self.buffer = deque(maxlen=self.max_len)
        self.step = step
        self.offsets = np.sort(np.arange(-past_dt, 0.001, step))
    
    def push(self, timestamp, speed, slope):
        self.buffer.append({'ts': timestamp, 'speed': speed, 'slope': slope})
        
    def get_history_features(self, current_ts):
        if len(self.buffer) < 2: return None
        if self.buffer[-1]['ts'] - self.buffer[0]['ts'] < self.past_dt - 0.5: return None
        
        ts_arr = np.array([x['ts'] for x in self.buffer])
        spd_arr = np.array([x['speed'] for x in self.buffer])
        slp_arr = np.array([x['slope'] for x in self.buffer])
        
        target_ts = current_ts + self.offsets
        
        if target_ts[0] < ts_arr[0] or target_ts[-1] > ts_arr[-1] + 0.1:
            return None
            
        hist_speed = np.interp(target_ts, ts_arr, spd_arr)
        hist_slope = np.interp(target_ts, ts_arr, slp_arr)
        
        return hist_speed, hist_slope

def draw_plot_on_image(img, preds, gts, future_times):
    """
    使用 Matplotlib 画图并转为 OpenCV 图像叠加
    改进：固定布局防止闪烁
    """
    # 1. 创建 Figure，固定 DPI 和 尺寸
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    
    # 2. 【关键】手动固定边距，替代 tight_layout，防止边框跳动
    # left, bottom, right, top 是相对 figure 尺寸的比例 (0-1)
    fig.subplots_adjust(left=0.22, bottom=0.20, right=0.95, top=0.85)
    
    # 3. 画线
    ax.plot(future_times, gts, 'g-', label='GT', linewidth=2, alpha=0.8)
    ax.plot(future_times, preds, 'r--', label='Pred', linewidth=2)
    
    # 4. 【关键】固定 Y 轴范围和刻度格式
    # 范围根据你的数据分布设定，比如 -0.15 到 0.15
    Y_LIMIT = 0.2
    ax.set_ylim(-Y_LIMIT, Y_LIMIT)
    
    # 强制刻度显示 2 位小数，防止 "-0.1" 和 "-0.12" 长度不同导致抖动
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    
    # 5. 固定 X 轴
    ax.set_xlim(future_times[0]-0.1, future_times[-1]+0.1)
    
    # 6. 设置标签
    ax.set_title("Future Slope (3s)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Slope (rad)", fontsize=10)
    
    # 固定图例位置
    ax.legend(loc='upper right', fontsize='small', framealpha=0.8)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # 转为 numpy
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    plot_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    plot_img = plot_img.reshape(int(height), int(width), 3)
    
    # 必须显式关闭 figure，释放内存
    plt.close(fig)
    
    # 叠加到原图右下角
    h, w, _ = img.shape
    ph, pw, _ = plot_img.shape
    
    x_offset = w - pw - 20
    y_offset = h - ph - 20
    
    if y_offset < 0 or x_offset < 0: return img # 防止图比原图大
    
    roi = img[y_offset:y_offset+ph, x_offset:x_offset+pw]
    cv2.addWeighted(plot_img, 0.85, roi, 0.15, 0, roi)
    img[y_offset:y_offset+ph, x_offset:x_offset+pw] = roi
    
    return img

def visualize_sequence(csv_path, output_path):
    # 1. 加载配置和模型
    if not os.path.exists(SCALER_PATH):
        print("Scaler params not found!")
        return
    with open(SCALER_PATH, "r") as f:
        stats = json.load(f)
        
    # 实例化模型
    model = LaneSlopeNet(history_steps=7, future_steps=6).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        st = torch.load(MODEL_PATH, map_location=DEVICE)
        st = {k.replace('module.', ''): v for k, v in st.items()}
        model.load_state_dict(st, strict=False)
        model.eval()
    else:
        print("Model not found!")
        return

    # 2. 读取数据
    print(f"Processing {csv_path}...")
    df = pd.read_csv(csv_path)
    req_cols = ['slope_rad', 'speed_mps', 'timestamp', 'left_img']
    if not all(c in df.columns for c in req_cols):
        print("CSV missing columns.")
        return
        
    df = df.dropna(subset=req_cols).sort_values('timestamp')
    timestamps = df['timestamp'].values
    slopes = df['slope_rad'].values
    speeds = df['speed_mps'].values
    img_paths = df['left_img'].values
    
    # 3. 视频准备
    first_img_path = os.path.join(MASK_ROOT, "../", img_paths[0])
    if not os.path.exists(first_img_path):
         first_img_path = os.path.join("dataset_root", img_paths[0])
         
    frame = cv2.imread(first_img_path)
    if frame is None:
        print(f"Cannot read image: {first_img_path}")
        return
    h, w = frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))
    
    # 4. 初始化 Buffer
    history_buf = HistoryBuffer(past_dt=3.0, step=0.5)
    fut_offsets = np.arange(0.5, 3.1, 0.5)
    
    # 【新增】预测平滑 Buffer
    pred_smooth_buf = deque(maxlen=3) # 平滑最近3帧的预测
    
    print("Generating video...")
    with torch.no_grad():
        for i in tqdm(range(len(df))):
            curr_t = timestamps[i]
            curr_spd = speeds[i]
            curr_slp = slopes[i]
            img_rel = img_paths[i]
            print(f"Time: {curr_t:.2f}s, Speed: {curr_spd:.2f} m/s, Slope: {curr_slp:.4f} rad")
            
            history_buf.push(curr_t, curr_spd, curr_slp)
            
            # 读取图片
            full_img_path = os.path.join("dataset_root", img_rel)
            frame = cv2.imread(full_img_path)
            if frame is None: continue
            
            hist_res = history_buf.get_history_features(curr_t)
            if hist_res is not None:
                h_spd, h_slp = hist_res
                
                # Mask 处理
                mask_rel = img_rel.replace(".jpg", ".png").replace(".jpeg", ".png")
                mask_path = os.path.join(MASK_ROOT, mask_rel)
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask.shape != INPUT_SIZE: mask = cv2.resize(mask, INPUT_SIZE)
                else:
                    mask = np.zeros(INPUT_SIZE, dtype=np.uint8)
                
                mask_t = torch.from_numpy(mask).float() / 255.0
                mask_t = mask_t.unsqueeze(0).unsqueeze(0).to(DEVICE)
                
                norm_h_spd = (h_spd - stats['speed_mean']) / stats['speed_std']
                norm_h_slp = (h_slp - stats['slope_mean']) / stats['slope_std']
                norm_h_spd = np.clip(norm_h_spd, -5.0, 5.0)
                hist_t = torch.tensor(np.stack([norm_h_spd, norm_h_slp], axis=1), dtype=torch.float32).unsqueeze(0).to(DEVICE)
                curr_slp_norm_t = torch.tensor([norm_h_slp[-1]], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                
                # 推理
                pred_norm = model(mask_t, hist_t, curr_slp_norm_t)
                pred_raw = pred_norm.cpu().numpy().flatten() * stats['slope_std'] + stats['slope_mean']
                
                # 【新增】平滑处理：取最近几次预测的平均值
                pred_smooth_buf.append(pred_raw)
                pred_smoothed = np.mean(pred_smooth_buf, axis=0)
                
                # 获取 GT
                gt_raw = np.interp(curr_t + fut_offsets, timestamps, slopes)
                
                # 绘图 (使用平滑后的预测值)
                frame = draw_plot_on_image(frame, pred_smoothed, gt_raw, fut_offsets)
                
                # 叠加数值
                text_color = (0, 255, 255) # 黄色
                cv2.putText(frame, f"Speed: {curr_spd:.1f} m/s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                cv2.putText(frame, f"Slope: {curr_slp:.3f} rad", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            else:
                cv2.putText(frame, "Initializing Buffer...", (w//2-100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            out.write(frame)
            
    out.release()
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    # 自动寻找第一个 CSV 文件进行测试
    all_csvs = glob.glob("dataset_root/**/*.csv", recursive=True)
    if all_csvs:
        # 为了演示效果，尽量选一个文件较大的（时间长）
        all_csvs.sort(key=lambda x: os.path.getsize(x), reverse=True)
        test_csv = all_csvs[2] 
        print(f"Visualizing {test_csv}")
        visualize_sequence(test_csv, "vis_result_stable.mp4")
    else:
        print("No CSV files found.")