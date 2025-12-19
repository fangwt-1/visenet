import numpy as np
import cv2
import torch
import torch.nn as nn
import json
import os
import sys
from collections import deque
from sklearn.linear_model import RANSACRegressor

# === 引入 YOLOP (假设 standalone_yolop.py 在同一目录或 Python 路径下) ===
# 如果报错找不到模块，请确保 standalone_yolop.py 在 visenet-fwt_lane 根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from standalone_yolop import YOLOPTester
except ImportError:
    print("Warning: Could not import YOLOPTester from standalone_yolop.py")
    YOLOPTester = None

# === 1. 定义你的 LaneSlopeNet (Residual Version) ===
class LaneSlopeNet(nn.Module):
    def __init__(self, input_size=(256, 256)):
        super(LaneSlopeNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4))
        )
        self.flatten_dim = 64 * 4 * 4
        
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim + 2, 128), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, mask, speed, current_slope):
        x = self.features(mask)
        x = x.view(-1, self.flatten_dim)
        combined = torch.cat([x, speed, current_slope], dim=1)
        delta = self.fc(combined)
        return current_slope + delta  # 残差连接

# === 2. 几何坡度估计器 (升级版：基于 Mask) ===
class LaneSlopeEstimator:
    def __init__(self, cam_matrix):
        self.fx = cam_matrix[0, 0]
        self.fy = cam_matrix[1, 1]
        self.cx = cam_matrix[0, 2]
        self.cy = cam_matrix[1, 2]

    def get_intersection(self, line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0: return None
        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
        return [px, py]

    def fit_lane_line(self, lines, img_w):
        left_pts, right_pts = [], []
        for x1, y1, x2, y2 in lines:
            if x2 == x1: continue
            k = (y2 - y1) / (x2 - x1)
            if abs(k) < 0.3 or abs(k) > 5.0: continue
            if k < 0 and x1 < img_w * 0.6:
                left_pts.append([x1, y1]); left_pts.append([x2, y2])
            elif k > 0 and x1 > img_w * 0.4:
                right_pts.append([x1, y1]); right_pts.append([x2, y2])
                
        def fit_line(pts):
            if len(pts) < 2: return None
            [vx, vy, x, y] = cv2.fitLine(np.array(pts), cv2.DIST_L2, 0, 0.01, 0.01)
            k = vy / (vx + 1e-6)
            return [float(x - 1000), float(y - k*1000), float(x + 1000), float(y + k*1000)]

        return fit_line(left_pts), fit_line(right_pts)

    def run_from_mask(self, mask):
        """ 从 YOLOP 的二值 Mask 中提取线段并计算几何坡度 """
        h, w = mask.shape[:2]
        # 边缘检测 + 霍夫变换提取线段
        edges = cv2.Canny(mask * 255, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=30, maxLineGap=20)
        
        raw_lines = []
        if lines is not None:
            for line in lines: raw_lines.append(line[0])
            
        if not raw_lines: return 0.0
        
        left_line, right_line = self.fit_lane_line(raw_lines, w)
        if left_line and right_line:
            vp = self.get_intersection(left_line, right_line)
            if vp:
                u, v = vp
                if -h < v < h * 2: 
                    return float(-np.arctan((v - self.cy) / self.fy))
        return 0.0

# === 3. 环境融合主类 (集成 Visenet2) ===
class EnvironmentFusion:
    def __init__(self, cam_matrix, model_path="visenet2_best.pth", scaler_path="scaler_params.json"):
        self.K = cam_matrix
        self.cx = cam_matrix[0, 2]
        self.cy = cam_matrix[1, 2]
        self.fx = cam_matrix[0, 0]
        self.fy = cam_matrix[1, 1]
        
        self.lane_estimator = LaneSlopeEstimator(cam_matrix)
        self.gps_buffer = deque(maxlen=10)
        self.last_valid_dist = None
        
        # --- 初始化 AI 模型 ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. YOLOP
        if YOLOPTester:
            self.yolop = YOLOPTester(device=str(self.device))
        else:
            self.yolop = None
            
        # 2. LaneSlopeNet (Visenet2)
        print(f"Loading LaneSlopeNet from {model_path}...")
        self.slope_net = LaneSlopeNet().to(self.device)
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            # 处理 DDP 保存的权重 (去除 module. 前缀)
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.slope_net.load_state_dict(new_state_dict)
            self.slope_net.eval()
            print("LaneSlopeNet loaded successfully.")
        else:
            print(f"Warning: {model_path} not found! Prediction will be random.")

        # 3. 加载标准化参数
        if os.path.exists(scaler_path):
            with open(scaler_path, "r") as f:
                self.scaler = json.load(f)
            print(f"Scaler loaded: {self.scaler}")
        else:
            self.scaler = {"speed_mean": 0, "speed_std": 1, "slope_mean": 0, "slope_std": 1}
            print("Warning: Using default scaler.")

    def get_front_vehicle_dist(self, detections, depth_map):
        # ... (保持原有的车辆测距逻辑不变) ...
        # 注意：这里的 detections 现在由 fusion_slope 返回，或者你需要单独调 YOLOP
        min_dist = 100.0 
        found_valid_obj = False
        h, w = depth_map.shape
        
        for obj in detections:
            # bbox 格式: (x1, y1, x2, y2), obj 是 (box, score) 元组
            # 这里需要适配 yolop 返回的格式
            box, score = obj
            x1, y1, x2, y2 = box
            
            # ... 简单的 ROI 深度提取逻辑 ...
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)
            w_box, h_box = x2-x1, y2-y1
            
            # 简单取中心区域
            crop = 0.2
            rx1, ry1 = int(x1 + w_box*crop), int(y1 + h_box*crop)
            rx2, ry2 = int(x2 - w_box*crop), int(y2 - h_box*crop)
            
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(w, rx2), min(h, ry2)
            
            if rx2 > rx1 and ry2 > ry1:
                roi = depth_map[ry1:ry2, rx1:rx2]
                valid = roi[(roi > 2.0) & (roi < 80.0)]
                if len(valid) > 5:
                    d = np.percentile(valid, 20)
                    if d < min_dist:
                        min_dist = d
                        found_valid_obj = True
        
        # 平滑逻辑
        final_dist = 100.0
        if found_valid_obj:
            if self.last_valid_dist is not None:
                alpha = 0.6 if abs(min_dist - self.last_valid_dist) < 5.0 else 0.1
                final_dist = alpha * min_dist + (1 - alpha) * self.last_valid_dist
            else:
                final_dist = min_dist
            self.last_valid_dist = final_dist
        else:
            if self.last_valid_dist is not None:
                final_dist = self.last_valid_dist * 0.95 + 100.0 * 0.05
                self.last_valid_dist = final_dist
        return final_dist

    def estimate_depth_slope(self, depth_map):
        """ 3D 深度图重建地面坡度 (保持不变) """
        h, w = depth_map.shape
        roi = depth_map[int(h*0.6):h, int(w*0.3):int(w*0.7)]
        roi_sub = roi[::5, ::5]
        valid_mask = (roi_sub > 1.0) & (roi_sub < 50.0)
        zs = roi_sub[valid_mask]
        if len(zs) < 50: return 0.0
        grid_v, _ = np.indices(roi_sub.shape)
        grid_v = grid_v * 5 + int(h*0.6)
        vs = grid_v[valid_mask]
        ys = (vs - self.cy) * zs / self.fy
        try:
            ransac = RANSACRegressor().fit(zs.reshape(-1, 1), ys)
            k = ransac.estimator_.coef_[0]
            return float(np.arctan(-k)) 
        except:
            return 0.0

    def calculate_gps_slope(self, current_gps):
        """ GPS 差分 (保持不变) """
        if current_gps is None: return None
        self.gps_buffer.append(current_gps)
        if len(self.gps_buffer) < 2: return None
        old = self.gps_buffer[0]; curr = self.gps_buffer[-1]
        d_alt = curr['alt'] - old['alt']
        R = 6371000
        d_lat = np.radians(curr['lat'] - old['lat'])
        d_lon = np.radians(curr['lon'] - old['lon'])
        a = np.sin(d_lat/2)**2 + np.cos(np.radians(old['lat'])) * np.cos(np.radians(curr['lat'])) * np.sin(d_lon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        dist = R * c
        if dist < 2.0: return None
        return float(np.arctan(d_alt / dist))

    def predict_with_visenet(self, mask, speed, current_slope):
        """ 使用 Visenet2 进行预测 """
        # 1. Mask 预处理: 640x640 -> 256x256, Normalized
        mask_input = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask_input).float() / 255.0 # 0-1
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device) # [1, 1, 256, 256]
        
        # 2. 数值标准化
        stats = self.scaler
        norm_speed = (speed - stats['speed_mean']) / stats['speed_std']
        norm_curr_slope = (current_slope - stats['slope_mean']) / stats['slope_std']
        
        speed_tensor = torch.tensor([[norm_speed]], dtype=torch.float32).to(self.device)
        curr_slope_tensor = torch.tensor([[norm_curr_slope]], dtype=torch.float32).to(self.device)
        
        # 3. 推理
        with torch.no_grad():
            # 输出是 Normalized Slope (Residual 结构已经在 forward 里处理了加法)
            # 但注意：Dataset 里的 label 是 Normalized 的 Abs Slope
            # 模型 forward 返回的是 Normalized Abs Slope
            pred_norm = self.slope_net(mask_tensor, speed_tensor, curr_slope_tensor)
            
        # 4. 反标准化
        pred_slope = pred_norm.item() * stats['slope_std'] + stats['slope_mean']
        return pred_slope

    def fusion_slope(self, img, depth_map, speed_mps, imu_pitch=None, gps_data=None):
        """
        主流程:
        1. YOLOP -> Mask & Detections
        2. Mask -> 几何坡度 (s_2d)
        3. Depth -> 3D坡度 (s_3d)
        4. 融合 -> current_slope
        5. Visenet2 -> future_slope
        """
        # A. 运行 YOLOP
        yolop_res = self.yolop.infer(img) # 假设输入已经是路径或YOLOP支持numpy? 
        # 注意: standalone_yolop.py 的 infer 接收的是路径。如果 img 是 numpy，需要修改 yolop 的 infer。
        # 这里假设 img 是路径字符串。如果是 numpy，请修改 YOLOPTester.infer 接收 numpy。
        # 为了兼容，我们假设 environment.py 里传入的是 cv2 image (numpy)
        # 我们需要简单修改 YOLOPTester 或者在这里适配
        # 临时适配：调用 YOLOP 内部处理逻辑
        
        # [临时 YOLOP 推理逻辑]
        img_h, img_w = img.shape[:2]
        img_in = cv2.resize(img, (640, 640))
        img_in = img_in.astype(np.float32) / 255.0
        img_in = (img_in - self.yolop.normalize_mean) / self.yolop.normalize_std
        img_in = img_in.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img_in).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            det_out, da_seg_out, ll_seg_out = self.yolop.model(img_tensor)
            
        # Mask 后处理
        ll_seg_mask = ll_seg_out[0].argmax(0).cpu().numpy().astype(np.uint8)
        # Resize 回原图计算 Hough，Resize 到 256 给 Visenet
        # Mask 用于几何计算 (resize to original)
        mask_orig = cv2.resize(ll_seg_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        
        # B. 几何坡度 (基于 YOLOP Mask)
        s_2d = self.lane_estimator.run_from_mask(mask_orig)
        
        # C. 3D 深度坡度
        s_3d = self.estimate_depth_slope(depth_map)
        
        # D. GPS 坡度
        s_gps = self.calculate_gps_slope(gps_data)
        
        # E. 融合当前坡度
        val_list = []
        w_list = []
        
        if abs(s_2d) > 0.001: 
            val_list.append(s_2d); w_list.append(0.3)
        if abs(s_3d) > 0.001:
            val_list.append(s_3d); w_list.append(0.3)
        if s_gps is not None:
            val_list.append(s_gps); w_list.append(0.4)
        if imu_pitch is not None:
            val_list.append(imu_pitch); w_list.append(0.5)
            
        if not val_list:
            current_fused_slope = 0.0
        else:
            current_fused_slope = np.average(val_list, weights=w_list)
            
        # F. AI 预测未来坡度 (Visenet2)
        # 传入: (Mask, Speed, CurrentFused)
        # 注意: 如果 Mask 质量太差(全黑)，Visenet 会依靠 CurrentFused 进行惯性预测
        future_slope = self.predict_with_visenet(mask_orig, speed_mps, current_fused_slope)
        
        # 返回: 当前融合坡度, AI预测坡度, 车辆检测结果(这里det_out还没处理NMS，为简化略过，返回空)
        return current_fused_slope, future_slope, []