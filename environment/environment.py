import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor
from collections import deque
import math

# === 1. 模拟 LaneDet 模型 (基于 OpenCV) ===
class LaneDetModel:
    """
    模拟 LaneDet 模型的功能：输入图像，输出 2D 车道线参数。
    由于无法加载外部权重，这里使用鲁棒的传统图像处理算法替代。
    """
    def __init__(self):
        pass

    def detect(self, img):
        """
        返回: list of lines [x1, y1, x2, y2]
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. ROI 掩码 (只关注下半部分)
        mask = np.zeros_like(gray)
        points = np.array([
            [(0, h), (w//2 - 50, h//2 + 50), (w//2 + 50, h//2 + 50), (w, h)]
        ], dtype=np.int32)
        cv2.fillPoly(mask, points, 255)
        masked_gray = cv2.bitwise_and(gray, mask)
        
        # 2. 边缘检测
        edges = cv2.Canny(masked_gray, 50, 150)
        
        # 3. 霍夫直线变换
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=20)
        
        output_lines = []
        if lines is not None:
            for line in lines:
                output_lines.append(line[0])
        return output_lines

# === 2. 2D 车道线坡度估计器 ===
class LaneSlopeEstimator:
    def __init__(self, cam_matrix):
        self.fx = cam_matrix[0, 0]
        self.fy = cam_matrix[1, 1]
        self.cx = cam_matrix[0, 2]
        self.cy = cam_matrix[1, 2]
        self.model = LaneDetModel() # 使用模拟的 LaneDet

    def get_intersection(self, line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0: return None
        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
        return [px, py]

    def fit_lane_line(self, lines, img_w):
        """ 将零散线段拟合为左右两条主车道线 """
        left_pts = []
        right_pts = []
        
        for x1, y1, x2, y2 in lines:
            if x2 == x1: continue
            k = (y2 - y1) / (x2 - x1)
            if abs(k) < 0.3 or abs(k) > 5.0: continue
            
            # 简单的左右分类
            if k < 0 and x1 < img_w * 0.6:
                left_pts.append([x1, y1])
                left_pts.append([x2, y2])
            elif k > 0 and x1 > img_w * 0.4:
                right_pts.append([x1, y1])
                right_pts.append([x2, y2])
                
        def fit_line(pts):
            if len(pts) < 2: return None
            [vx, vy, x, y] = cv2.fitLine(np.array(pts), cv2.DIST_L2, 0, 0.01, 0.01)
            k = vy / (vx + 1e-6)
            # 构造一条长线段用于计算
            return [float(x - 1000), float(y - k*1000), float(x + 1000), float(y + k*1000)]

        return fit_line(left_pts), fit_line(right_pts)

    def run(self, img):
        """
        逻辑: LaneDet -> 2D Lines -> Vanishing Point -> Slope
        """
        raw_lines = self.model.detect(img)
        if not raw_lines: return 0.0
        
        h, w = img.shape[:2]
        left_line, right_line = self.fit_lane_line(raw_lines, w)
        
        if left_line and right_line:
            vp = self.get_intersection(left_line, right_line)
            if vp:
                u, v = vp
                # 消失点合理性检查
                if -h < v < h * 2: 
                    # 映射关系: pitch = -arctan((v_vp - cy) / fy)
                    # 负号是因为图像坐标系Y向下，VP向上移(v减小)代表上坡
                    return float(-np.arctan((v - self.cy) / self.fy))
        
        return 0.0

# === 3. 环境融合与未来预测 ===
class EnvironmentFusion:
    def __init__(self, cam_matrix):
        self.K = cam_matrix
        self.cx = cam_matrix[0, 2]
        self.cy = cam_matrix[1, 2]
        self.fx = cam_matrix[0, 0]
        self.fy = cam_matrix[1, 1]
        
        self.lane_estimator = LaneSlopeEstimator(cam_matrix)
        
        # 历史数据 Buffer，用于时序预测
        self.slope_history = deque(maxlen=50) # 存最近50帧的融合坡度
        self.gps_buffer = deque(maxlen=10)    # 存最近10帧GPS用于差分
        
        self.last_valid_dist = None

    def get_front_vehicle_dist(self, detections, depth_map):
        # ... (保持原有的车辆测距逻辑不变) ...
        min_dist = 100.0 
        found_valid_obj = False
        h, w = depth_map.shape
        
        for obj in detections:
            x1, y1, x2, y2 = obj['bbox']
            w_box = x2 - x1
            h_box = y2 - y1
            crop = 0.3
            rx1, ry1 = int(x1 + w_box*crop), int(y1 + h_box*crop)
            rx2, ry2 = int(x2 - w_box*crop), int(y2 - h_box*crop)
            
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(w, rx2), min(h, ry2)
            
            if rx2 > rx1 and ry2 > ry1:
                roi = depth_map[ry1:ry2, rx1:rx2]
                valid = roi[(roi > 2.0) & (roi < 80.0)]
                if len(valid) > 5:
                    d = np.percentile(valid, 20)
                    obj['dist'] = d
                    if d < min_dist:
                        min_dist = d
                        found_valid_obj = True
            if 'dist' not in obj: obj['dist'] = -1

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
        """ 3D 深度图重建地面坡度 """
        h, w = depth_map.shape
        roi = depth_map[int(h*0.6):h, int(w*0.3):int(w*0.7)]
        roi_sub = roi[::5, ::5]
        valid_mask = (roi_sub > 1.0) & (roi_sub < 50.0)
        zs = roi_sub[valid_mask]
        
        if len(zs) < 50: return 0.0
        
        grid_v, _ = np.indices(roi_sub.shape)
        grid_v = grid_v * 5 + int(h*0.6)
        vs = grid_v[valid_mask]
        
        # Y = (v - cy) * Z / fy
        # 拟合 Z vs Y 的斜率 k, slope = arctan(-k)
        ys = (vs - self.cy) * zs / self.fy
        
        try:
            ransac = RANSACRegressor().fit(zs.reshape(-1, 1), ys)
            k = ransac.estimator_.coef_[0]
            return float(np.arctan(-k)) 
        except:
            return 0.0

    def calculate_gps_slope(self, current_gps):
        """
        通过 GPS 差分计算坡度
        current_gps: {'lat': float, 'lon': float, 'alt': float}
        """
        if current_gps is None: return None
        
        self.gps_buffer.append(current_gps)
        if len(self.gps_buffer) < 2: return None
        
        # 取最早和最新的数据进行差分，减少噪声
        old = self.gps_buffer[0]
        curr = self.gps_buffer[-1]
        
        d_alt = curr['alt'] - old['alt']
        
        # 简单的经纬度转米 (近似)
        R = 6371000
        d_lat = np.radians(curr['lat'] - old['lat'])
        d_lon = np.radians(curr['lon'] - old['lon'])
        a = np.sin(d_lat/2)**2 + np.cos(np.radians(old['lat'])) * np.cos(np.radians(curr['lat'])) * np.sin(d_lon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        dist = R * c
        
        if dist < 2.0: return None # 移动距离太小，误差大
        
        return float(np.arctan(d_alt / dist))

    def predict_future_slope(self, current_fused_slope):
        """
        预测未来5秒的坡度。
        原理：基于历史坡度的变化趋势（一阶线性回归）进行外推。
        假设帧率为 20fps，5秒对应 100 帧。
        """
        self.slope_history.append(current_fused_slope)
        
        if len(self.slope_history) < 10:
            return current_fused_slope # 数据不够，不预测
        
        # 线性拟合 y = ax + b
        y = np.array(self.slope_history)
        x = np.arange(len(y))
        
        try:
            # 使用最近 20 帧拟合趋势
            fit_len = min(len(y), 20)
            coeffs = np.polyfit(x[-fit_len:], y[-fit_len:], 1) 
            slope_change_rate = coeffs[0]
            current_trend = coeffs[1] + coeffs[0] * x[-1]
            
            # 外推: 假设采样间隔 dt (约0.05s)
            # future_steps = 5.0 / 0.05 = 100
            # 注意：简单的线性外推在长达5秒的时间里可能很不准，这里做一个衰减
            # 实际上未来坡度更多取决于地图信息，纯感知只能预测“可见范围”内的趋势
            future_steps = 100 
            pred_val = current_trend + slope_change_rate * future_steps * 0.5 # 0.5为保守系数
            
            # 限制幅度，防止飞出天际
            return np.clip(pred_val, -0.3, 0.3) 
        except:
            return current_fused_slope

    def fusion_slope(self, img, depth_map, imu_pitch=None, gps_data=None):
        # 1. 2D 视觉坡度 (模拟 LaneDet)
        s_2d = self.lane_estimator.run(img)
        
        # 2. 3D 深度坡度
        s_3d = self.estimate_depth_slope(depth_map)
        
        # 3. GPS 差分坡度
        s_gps = self.calculate_gps_slope(gps_data)
        
        # 4. 融合逻辑
        # 权重分配：
        # - IMU: 最稳定，但包含车辆姿态（俯仰），不完全等于道路坡度
        # - Visual (2D/3D): 容易受颠簸影响，但能反映相对路面
        # - GPS: 长期来看最准，但短期噪声大，更新慢
        
        val_list = []
        w_list = []
        
        # 基础权重
        if abs(s_2d) > 0.001: 
            val_list.append(s_2d)
            w_list.append(0.3)
            
        if abs(s_3d) > 0.001:
            val_list.append(s_3d)
            w_list.append(0.3)
            
        if s_gps is not None:
            val_list.append(s_gps)
            w_list.append(0.4)
            
        if imu_pitch is not None:
            val_list.append(imu_pitch)
            w_list.append(0.5) # IMU 权重较高
            
        if not val_list:
            final_slope = 0.0
        else:
            final_slope = np.average(val_list, weights=w_list)
            
        # 5. 未来预测
        future_slope_5s = self.predict_future_slope(final_slope)
        
        return final_slope, future_slope_5s