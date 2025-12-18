import cv2
import numpy as np
import collections
import math
import sys
import os
from collections import deque
from scipy.spatial.transform import Rotation as R_scipy

# === 引入项目模块 ===
from perception.perception import ObjectDetector, DepthEstimator
from environment.environment import EnvironmentFusion

# === 兼容性处理 ===
HAS_ROS = False
try:
    import rosbag
    from cv_bridge import CvBridge
    HAS_ROS = True
except ImportError:
    try:
        from rosbags.rosbag1 import Reader
        from rosbags.typesys import Stores, get_typestore
        HAS_ROS = False
    except ImportError:
        pass

class SimpleCvBridge:
    def imgmsg_to_cv2(self, msg):
        dtype = np.uint8
        if hasattr(msg, 'encoding') and '16UC1' in msg.encoding:
            dtype = np.uint16
        img = np.frombuffer(msg.data, dtype=dtype)
        if len(img) == msg.height * msg.width * 3:
            img = img.reshape(msg.height, msg.width, 3)
            if hasattr(msg, 'encoding') and 'rgb' in msg.encoding.lower():
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif len(img) == msg.height * msg.width:
            img = img.reshape(msg.height, msg.width)
        return img

def get_time_sec(header):
    if hasattr(header.stamp, 'to_sec'):
        return header.stamp.to_sec()
    else:
        return header.stamp.sec + header.stamp.nanosec * 1e-9

# === 动态 IMU 对齐器 (与 run.py 保持一致) ===
class DynamicImuAligner:
    def __init__(self):
        self.bias = 0.0
        self.is_initialized = False
        self.gps_buffer = collections.deque(maxlen=100) # 存储用于差分的数据
        
    def update(self, raw_imu_pitch, current_gps, timestamp):
        """
        利用 GPS 数据动态校准 IMU Bias。
        返回: 校准后的 IMU Pitch
        """
        if current_gps is None:
            return raw_imu_pitch - self.bias
            
        self.gps_buffer.append((timestamp, current_gps))
        
        # 1. 尝试计算可靠的 GPS 坡度 (回溯 > 5m)
        gps_slope = None
        MIN_DIST = 5.0
        curr_g = self.gps_buffer[-1][1]
        best_prev = None
        
        # 倒序查找
        for i in range(len(self.gps_buffer)-2, -1, -1):
            t, prev_g = self.gps_buffer[i]
            # 简单距离计算
            R = 6371000
            d_lat = np.radians(curr_g['lat'] - prev_g['lat'])
            d_lon = np.radians(curr_g['lon'] - prev_g['lon'])
            dist = R * math.sqrt(d_lat**2 + (math.cos(math.radians(prev_g['lat'])) * d_lon)**2)
            
            if dist > MIN_DIST:
                best_prev = (t, prev_g, dist)
                break
        
        valid_gps_slope = False
        if best_prev:
            prev_t, prev_g, dist = best_prev
            dt = timestamp - prev_t
            speed = dist / (dt + 1e-6)
            
            # 只有当速度 > 3m/s (约10km/h) 时，GPS 坡度才准
            if speed > 3.0:
                d_alt = curr_g['alt'] - prev_g['alt']
                gps_slope = math.atan(d_alt / dist)
                
                # 排除异常值 (比如坡度 > 20度)
                if abs(gps_slope) < math.radians(20):
                    valid_gps_slope = True
        
        # 2. 更新 Bias
        if valid_gps_slope:
            # 当前的瞬时偏差 = 原始IMU - 真实GPS坡度
            instant_bias = raw_imu_pitch - gps_slope
            
            if not self.is_initialized:
                self.bias = instant_bias
                self.is_initialized = True
            else:
                # 互补滤波，缓慢更新
                alpha = 0.005 
                self.bias = (1 - alpha) * self.bias + alpha * instant_bias
                
        return raw_imu_pitch - self.bias

# === 绘图器 ===
class SlopePlotter:
    def __init__(self, width=600, height=300, max_len=100):
        self.width = width
        self.height = height
        self.max_len = max_len
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 数据缓存
        self.data = {
            '2D_Lane': deque(maxlen=max_len),
            '3D_Depth': deque(maxlen=max_len),
            'IMU(Corr)': deque(maxlen=max_len), # 改名：修正后的IMU
            'GPS': deque(maxlen=max_len),
            'Fused': deque(maxlen=max_len),
            'Pred_5s': deque(maxlen=max_len)
        }
        
        # 颜色定义 (BGR)
        self.colors = {
            '2D_Lane': (0, 255, 0),    # Green
            '3D_Depth': (0, 255, 255), # Yellow
            'IMU(Corr)': (255, 0, 255),# Magenta
            'GPS': (255, 100, 0),      # Blue-ish
            'Fused': (255, 255, 255),  # White
            'Pred_5s': (0, 0, 255)     # Red
        }
        
        self.y_range = 8.0 # +/- 8 degrees

    def update(self, s_2d, s_3d, s_imu, s_gps, s_fused, s_pred):
        def to_deg(r): return math.degrees(r) if r is not None else 0.0
        self.data['2D_Lane'].append(to_deg(s_2d))
        self.data['3D_Depth'].append(to_deg(s_3d))
        self.data['IMU(Corr)'].append(to_deg(s_imu))
        self.data['GPS'].append(to_deg(s_gps))
        self.data['Fused'].append(to_deg(s_fused))
        self.data['Pred_5s'].append(to_deg(s_pred))

    def draw(self):
        self.canvas.fill(30)
        cy = self.height // 2
        # 网格
        for i in range(1, 4):
            offset = int((self.height/2) * (i/4))
            cv2.line(self.canvas, (0, cy+offset), (self.width, cy+offset), (60,60,60), 1)
            cv2.line(self.canvas, (0, cy-offset), (self.width, cy-offset), (60,60,60), 1)
        cv2.line(self.canvas, (0, cy), (self.width, cy), (150, 150, 150), 1)
        
        # 图例
        legend_x = 10
        for i, (key, color) in enumerate(self.colors.items()):
            y_pos = 20 + i * 20
            cv2.line(self.canvas, (legend_x, y_pos), (legend_x+20, y_pos), color, 2)
            cv2.putText(self.canvas, key, (legend_x+25, y_pos+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        # 曲线
        for key, points in self.data.items():
            if len(points) < 2: continue
            pts_array = []
            for i, val in enumerate(points):
                x = int(i * (self.width / self.max_len))
                val_clip = max(-self.y_range, min(self.y_range, val))
                y = int(cy - (val_clip / self.y_range) * (cy - 10))
                pts_array.append([x, y])
            
            cv2.polylines(self.canvas, [np.array(pts_array)], False, self.colors[key], 2)
            # 显示最新值
            last_val = points[-1]
            cv2.putText(self.canvas, f"{last_val:.1f}", (pts_array[-1][0]-30, pts_array[-1][1]-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors[key], 1)
        return self.canvas

# === 主系统 ===
class SlopeAnalysisSystem:
    def __init__(self, bag_path, raw_calib):
        self.bag_path = bag_path
        
        if HAS_ROS:
            self.bridge = CvBridge()
        else:
            self.bridge = SimpleCvBridge()
            self.typestore = get_typestore(Stores.ROS1_NOETIC)
        
        self.map1x, self.map1y, self.map2x, self.map2y, P1, P2 = self.init_rectification(raw_calib)
        new_fx = P1[0, 0]
        new_baseline = abs(P2[0, 3] / P2[0, 0]) if new_fx != 0 else 0.12
            
        print(f"[Init] Params -> fx: {new_fx:.2f}, baseline: {new_baseline:.4f}m")

        self.depth_net = DepthEstimator(method='depth_anything_raft', baseline=new_baseline, focal_length=new_fx)
        self.K_new = P1[:3, :3]
        self.fusion = EnvironmentFusion(cam_matrix=self.K_new)
        self.plotter = SlopePlotter(width=800, height=400)

        # === 核心修改：使用对齐器 ===
        self.imu_aligner = DynamicImuAligner()
        self.last_corrected_pitch = 0.0
        self.current_gps = None
        
        # 独立的GPS Buffer用于画图 (逻辑也得升级成长距离)
        self.vis_gps_buffer = deque(maxlen=100)

        self.left_queue = collections.deque()
        self.right_queue = collections.deque()
        self.sync_threshold = 0.1

        self.left_topic = '/zed_node/rgb/left_image'
        self.right_topic = '/zed_node/rgb/right_image'
        self.imu_topic = '/imu/data'
        self.gps_topic = '/fix' 

        self.output_video_path = "slope_vis_result.mp4"
        self.video_writer = None

    def init_rectification(self, calib):
        K1, D1, K2, D2 = calib['K1'], calib['D1'], calib['K2'], calib['D2']
        R, T, img_size = calib['R'], calib['T'], calib['img_size']
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, img_size, R, T, alpha=-1)
        m1x, m1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_16SC2)
        m2x, m2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_16SC2)
        return m1x, m1y, m2x, m2y, P1, P2

    def process_bag(self):
        print(f"Opening {self.bag_path}...")
        if not os.path.exists(self.bag_path):
            print("[Error] Bag file not found.")
            return

        target_topics = [self.left_topic, self.right_topic, self.imu_topic, self.gps_topic]

        if not HAS_ROS:
            from rosbags.rosbag1 import Reader
            with Reader(self.bag_path) as reader:
                for connection, timestamp, rawdata in reader.messages():
                    if connection.topic not in target_topics: continue
                    msg = self.typestore.deserialize_ros1(rawdata, connection.msgtype)
                    self.handle_message(connection.topic, msg, timestamp * 1e-9)
        else:
            import rosbag
            bag = rosbag.Bag(self.bag_path)
            for topic, msg, t in bag.read_messages(topics=target_topics):
                self.handle_message(topic, msg, t.to_sec())
            bag.close()

        if self.video_writer:
            self.video_writer.release()
            print(f"Saved visualization to {self.output_video_path}")
        cv2.destroyAllWindows()

    def handle_message(self, topic, msg, ts):
        if topic == self.imu_topic:
            q = msg.orientation
            sinp = 2 * (q.w * q.y - q.z * q.x)
            raw_pitch = math.copysign(math.pi/2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
            
            # 使用对齐器更新 IMU Pitch
            self.last_corrected_pitch = self.imu_aligner.update(raw_pitch, self.current_gps, ts)
            
        elif topic == self.gps_topic:
            if hasattr(msg, 'latitude'):
                self.current_gps = {
                    'lat': msg.latitude,
                    'lon': msg.longitude,
                    'alt': msg.altitude,
                    'ts': ts # 存一下时间方便画图计算
                }
                
        elif topic == self.left_topic:
            self.left_queue.append(msg)
        elif topic == self.right_topic:
            self.right_queue.append(msg)
        self.sync_and_process()

    def sync_and_process(self):
        while self.left_queue and self.right_queue:
            l_msg = self.left_queue[0]
            r_msg = self.right_queue[0]
            l_ts = get_time_sec(l_msg.header)
            r_ts = get_time_sec(r_msg.header)
            
            if abs(l_ts - r_ts) < self.sync_threshold:
                self.left_queue.popleft()
                self.right_queue.popleft()
                self.process_frame(l_msg, r_msg)
            elif l_ts < r_ts:
                self.left_queue.popleft()
            else:
                self.right_queue.popleft()

    def draw_lanes_and_vp(self, img):
        vis_img = img.copy()
        h, w = img.shape[:2]
        estimator = self.fusion.lane_estimator
        
        raw_lines = estimator.model.detect(img)
        if raw_lines:
            for line in raw_lines:
                x1, y1, x2, y2 = line
                cv2.line(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        left_line, right_line = estimator.fit_lane_line(raw_lines, w)
        vp = None
        if left_line:
            cv2.line(vis_img, (int(left_line[0]), int(left_line[1])), 
                     (int(left_line[2]), int(left_line[3])), (255, 0, 0), 3)
        if right_line:
            cv2.line(vis_img, (int(right_line[0]), int(right_line[1])), 
                     (int(right_line[2]), int(right_line[3])), (0, 0, 255), 3)
            
        if left_line and right_line:
            vp = estimator.get_intersection(left_line, right_line)
            if vp:
                cv2.circle(vis_img, (int(vp[0]), int(vp[1])), 10, (0, 255, 255), -1)
                cv2.putText(vis_img, "VP", (int(vp[0])+15, int(vp[1])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return vis_img

    def calculate_vis_gps_slope(self):
        """ 用于画图的 GPS 坡度，也采用长距离策略 """
        if self.current_gps is None: return 0.0
        self.vis_gps_buffer.append(self.current_gps)
        
        MIN_DIST = 5.0
        curr = self.vis_gps_buffer[-1]
        best_prev = None
        
        for i in range(len(self.vis_gps_buffer)-2, -1, -1):
            prev = self.vis_gps_buffer[i]
            R = 6371000
            d_lat = np.radians(curr['lat'] - prev['lat'])
            d_lon = np.radians(curr['lon'] - prev['lon'])
            dist = R * math.sqrt(d_lat**2 + (math.cos(math.radians(prev['lat'])) * d_lon)**2)
            if dist > MIN_DIST:
                best_prev = (prev, dist)
                break
                
        if best_prev:
            prev, dist = best_prev
            dt = curr['ts'] - prev['ts']
            speed = dist / (dt + 1e-6)
            if speed > 3.0: # 只有速度够快才显示
                d_alt = curr['alt'] - prev['alt']
                slope = math.atan(d_alt / dist)
                if abs(slope) < math.radians(20):
                    return slope
        return 0.0

    def process_frame(self, msg_l, msg_r):
        try:
            raw_l = self.bridge.imgmsg_to_cv2(msg_l)
            raw_r = self.bridge.imgmsg_to_cv2(msg_r)
            img_l = cv2.remap(raw_l, self.map1x, self.map1y, cv2.INTER_LINEAR)
            img_r = cv2.remap(raw_r, self.map2x, self.map2y, cv2.INTER_LINEAR)

            depth, _ = self.depth_net.compute_depth(img_l, img_r, cam_matrix=self.K_new)

            s_2d = self.fusion.lane_estimator.run(img_l)
            s_3d = self.fusion.estimate_depth_slope(depth)
            
            # 这里取修正后的 pitch
            s_imu = self.last_corrected_pitch
            s_gps_vis = self.calculate_vis_gps_slope()

            fused, pred = self.fusion.fusion_slope(img_l, depth, s_imu, gps_data=self.current_gps)

            self.plotter.update(s_2d, s_3d, s_imu, s_gps_vis, fused, pred)

            # 绘图
            lane_vis = self.draw_lanes_and_vp(img_l)
            
            # 显示 Bias
            bias_deg = math.degrees(self.imu_aligner.bias)
            cv2.putText(lane_vis, f"IMU Bias: {bias_deg:.2f} deg", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.putText(lane_vis, f"2D: {math.degrees(s_2d):.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(lane_vis, f"GPS(>5m): {math.degrees(s_gps_vis):.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
            cv2.putText(lane_vis, f"IMU(Corr): {math.degrees(s_imu):.1f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_vis = cv2.applyColorMap(255 - depth_norm, cv2.COLORMAP_MAGMA)
            cv2.putText(depth_vis, f"3D: {math.degrees(s_3d):.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(depth_vis, f"Fused: {math.degrees(fused):.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(depth_vis, f"Pred: {math.degrees(pred):.1f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            graph_vis = self.plotter.draw()
            
            top_row = np.hstack([lane_vis, depth_vis])
            if graph_vis.shape[1] != top_row.shape[1]:
                graph_vis = cv2.resize(graph_vis, (top_row.shape[1], 400))
            
            final_frame = np.vstack([top_row, graph_vis])
            scale = 0.6
            final_frame = cv2.resize(final_frame, (0,0), fx=scale, fy=scale)

            cv2.imshow("Slope Prediction Analysis", final_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sys.exit(0)
            
            if self.video_writer is None:
                h, w = final_frame.shape[:2]
                self.video_writer = cv2.VideoWriter(self.output_video_path, 
                                                  cv2.VideoWriter_fourcc(*'mp4v'), 
                                                  20.0, (w, h))
            self.video_writer.write(final_frame)

        except Exception as e:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    bag_path = "/media/fwt/fangwt/data/21/record_20251121_144817_0.bag"
    
    IMG_SIZE = (1280, 720) 
    K1 = np.array([[266.25368463, 0.0, 315.6226601], [0.0, 266.94010561, 169.67651244], [0.0, 0.0, 0.5]])*2
    D1 = np.array([-0.10957469, 0.09339573, 0.0010364, -0.00279321, 0.0])
    K2 = np.array([[262.47891389, 0.0, 318.21211774], [0.0, 264.54928097, 181.65572132], [0.0, 0.0, 0.5]])*2
    D2 = np.array([-0.01527386, -0.01077888, -0.00009371, -0.00227572, 0.0])
    q_val = [-0.00049487, -0.00871143, -0.00105701, 0.99996137]
    R_mat = R_scipy.from_quat(q_val).as_matrix()
    T_val = np.array([[-0.11865043], [-0.00021856], [0.0]]) 

    raw_calib_data = {'K1': K1, 'D1': D1, 'K2': K2, 'D2': D2, 'R': R_mat, 'T': T_val, 'img_size': IMG_SIZE}

    sys_inst = SlopeAnalysisSystem(bag_path, raw_calib_data)
    sys_inst.process_bag()