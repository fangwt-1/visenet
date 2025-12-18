from perception.perception import ObjectDetector, DepthEstimator
from driver.driver import DriverProfiler, Predictor
from environment.environment import EnvironmentFusion
import sys
import cv2
import numpy as np
import math
import collections
from scipy.spatial.transform import Rotation as R_scipy
import os

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
        sys.exit(1)

# === 简单的 CvBridge 替代品 ===
class SimpleCvBridge:
    def imgmsg_to_cv2(self, msg):
        import numpy as np
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

# === 新增：动态 IMU 对齐器 ===
class DynamicImuAligner:
    def __init__(self):
        self.bias = 0.0
        self.is_initialized = False
        self.gps_buffer = collections.deque(maxlen=50) # 存储用于差分的数据
        self.last_update_time = 0
        
    def update(self, raw_imu_pitch, current_gps, timestamp):
        """
        利用 GPS 数据动态校准 IMU Bias。
        返回: 校准后的 IMU Pitch
        """
        if current_gps is None:
            # 如果没有 GPS，只能减去当前的已知 Bias
            return raw_imu_pitch - self.bias
            
        self.gps_buffer.append((timestamp, current_gps))
        
        # 1. 尝试计算可靠的 GPS 坡度
        gps_slope = None
        
        # 向前回溯找一个距离 > 5m 的点
        MIN_DIST = 5.0
        curr_g = self.gps_buffer[-1][1]
        best_prev = None
        
        for i in range(len(self.gps_buffer)-2, -1, -1):
            t, prev_g = self.gps_buffer[i]
            # 简单距离计算
            R = 6371000
            d_lat = np.radians(curr_g['lat'] - prev_g['lat'])
            d_lon = np.radians(curr_g['lon'] - prev_g['lon'])
            # 平面近似即可，短距离误差小
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
            # 当前的瞬时偏差
            instant_bias = raw_imu_pitch - gps_slope
            
            if not self.is_initialized:
                # 初始化：直接接受第一个有效值
                self.bias = instant_bias
                self.is_initialized = True
                print(f"[IMU Align] Initialized Bias to {math.degrees(self.bias):.2f} deg")
            else:
                # 运行时：缓慢更新 (互补滤波)
                # alpha 越小，Bias 变化越慢，越抗噪；alpha 越大，收敛越快
                # 刚开始可能漂移大，我们用一个自适应的 alpha? 
                # 这里使用固定的小系数，保证平滑
                alpha = 0.005 
                self.bias = (1 - alpha) * self.bias + alpha * instant_bias
                
        # 3. 返回校正值
        return raw_imu_pitch - self.bias

class VehiclePredictionSystem:
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
            
        print(f"[Init] Rectified Params -> fx: {new_fx:.2f}, baseline: {new_baseline:.4f}m")

        self.depth_net = DepthEstimator(method='depth_anything_raft', baseline=new_baseline, focal_length=new_fx)
        self.detector = ObjectDetector(device=0)
        
        self.K_new = P1[:3, :3]
        self.fusion = EnvironmentFusion(cam_matrix=self.K_new)
        self.profiler = DriverProfiler()
        self.predictor = Predictor()
        
        # === 修改：使用 DynamicImuAligner 替代简单的变量 ===
        self.imu_aligner = DynamicImuAligner()
        self.last_corrected_pitch = 0.0
        
        self.current_gps = None 
        
        self.frame_count = 0
        self.left_queue = collections.deque()
        self.right_queue = collections.deque()
        self.sync_threshold = 0.1

        self.left_topic = '/zed_node/rgb/left_image'
        self.right_topic = '/zed_node/rgb/right_image'
        self.imu_topic = '/imu/data'
        self.gps_topic = '/fix' 
        
        self.video_writer = None
        self.output_video_path = "output_result.mp4"

    def init_rectification(self, calib):
        K1, D1 = calib['K1'], calib['D1']
        K2, D2 = calib['K2'], calib['D2']
        R, T = calib['R'], calib['T']
        img_size = calib['img_size']

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, D1, K2, D2, img_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1 
        )
        m1x, m1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_16SC2)
        m2x, m2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_16SC2)
        return m1x, m1y, m2x, m2y, P1, P2
        
    def process_bag(self):
        print(f"Opening {self.bag_path}...")
        if not os.path.exists(self.bag_path):
            print(f"[Error] Bag file not found.")
            return

        target_topics = [self.left_topic, self.right_topic, self.imu_topic, self.gps_topic]

        if not HAS_ROS:
            try:
                with Reader(self.bag_path) as reader:
                    print("[Info] Scanning topics in bag...")
                    for connection, timestamp, rawdata in reader.messages():
                        if connection.topic not in target_topics: continue
                        msg = self.typestore.deserialize_ros1(rawdata, connection.msgtype)
                        # 传入时间戳
                        ts = timestamp * 1e-9
                        self.handle_message(connection.topic, msg, ts)
            except Exception as e:
                import traceback
                traceback.print_exc()
        else:
            import rosbag
            bag = rosbag.Bag(self.bag_path)
            for topic, msg, t in bag.read_messages(topics=target_topics):
                self.handle_message(topic, msg, t.to_sec())
            bag.close()

        if self.video_writer is not None:
            self.video_writer.release()
            print(f"\n[Info] Video saved to {self.output_video_path}")
        cv2.destroyAllWindows()

    def handle_message(self, topic, msg, ts):
        if topic == self.imu_topic:
            q = msg.orientation
            sinp = 2 * (q.w * q.y - q.z * q.x)
            raw_pitch = math.copysign(math.pi/2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
            
            # === 使用动态对齐器更新 Pitch ===
            # 注意：这里需要传入 GPS 和 时间戳
            self.last_corrected_pitch = self.imu_aligner.update(raw_pitch, self.current_gps, ts)
            
        elif topic == self.gps_topic:
            if hasattr(msg, 'latitude'):
                self.current_gps = {
                    'lat': msg.latitude,
                    'lon': msg.longitude,
                    'alt': msg.altitude
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
                self.try_process(l_msg, r_msg)
            elif l_ts < r_ts:
                self.left_queue.popleft()
            else:
                self.right_queue.popleft()

    def try_process(self, msg_l, msg_r):
        self.frame_count += 1
        try:
            raw_l = self.bridge.imgmsg_to_cv2(msg_l)
            raw_r = self.bridge.imgmsg_to_cv2(msg_r)
            img_l = cv2.remap(raw_l, self.map1x, self.map1y, cv2.INTER_LINEAR)
            img_r = cv2.remap(raw_r, self.map2x, self.map2y, cv2.INTER_LINEAR)
            
            dets = self.detector.detect(img_l)
            
            depth, vis_pack = self.depth_net.compute_depth(img_l, img_r, cam_matrix=self.K_new)
            
            dist = self.fusion.get_front_vehicle_dist(dets, depth)
            
            # 使用修正后的 pitch
            slope, pred_slope_5s = self.fusion.fusion_slope(
                img_l, depth, self.last_corrected_pitch, gps_data=self.current_gps
            )
            
            self.visualize(img_l, vis_pack, dets, slope, pred_slope_5s, dist)
            
        except Exception as e:
            print(f"[Process Error] {e}")
            import traceback
            traceback.print_exc()

    def visualize(self, img, vis_pack, dets, slope, pred_slope, dist):
        main_vis = img.copy()
        cv2.putText(main_vis, f"Slope: {math.degrees(slope):.1f} deg", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(main_vis, f"Pred(5s): {math.degrees(pred_slope):.1f} deg", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2)
        cv2.putText(main_vis, f"Front Dist: {dist:.1f} m", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        
        # 显示当前的 Bias 值，方便调试
        current_bias_deg = math.degrees(self.imu_aligner.bias)
        cv2.putText(main_vis, f"IMU Bias: {current_bias_deg:.1f} deg", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        
        for d in dets:
            x1, y1, x2, y2 = d['bbox']
            cv2.rectangle(main_vis, (x1, y1), (x2, y2), (0,255,0), 2)
            val = d.get('dist', -1)
            if val > 0:
                cv2.putText(main_vis, f"{val:.1f}m", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        seg_vis = vis_pack.get('seg', img)    
        mono_vis = vis_pack.get('mono', img)
        stereo_vis = vis_pack.get('stereo', img)
        fused_vis = vis_pack.get('fused', img) 
        cloud_vis = vis_pack.get('cloud', img)

        top_row = np.hstack([main_vis, seg_vis, mono_vis])
        bot_row = np.hstack([stereo_vis, fused_vis, cloud_vis])
        grid_frame = np.vstack([top_row, bot_row])
        scale = 0.5
        grid_frame = cv2.resize(grid_frame, (0,0), fx=scale, fy=scale)
        h_grid, w_grid = grid_frame.shape[:2]

        if self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.output_video_path, fourcc, 20.0, (w_grid, h_grid))
        
        self.video_writer.write(grid_frame)
        cv2.imshow("Multi-View System", grid_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if self.video_writer: self.video_writer.release()
            sys.exit(0)

if __name__ == "__main__":
    bag_path = "/media/fwt/fangwt/data/21/record_20251121_183150_1.bag"
    
    IMG_SIZE = (1280, 720) 
    K1 = np.array([[266.25368463, 0.0, 315.6226601], [0.0, 266.94010561, 169.67651244], [0.0, 0.0, 0.5]])*2
    D1 = np.array([-0.10957469, 0.09339573, 0.0010364, -0.00279321, 0.0])
    K2 = np.array([[262.47891389, 0.0, 318.21211774], [0.0, 264.54928097, 181.65572132], [0.0, 0.0, 0.5]])*2
    D2 = np.array([-0.01527386, -0.01077888, -0.00009371, -0.00227572, 0.0])
    q_val = [-0.00049487, -0.00871143, -0.00105701, 0.99996137]
    R_mat = R_scipy.from_quat(q_val).as_matrix()
    T_val = np.array([[-0.11865043], [-0.00021856], [0.0]]) 

    raw_calib_data = {'K1': K1, 'D1': D1, 'K2': K2, 'D2': D2, 'R': R_mat, 'T': T_val, 'img_size': IMG_SIZE}

    sys_inst = VehiclePredictionSystem(bag_path, raw_calib_data)
    sys_inst.process_bag()