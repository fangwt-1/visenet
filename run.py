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
        
        self.last_imu_pitch = 0.0
        self.frame_count = 0
        self.left_queue = collections.deque()
        self.right_queue = collections.deque()
        self.sync_threshold = 0.1

        self.left_topic = '/zed_node/rgb/left_image'
        self.right_topic = '/zed_node/rgb/right_image'
        self.imu_topic = '/imu/data'
        
        # === 视频录制 ===
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

        if not HAS_ROS:
            try:
                with Reader(self.bag_path) as reader:
                    print("[Info] Scanning topics...")
                    all_topics = list(reader.topics.keys())
                    for t in all_topics:
                        if 'left' in t and 'image' in t and 'rect' not in t: self.left_topic = t
                        if 'right' in t and 'image' in t and 'rect' not in t: self.right_topic = t
                        if 'imu' in t: self.imu_topic = t
                    
                    target_topics = [self.left_topic, self.right_topic, self.imu_topic]
                    print(f"[Info] Running processing loop...")

                    for connection, timestamp, rawdata in reader.messages():
                        if connection.topic not in target_topics: continue
                        msg = self.typestore.deserialize_ros1(rawdata, connection.msgtype)
                        self.handle_message(connection.topic, msg)
            except Exception as e:
                import traceback
                traceback.print_exc()
        else:
            pass

        if self.video_writer is not None:
            self.video_writer.release()
            print(f"\n[Info] Video saved to {self.output_video_path}")
        cv2.destroyAllWindows()

    def handle_message(self, topic, msg):
        if topic == self.imu_topic:
            q = msg.orientation
            sinp = 2 * (q.w * q.y - q.z * q.x)
            self.last_imu_pitch = math.copysign(math.pi/2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
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
            slope = self.fusion.fusion_slope(img_l, depth, self.last_imu_pitch)
            
            self.visualize(img_l, vis_pack, dets, slope, dist)
            
        except Exception as e:
            print(f"[Process Error] {e}")
            import traceback
            traceback.print_exc()

    def visualize(self, img, vis_pack, dets, slope, dist):
        # 1. 主图（左上角）：原图 + 检测框 + 数据
        main_vis = img.copy()
        cv2.putText(main_vis, f"Slope: {math.degrees(slope):.1f} deg", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(main_vis, f"Front Dist: {dist:.1f} m", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        for d in dets:
            x1, y1, x2, y2 = d['bbox']
            cv2.rectangle(main_vis, (x1, y1), (x2, y2), (0,255,0), 2)
            val = d.get('dist', -1)
            if val > 0:
                cv2.putText(main_vis, f"{val:.1f}m", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(main_vis, "Original + Detection", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # 2. 获取其他图层
        seg_vis = vis_pack.get('seg', img)    
        mono_vis = vis_pack.get('mono', img)
        stereo_vis = vis_pack.get('stereo', img)
        fused_vis = vis_pack.get('fused', img) 
        cloud_vis = vis_pack.get('cloud', img)

        # 3. 拼接 (2行3列)
        # 第一行: 原图 | 语义分割 | 单目深度
        top_row = np.hstack([main_vis, seg_vis, mono_vis])
        # 第二行: 双目视差 | 融合深度 | 3D点云
        bot_row = np.hstack([stereo_vis, fused_vis, cloud_vis])
        
        # 最终大图
        grid_frame = np.vstack([top_row, bot_row])
        
        # 4. 缩放 (太大了，缩小 0.5 倍)
        scale = 0.5
        grid_frame = cv2.resize(grid_frame, (0,0), fx=scale, fy=scale)
        h_grid, w_grid = grid_frame.shape[:2]

        # 5. 视频保存
        if self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.output_video_path, fourcc, 20.0, (w_grid, h_grid))
        
        self.video_writer.write(grid_frame)

        # 6. 显示
        cv2.imshow("Multi-View System", grid_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if self.video_writer: self.video_writer.release()
            sys.exit(0)

if __name__ == "__main__":
    bag_path = "/media/fwt/fangwt/data/21/record_20251121_181236_1.bag"
    
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