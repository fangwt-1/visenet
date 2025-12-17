from perception.perception import ObjectDetector, DepthEstimator
from driver.driver import DriverProfiler, Predictor
from environment.environment import EnvironmentFusion
import sys
import cv2
import numpy as np
import torch.nn.functional as F
import math
import collections
from scipy.spatial.transform import Rotation as R_scipy

try:
    import rosbag
    from cv_bridge import CvBridge
    import rospy
    from sensor_msgs.msg import Image as RosImage
    from sensor_msgs.msg import Imu
    from geometry_msgs.msg import TwistStamped
except ImportError:
    print("[Warning] ROS libraries not found. Run in a ROS environment for .bag files.")
    

class VehiclePredictionSystem:
    def __init__(self, bag_path, raw_calib):
        self.bag_path = bag_path
        self.bridge = CvBridge()
        
        # 1. 初始化极线校正 (新增步骤)
        # 这会计算出校正后的投影矩阵 P1，我们需要用 P1 的 fx 和 baseline 来更新后续算法
        self.map1x, self.map1y, self.map2x, self.map2y, P1, P2 = self.init_rectification(raw_calib)
        
        # 从 P1 获取焦距 fx (P1 和 P2 的 fx 是一样的)
        new_fx = P1[0, 0]
        
        # ✅✅✅ 关键修改：从 P2 计算基线
        # P2[0, 3] = fx * Tx (单位通常是 像素 * 米)
        # 所以 Baseline = abs(P2[0, 3] / fx)
        if new_fx != 0:
            new_baseline = abs(P2[0, 3] / P2[0, 0])
        else:
            new_baseline = 0.12 # 防止除零的默认值
            
        print(f"[Init] Rectified Params -> fx: {new_fx:.2f}, baseline: {new_baseline:.4f}m")

        # 2. 初始化深度网络 (使用新的 fx 和 baseline)
        self.depth_net = DepthEstimator(method='auto', baseline=new_baseline, 
                                        focal_length=new_fx, onnx_path='/home/fwt/visenet/crestereo.onnx')
        
        self.detector = ObjectDetector(device=0)
        
        # 3. 初始化融合层 (注意：这里要传校正后的内参矩阵 K_new)
        # 构造一个新的 3x3 K 矩阵传给 EnvironmentFusion
        K_new = P1[:3, :3]
        self.fusion = EnvironmentFusion(cam_matrix=K_new)
        
        self.profiler = DriverProfiler()
        self.predictor = Predictor()
        
        self.last_imu_pitch = 0.0
        self.frame_count = 0
        self.left_queue = collections.deque()
        self.right_queue = collections.deque()
        self.sync_threshold = 0.05

    def init_rectification(self, calib):
        """ 根据图片中的参数计算校正映射表 """
        # 提取参数
        K1, D1 = calib['K1'], calib['D1']
        K2, D2 = calib['K2'], calib['D2']
        R, T = calib['R'], calib['T']
        img_size = calib['img_size'] # (W, H)

        # 计算校正旋转矩阵
        # alpha=0 表示自动裁剪黑边
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, D1, K2, D2, img_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, 
            alpha=-1 
        )

        m1x, m1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_16SC2)
        m2x, m2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_16SC2)
        
        # ✅ 修改这里：把 P2 也返回出去
        return m1x, m1y, m2x, m2y, P1, P2
        
    def process_bag(self):
        print(f"Opening {self.bag_path}...")
        try:
            bag = rosbag.Bag(self.bag_path)
        except Exception as e:
            print(f"Error: {e}")
            return
        
        left_topic = '/zed_node/rgb/left_image'
        right_topic = '/zed_node/rgb/right_image'
        imu_topic = '/imu/data'
        
        print("Start loop...")
        for topic, msg, t in bag.read_messages(topics=[left_topic, right_topic, imu_topic]):
            if topic == imu_topic:
                q = msg.orientation
                sinp = 2 * (q.w * q.y - q.z * q.x)
                self.last_imu_pitch = math.copysign(math.pi/2, sinp) if abs(sinp)>=1 else math.asin(sinp)
            elif topic == left_topic:
                self.left_queue.append(msg)
            elif topic == right_topic:
                self.right_queue.append(msg)
            self.sync_and_process()
        bag.close()
        cv2.destroyAllWindows()

    def sync_and_process(self):
        while self.left_queue and self.right_queue:
            l_msg = self.left_queue[0]
            r_msg = self.right_queue[0]
            l_ts = l_msg.header.stamp.to_sec()
            r_ts = r_msg.header.stamp.to_sec()
            
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
        # if self.frame_count % 2 != 0: return # 如果卡顿可以取消注释跳帧

        try:
            # 1. 读取原始图像
            raw_l = cv2.cvtColor(self.bridge.imgmsg_to_cv2(msg_l), cv2.COLOR_RGB2BGR)
            raw_r = cv2.cvtColor(self.bridge.imgmsg_to_cv2(msg_r), cv2.COLOR_RGB2BGR)
            
            # 2. 【关键】执行极线校正 (Remap)
            # 如果原始图像尺寸很大，建议先 resize 到 640x360 再传进来，或者确保标定参数匹配原图
            # 这里假设 raw_l 的尺寸和标定时的 img_size 一致
            img_l = cv2.remap(raw_l, self.map1x, self.map1y, cv2.INTER_LINEAR)
            img_r = cv2.remap(raw_r, self.map2x, self.map2y, cv2.INTER_LINEAR)
            
            # 3. 后续处理 (使用校正后的图像)
            dets = self.detector.detect(img_l)
            depth, disp_vis = self.depth_net.compute_depth(img_l, img_r)
            dist = self.fusion.get_front_vehicle_dist(dets, depth)
            slope = self.fusion.fusion_slope(img_l, depth, self.last_imu_pitch)
            
            self.visualize(img_l, disp_vis, dets, slope, dist)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            pass

    def visualize(self, img, disp_vis, dets, slope, dist):
        vis = img.copy()
        cv2.putText(vis, f"Slope: {math.degrees(slope):.1f} deg", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(vis, f"Front Dist: {dist:.1f} m", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        
        for d in dets:
            x1, y1, x2, y2 = d['bbox']
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
            val = d.get('dist', -1)
            if val > 0:
                cv2.putText(vis, f"{val:.1f}m", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        if disp_vis is not None:
            disp_vis = cv2.resize(disp_vis, (vis.shape[1], vis.shape[0]))
            final = np.vstack([vis, disp_vis])
        else:
            final = vis
            
        cv2.imshow("System", final)
        cv2.waitKey(1)

if __name__ == "__main__":
    bag_path = "/media/fwt/fangwt/data/21/record_20251121_144817_0.bag"
    

    IMG_SIZE = (1280, 720) 

    # 左相机
    K1 = np.array([
        [266.25368463, 0.0,          315.6226601],
        [0.0,          266.94010561, 169.67651244],
        [0.0,          0.0,          0.5]
    ])*2
    D1 = np.array([-0.10957469, 0.09339573, 0.0010364, -0.00279321, 0.0])

    # 右相机
    K2 = np.array([
        [262.47891389, 0.0,          318.21211774],
        [0.0,          264.54928097, 181.65572132],
        [0.0,          0.0,          0.5]
    ])*2
    D2 = np.array([-0.01527386, -0.01077888, -0.00009371, -0.00227572, 0.0])

    # 外参：四元数转旋转矩阵
    # q = [x, y, z, w]
    q_val = [-0.00049487, -0.00871143, -0.00105701, 0.99996137]
    R_mat = R_scipy.from_quat(q_val).as_matrix()

    # 外参：平移向量
    # 【修正】: 图片中 Z=-0.118 显然是笔误，已修正为 0.0
    T_val = np.array([[-0.11865043], 
                      [-0.00021856], 
                      [0.0]]) 

    # 打包参数
    raw_calib_data = {
        'K1': K1, 'D1': D1,
        'K2': K2, 'D2': D2,
        'R': R_mat,
        'T': T_val,
        'img_size': IMG_SIZE
    }

    # 启动系统
    # 注意：我们不再传简单的 calib 字典，而是传包含所有 Raw 数据的字典
    sys = VehiclePredictionSystem(bag_path, raw_calib_data)
    sys.process_bag()