import numpy as np
from sklearn.linear_model import RANSACRegressor

class LaneSlopeEstimator:
    def __init__(self, args=None, weights_path=None):
        try:
            self.args = define_args()
            self.args.resize_h = 360
            self.args.resize_w = 480
        except:
            class Args: pass
            self.args = Args()
            self.args.resize_h = 360
            self.args.resize_w = 480
        
        self.y_ref = 20.0 

    def run(self, img, last_imu_pitch):
        return last_imu_pitch if last_imu_pitch else 0.0

class EnvironmentFusion:
    def __init__(self, cam_matrix):
        self.K = cam_matrix
        self.cx = cam_matrix[0, 2]
        self.cy = cam_matrix[1, 2]
        self.fx = cam_matrix[0, 0]
        self.fy = cam_matrix[1, 1]
        self.lane_estimator = LaneSlopeEstimator()
        self.last_valid_dist = None

    def get_front_vehicle_dist(self, detections, depth_map):
        min_dist = 100.0 
        found_valid_obj = False
        
        for obj in detections:
            x1, y1, x2, y2 = obj['bbox']
            h, w = depth_map.shape
            
            w_box = x2 - x1
            h_box = y2 - y1
            crop_factor = 0.3 
            
            roi_x1 = int(x1 + w_box * crop_factor)
            roi_x2 = int(x2 - w_box * crop_factor)
            roi_y1 = int(y1 + h_box * crop_factor)
            roi_y2 = int(y2 - h_box * crop_factor)
            
            roi_x1 = max(0, roi_x1)
            roi_y1 = max(0, roi_y1)
            roi_x2 = min(w, roi_x2)
            roi_y2 = min(h, roi_y2)
            
            if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                roi = depth_map[roi_y1:roi_y2, roi_x1:roi_x2]
                valid_roi = roi[(roi > 3.0) & (roi < 20.0)]
                
                if len(valid_roi) > 5:
                    dist = np.percentile(valid_roi, 20) 
                    obj['dist'] = dist
                    if dist < min_dist: 
                        min_dist = dist
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

    def fusion_slope(self, img, depth_map, imu_pitch=None):
        s_depth = self.estimate_depth_slope(depth_map)
        s_lane = self.lane_estimator.run(img, imu_pitch)
        final = 0.5 * s_depth + 0.5 * s_lane
        if imu_pitch is not None:
            final = 0.4 * final + 0.6 * imu_pitch
        return final
