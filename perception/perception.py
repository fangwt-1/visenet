import cv2
import torch
import numpy as np
import os
import sys
from ultralytics import YOLO
from sklearn.linear_model import RANSACRegressor

# === 依赖检查 ===
try:
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
except ImportError:
    raft_large = None
    print("[Error] torchvision is missing or too old.")

try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt', device=0):
        self.device = device
        try:
            self.model = YOLO(model_path)
            self.classes_of_interest = ['car', 'truck', 'bus']
            self.coco_map = { 2: 'car', 5: 'bus', 7: 'truck' }
        except Exception as e:
            print(f"[Error] YOLO load failed: {e}")
            self.model = None

    def detect(self, image):
        if self.model is None: return []
        temp_path = "/tmp/yolo_temp_frame.jpg"
        try:
            cv2.imwrite(temp_path, image)
            results = self.model.predict(source=temp_path, verbose=False, device=self.device)
        except Exception as e:
            return []

        detections = []
        for result in results:
            data_to_parse = None
            if hasattr(result, 'boxes'):
                 if hasattr(result.boxes, 'data'):
                     data_to_parse = result.boxes.data.cpu().numpy()
            
            if data_to_parse is None: continue

            for row in data_to_parse:
                if len(row) < 6: continue
                x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                conf = float(row[4])
                cls_id = int(row[5])
                
                name = self.coco_map.get(cls_id, 'unknown')
                if name == 'unknown' and hasattr(self.model, 'names') and cls_id in self.model.names:
                    name = self.model.names[cls_id]

                if name in self.classes_of_interest and conf > 0.4:
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'class': name,
                        'conf': conf
                    })
        return detections
    
class DepthEstimator:
    def __init__(self, method='depth_anything_raft', baseline=0.54, focal_length=721.5, onnx_path=None):
        self.baseline = baseline
        self.focal_length = focal_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[Init] Depth Estimator initializing...")

        # 1. RAFT
        self.raft = None
        self.raft_transforms = None
        if raft_large is not None:
            try:
                weights = Raft_Large_Weights.DEFAULT
                self.raft = raft_large(weights=weights, progress=False).to(self.device)
                self.raft_transforms = weights.transforms()
                self.raft.eval()
                print("[Init] RAFT loaded.")
            except Exception as e:
                print(f"[Error] Failed to load RAFT: {e}")

        # 2. Transformers
        self.mono_model = None
        self.mono_processor = None
        self.seg_model = None
        self.seg_processor = None
        
        if HAS_TRANSFORMERS:
            try:
                checkpoint_depth = "LiheYoung/depth-anything-small-hf"
                self.mono_processor = AutoImageProcessor.from_pretrained(checkpoint_depth)
                self.mono_model = AutoModelForDepthEstimation.from_pretrained(checkpoint_depth).to(self.device)
                self.mono_model.eval()
                print("[Init] Depth Anything loaded.")
            except Exception as e:
                print(f"[Error] DepthAnything load failed: {e}")

            try:
                checkpoint_seg = "nvidia/segformer-b0-finetuned-ade-512-512"
                print("[Init] Loading SegFormer...")
                self.seg_processor = SegformerImageProcessor.from_pretrained(checkpoint_seg)
                self.seg_model = SegformerForSemanticSegmentation.from_pretrained(checkpoint_seg).to(self.device)
                self.seg_model.eval()
                print("[Init] SegFormer loaded.")
            except Exception as e:
                print(f"[Error] SegFormer load failed: {e}")

        # 3. RANSAC
        self.ransac = RANSACRegressor(min_samples=20, residual_threshold=2.0, random_state=42)
        self.scale_smooth = 1.0  
        self.shift_smooth = 0.0
        self.is_initialized = False
        self.alpha = 0.15 

    def get_sky_mask(self, img):
        if self.seg_model is None: return None
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            inputs = self.seg_processor(images=img_rgb, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.seg_model(**inputs)
                logits = outputs.logits 
            
            upsampled_logits = torch.nn.functional.interpolate(
                logits, size=(h, w), mode="bilinear", align_corners=False
            )
            pred_seg = upsampled_logits.argmax(dim=1)[0] 
            sky_mask = (pred_seg == 2).cpu().numpy() # Class 2 = Sky
            return sky_mask
        except Exception:
            return None

    def run_raft_stereo(self, img_l, img_r):
        if self.raft is None: return None, None
        t_l = torch.from_numpy(cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
        t_r = torch.from_numpy(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
        img1_batch = t_l.unsqueeze(0).to(self.device)
        img2_batch = t_r.unsqueeze(0).to(self.device)
        img1_batch, img2_batch = self.raft_transforms(img1_batch, img2_batch)
        with torch.no_grad():
            list_of_flows = self.raft(img1_batch, img2_batch)
            predicted_flow = list_of_flows[-1][0] 
        flow_np = predicted_flow.cpu().numpy().transpose(1, 2, 0)
        flow_x, flow_y = flow_np[..., 0], flow_np[..., 1]
        valid_y = np.abs(flow_y) < 6.0 
        valid_x = (flow_x < -0.5) & (flow_x > -400.0) 
        mask = valid_y & valid_x
        return -flow_x, mask

    def run_depth_anything(self, img):
        if self.mono_model is None: return np.zeros(img.shape[:2], dtype=np.float32)
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            inputs = self.mono_processor(images=image_rgb, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.mono_model(**inputs)
                predicted_depth = outputs.predicted_depth
            h, w = img.shape[:2]
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
            )
            return prediction.squeeze().cpu().numpy()
        except Exception:
            return np.zeros(img.shape[:2], dtype=np.float32)

    def render_point_cloud(self, depth, color_img, cam_matrix):
        """ 基于 NumPy 的快速点云渲染器 """
        scale_factor = 0.25
        small_depth = cv2.resize(depth, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
        small_img = cv2.resize(color_img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        
        h, w = small_depth.shape
        fx = cam_matrix[0, 0] * scale_factor
        fy = cam_matrix[1, 1] * scale_factor
        cx = cam_matrix[0, 2] * scale_factor
        cy = cam_matrix[1, 2] * scale_factor

        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        mask = (small_depth > 0.1) & (small_depth < 80.0)
        
        z_3d = small_depth[mask]
        x_3d = (xx[mask] - cx) * z_3d / fx
        y_3d = (yy[mask] - cy) * z_3d / fy
        colors = small_img[mask] 

        if len(z_3d) == 0:
            return np.zeros_like(color_img)

        # 变换: 俯视30度，后退10米，抬高3米
        theta = np.radians(30)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        y_rot = y_3d * cos_t - z_3d * sin_t
        z_rot = y_3d * sin_t + z_3d * cos_t
        x_rot = x_3d

        z_final = z_rot + 10.0 
        y_final = y_rot + 3.0
        x_final = x_rot

        v_fx, v_fy = 400.0, 400.0
        v_cx, v_cy = w / 2, h / 2
        
        valid_z = z_final > 1.0
        
        x_proj = (x_final[valid_z] * v_fx / z_final[valid_z]) + v_cx
        y_proj = (y_final[valid_z] * v_fy / z_final[valid_z]) + v_cy
        c_proj = colors[valid_z]
        z_vals = z_final[valid_z]

        # 绘制
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        x_idx = x_proj.astype(np.int32)
        y_idx = y_proj.astype(np.int32)
        
        in_bounds = (x_idx >= 0) & (x_idx < w) & (y_idx >= 0) & (y_idx < h)
        
        x_vis = x_idx[in_bounds]
        y_vis = y_idx[in_bounds]
        c_vis = c_proj[in_bounds]
        z_vis = z_vals[in_bounds]
        
        sort_idx = np.argsort(z_vis)[::-1]
        
        canvas[y_vis[sort_idx], x_vis[sort_idx]] = c_vis[sort_idx]
        
        canvas_large = cv2.resize(canvas, (color_img.shape[1], color_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        return canvas_large

    def compute_depth(self, img_l, img_r, cam_matrix=None):
        mono_depth_raw = self.run_depth_anything(img_l)
        stereo_disp, stereo_mask = self.run_raft_stereo(img_l, img_r)
        sky_mask = self.get_sky_mask(img_l)
        
        final_mask = stereo_mask
        if sky_mask is not None and stereo_mask is not None:
            final_mask = stereo_mask & (~sky_mask)

        safe_mono = mono_depth_raw.copy()
        safe_mono[safe_mono < 0.1] = 0.1
        # mono_inv = 1.0 / safe_mono 
        mono_inv = safe_mono
        
        valid_fit = False
        new_scale, new_shift = self.scale_smooth, self.shift_smooth

        if stereo_disp is not None and np.sum(final_mask) > 30:
            y_idx, x_idx = np.where(final_mask)
            if len(y_idx) > 2000:
                indices = np.random.choice(len(y_idx), 2000, replace=False)
                y_idx, x_idx = y_idx[indices], x_idx[indices]
            
            X = mono_inv[y_idx, x_idx].reshape(-1, 1)
            y = stereo_disp[y_idx, x_idx]
            try:
                self.ransac.fit(X, y)
                s = self.ransac.estimator_.coef_[0]
                i = self.ransac.estimator_.intercept_
                if s > 0.01: 
                    new_scale, new_shift = s, i
                    valid_fit = True
            except Exception: pass
        
        if valid_fit:
            if not self.is_initialized:
                self.scale_smooth, self.shift_smooth = new_scale, new_shift
                self.is_initialized = True
            else:
                self.scale_smooth = (1 - self.alpha) * self.scale_smooth + self.alpha * new_scale
                self.shift_smooth = (1 - self.alpha) * self.shift_smooth + self.alpha * new_shift
        
        abs_disp_map = self.scale_smooth * mono_inv + self.shift_smooth
        abs_disp_map[abs_disp_map < 0.5] = 0.5 
        final_depth = (self.focal_length * self.baseline) / abs_disp_map
        final_depth[final_depth > 150] = 0
        final_depth[final_depth < 0] = 0
        if sky_mask is not None:
            final_depth[sky_mask] = 0

        vis_pack = {}
        
        # 1. 渲染点云
        if cam_matrix is not None:
            vis_pack['cloud'] = self.render_point_cloud(final_depth, img_l, cam_matrix)
            cv2.putText(vis_pack['cloud'], "3D Point Cloud", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        else:
            vis_pack['cloud'] = np.zeros_like(img_l)

        # 2. 语义分割
        vis_pack['seg'] = img_l.copy()
        if sky_mask is not None:
            color_mask = np.zeros_like(img_l)
            color_mask[sky_mask] = [255, 0, 0] 
            vis_pack['seg'] = cv2.addWeighted(vis_pack['seg'], 0.7, color_mask, 0.3, 0)
        cv2.putText(vis_pack['seg'], "Sky Segmentation", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # 3. 最终融合深度 (Final Depth)
        fused_norm = cv2.normalize(final_depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        fused_norm = 255 - fused_norm 
        fused_vis = cv2.applyColorMap(fused_norm, cv2.COLORMAP_MAGMA) 
        if sky_mask is not None: fused_vis[sky_mask] = 0
        cv2.putText(fused_vis, f"Final Depth (Fused)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        vis_pack['fused'] = fused_vis

        # 4. 单目深度 (Mono) [Added]
        mono_norm = cv2.normalize(mono_depth_raw, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        mono_vis = cv2.applyColorMap(mono_norm, cv2.COLORMAP_MAGMA)
        cv2.putText(mono_vis, "Monocular Depth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        vis_pack['mono'] = mono_vis

        # 5. 双目视差 (Stereo) [Added]
        if stereo_disp is not None:
            stereo_norm = cv2.normalize(stereo_disp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            stereo_vis = cv2.applyColorMap(stereo_norm, cv2.COLORMAP_JET)
        else:
            stereo_vis = np.zeros_like(img_l)
        cv2.putText(stereo_vis, "Stereo Disparity", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        vis_pack['stereo'] = stereo_vis

        return final_depth, vis_pack