
import cv2
import torch
from ultralytics import YOLO
import os
import numpy as np

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt', device=0):
        self.device = device
        print(f"[Init] Loading YOLOv8 model: {model_path} on GPU {self.device}...")
        try:
            self.model = YOLO(model_path)
            # 我们只关心这些类别
            self.classes_of_interest = ['car', 'truck', 'bus']
            
            # 类别ID映射 (COCO标准)
            self.coco_map = { 2: 'car', 5: 'bus', 7: 'truck' }
        except Exception as e:
            print(f"[Error] YOLO load failed: {e}")
            self.model = None

    def detect(self, image):
        if self.model is None: return []
        
        # 1. 临时文件大法 (避开 numpy 格式 ValueError 问题)
        temp_path = "/tmp/yolo_temp_frame.jpg"
        try:
            cv2.imwrite(temp_path, image)
            results = self.model.predict(source=temp_path, verbose=False, device=self.device)
        except Exception as e:
            print(f"[Detector Error] YOLO predict failed: {e}")
            return []

        detections = []
        for result in results:
            data_to_parse = None
            if isinstance(result, torch.Tensor):
                data_to_parse = result.cpu().numpy()
            elif hasattr(result, 'boxes'):
                 if hasattr(result.boxes, 'data'):
                     data_to_parse = result.boxes.data.cpu().numpy()
                 else:
                     continue

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
    def __init__(self, method='auto', baseline=0.54, focal_length=721.5, onnx_path='crestereo.onnx'):
        self.baseline = baseline
        self.focal_length = focal_length
        self.method = method
        self.net = None
        self.input_size = (1280, 720) # W, H
        
        # 1. 尝试加载 CREStereo ONNX (强制使用 OpenCV DNN)
        if method == 'auto' or method == 'crestereo':
            if os.path.exists(onnx_path):
                print(f"[Init] Found CREStereo ONNX: {onnx_path}")
                try:
                    self.net = cv2.dnn.readNetFromONNX(onnx_path)
                    # 尝试启用 CUDA
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    self.method = 'cv2_dnn'
                    print("[Init] CREStereo loaded via OpenCV DNN.")
                except Exception as e:
                    print(f"[Warning] OpenCV DNN load failed: {e}")
                    self.method = 'sgbm'
            else:
                print(f"[Info] {onnx_path} not found. Using SGBM.")
                self.method = 'sgbm'
        
        # 2. SGBM 初始化 (兜底)
        self.init_sgbm()

    def init_sgbm(self):
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=128, blockSize=9,
            P1=8*3*9**2, P2=32*3*9**2,
            disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=32
        )

    def compute_depth(self, img_l, img_r):
        h_orig, w_orig = img_l.shape[:2]

        # === 路径 B: OpenCV DNN ===
        if self.method == 'cv2_dnn' and self.net is not None:
            # try:
                # 预处理
            l_in = cv2.resize(img_l, self.input_size)
            r_in = cv2.resize(img_r, self.input_size)
            
            # 转换: (H, W, C) -> (1, C, H, W)
            blob_l = cv2.dnn.blobFromImage(l_in, 1.0/255.0, self.input_size, (0,0,0), swapRB=False, crop=False)
            blob_r = cv2.dnn.blobFromImage(r_in, 1.0/255.0, self.input_size, (0,0,0), swapRB=False, crop=False)
            
            # 尝试设置输入
            # 如果是 CREStereo，通常有两个输入
            # OpenCV 4.x 有时候需要明确的名字，有时候可以按顺序
            # 我们这里尝试暴力枚举常见名字
            # 如果失败，会自动捕获异常并回退 SGBM
            
            names_to_try = [ 
                ("init_left", "init_right")
            ]
            
            success = False
            for n1, n2 in names_to_try:
                # try:
                self.net.setInput(blob_l, n1)
                self.net.setInput(blob_r, n2)
                # 尝试一次极其轻量的 forward 检查是否报错
                # 这里我们直接运行，如果不报错就认为名字是对的
                output = self.net.forward()
                success = True
                # except Exception:
                #     continue
            
            if not success:
                # 如果所有名字都失败了，抛出异常回退 SGBM
                # 这是一个已知问题：OpenCV 对某些 ONNX 模型的输入层解析有问题
                raise RuntimeError("Failed to set inputs for ONNX model")

            # 解析输出
            if output.ndim == 4: disp = output[0, 0, :, :]
            elif output.ndim == 3: disp = output[0, :, :]
            else: disp = output
            
            return self.process_disparity(disp, w_orig, h_orig)

            # except Exception as e:
            #     # 这一行是关键：只要 DL 失败，立刻切回 SGBM，保证不崩
            #     # print(f"[Error] OpenCV DNN inference failed: {e}. Switching to SGBM.")
            #     self.method = 'sgbm'
            #     return self.compute_depth(img_l, img_r)

        # === 路径 C: SGBM (默认) ===
        else:
            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            disparity = self.stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
            return self.process_disparity(disparity, w_orig, h_orig, is_sgbm=True)

    def process_disparity(self, disp, w_orig, h_orig, is_sgbm=False):
        # Resize 回原图
        if disp.shape[:2] != (h_orig, w_orig):
            scale = w_orig / disp.shape[1]
            disp = cv2.resize(disp, (w_orig, h_orig)) * scale

        # 可视化图
        if is_sgbm:
            mask = disp > 0
            disp_vis = np.zeros_like(disp, dtype=np.uint8)
            if mask.any():
                norm_disp = (disp - disp[mask].min()) / (disp[mask].max() - disp[mask].min() + 1e-5) * 255
                disp_vis[mask] = norm_disp[mask].astype(np.uint8)
        else:
            disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_MAGMA)

        # 计算深度
        depth = np.zeros_like(disp)
        mask = disp > 0.1
        depth[mask] = (self.focal_length * self.baseline) / disp[mask]
        depth[depth > 80] = 0
        depth[depth < 0] = 0
        
        return depth, disp_vis