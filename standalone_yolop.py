import torch
import cv2
import numpy as np
import argparse
import os
import sys
import torchvision

class YOLOPTester:
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading YOLOP on {self.device}...")
        
        # 直接从 PyTorch Hub 加载，不需要本地代码文件
        # trust_repo=True 是新版 torch 的安全要求
        try:
            self.model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True, trust_repo=True)
            self.model.to(self.device)
            self.model.eval()
            print("YOLOP loaded successfully!")
        except Exception as e:
            print(f"Error loading YOLOP: {e}")
            sys.exit(1)
            
        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]

    def infer(self, img_path):
        # 1. 读取图像
        img0 = cv2.imread(img_path)
        if img0 is None:
            print(f"Failed to read {img_path}")
            return None
        
        h0, w0 = img0.shape[:2]
        
        # 2. 预处理
        # YOLOP 要求输入为 640x640
        img = cv2.resize(img0, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = (img - self.normalize_mean) / self.normalize_std
        img = img.transpose(2, 0, 1) # HWC -> CHW
# 加上 .float() 强制转换为 32位浮点数
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(self.device)        
        # 3. 推理
        with torch.no_grad():
            # det_out: 车辆检测 [1, n, 6]
            # da_seg_out: 可行驶区域 [1, 2, 640, 640]
            # ll_seg_out: 车道线 [1, 2, 640, 640]
            det_out, da_seg_out, ll_seg_out = self.model(img_tensor)
            
        # 4. 后处理 - 车道线 (Lane Lines)
        # 取第二个通道作为车道线置信度，或者直接 argmax
        ll_seg_mask = ll_seg_out[0].argmax(0).cpu().numpy().astype(np.uint8)
        # 恢复到原图尺寸
        ll_seg_mask = cv2.resize(ll_seg_mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
        
        # 5. 后处理 - 车辆检测 (Detections)
        # YOLOP 的 det_out 已经是 [xy, wh, conf, cls] 格式，但需要 NMS
        if det_out[0].ndim == 3: det_out = det_out[0] # handle batch dimension
        
        # 筛选置信度
        conf_thres = 0.3
        iou_thres = 0.45
        
        pred = det_out[0]
        mask = pred[:, 4] > conf_thres
        pred = pred[mask]
        
        det_boxes = []
        if len(pred) > 0:
            # xywh -> xyxy
            boxes = self.xywh2xyxy(pred[:, :4])
            scores = pred[:, 4]
            # NMS
            keep = torchvision.ops.nms(boxes, scores, iou_thres)
            boxes = boxes[keep]
            scores = scores[keep]
            
            # 缩放坐标回原图
            scale_x = w0 / 640
            scale_y = h0 / 640
            boxes[:, 0] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 2] *= scale_x
            boxes[:, 3] *= scale_y
            
            for box, score in zip(boxes.cpu().numpy(), scores.cpu().numpy()):
                det_boxes.append((box.astype(int), score))

        return img0, ll_seg_mask, det_boxes

    def xywh2xyxy(self, x):
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def visualize_and_extract_lines(self, img, mask, boxes):
        vis_img = img.copy()
        h, w = img.shape[:2]
        
        # A. 绘制车辆
        for box, score in boxes:
            cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            cv2.putText(vis_img, f"Car {score:.2f}", (box[0], box[1]-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # B. 绘制车道线 (从 Mask 提取线段 - 模拟你的坡度计算需求)
        # 1. 为了效果好，只取下半部分
        roi_mask = np.zeros_like(mask)
        roi_mask[int(h*0.5):, :] = 1
        masked_seg = mask * roi_mask
        
        # 2. 边缘检测
        edges = cv2.Canny(masked_seg * 255, 50, 150)
        
        # 3. 霍夫变换提取线段
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=50, maxLineGap=20)
        
        # 4. 绘制
        # 先把分割掩码画成半透明绿色
        color_mask = np.zeros_like(img)
        color_mask[mask == 1] = [0, 255, 0]
        vis_img = cv2.addWeighted(vis_img, 0.7, color_mask, 0.3, 0)
        
        # 再画提取出来的线段（红色），验证能否用于计算消失点
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
        cv2.putText(vis_img, f"Lanes: {len(lines) if lines is not None else 0}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return vis_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='dataset_root/images', help='Path to image or dir')
    parser.add_argument('--save', action='store_true', help='Save result to disk')
    args = parser.parse_args()

    tester = YOLOPTester()
    
    # 获取图片列表
    if os.path.isdir(args.source):
        # 递归查找图片
        files = []
        for root, _, filenames in os.walk(args.source):
            for f in filenames:
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    files.append(os.path.join(root, f))
        files.sort()
        # 限制数量，防止太多
        if len(files) > 100: files = files[:100]
    else:
        files = [args.source]

    print(f"Found {len(files)} images.")
    
    for f in files:
        print(f"Processing {os.path.basename(f)}...")
        
        # 推理
        result = tester.infer(f)
        if result is None: continue
        
        orig_img, lane_mask, det_boxes = result
        
        # 可视化
        res_img = tester.visualize_and_extract_lines(orig_img, lane_mask, det_boxes)
        
        # 显示
        cv2.imshow("YOLOP Standalone Test", res_img)
        
        if args.save:
            if not os.path.exists("yolop_results"): os.makedirs("yolop_results")
            save_path = os.path.join("yolop_results", "res_" + os.path.basename(f))
            cv2.imwrite(save_path, res_img)
        
        key = cv2.waitKey(0)
        if key == 27: # ESC
            break
            
    cv2.destroyAllWindows()