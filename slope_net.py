import torch
import torch.nn as nn

class LaneSlopeNet(nn.Module):
    def __init__(self, input_size=(256, 256)):
        super(LaneSlopeNet, self).__init__()
        
        # 1. CNN 特征提取 (保持不变)
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
        
        # 2. 全连接层预测 "坡度变化量 (Delta)"
        # 输入: 视觉特征 + 速度 + 当前坡度
        # 我们依然把 current_slope 放进输入，因为"当前是陡坡"还是"当前是平路"，
        # 对"接下来会怎么变"是有影响的（物理约束）。
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim + 2, 128), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # 输出 delta_slope
        )
        
        # 初始化权重 (可选优化)
        # 将最后一层初始化为接近0，这样模型刚开始训练时，
        # output ≈ current_slope + 0，相当于直接复用当前值，Loss起点会很低。
        nn.init.normal_(self.fc[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc[-1].bias, 0.0)

    def forward(self, mask, speed, current_slope):
        """
        mask: [B, 1, H, W]
        speed: [B, 1]
        current_slope: [B, 1]
        """
        # CNN 特征
        x = self.features(mask)
        x = x.view(-1, self.flatten_dim)
        
        # 拼接: [Visual_Feat, Speed, Current_Slope]
        combined = torch.cat([x, speed, current_slope], dim=1)
        
        # 预测变化量 (Delta)
        # 这里的 delta 是在"标准化空间"下的变化量
        delta = self.fc(combined)
        
        # === 残差连接 ===
        # 最终预测 = 当前坡度 + 预测的变化量
        out = current_slope + delta
        
        return out