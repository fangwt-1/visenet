import torch
import torch.nn as nn
import torch.nn.functional as F

class LaneSlopeNet(nn.Module):
    def __init__(self, input_size=(256, 256), history_steps=7, future_steps=6, dropout_p=0.2):
        """
        Args:
            history_steps: 历史时间步数量 (包含当前时刻)
            future_steps: 预测未来时间步数量
            dropout_p: Dropout 概率 (默认 0.5)
        """
        super(LaneSlopeNet, self).__init__()
        
        self.future_steps = future_steps
        
        # === 1. 视觉分支 (CNN) ===
        # 增加 Dropout2d 防止卷积层过拟合
        # Dropout2d 会随机将整个通道置零，有助于增强空间特征的独立性
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout2d(p=0.1),  # 浅层网络使用较小的 dropout
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(p=0.1)
        )
        self.visual_dim = 64 * 4 * 4
        
        # === 2. 时序分支 (处理历史速度和坡度) ===
        # 输入维度: 2 (Speed + Slope)
        self.rnn_hidden_dim = 64
        self.rnn_num_layers = 3
        # 在 GRU 内部添加 dropout (仅在 num_layers > 1 时有效，作用于层与层之间)
        self.ts_encoder = nn.GRU(
            input_size=2, 
            hidden_size=self.rnn_hidden_dim, 
            num_layers=self.rnn_num_layers, 
            batch_first=True,
            dropout=dropout_p if self.rnn_num_layers > 1 else 0  # 防止单层报错，虽然这里写死是3层
        )
        
        # === 3. 融合与预测 ===
        # 输入: 视觉特征 + 时序特征
        concat_dim = self.visual_dim + self.rnn_hidden_dim
        
        self.head = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p), # 主要的全连接层 Dropout
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p), # 增加第二个 Dropout 进一步正则化
            
            # 输出未来 steps 的增量
            nn.Linear(128, future_steps)
        )
        
        # 初始化输出层为0，利用残差连接思想，初始预测等于当前坡度
        nn.init.constant_(self.head[-1].bias, 0.0)
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=0.001)

    def forward(self, mask, history_data, current_slope_val):
        """
        mask: [B, 1, H, W]
        history_data: [B, T_hist, 2]  (包含标准化的 speed 和 slope)
        current_slope_val: [B, 1] (非标准化或标准化的当前坡度值，用于残差连接)
        """
        # 1. 视觉特征
        x_vis = self.features(mask)
        x_vis = x_vis.view(-1, self.visual_dim)
        # 2. 时序特征
        # history_data: [Batch, Sequence, Feature]
        # output: [Batch, Seq, Hidden], h_n: [num_layers, Batch, Hidden]
        self.ts_encoder.flatten_parameters() # 使得在多 GPU 或特定环境下更紧凑，有时有助于内存
        _, h_n = self.ts_encoder(history_data)
        x_ts = h_n[-1] # 取最后一个 Stacked Layer 的最后一个时间步隐状态 [Batch, Hidden]
        # 3. 融合
        combined = torch.cat([x_vis, x_ts], dim=1)
        # 4. 预测增量 (Delta)
        # 预测未来每个时刻相对于"当前时刻"的坡度变化量
        deltas = self.head(combined) # [B, future_steps]
        # 5. 残差连接输出
        # 最终预测 = 当前坡度 + 预测的变化量
        out = current_slope_val + deltas
        
        return out

class SlopeTrendLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1, gamma=0.7):
        super(SlopeTrendLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha  # 基础数值权重
        self.beta = beta    # 趋势方向权重
        self.gamma = gamma  # 变化率（梯度）权重

    def forward(self, pred, target):
        # 1. 基础 MSE 损失
        mse_loss = self.mse(pred, target)

        # 2. 趋势一致性损失 (余弦相似度)
        # 关注预测曲线和真值曲线在方向上的重合度
        cos_sim = F.cosine_similarity(pred, target, dim=1)
        cos_loss = torch.mean(1 - cos_sim)

        # 3. 梯度（变化率）损失
        # 计算相邻帧之间的差值，确保预测的“坡度变化趋势”与真值一致
        pred_grad = pred[:, 1:] - pred[:, :-1]
        target_grad = target[:, 1:] - target[:, :-1]
        grad_loss = self.mse(pred_grad, target_grad)
        return self.alpha * mse_loss + self.beta * cos_loss + self.gamma * grad_loss
    
# === 使用示例与 L2 正则化建议 ===
if __name__ == "__main__":
    # 实例化模型
    model = LaneSlopeNet(dropout_p=0.5)
    
    # 创建假数据测试运行
    B, T_hist, T_future = 8, 7, 6
    mask = torch.randn(B, 1, 256, 256)
    history = torch.randn(B, T_hist, 2)
    current_slope = torch.randn(B, 1)
    
    # 训练模式 (Dropout 开启)
    model.train()
    output = model(mask, history, current_slope)
    print("Train Output shape:", output.shape)
    
    # 评估模式 (Dropout 关闭，非常重要！)
    model.eval()
    with torch.no_grad():
        output_eval = model(mask, history, current_slope)
    print("Eval Output shape:", output_eval.shape)

    # === 关键建议：在优化器中加入 L2 正则化 (Weight Decay) ===
    # 这也是防止过拟合最重要的手段之一
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=1e-4, 
        weight_decay=1e-4 # <--- 这里的 weight_decay 就是 L2 正则化系数
    )
    print("Optimizer initialized with weight decay (L2 Regularization)")