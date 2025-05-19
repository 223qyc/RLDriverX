import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class RadarEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(RadarEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.network(x)

class VisualEncoder(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=64):
        super(VisualEncoder, self).__init__()
        # 增大步长和核大小，减小特征图尺寸
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=16, stride=8)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=8, stride=4)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        # 计算256x256输入后到达fc层时的特征尺寸
        # 256x256 -> 31x31 (conv1) -> 6x6 (conv2) -> 2x2 (conv3)
        self.fc = nn.Linear(64 * 2 * 2, hidden_dim)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

class MultiModalNetwork(nn.Module):
    def __init__(self, radar_dim, visual_shape, action_dim, hidden_dim=64):
        super(MultiModalNetwork, self).__init__()
        
        # 雷达编码器
        self.radar_encoder = RadarEncoder(radar_dim, hidden_dim)
        
        # 视觉编码器
        self.visual_encoder = VisualEncoder(visual_shape[0], hidden_dim)
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # 策略头（Actor）
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出范围限制在[-1, 1]
        )
        
        # 价值头（Critic）
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, radar_input, visual_input):
        # 编码雷达数据
        radar_features = self.radar_encoder(radar_input)
        
        # 转换视觉输入维度 (H, W, C) -> (C, H, W)
        if visual_input.shape[-1] == 3:  # 如果通道在最后
            visual_input = visual_input.permute(0, 3, 1, 2)
        
        # 编码视觉数据
        visual_features = self.visual_encoder(visual_input)
        
        # 特征融合
        combined_features = torch.cat([radar_features, visual_features], dim=1)
        fused_features = self.fusion(combined_features)
        
        # 输出动作和价值
        actions = self.actor(fused_features)
        value = self.critic(fused_features)
        
        return actions, value
    
    def get_action(self, radar_input, visual_input):
        with torch.no_grad():
            actions, _ = self.forward(radar_input, visual_input)
        return actions
    
    def get_value(self, radar_input, visual_input):
        with torch.no_grad():
            _, value = self.forward(radar_input, visual_input)
        return value

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer) 