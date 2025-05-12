import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .config import BaseConfig

class MultiModalDQN(nn.Module):
    """
    多模态DQN网络，融合视觉、Lidar和向量输入
    """
    def __init__(self, vector_dim: int, lidar_dim: int, action_dim: int):
        """
        初始化多模态DQN网络
        :param vector_dim: 向量状态的维度
        :param lidar_dim: Lidar扫描数据的维度
        :param action_dim: 动作空间维度
        """
        super(MultiModalDQN, self).__init__()

        # 视觉分支 (CNN)
        # 输入大小: (N, C, H, W) = (N, BaseConfig.VISUAL_INPUT_CHANNELS, BaseConfig.VISUAL_RESIZE_DIM[0], BaseConfig.VISUAL_RESIZE_DIM[1])
        # 这里参考了Nature DQN的CNN结构，但做了轻量化处理
        self.conv1 = nn.Conv2d(BaseConfig.VISUAL_INPUT_CHANNELS, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        # 计算卷积层输出的大小
        def conv_output_size(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        # 计算CNN输出的特征维度
        conv_h = conv_output_size(BaseConfig.VISUAL_RESIZE_DIM[0], 8, 4)
        conv_h = conv_output_size(conv_h, 4, 2)
        conv_h = conv_output_size(conv_h, 3, 1)
        
        conv_w = conv_output_size(BaseConfig.VISUAL_RESIZE_DIM[1], 8, 4)
        conv_w = conv_output_size(conv_w, 4, 2)
        conv_w = conv_output_size(conv_w, 3, 1)
        
        self.visual_feature_dim = conv_w * conv_h * 32
        self.visual_fc = nn.Linear(self.visual_feature_dim, 128)

        # Lidar分支 (MLP)
        self.lidar_fc1 = nn.Linear(lidar_dim, 128)
        self.lidar_fc2 = nn.Linear(128, 64)

        # 向量分支 (MLP)
        self.vector_fc1 = nn.Linear(vector_dim, 128)
        self.vector_fc2 = nn.Linear(128, 64)

        # 融合层
        # 融合后的维度 = 视觉特征维度 + Lidar特征维度 + 向量特征维度
        fusion_dim = 128 + 64 + 64  # 更新为新的特征维度
        self.fusion_fc1 = nn.Linear(fusion_dim, 512)
        self.fusion_fc2 = nn.Linear(512, action_dim)  # 输出最终的Q值

        # 优化的权重初始化
        self._initialize_weights()


    def _initialize_weights(self):
        """
        初始化网络权重，分别使用了Kaiming & Xavier初始化
        """
        # CNN层使用Kaiming初始化
        for m in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        # 全连接层使用Xavier初始化
        for m in [self.visual_fc, self.lidar_fc1, self.lidar_fc2, 
                 self.vector_fc1, self.vector_fc2, self.fusion_fc1]:
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
            
        # 输出层使用较小的初始值
        nn.init.xavier_uniform_(self.fusion_fc2.weight, gain=0.01)
        nn.init.constant_(self.fusion_fc2.bias, 0)

    def forward(self, state_dict: dict) -> torch.Tensor:
        """前向传播，接收包含多模态数据的字典"""
        try:
            # 处理视觉输入
            visual_input = state_dict['visual']
            x_visual = F.relu(self.conv1(visual_input))
            x_visual = F.relu(self.conv2(x_visual))
            x_visual = F.relu(self.conv3(x_visual))
            x_visual = x_visual.view(x_visual.size(0), -1)  # Flatten
            visual_features = F.relu(self.visual_fc(x_visual))

            # 处理Lidar输入
            lidar_input = state_dict['lidar']
            x_lidar = F.relu(self.lidar_fc1(lidar_input))
            lidar_features = F.relu(self.lidar_fc2(x_lidar))

            # 处理向量输入
            vector_input = state_dict['vector']
            x_vector = F.relu(self.vector_fc1(vector_input))
            vector_features = F.relu(self.vector_fc2(x_vector))

            # 融合特征
            fused_features = torch.cat((visual_features, lidar_features, vector_features), dim=1)

            # 输出Q值
            x = F.relu(self.fusion_fc1(fused_features))
            q_values = self.fusion_fc2(x)

            return q_values
            
        except Exception as e:
            print(f"MultiModalDQN前向传播错误: {e}")
            # 如果出错，尝试返回一个全零张量作为安全值
            batch_size = state_dict.get('vector', torch.zeros(1)).size(0)
            return torch.zeros((batch_size, self.fusion_fc2.out_features), 
                              device=state_dict.get('vector', torch.zeros(1)).device) 