"""
第三阶段TD3智能体的神经网络模块
包含Actor、Critic、状态编码器等网络组件
"""

from collections import deque
import random
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class RadarVectorEncoder(nn.Module):
    """
    雷达和向量状态编码器
    用于处理低维度的雷达传感器数据和车辆状态信息

    输入:
        - radar: 雷达传感器读数（检测周围障碍物的距离）
        - vector: 车辆状态向量（速度、角度、目标距离等）
    """

    def __init__(self, radar_dim: int, vector_dim: int, hidden_dim: int):
        super().__init__()
        # 简单的两层全连接网络，使用LayerNorm增强稳定性
        self.network = nn.Sequential(
            nn.Linear(radar_dim + vector_dim, hidden_dim),  # 合并雷达和向量输入
            nn.LayerNorm(hidden_dim),                        # 层归一化，稳定训练
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, radar: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """前向传播：将雷达和向量数据拼接后通过网络"""
        return self.network(torch.cat([radar, vector], dim=1))


class VisualEncoder(nn.Module):
    """
    视觉编码器
    处理自上而下的俯视图观测，用于多模态策略

    使用CNN提取视觉特征，适合处理空间信息如障碍物位置、目标位置等
    """

    def __init__(self, input_channels: int = 3, hidden_dim: int = 128):
        super().__init__()
        # CNN特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),  # 第一层卷积
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),              # 第二层卷积
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),              # 第三层卷积
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # 自适应池化，输出固定尺寸
            nn.Flatten(),                   # 展平为一维向量
        )
        # 特征投影层：将CNN输出映射到隐藏维度
        self.projection = nn.Sequential(
            nn.Linear(64 * 4 * 4, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, visual: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            visual: 视觉输入张量，形状应为 (batch, channels, height, width) 或 (batch, height, width, channels)

        返回:
            编码后的特征向量
        """
        if visual.dim() != 4:
            raise ValueError("视觉输入必须是4D张量。")

        # 处理可能的通道顺序问题（HWC -> CHW）
        if visual.shape[1] != 3 and visual.shape[-1] == 3:
            visual = visual.permute(0, 3, 1, 2)

        return self.projection(self.features(visual))


class StateEncoder(nn.Module):
    """
    状态编码器
    融合雷达/向量特征与可选的视觉特征，生成统一的状态表示

    支持两种模式：
    1. 纯雷达+向量模式（低维输入）
    2. 多模态模式（雷达+向量+视觉）
    """

    def __init__(
        self,
        radar_dim: int,
        vector_dim: int,
        hidden_dim: int = 192,
        use_visual: bool = True,
        visual_channels: int = 3,
    ):
        super().__init__()
        self.use_visual = use_visual

        # 雷达和向量编码器（始终使用）
        self.radar_vector_encoder = RadarVectorEncoder(radar_dim, vector_dim, hidden_dim)

        # 计算融合层的输入维度
        fusion_input_dim = hidden_dim

        # 如果启用视觉输入，添加视觉编码器
        if use_visual:
            self.visual_encoder = VisualEncoder(visual_channels, hidden_dim)
            fusion_input_dim += hidden_dim  # 视觉特征也贡献hidden_dim维度
        else:
            self.visual_encoder = None

        # 特征融合层：将不同模态的特征合并
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        radar: torch.Tensor,
        vector: torch.Tensor,
        visual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播：编码并融合多模态状态信息

        参数:
            radar: 雷达数据
            vector: 向量状态数据
            visual: 视觉数据（可选）

        返回:
            融合后的状态特征向量
        """
        # 编码雷达和向量数据
        features = [self.radar_vector_encoder(radar, vector)]

        # 如果启用视觉且提供了视觉数据，编码并加入特征列表
        if self.use_visual and visual is not None:
            features.append(self.visual_encoder(visual))

        # 拼接所有特征并通过融合层
        return self.fusion(torch.cat(features, dim=1))


class Actor(nn.Module):
    """
    TD3策略网络（Actor）
    负责根据状态输出动作

    Actor的目标是选择能够最大化期望奖励的动作
    使用Tanh激活确保输出动作在[-1, 1]范围内
    """

    def __init__(
        self,
        radar_dim: int,
        vector_dim: int,
        action_dim: int,
        hidden_dim: int = 192,
        use_visual: bool = True,
    ):
        super().__init__()
        # 状态编码器
        self.encoder = StateEncoder(
            radar_dim=radar_dim,
            vector_dim=vector_dim,
            hidden_dim=hidden_dim,
            use_visual=use_visual,
        )

        # 策略输出头（动作网络）
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),  # 输出动作维度
            nn.Tanh(),                          # 将输出限制在[-1, 1]范围
        )

    def forward(
        self,
        radar: torch.Tensor,
        vector: torch.Tensor,
        visual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播：根据状态输出动作

        参数:
            radar: 雷达观测
            vector: 向量状态
            visual: 视觉观测（可选）

        返回:
            动作向量 [油门, 转向]，范围[-1, 1]
        """
        features = self.encoder(radar, vector, visual)
        return self.policy_head(features)


class Critic(nn.Module):
    """
    Q网络（Critic）
    评估在给定状态下采取某动作的期望价值（Q值）

    Critic的输入是状态和动作，输出是对未来奖励的估计
    TD3使用两个独立的Critic网络来减少价值过估计
    """

    def __init__(
        self,
        radar_dim: int,
        vector_dim: int,
        action_dim: int,
        hidden_dim: int = 192,
        use_visual: bool = True,
    ):
        super().__init__()
        # 状态编码器（与Actor共享架构但参数独立）
        self.encoder = StateEncoder(
            radar_dim=radar_dim,
            vector_dim=vector_dim,
            hidden_dim=hidden_dim,
            use_visual=use_visual,
        )

        # 价值输出头：输入是状态特征+动作
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),  # 状态特征与动作拼接
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 输出单个Q值
        )

    def forward(
        self,
        radar: torch.Tensor,
        vector: torch.Tensor,
        visual: Optional[torch.Tensor],
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播：计算Q值

        参数:
            radar: 雷达观测
            vector: 向量状态
            visual: 视觉观测
            action: 待评估的动作

        返回:
            Q值估计（期望的未来累积奖励）
        """
        features = self.encoder(radar, vector, visual)
        return self.value_head(torch.cat([features, action], dim=1))


class TwinCritic(nn.Module):
    """
    双Q网络
    TD3算法的核心组件，使用两个独立的Critic网络

    使用双网络的原因：
    - 在计算目标Q值时取两个网络的最小值
    - 这可以减少Q值的过估计问题
    - 过估计会导致策略学习到次优动作
    """

    def __init__(
        self,
        radar_dim: int,
        vector_dim: int,
        action_dim: int,
        hidden_dim: int = 192,
        use_visual: bool = True,
    ):
        super().__init__()
        # 两个独立的Critic网络
        self.q1 = Critic(radar_dim, vector_dim, action_dim, hidden_dim, use_visual)
        self.q2 = Critic(radar_dim, vector_dim, action_dim, hidden_dim, use_visual)

    def forward(
        self,
        radar: torch.Tensor,
        vector: torch.Tensor,
        visual: Optional[torch.Tensor],
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：同时计算两个Q值

        返回:
            (q1_value, q2_value)：两个网络的Q值估计
        """
        return (
            self.q1(radar, vector, visual, action),
            self.q2(radar, vector, visual, action),
        )

    def q1_forward(
        self,
        radar: torch.Tensor,
        vector: torch.Tensor,
        visual: Optional[torch.Tensor],
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        只使用第一个Critic计算Q值
        用于Actor更新时的梯度计算
        """
        return self.q1(radar, vector, visual, action)


class ReplayBuffer:
    """
    经验回放池
    存储和采样历史经验，支持高效的强化学习训练

    经验回放的作用：
    1. 打破样本间的相关性（时序相关性会导致训练不稳定）
    2. 提高数据利用效率（一个经验可以被多次使用）
    3. 支持批量训练，提高计算效率
    """

    def __init__(self, capacity: int):
        """
        初始化回放池

        参数:
            capacity: 最大容量，超过时会丢弃最旧的经验
        """
        self.capacity = capacity
        # 使用双端队列实现，自动处理容量溢出
        self.buffer: Deque[Tuple[Dict[str, np.ndarray], np.ndarray, float, Dict[str, np.ndarray], float]] = deque(
            maxlen=capacity
        )

    def push(
        self,
        state: Dict[str, np.ndarray],      # 当前状态
        action: np.ndarray,                 # 采取的动作
        reward: float,                      # 获得的奖励
        next_state: Dict[str, np.ndarray],  # 下一状态
        done: float,                        # 是否终止
    ) -> None:
        """
        将一条经验添加到回放池

        注意：状态会被深拷贝，避免后续修改影响存储的数据
        """
        copied_state = {key: np.array(value, copy=True) for key, value in state.items()}
        copied_next_state = {key: np.array(value, copy=True) for key, value in next_state.items()}
        self.buffer.append(
            (copied_state, np.array(action, copy=True), float(reward), copied_next_state, float(done))
        )

    def sample(
        self, batch_size: int
    ) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray, np.ndarray, List[Dict[str, np.ndarray]], np.ndarray]:
        """
        从回放池随机采样一批经验

        随机采样打乱了时序相关性，使训练更加稳定

        参数:
            batch_size: 采样数量

        返回:
            (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return list(states), np.stack(actions), np.asarray(rewards), list(next_states), np.asarray(dones)

    def __len__(self) -> int:
        """返回当前存储的经验数量"""
        return len(self.buffer)