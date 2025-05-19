import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List
import random
from ..models import MultiModalNetwork, ReplayBuffer

class Agent:
    def __init__(self, 
                 radar_dim: int,
                 visual_shape: Tuple[int, int, int],
                 action_dim: int,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 tau: float = 0.005,
                 hidden_dim: int = 64):
        
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        
        # 创建网络
        self.actor_critic = MultiModalNetwork(
            radar_dim=radar_dim,
            visual_shape=visual_shape,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(device)
        
        self.actor_critic_target = MultiModalNetwork(
            radar_dim=radar_dim,
            visual_shape=visual_shape,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(device)
        
        # 复制参数到目标网络
        self.actor_critic_target.load_state_dict(self.actor_critic.state_dict())
        
        # 创建优化器
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # 创建经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # 初始化噪声参数
        self.noise_scale = 1.0
        self.noise_decay = 0.995
        self.min_noise_scale = 0.01
    
    def select_action(self, state: Dict[str, np.ndarray], evaluate: bool = False) -> np.ndarray:
        # 准备输入数据
        radar_input = torch.FloatTensor(state['radar']).unsqueeze(0).to(self.device)
        visual_input = torch.FloatTensor(state['visual']).unsqueeze(0).to(self.device)
        
        # 获取动作
        with torch.no_grad():
            action = self.actor_critic.get_action(radar_input, visual_input)
        
        # 添加探索噪声
        if not evaluate:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = action.cpu().numpy() + noise
            action = np.clip(action, -1, 1)
            
            # 衰减噪声
            self.noise_scale = max(self.min_noise_scale, self.noise_scale * self.noise_decay)
        else:
            action = action.cpu().numpy()
        
        return action.squeeze()
    
    def update(self) -> Tuple[float, float]:
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0
        
        # 从缓冲区采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 准备数据 - 使用numpy.array()预处理
        radar_inputs = torch.FloatTensor(np.array([s['radar'] for s in states])).to(self.device)
        visual_inputs = torch.FloatTensor(np.array([s['visual'] for s in states])).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_radar_inputs = torch.FloatTensor(np.array([s['radar'] for s in next_states])).to(self.device)
        next_visual_inputs = torch.FloatTensor(np.array([s['visual'] for s in next_states])).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        
        # 计算当前Q值
        current_actions, current_values = self.actor_critic(radar_inputs, visual_inputs)
        
        # 计算目标Q值
        with torch.no_grad():
            next_actions, next_values = self.actor_critic_target(next_radar_inputs, next_visual_inputs)
            target_values = rewards + (1 - dones) * self.gamma * next_values
        
        # 计算损失
        value_loss = F.mse_loss(current_values, target_values)
        policy_loss = -current_values.mean()
        
        # 更新网络
        self.optimizer.zero_grad()
        (value_loss + policy_loss).backward()
        self.optimizer.step()
        
        # 软更新目标网络
        self._soft_update()
        
        return value_loss.item(), policy_loss.item()
    
    def _soft_update(self):
        for target_param, param in zip(self.actor_critic_target.parameters(), self.actor_critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def save(self, path: str):
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'actor_critic_target_state_dict': self.actor_critic_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'noise_scale': self.noise_scale
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.actor_critic_target.load_state_dict(checkpoint['actor_critic_target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.noise_scale = checkpoint['noise_scale'] 