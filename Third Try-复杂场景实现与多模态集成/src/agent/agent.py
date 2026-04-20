"""
第三阶段TD3智能体模块
用于驾驶任务的TD3（Twin Delayed DDPG）算法实现
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from ..models import Actor, ReplayBuffer, TwinCritic


class Agent:
    """
    TD3智能体实现
    针对驾驶任务进行优化的最小化TD3实现

    TD3算法的核心特点：
    1. 使用双Q网络（Twin Critics）减少价值过估计
    2. 延迟策略更新（Delayed Policy Updates）
    3. 目标策略平滑（Target Policy Smoothing）
    """

    def __init__(
        self,
        radar_dim: int,                    # 雷达传感器维度（射线数量）
        vector_dim: int,                   # 向量状态维度（速度、角度等）
        visual_shape: Tuple[int, int, int], # 视觉输入形状（通道、高度、宽度）
        action_dim: int,                   # 动作维度（油门、转向）
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 3e-4,       # 学习率
        gamma: float = 0.99,               # 折扣因子
        buffer_size: int = 150_000,        # 经验回放池容量
        batch_size: int = 128,             # 批大小
        tau: float = 0.005,                # 软更新系数
        hidden_dim: int = 192,             # 隐藏层维度
        policy_noise: float = 0.18,        # 目标策略噪声
        noise_clip: float = 0.35,          # 噪声裁剪范围
        exploration_noise: float = 0.12,   # 探索噪声
        policy_delay: int = 2,             # 策略更新延迟（每N次critic更新后更新actor）
        warmup_steps: int = 3_000,         # 预热步数（随机探索阶段）
        use_visual: bool = True,           # 是否使用视觉输入
    ):
        # 初始化计算设备
        self.device = torch.device(device)

        # TD3核心参数
        self.gamma = gamma                  # 折扣因子，用于计算未来奖励的权重
        self.batch_size = batch_size        # 训练批大小
        self.tau = tau                      # 软更新系数，用于目标网络的渐进更新
        self.policy_noise = policy_noise    # 目标策略噪声，用于减少Q值过估计
        self.noise_clip = noise_clip        # 噪声裁剪上限
        self.exploration_noise = exploration_noise  # 探索噪声幅度
        self.policy_delay = policy_delay    # 策略更新延迟步数
        self.warmup_steps = warmup_steps    # 预热阶段的步数
        self.use_visual = use_visual        # 是否启用视觉观测分支
        self.action_dim = action_dim        # 动作空间维度
        self.hidden_dim = hidden_dim        # 网络隐藏层维度
        self.total_it = 0                   # 总更新迭代次数计数器
        self.total_steps = 0                # 总步数计数器（用于预热判断）

        # 初始化Actor网络（策略网络）
        # Actor负责输出动作，决定智能体的行为
        self.actor = Actor(
            radar_dim=radar_dim,
            vector_dim=vector_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            use_visual=use_visual,
        ).to(self.device)

        # 初始化Actor目标网络
        # 目标网络用于稳定训练，通过软更新逐渐同步
        self.actor_target = Actor(
            radar_dim=radar_dim,
            vector_dim=vector_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            use_visual=use_visual,
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())  # 初始时与主网络同步

        # 初始化Critic网络（价值网络）
        # Critic负责评估动作的价值，TD3使用双Critic减少过估计
        self.critic = TwinCritic(
            radar_dim=radar_dim,
            vector_dim=vector_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            use_visual=use_visual,
        ).to(self.device)

        # 初始化Critic目标网络
        self.critic_target = TwinCritic(
            radar_dim=radar_dim,
            vector_dim=vector_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            use_visual=use_visual,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 初始化优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # 初始化经验回放池
        # 用于存储和采样历史经验，支持高效学习
        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state: Dict[str, np.ndarray], evaluate: bool = False) -> np.ndarray:
        """
        根据当前状态选择动作

        参数:
            state: 当前观测状态，包含radar、vector、visual等信息
            evaluate: 是否为评估模式（评估时不添加探索噪声）

        返回:
            选择的动作数组 [油门, 转向]
        """
        # 预热阶段使用随机动作采样
        # 这有助于在策略网络稳定前收集多样化的经验
        if not evaluate and self.total_steps < self.warmup_steps:
            action = np.array(
                [
                    np.random.uniform(0.0, 1.0),   # 油门：0-1范围
                    np.random.uniform(-1.0, 1.0),  # 转向：-1到1范围
                ],
                dtype=np.float32,
            )
            self.total_steps += 1
            return action

        # 将状态转换为张量格式
        radar, vector, visual = self._state_to_tensors(state)

        # 使用Actor网络推理动作
        with torch.no_grad():
            action = self.actor(radar, vector, visual).cpu().numpy()[0]

        # 训练模式下添加探索噪声
        # 有助于探索更多动作空间，避免过早收敛到局部最优
        if not evaluate:
            action += np.random.normal(0.0, self.exploration_noise, size=self.action_dim)
            action = np.clip(action, -1.0, 1.0)  # 确保动作在合法范围内

        self.total_steps += 1
        return action.astype(np.float32)

    def update(self) -> Tuple[Optional[float], Optional[float]]:
        """
        执行一次TD3更新步骤

        TD3更新流程：
        1. 从经验回放池采样一批数据
        2. 计算目标Q值（使用目标网络和噪声平滑）
        3. 更新Critic网络
        4. 延迟更新Actor网络（每policy_delay次更新一次）
        5. 软更新目标网络

        返回:
            (critic_loss, actor_loss)：Critic损失和Actor损失，如果数据不足则返回None
        """
        # 检查经验回放池是否有足够的数据
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        self.total_it += 1

        # 从经验回放池采样一批经验
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # 将数据转换为张量并移到计算设备
        radar, vector, visual = self._batch_states_to_tensors(states)
        next_radar, next_vector, next_visual = self._batch_states_to_tensors(next_states)
        actions_tensor = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # 计算目标Q值
        # TD3的核心创新：添加噪声并进行裁剪，减少Q值过估计
        with torch.no_grad():
            # 添加目标策略平滑噪声
            noise = torch.randn_like(actions_tensor) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)

            # 使用目标Actor网络计算下一状态的动作
            next_actions = self.actor_target(next_radar, next_vector, next_visual)
            next_actions = (next_actions + noise).clamp(-1.0, 1.0)

            # 使用双Critic计算目标Q值，取最小值以减少过估计
            target_q1, target_q2 = self.critic_target(
                next_radar,
                next_vector,
                next_visual,
                next_actions,
            )
            target_q = torch.min(target_q1, target_q2)

            # 计算贝尔曼目标值
            target_values = rewards_tensor + (1.0 - dones_tensor) * self.gamma * target_q

        # 计算当前Q值并更新Critic
        current_q1, current_q2 = self.critic(radar, vector, visual, actions_tensor)
        critic_loss = F.mse_loss(current_q1, target_values) + F.mse_loss(current_q2, target_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 延迟更新Actor（每policy_delay次Critic更新后执行一次）
        actor_loss = None
        if self.total_it % self.policy_delay == 0:
            # Actor的目标是最大化Critic估计的Q值
            policy_actions = self.actor(radar, vector, visual)
            actor_loss_tensor = -self.critic.q1_forward(radar, vector, visual, policy_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss_tensor.backward()
            self.actor_optimizer.step()

            # 软更新目标网络
            # 目标网络参数逐渐向主网络靠近，保持训练稳定性
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            actor_loss = float(actor_loss_tensor.item())

        return float(critic_loss.item()), actor_loss

    def save(self, path: str) -> None:
        """
        保存智能体的所有状态到文件

        参数:
            path: 保存路径
        """
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "total_it": self.total_it,
                "total_steps": self.total_steps,
                "use_visual": self.use_visual,
                "hidden_dim": self.hidden_dim,
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        从文件加载智能体状态

        参数:
            path: 加载路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.total_it = checkpoint.get("total_it", 0)
        self.total_steps = checkpoint.get("total_steps", 0)

    @staticmethod
    def checkpoint_metadata(path: str) -> Dict[str, object]:
        """
        从保存的checkpoint文件中提取元数据

        用于在不加载完整模型的情况下获取网络配置信息

        参数:
            path: checkpoint文件路径

        返回:
            包含hidden_dim和use_visual的字典
        """
        checkpoint = torch.load(path, map_location="cpu")
        actor_state = checkpoint["actor"]
        hidden_dim = checkpoint.get("hidden_dim")
        if hidden_dim is None:
            hidden_dim = int(actor_state["policy_head.0.weight"].shape[0])
        use_visual = checkpoint.get(
            "use_visual",
            any(name.startswith("encoder.visual_encoder") for name in actor_state.keys()),
        )
        return {
            "hidden_dim": int(hidden_dim),
            "use_visual": bool(use_visual),
        }

    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module) -> None:
        """
        软更新目标网络参数

        目标网络参数 = tau * 主网络参数 + (1 - tau) * 目标网络参数
        这种渐进更新方式有助于保持训练稳定性

        参数:
            source: 主网络（源网络）
            target: 目标网络（被更新的网络）
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)

    def _state_to_tensors(self, state: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        将单个状态字典转换为张量格式

        参数:
            state: 状态字典，包含radar、vector、visual

        返回:
            (radar_tensor, vector_tensor, visual_tensor)
        """
        radar = torch.as_tensor(state["radar"], dtype=torch.float32, device=self.device).unsqueeze(0)
        vector = torch.as_tensor(state["vector"], dtype=torch.float32, device=self.device).unsqueeze(0)
        visual = None
        if self.use_visual:
            visual = torch.as_tensor(state["visual"], dtype=torch.float32, device=self.device).unsqueeze(0)
        return radar, vector, visual

    def _batch_states_to_tensors(
        self, states: List[Dict[str, np.ndarray]]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        将一批状态字典转换为张量格式

        参数:
            states: 状态列表

        返回:
            (radar_tensor, vector_tensor, visual_tensor)
        """
        radar = torch.as_tensor(np.stack([state["radar"] for state in states]), dtype=torch.float32, device=self.device)
        vector = torch.as_tensor(np.stack([state["vector"] for state in states]), dtype=torch.float32, device=self.device)
        visual = None
        if self.use_visual:
            visual = torch.as_tensor(
                np.stack([state["visual"] for state in states]),
                dtype=torch.float32,
                device=self.device,
            )
        return radar, vector, visual