"""
训练和评估指标辅助模块
提供指标记录、保存和可视化功能

功能包括：
- 记录每回合的奖励、长度、损失等指标
- 生成训练曲线图表
- 计算和保存统计摘要
"""

import json
import os
import tempfile
from typing import Dict, List, Optional

# 设置matplotlib临时目录，避免权限问题
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "rl_driverx_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "rl_driverx_cache"))

import matplotlib.pyplot as plt
import numpy as np


class MetricsRecorder:
    """
    指标记录器

    记录训练/评估过程中的各项指标，并提供可视化和保存功能

    记录的指标包括：
    - episode_rewards: 每回合的累计奖励
    - episode_lengths: 每回合的步数
    - critic_losses/actor_losses: TD3网络损失
    - collision_counts: 碰撞次数
    - target_reached: 是否成功到达目标
    - final_distances/min_distances: 距离相关指标
    - stage_ids: 课程学习阶段ID
    """

    def __init__(self):
        """初始化各指标列表"""
        self.episode_rewards: List[float] = []     # 每回合奖励
        self.episode_lengths: List[int] = []       # 每回合长度
        self.critic_losses: List[float] = []       # Critic损失
        self.actor_losses: List[float] = []        # Actor损失
        self.collision_counts: List[int] = []      # 碰撞次数
        self.target_reached: List[bool] = []       # 成功标志
        self.final_distances: List[float] = []     # 最终距离
        self.min_distances: List[float] = []       # 最小距离
        self.stage_ids: List[int] = []             # 阶段ID
        self.current_step_rewards: List[float] = []  # 当前回合的每步奖励

    def start_episode(self) -> None:
        """开始新回合，清空每步奖励记录"""
        self.current_step_rewards = []

    def add_step_reward(self, reward: float) -> None:
        """
        记录每步奖励

        参数:
            reward: 当前步获得的奖励
        """
        self.current_step_rewards.append(float(reward))

    def add_episode_data(
        self,
        reward: float,
        length: int,
        critic_loss: Optional[float],
        actor_loss: Optional[float],
        collisions: int,
        target_reached: bool,
        final_distance: float,
        min_distance: float,
        stage_id: int,
    ) -> None:
        """
        记录回合结束时的完整数据

        参数:
            reward: 回合累计奖励
            length: 回合步数
            critic_loss: Critic网络平均损失
            actor_loss: Actor网络平均损失
            collisions: 回合中碰撞次数
            target_reached: 是否成功到达目标
            final_distance: 最终到目标的距离
            min_distance: 回合中最小距离
            stage_id: 当前课程学习阶段
        """
        self.episode_rewards.append(float(reward))
        self.episode_lengths.append(int(length))
        self.critic_losses.append(0.0 if critic_loss is None else float(critic_loss))
        self.actor_losses.append(0.0 if actor_loss is None else float(actor_loss))
        self.collision_counts.append(int(collisions))
        self.target_reached.append(bool(target_reached))
        self.final_distances.append(float(final_distance))
        self.min_distances.append(float(min_distance))
        self.stage_ids.append(int(stage_id))

    def save_metrics(self, save_dir: str) -> Dict:
        """
        保存指标数据到JSON文件

        参数:
            save_dir: 保存目录

        返回:
            包含完整指标数据的字典
        """
        os.makedirs(save_dir, exist_ok=True)
        summary = self._summary()

        # 构建完整指标数据
        payload = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "critic_losses": self.critic_losses,
            "actor_losses": self.actor_losses,
            "collision_counts": self.collision_counts,
            "target_reached": self.target_reached,
            "final_distances": self.final_distances,
            "min_distances": self.min_distances,
            "stage_ids": self.stage_ids,
            "summary": summary,
        }

        # 保存完整数据
        with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

        # 保存摘要数据
        with open(os.path.join(save_dir, "metrics_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

        return payload

    def plot_metrics(self, save_dir: str) -> None:
        """
        生成并保存训练曲线图表

        参数:
            save_dir: 图表保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        self._plot_rewards_and_lengths(save_dir)      # 奖励和长度曲线
        self._plot_losses(save_dir)                   # 损失曲线
        self._plot_success_and_collisions(save_dir)   # 成功率和碰撞
        self._plot_distance_metrics(save_dir)         # 距离指标
        self._plot_combined_metrics(save_dir)         # 综合指标

    def _summary(self) -> Dict:
        """
        计算指标统计摘要

        返回:
            包含各项统计指标的字典
        """
        success_rate = float(np.mean(self.target_reached)) if self.target_reached else 0.0
        return {
            "mean_reward": float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
            "std_reward": float(np.std(self.episode_rewards)) if len(self.episode_rewards) > 1 else 0.0,
            "mean_length": float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0,
            "mean_final_distance": float(np.mean(self.final_distances)) if self.final_distances else 0.0,
            "mean_min_distance": float(np.mean(self.min_distances)) if self.min_distances else 0.0,
            "success_rate": success_rate,
            "total_collisions": int(np.sum(self.collision_counts)) if self.collision_counts else 0,
        }

    def _plot_rewards_and_lengths(self, save_dir: str) -> None:
        """绘制奖励和回合长度曲线"""
        if not self.episode_rewards:
            return

        plt.figure(figsize=(12, 5))

        # 奖励曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards, color="#2f65e0")
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True, alpha=0.3)

        # 长度曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.episode_lengths, color="#31aa70")
        plt.title("Episode Lengths")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "rewards_lengths.png"), dpi=150)
        plt.close()

    def _plot_losses(self, save_dir: str) -> None:
        """绘制网络损失曲线"""
        if not self.critic_losses:
            return

        plt.figure(figsize=(12, 5))

        # Critic损失
        plt.subplot(1, 2, 1)
        plt.plot(self.critic_losses, color="#c1604d")
        plt.title("Critic Loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)

        # Actor损失
        plt.subplot(1, 2, 2)
        plt.plot(self.actor_losses, color="#9558d3")
        plt.title("Actor Loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "losses.png"), dpi=150)
        plt.close()

    def _plot_success_and_collisions(self, save_dir: str) -> None:
        """绘制碰撞次数和成功率曲线"""
        if not self.target_reached:
            return

        plt.figure(figsize=(12, 5))

        # 碰撞柱状图
        plt.subplot(1, 2, 1)
        plt.bar(range(len(self.collision_counts)), self.collision_counts, color="#c1604d")
        plt.title("Collisions")
        plt.xlabel("Episode")
        plt.ylabel("Count")

        # 成功率滑动平均
        plt.subplot(1, 2, 2)
        running_success = self._running_average(np.asarray(self.target_reached, dtype=np.float32), window=10)
        plt.plot(running_success, color="#31aa70")
        plt.title("Success Rate (Running Average)")
        plt.xlabel("Episode")
        plt.ylabel("Success Rate")
        plt.ylim(0.0, 1.05)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "collision_success.png"), dpi=150)
        plt.close()

    def _plot_distance_metrics(self, save_dir: str) -> None:
        """绘制距离相关指标曲线"""
        if not self.final_distances:
            return

        plt.figure(figsize=(12, 5))

        # 最终距离
        plt.subplot(1, 2, 1)
        plt.plot(self.final_distances, color="#f0a202")
        plt.title("Final Distance To Target")
        plt.xlabel("Episode")
        plt.ylabel("Distance")
        plt.grid(True, alpha=0.3)

        # 最小距离
        plt.subplot(1, 2, 2)
        plt.plot(self.min_distances, color="#4b83f5")
        plt.title("Minimum Distance To Target")
        plt.xlabel("Episode")
        plt.ylabel("Distance")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "distance_metrics.png"), dpi=150)
        plt.close()

    def _plot_combined_metrics(self, save_dir: str) -> None:
        """绘制综合归一化指标曲线"""
        if not self.episode_rewards:
            return

        plt.figure(figsize=(12, 6))
        plt.plot(self._normalize(self.episode_rewards), label="Reward", color="#2f65e0")
        plt.plot(self._normalize(self.episode_lengths), label="Length", color="#31aa70")
        plt.plot(self._normalize(self.final_distances, invert=True), label="Final Distance", color="#f0a202")
        plt.plot(self._normalize(self.collision_counts, invert=True), label="Collision Score", color="#c1604d")
        plt.title("Combined Normalized Metrics")
        plt.xlabel("Episode")
        plt.ylabel("Normalized Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "combined_metrics.png"), dpi=150)
        plt.close()

    @staticmethod
    def _normalize(values: List[float], invert: bool = False) -> np.ndarray:
        """
        归一化指标值到[0, 1]范围

        参数:
            values: 原始值列表
            invert: 是否反转（用于"越小越好"的指标）

        返回:
            归一化后的数组
        """
        array = np.asarray(values, dtype=np.float32)
        if array.size == 0:
            return array
        min_value = np.min(array)
        max_value = np.max(array)
        if np.isclose(max_value, min_value):
            normalized = np.ones_like(array)
        else:
            normalized = (array - min_value) / (max_value - min_value)
        if invert:
            normalized = 1.0 - normalized
        return normalized

    @staticmethod
    def _running_average(values: np.ndarray, window: int = 10) -> np.ndarray:
        """
        计算滑动平均

        参数:
            values: 原始值数组
            window: 窗口大小

        返回:
            滑动平均后的数组
        """
        if values.size == 0:
            return values
        result = np.zeros_like(values, dtype=np.float32)
        for index in range(values.size):
            start = max(0, index - window + 1)
            result[index] = float(np.mean(values[start : index + 1]))
        return result