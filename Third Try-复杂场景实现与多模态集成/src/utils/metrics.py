import numpy as np
import json
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple

class MetricsRecorder:
    """用于记录和分析训练与测试指标的工具"""
    
    def __init__(self):
        # 训练指标
        self.episode_rewards = []
        self.episode_lengths = []
        self.value_losses = []
        self.policy_losses = []
        self.collision_counts = []
        self.target_reached = []
        self.distance_to_target = []
        self.avg_speed = []
        self.avg_rotation = []
        
        # 步级指标
        self.current_episode_rewards = []
        self.current_episode_distances = []
        self.current_episode_speeds = []
        self.current_episode_rotations = []
        
    def reset_episode_metrics(self):
        """重置当前回合的步级指标"""
        self.current_episode_rewards = []
        self.current_episode_distances = []
        self.current_episode_speeds = []
        self.current_episode_rotations = []
        
    def add_episode_data(self, 
                        reward: float, 
                        length: int, 
                        value_loss: float, 
                        policy_loss: float,
                        collisions: int,
                        target_reached: bool):
        """添加回合级指标"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.value_losses.append(value_loss)
        self.policy_losses.append(policy_loss)
        self.collision_counts.append(collisions)
        self.target_reached.append(target_reached)
        
        # 计算平均距离、速度和旋转
        if self.current_episode_distances:
            self.distance_to_target.append(np.mean(self.current_episode_distances))
        if self.current_episode_speeds:
            self.avg_speed.append(np.mean(self.current_episode_speeds))
        if self.current_episode_rotations:
            self.avg_rotation.append(np.mean(self.current_episode_rotations))
        
        self.reset_episode_metrics()
        
    def add_step_data(self, 
                     reward: float, 
                     distance: float,
                     speed: float,
                     rotation: float):
        """添加步级指标"""
        self.current_episode_rewards.append(reward)
        self.current_episode_distances.append(distance)
        self.current_episode_speeds.append(speed)
        self.current_episode_rotations.append(abs(rotation))
        
    def save_metrics(self, save_dir: str):
        """保存指标到JSON文件"""
        # 确保保存目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 创建指标字典
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'value_losses': self.value_losses,
            'policy_losses': self.policy_losses,
            'collision_counts': self.collision_counts,
            'target_reached': self.target_reached,
            'distance_to_target': self.distance_to_target,
            'avg_speed': self.avg_speed,
            'avg_rotation': self.avg_rotation,
            'summary': {
                'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
                'std_reward': np.std(self.episode_rewards) if len(self.episode_rewards) > 1 else 0,
                'mean_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
                'std_length': np.std(self.episode_lengths) if len(self.episode_lengths) > 1 else 0,
                'total_collisions': sum(self.collision_counts) if self.collision_counts else 0,
                'success_rate': sum(self.target_reached) / len(self.target_reached) if self.target_reached else 0,
            }
        }
        
        # 将所有数组转换为列表，以便JSON序列化
        metrics_json = {k: v if not isinstance(v, np.ndarray) else v.tolist() for k, v in metrics.items()}
        
        try:
            # 保存完整指标
            metrics_path = os.path.join(save_dir, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics_json, f, indent=4)
            
            # 保存摘要指标
            summary_path = os.path.join(save_dir, 'metrics_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(metrics_json['summary'], f, indent=4)
        except Exception as e:
            print(f"保存指标时出错: {e}")
        
        return metrics
    
    def plot_metrics(self, save_dir: str):
        """绘制指标图表"""
        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # 创建奖励和长度图表
            self._plot_rewards_lengths(save_dir)
            
            # 创建损失图表
            self._plot_losses(save_dir)
            
            # 创建碰撞和成功率图表
            self._plot_collision_success(save_dir)
            
            # 创建距离、速度和旋转图表
            self._plot_distance_speed_rotation(save_dir)
            
            # 创建综合指标图
            self._plot_combined_metrics(save_dir)
        except Exception as e:
            print(f"绘制指标图表时出错: {e}")
        
    def _plot_rewards_lengths(self, save_dir: str):
        """绘制奖励和回合长度图表"""
        if not self.episode_rewards or not self.episode_lengths:
            return
            
        try:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.plot(self.episode_rewards, 'b-')
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(self.episode_lengths, 'g-')
            plt.title('Episode Lengths')
            plt.xlabel('Episode')
            plt.ylabel('Length')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'rewards_lengths.png'))
            plt.close()
        except Exception as e:
            print(f"绘制奖励和长度图表时出错: {e}")
        
    def _plot_losses(self, save_dir: str):
        """绘制损失图表"""
        if not self.value_losses or not self.policy_losses:
            return
        
        try:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.plot(self.value_losses, 'r-')
            plt.title('Value Losses')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(self.policy_losses, 'm-')
            plt.title('Policy Losses')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'losses.png'))
            plt.close()
        except Exception as e:
            print(f"绘制损失图表时出错: {e}")
        
    def _plot_collision_success(self, save_dir: str):
        """绘制碰撞和成功率图表"""
        if not self.collision_counts or not self.target_reached:
            return
            
        try:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.bar(range(len(self.collision_counts)), self.collision_counts, color='r')
            plt.title('Collision Counts')
            plt.xlabel('Episode')
            plt.ylabel('Collisions')
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            success_rates = []
            window_size = min(5, len(self.target_reached))
            if window_size > 0:
                for i in range(len(self.target_reached)):
                    end_idx = min(i + window_size, len(self.target_reached))
                    window = self.target_reached[i:end_idx]
                    if window:
                        success_rates.append(sum(window) / len(window))
                    else:
                        success_rates.append(0)
                
                plt.plot(success_rates, 'g-')
                plt.title('Success Rate (Running Average)')
                plt.xlabel('Episode')
                plt.ylabel('Success Rate')
                plt.ylim([0, 1.1])
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'collision_success.png'))
            plt.close()
        except Exception as e:
            print(f"绘制碰撞和成功率图表时出错: {e}")
        
    def _plot_distance_speed_rotation(self, save_dir: str):
        """绘制距离、速度和旋转图表"""
        # 检查是否有足够的数据
        has_distance = len(self.distance_to_target) > 0
        has_speed = len(self.avg_speed) > 0
        has_rotation = len(self.avg_rotation) > 0
        
        if not (has_distance or has_speed or has_rotation):
            return
            
        try:
            plt.figure(figsize=(15, 5))
            
            # 绘制可用的数据
            subplot_count = sum([has_distance, has_speed, has_rotation])
            current_plot = 1
            
            if has_distance:
                plt.subplot(1, subplot_count, current_plot)
                plt.plot(self.distance_to_target, 'b-')
                plt.title('Average Distance to Target')
                plt.xlabel('Episode')
                plt.ylabel('Distance')
                plt.grid(True)
                current_plot += 1
            
            if has_speed:
                plt.subplot(1, subplot_count, current_plot)
                plt.plot(self.avg_speed, 'g-')
                plt.title('Average Speed')
                plt.xlabel('Episode')
                plt.ylabel('Speed')
                plt.grid(True)
                current_plot += 1
            
            if has_rotation:
                plt.subplot(1, subplot_count, current_plot)
                plt.plot(self.avg_rotation, 'r-')
                plt.title('Average Rotation')
                plt.xlabel('Episode')
                plt.ylabel('Rotation')
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'distance_speed_rotation.png'))
            plt.close()
        except Exception as e:
            print(f"绘制距离、速度和旋转图表时出错: {e}")
        
    def _plot_combined_metrics(self, save_dir: str):
        """绘制综合指标图表"""
        # 只取有数据的指标
        metrics = []
        labels = []
        
        if self.episode_rewards:
            metrics.append(self.episode_rewards)
            labels.append('Rewards')
        
        if self.episode_lengths:
            # 归一化长度以便比较
            max_length = max(self.episode_lengths) if self.episode_lengths else 1
            if max_length > 0:
                normalized_lengths = np.array(self.episode_lengths) / max_length
                metrics.append(normalized_lengths.tolist())
                labels.append('Normalized Lengths')
        
        if self.avg_speed:
            # 归一化速度
            max_speed = max(self.avg_speed) if self.avg_speed else 1
            if max_speed > 0:
                normalized_speed = np.array(self.avg_speed) / max_speed
                metrics.append(normalized_speed.tolist())
                labels.append('Normalized Speed')
        
        if not metrics:
            return
        
        try:
            plt.figure(figsize=(12, 8))
            
            # 获取最短长度以确保所有数据可比较
            min_length = min(len(m) for m in metrics) if metrics else 0
            
            if min_length > 0:
                # 绘制所有指标
                for i, (metric, label) in enumerate(zip(metrics, labels)):
                    plt.plot(metric[:min_length], label=label)
                
                plt.title('Combined Metrics')
                plt.xlabel('Episode')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'combined_metrics.png'))
            plt.close()
        except Exception as e:
            print(f"绘制综合指标图表时出错: {e}") 