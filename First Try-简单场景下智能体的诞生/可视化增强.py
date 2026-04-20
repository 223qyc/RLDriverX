"""
增强可视化模块
提供训练过程的详细可视化和分析功能

主要功能：
- EnhancedVisualizer: 增强版可视化器，记录并分析回合数据
- 3D Q值可视化：展示不同位置下各动作的Q值分布
- 策略可视化：展示学习到的策略在环境中的分布
- 训练进度可视化：奖励、探索率、损失等曲线
- 回合渲染动画：生成带轨迹和Q值信息的动画
"""

import os
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Arrow, Polygon
from collections import deque, defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from datetime import datetime
import warnings
from main import *

warnings.filterwarnings('ignore')

# 设置随机种子，确保结果可复现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 创建必要的输出目录
os.makedirs("models", exist_ok=True)
os.makedirs("videos", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

# 自定义颜色映射，用于可视化
car_cmap = LinearSegmentedColormap.from_list('car_cmap', ['#FF6B6B', '#4ECDC4', '#45B7D1'])
q_cmap = LinearSegmentedColormap.from_list('q_cmap', ['#2C3E50', '#3498DB', '#1ABC9C'])


class EnhancedVisualizer:
    """
    增强版可视化器

    功能：
    - 记录每回合的轨迹、动作、奖励数据
    - 统计成功率、轨迹长度、奖励分布
    - 生成回合动画（带轨迹和Q值）
    """

    def __init__(self, env, agent):
        """
        初始化可视化器

        参数:
            env: 环境实例
            agent: 智能体实例
        """
        self.env = env
        self.agent = agent
        self.episode_data = defaultdict(list)  # 存储所有回合数据
        self.current_trajectory = []            # 当前回合轨迹
        self.current_actions = []               # 当前回合动作序列
        self.current_rewards = []               # 当前回合奖励序列

    def reset_recording(self):
        """开始新回合前清空记录"""
        self.current_trajectory = []
        self.current_actions = []
        self.current_rewards = []

    def record_step(self, x, y, action, reward):
        """
        记录每一步的数据

        参数:
            x, y: 小车位置
            action: 采取的动作
            reward: 获得的奖励
        """
        self.current_trajectory.append((x, y))
        self.current_actions.append(action)
        self.current_rewards.append(reward)

    def save_episode_data(self, episode, success):
        """
        保存回合结束时的数据

        参数:
            episode: 回合编号
            success: 是否成功到达目标
        """
        self.episode_data['episode'].append(episode)
        self.episode_data['trajectory'].append(self.current_trajectory)
        self.episode_data['actions'].append(self.current_actions)
        self.episode_data['rewards'].append(self.current_rewards)
        self.episode_data['success'].append(success)
        self.episode_data['length'].append(len(self.current_trajectory))
        self.episode_data['total_reward'].append(sum(self.current_rewards))

    def plot_episode_stats(self):
        """
        绘制回合统计图表

        包括：成功率、轨迹长度分布、奖励分布、动作频率
        """
        df = pd.DataFrame(self.episode_data)

        plt.figure(figsize=(15, 10))

        # 成功/失败比例
        plt.subplot(2, 2, 1)
        success_rate = df['success'].mean()
        plt.bar(['Success', 'Failure'], [success_rate, 1 - success_rate], color=['#2ecc71', '#e74c3c'])
        plt.title(f'Success Rate ({success_rate:.1%})')

        # 轨迹长度分布直方图
        plt.subplot(2, 2, 2)
        sns.histplot(df['length'], bins=20, kde=True, color='#3498db')
        plt.title('Trajectory Length Distribution')

        # 奖励分布箱线图（按成功/失败分组）
        plt.subplot(2, 2, 3)
        sns.boxplot(x='success', y='total_reward', data=df, palette=['#e74c3c', '#2ecc71'])
        plt.title('Reward Distribution by Outcome')

        # 动作频率统计
        plt.subplot(2, 2, 4)
        all_actions = [a for sublist in df['actions'] for a in sublist]
        action_counts = pd.Series(all_actions).value_counts().sort_index()
        action_counts.plot(kind='bar', color=['#9b59b6', '#3498db', '#1abc9c'])
        plt.title('Action Frequency Distribution')

        plt.tight_layout()
        plt.savefig('visualizations/episode_stats.png')
        plt.close()

    def render_episode(self, episode_idx, save_path=None):
        """
        渲染指定回合的动画

        动画包含：环境可视化、轨迹、当前动作、Q值信息

        参数:
            episode_idx: 回合编号
            save_path: 视频保存路径（可选）

        返回:
            动画对象
        """
        if episode_idx >= len(self.episode_data['episode']):
            print(f"Episode {episode_idx} not found in recorded data")
            return

        fig, ax = plt.subplots(figsize=(10, 10))
        env = self.env

        # 重置环境并获取记录数据
        env.reset()
        trajectory = self.episode_data['trajectory'][episode_idx]
        actions = self.episode_data['actions'][episode_idx]

        def update(frame):
            """动画更新函数"""
            ax.clear()

            # 更新小车位置并渲染环境
            env.car.x, env.car.y = trajectory[frame]
            env.render(ax=ax, step=frame)

            # 绘制轨迹线
            ax.plot(*zip(*trajectory[:frame + 1]), color='#9b59b6', linestyle='-', linewidth=2, alpha=0.7)

            # 显示当前动作
            action_text = ['Forward', 'Left', 'Right'][actions[frame]]
            ax.text(5, 20, f"Action: {action_text}", fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.7))

            # 显示各动作的Q值
            state = env._get_state()
            with torch.no_grad():
                q_values = self.agent.policy_net(state).numpy()
            q_text = "\n".join([f"Q{a}: {q:.2f}" for a, q in enumerate(q_values)])
            ax.text(5, 50, q_text, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7))

            return ax

        # 创建动画
        ani = animation.FuncAnimation(fig, update, frames=len(trajectory),
                                      interval=100, blit=False)

        if save_path:
            ani.save(save_path, writer='ffmpeg', fps=10, dpi=100)
            plt.close()
        else:
            plt.show()

        return ani


def evaluate(agent, num_episodes=20, render=True, save_video=False, visualizer=None):
    """
    增强版评估函数

    在评估过程中记录详细数据并生成增强可视化视频

    参数:
        agent: 待评估的智能体
        num_episodes: 评估回合数
        render: 是否实时渲染
        save_video: 是否保存视频
        visualizer: 可视化器实例（可选）

    返回:
        包含评估结果的字典
    """
    env = Environment()
    rewards = []
    successes = []
    lengths = []
    action_counts = [0, 0, 0]  # 记录各动作的使用次数

    if save_video:
        # 视频保存模式
        fig, ax = plt.subplots(figsize=(10, 10))
        writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Autonomous Car DQN'))

        with writer.saving(fig, 'videos/evaluation_enhanced.mp4', dpi=120):
            for episode in tqdm(range(num_episodes), desc="Evaluating Episodes"):
                state = env.reset()
                episode_reward = 0
                step = 0
                trajectory = []
                actions = []

                if visualizer:
                    visualizer.reset_recording()

                while True:
                    # 选择并执行动作
                    action = agent.select_action(state)
                    next_state, reward, done, _ = env.step(action)

                    # 记录数据
                    trajectory.append((env.car.x, env.car.y))
                    actions.append(action)
                    action_counts[action] += 1

                    if visualizer:
                        visualizer.record_step(env.car.x, env.car.y, action, reward)

                    # 增强渲染：环境 + 轨迹 + 动作指示
                    ax.clear()
                    env.render(ax=ax, step=step)

                    # 绘制轨迹
                    ax.plot(*zip(*trajectory), color='#9b59b6', linestyle='-',
                            linewidth=2, alpha=0.7, label='Trajectory')

                    # 绘制转向指示箭头
                    arrow_length = 15
                    if action == 1:  # 左转
                        ax.arrow(env.car.x, env.car.y,
                                 -arrow_length * math.sin(env.car.theta),
                                 arrow_length * math.cos(env.car.theta),
                                 head_width=5, head_length=7, fc='#e74c3c', ec='#c0392b')
                    elif action == 2:  # 右转
                        ax.arrow(env.car.x, env.car.y,
                                 arrow_length * math.sin(env.car.theta),
                                 -arrow_length * math.cos(env.car.theta),
                                 head_width=5, head_length=7, fc='#e74c3c', ec='#c0392b')

                    # 信息面板
                    info_text = (f"Episode: {episode + 1}\n"
                                 f"Step: {step}\n"
                                 f"Total Reward: {episode_reward:.1f}\n"
                                 f"Action: {['Forward', 'Left', 'Right'][action]}")
                    ax.text(5, ENV_HEIGHT - 80, info_text, fontsize=10,
                            bbox=dict(facecolor='white', alpha=0.7))

                    writer.grab_frame()

                    state = next_state
                    episode_reward += reward
                    step += 1

                    if done:
                        # 判断是否成功（奖励为500表示到达目标）
                        success = reward == 500
                        successes.append(success)
                        lengths.append(step)
                        if visualizer:
                            visualizer.save_episode_data(episode, success)
                        break

                rewards.append(episode_reward)
                print(f"Episode {episode + 1}: Reward={episode_reward:.1f}, "
                      f"Steps={step}, Success={'Yes' if success else 'No'}")

        plt.close(fig)
    else:
        # 非视频保存模式
        fig, ax = None, None
        if render:
            fig, ax = plt.subplots(figsize=(10, 10))

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            step = 0
            trajectory = []
            actions = []

            if visualizer:
                visualizer.reset_recording()

            while True:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)

                trajectory.append((env.car.x, env.car.y))
                actions.append(action)
                action_counts[action] += 1

                if visualizer:
                    visualizer.record_step(env.car.x, env.car.y, action, reward)

                if render:
                    ax.clear()
                    env.render(ax=ax, step=step)

                    # 绘制轨迹
                    ax.plot(*zip(*trajectory), color='#9b59b6', linestyle='-',
                            linewidth=2, alpha=0.7)

                    # 简化信息面板
                    ax.text(5, ENV_HEIGHT - 50,
                            f"Reward: {episode_reward:.1f}\nStep: {step}",
                            fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

                    plt.pause(0.01)

                state = next_state
                episode_reward += reward
                step += 1

                if done:
                    success = reward == 500
                    successes.append(success)
                    lengths.append(step)
                    if visualizer:
                        visualizer.save_episode_data(episode, success)
                    break

            rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward={episode_reward:.1f}, "
                  f"Steps={step}, Success={'Yes' if success else 'No'}")

        if fig is not None:
            plt.close(fig)

    # 打印评估总结
    print("\n=== Evaluation Summary ===")
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Success Rate: {np.mean(successes):.1%}")
    print(f"Average Steps: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print("Action Distribution:")
    for a, count in enumerate(action_counts):
        print(f"  {['Forward', 'Left', 'Right'][a]}: {count} ({count / sum(action_counts):.1%})")

    return {
        'rewards': rewards,
        'successes': successes,
        'lengths': lengths,
        'action_counts': action_counts
    }


def visualize_q_values_3d(agent, env):
    """
    3D Q值可视化

    在3D空间中展示不同位置下各动作的Q值分布
    可以直观理解智能体对不同位置的偏好

    参数:
        agent: 智能体实例
        env: 环境实例
    """
    from mpl_toolkits.mplot3d import Axes3D

    # 创建位置网格
    x_range = np.linspace(0, ENV_WIDTH, 15)
    y_range = np.linspace(0, ENV_HEIGHT, 15)
    theta_range = np.linspace(0, 2 * np.pi, 8)

    # 使用固定的传感器读数（假设无障碍物）
    sensor_readings = [SENSOR_RANGE] * NUM_SENSORS

    fig = plt.figure(figsize=(18, 12))

    for action in range(ACTION_SPACE):
        ax = fig.add_subplot(1, 3, action + 1, projection='3d')

        # 计算各位置的Q值
        q_values = np.zeros((len(x_range), len(y_range)))
        for i, x in enumerate(x_range):
            for j, y in enumerate(y_range):
                for k, theta in enumerate(theta_range):
                    # 设置小车位置和朝向
                    env.car.x, env.car.y, env.car.theta = x, y, theta

                    # 构造状态
                    state = torch.FloatTensor(sensor_readings + [
                        math.sqrt((x - env.goal_x) ** 2 + (y - env.goal_y) ** 2) / math.sqrt(
                            ENV_WIDTH ** 2 + ENV_HEIGHT ** 2),
                        (math.atan2(env.goal_y - y, env.goal_x - x) - theta) / np.pi
                    ])

                    # 计算Q值
                    with torch.no_grad():
                        q_values[i, j] += agent.policy_net(state)[action].item()

                # 平均不同角度的Q值
                q_values[i, j] /= len(theta_range)

        # 绘制3D表面图
        X, Y = np.meshgrid(x_range, y_range)
        surf = ax.plot_surface(X, Y, q_values.T, cmap=q_cmap, alpha=0.8)

        ax.set_title(f'Q-values for Action {["Forward", "Left", "Right"][action]}')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Q-value')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.savefig('visualizations/q_values_3d.png', dpi=120)
    plt.show()


def visualize_policy(agent, env):
    """
    策略可视化

    在2D环境地图上展示学习到的策略分布
    不同颜色表示不同位置下的最优动作

    参数:
        agent: 智能体实例
        env: 环境实例
    """
    x_range = np.linspace(0, ENV_WIDTH, 30)
    y_range = np.linspace(0, ENV_HEIGHT, 30)
    policy = np.zeros((len(x_range), len(y_range)))

    # 计算各位置的最优动作
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            env.car.x, env.car.y = x, y
            state = env._get_state()
            with torch.no_grad():
                actions = agent.policy_net(state)
                policy[i, j] = actions.argmax().item()

    # 绘制策略热力图
    plt.figure(figsize=(10, 8))
    plt.imshow(policy.T, origin='lower', cmap=car_cmap,
               extent=[0, ENV_WIDTH, 0, ENV_HEIGHT], alpha=0.6)

    # 绘制障碍物和目标
    for ox, oy in env.obstacles:
        plt.scatter(ox, oy, color='#e74c3c', s=OBSTACLE_RADIUS ** 2, alpha=0.7)
    plt.scatter(env.goal_x, env.goal_y, color='#2ecc71', s=GOAL_RADIUS ** 2, marker='*', label='Goal')

    # 添加动作图例
    action_labels = ['Forward', 'Left', 'Right']
    handles = [plt.Rectangle((0, 0), 1, 1, color=car_cmap(i / 2)) for i in range(3)]
    plt.legend(handles, action_labels, title='Optimal Action')

    plt.title('Learned Policy Visualization')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.colorbar(label='Action', ticks=[0, 1, 2])
    plt.grid(False)
    plt.savefig('visualizations/policy_map.png', dpi=120)
    plt.show()


def plot_training_progress(progress_data):
    """
    训练进度可视化

    绘制训练过程中的奖励、探索率、损失曲线

    参数:
        progress_data: 包含训练数据的字典
    """
    plt.figure(figsize=(15, 10))

    # 原始奖励曲线
    plt.subplot(2, 2, 1)
    plt.plot(progress_data['episode'], progress_data['reward'], color='#3498db')
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    # 平滑后的奖励曲线
    plt.subplot(2, 2, 2)
    smooth_window = max(1, len(progress_data['episode']) // 20)
    smooth_rewards = pd.Series(progress_data['reward']).rolling(smooth_window, min_periods=1).mean()
    plt.plot(progress_data['episode'], smooth_rewards, color='#9b59b6')
    plt.title(f'Smoothed Rewards (window={smooth_window})')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward')

    # 探索率曲线
    plt.subplot(2, 2, 3)
    plt.plot(progress_data['episode'], progress_data['epsilon'], color='#e74c3c')
    plt.title('Exploration Rate (ε)')
    plt.xlabel('Episode')
    plt.ylabel('ε')

    # 损失曲线
    plt.subplot(2, 2, 4)
    plt.plot(progress_data['episode'], progress_data['loss'], color='#2ecc71', alpha=0.7)
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('visualizations/training_progress.png', dpi=120)
    plt.show()


if __name__ == "__main__":
    # 初始化环境和智能体
    env = Environment()
    state_size = NUM_SENSORS + 2
    agent = Agent(state_size, ACTION_SPACE)
    visualizer = EnhancedVisualizer(env, agent)

    # 尝试加载预训练模型
    try:
        agent.load("models/best_model.pth")
        print("预训练模型已成功加载。")
    except:
        print("未找到预训练模型，将使用随机初始化的模型。")

    # 执行增强评估
    print("\n开始增强评估...")
    eval_results = evaluate(agent, num_episodes=5, render=True, save_video=True, visualizer=visualizer)

    # 生成可视化分析
    print("\n生成可视化分析...")
    visualizer.plot_episode_stats()
    visualize_q_values_3d(agent, env)
    visualize_policy(agent, env)

    # 渲染示例回合动画
    print("\n渲染示例episode...")
    visualizer.render_episode(0, save_path='videos/sample_episode.mp4')

    print("\n所有可视化结果已保存到 visualizations/ 目录")