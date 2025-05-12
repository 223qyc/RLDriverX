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

# 确保结果可以复现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 创建目录
os.makedirs("models", exist_ok=True)
os.makedirs("videos", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

# 自定义颜色映射
car_cmap = LinearSegmentedColormap.from_list('car_cmap', ['#FF6B6B', '#4ECDC4', '#45B7D1'])
q_cmap = LinearSegmentedColormap.from_list('q_cmap', ['#2C3E50', '#3498DB', '#1ABC9C'])


class EnhancedVisualizer:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.episode_data = defaultdict(list)
        self.current_trajectory = []
        self.current_actions = []
        self.current_rewards = []

    def reset_recording(self):
        self.current_trajectory = []
        self.current_actions = []
        self.current_rewards = []

    def record_step(self, x, y, action, reward):
        self.current_trajectory.append((x, y))
        self.current_actions.append(action)
        self.current_rewards.append(reward)

    def save_episode_data(self, episode, success):
        self.episode_data['episode'].append(episode)
        self.episode_data['trajectory'].append(self.current_trajectory)
        self.episode_data['actions'].append(self.current_actions)
        self.episode_data['rewards'].append(self.current_rewards)
        self.episode_data['success'].append(success)
        self.episode_data['length'].append(len(self.current_trajectory))
        self.episode_data['total_reward'].append(sum(self.current_rewards))

    def plot_episode_stats(self):
        df = pd.DataFrame(self.episode_data)

        plt.figure(figsize=(15, 10))

        # 成功率
        plt.subplot(2, 2, 1)
        success_rate = df['success'].mean()
        plt.bar(['Success', 'Failure'], [success_rate, 1 - success_rate], color=['#2ecc71', '#e74c3c'])
        plt.title(f'Success Rate ({success_rate:.1%})')

        # 轨迹长度分布
        plt.subplot(2, 2, 2)
        sns.histplot(df['length'], bins=20, kde=True, color='#3498db')
        plt.title('Trajectory Length Distribution')

        # 奖励分布
        plt.subplot(2, 2, 3)
        sns.boxplot(x='success', y='total_reward', data=df, palette=['#e74c3c', '#2ecc71'])
        plt.title('Reward Distribution by Outcome')

        # 动作频率
        plt.subplot(2, 2, 4)
        all_actions = [a for sublist in df['actions'] for a in sublist]
        action_counts = pd.Series(all_actions).value_counts().sort_index()
        action_counts.plot(kind='bar', color=['#9b59b6', '#3498db', '#1abc9c'])
        plt.title('Action Frequency Distribution')

        plt.tight_layout()
        plt.savefig('visualizations/episode_stats.png')
        plt.close()

    def render_episode(self, episode_idx, save_path=None):
        if episode_idx >= len(self.episode_data['episode']):
            print(f"Episode {episode_idx} not found in recorded data")
            return

        fig, ax = plt.subplots(figsize=(10, 10))
        env = self.env

        # 设置环境
        env.reset()
        trajectory = self.episode_data['trajectory'][episode_idx]
        actions = self.episode_data['actions'][episode_idx]

        def update(frame):
            ax.clear()

            # 绘制环境
            env.car.x, env.car.y = trajectory[frame]
            env.render(ax=ax, step=frame)

            # 绘制轨迹
            ax.plot(*zip(*trajectory[:frame + 1]), color='#9b59b6', linestyle='-', linewidth=2, alpha=0.7)

            # 添加动作信息
            action_text = ['Forward', 'Left', 'Right'][actions[frame]]
            ax.text(5, 20, f"Action: {action_text}", fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.7))

            # 添加Q值信息
            state = env._get_state()
            with torch.no_grad():
                q_values = self.agent.policy_net(state).numpy()
            q_text = "\n".join([f"Q{a}: {q:.2f}" for a, q in enumerate(q_values)])
            ax.text(5, 50, q_text, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7))

            return ax

        ani = animation.FuncAnimation(fig, update, frames=len(trajectory),
                                      interval=100, blit=False)

        if save_path:
            ani.save(save_path, writer='ffmpeg', fps=10, dpi=100)
            plt.close()
        else:
            plt.show()

        return ani


def evaluate(agent, num_episodes=20, render=True, save_video=False, visualizer=None):
    env = Environment()
    rewards = []
    successes = []
    lengths = []
    action_counts = [0, 0, 0]

    if save_video:
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
                    action = agent.select_action(state)
                    next_state, reward, done, _ = env.step(action)

                    trajectory.append((env.car.x, env.car.y))
                    actions.append(action)
                    action_counts[action] += 1

                    if visualizer:
                        visualizer.record_step(env.car.x, env.car.y, action, reward)

                    # 高级渲染
                    ax.clear()
                    env.render(ax=ax, step=step)

                    # 绘制轨迹
                    ax.plot(*zip(*trajectory), color='#9b59b6', linestyle='-',
                            linewidth=2, alpha=0.7, label='Trajectory')

                    # 绘制动作方向
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

                    # 添加信息面板
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
                        success = reward == 500  # 是否到达目标
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

                    # 添加信息
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
    """3D Q值可视化"""
    from mpl_toolkits.mplot3d import Axes3D

    # 创建网格
    x_range = np.linspace(0, ENV_WIDTH, 15)
    y_range = np.linspace(0, ENV_HEIGHT, 15)
    theta_range = np.linspace(0, 2 * np.pi, 8)

    # 选择一个固定的传感器读数
    sensor_readings = [SENSOR_RANGE] * NUM_SENSORS

    fig = plt.figure(figsize=(18, 12))

    for action in range(ACTION_SPACE):
        ax = fig.add_subplot(1, 3, action + 1, projection='3d')

        # 计算Q值
        q_values = np.zeros((len(x_range), len(y_range)))
        for i, x in enumerate(x_range):
            for j, y in enumerate(y_range):
                for k, theta in enumerate(theta_range):
                    env.car.x, env.car.y, env.car.theta = x, y, theta
                    state = torch.FloatTensor(sensor_readings + [
                        math.sqrt((x - env.goal_x) ** 2 + (y - env.goal_y) ** 2) / math.sqrt(
                            ENV_WIDTH ** 2 + ENV_HEIGHT ** 2),
                        (math.atan2(env.goal_y - y, env.goal_x - x) - theta) / np.pi
                    ])
                    with torch.no_grad():
                        q_values[i, j] += agent.policy_net(state)[action].item()
                q_values[i, j] /= len(theta_range)  # 平均不同角度的Q值

        # 3D表面图
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
    """策略可视化"""
    x_range = np.linspace(0, ENV_WIDTH, 30)
    y_range = np.linspace(0, ENV_HEIGHT, 30)
    policy = np.zeros((len(x_range), len(y_range)))

    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            env.car.x, env.car.y = x, y
            state = env._get_state()
            with torch.no_grad():
                actions = agent.policy_net(state)
                policy[i, j] = actions.argmax().item()

    plt.figure(figsize=(10, 8))
    plt.imshow(policy.T, origin='lower', cmap=car_cmap,
               extent=[0, ENV_WIDTH, 0, ENV_HEIGHT], alpha=0.6)

    # 绘制障碍物和目标
    for ox, oy in env.obstacles:
        plt.scatter(ox, oy, color='#e74c3c', s=OBSTACLE_RADIUS ** 2, alpha=0.7)
    plt.scatter(env.goal_x, env.goal_y, color='#2ecc71', s=GOAL_RADIUS ** 2, marker='*', label='Goal')

    # 添加图例
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
    """训练进度可视化"""
    plt.figure(figsize=(15, 10))

    # 奖励曲线
    plt.subplot(2, 2, 1)
    plt.plot(progress_data['episode'], progress_data['reward'], color='#3498db')
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    # 平滑后的奖励
    plt.subplot(2, 2, 2)
    smooth_window = max(1, len(progress_data['episode']) // 20)
    smooth_rewards = pd.Series(progress_data['reward']).rolling(smooth_window, min_periods=1).mean()
    plt.plot(progress_data['episode'], smooth_rewards, color='#9b59b6')
    plt.title(f'Smoothed Rewards (window={smooth_window})')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward')

    # 探索率
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

    # 加载预训练模型
    try:
        agent.load("models/best_model.pth")
        print("预训练模型已成功加载。")
    except:
        print("未找到预训练模型，将使用随机初始化的模型。")

    # 评估智能体
    print("\n开始增强评估...")
    eval_results = evaluate(agent, num_episodes=5, render=True, save_video=True, visualizer=visualizer)

    # 可视化分析
    print("\n生成可视化分析...")
    visualizer.plot_episode_stats()
    visualize_q_values_3d(agent, env)
    visualize_policy(agent, env)

    # 示例：渲染特定episode
    print("\n渲染示例episode...")
    visualizer.render_episode(0, save_path='videos/sample_episode.mp4')

    print("\n所有可视化结果已保存到 visualizations/ 目录")